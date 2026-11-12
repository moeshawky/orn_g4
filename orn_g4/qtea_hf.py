"""First-party Hugging Face tooling for Azrael QTEA models.

Provides end-to-end tooling for:
1. Remote Hugging Face repository auditing without weight downloads.
2. Direct download of inference-only QTEA payloads (excluding conversion checkpoints).
3. Lossless repacking of per-projection tensors into indexed SafeTensors shards.
4. Bitwise tensor verification preserving dtypes (FP8, uint8, FP32, BF16).
5. Safe publication to a staging branch (never touching main).
6. Complete Ornith model loading from empty skeleton + QTEA shards.
7. Inference benchmarking with separated load and inference memory accounting.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .errors import (
    QTEABenchmarkError,
    QTEAError,
    QTEALoadError,
    QTEAPackError,
    QTEAPublishError,
    QTEASourceError,
    QTEAVerifyError,
)
from .utils import sha256_file

DEFAULT_REPO = "moeshawky/Azrael-QTEA-9B"
DEFAULT_SOURCE_BRANCH = "main"
DEFAULT_STAGING_BRANCH = "qtea-sharded-v2"
DEFAULT_SHARD_MIB = 512

ALLOW_PATTERNS: List[str] = [
    "manifest.json",
    "encoded/*.safetensors",
    "preserved/preserved.safetensors",
    "assets/**",
    "ternary_runtime.py",
    "load_qtea_package.py",
    "provenance_lock.json",
    "README.md",
    "LICENSE",
]

DELETE_OLD_PATTERNS: List[str] = [
    "encoded/",
    "checkpoints/",
    "preserved/",
    "manifest.json",
    "*.tar.gz",
    "assets/model.safetensors.index.json",
]


def format_bytes(n: int) -> str:
    """Format bytes into human-readable GiB string."""
    return f"{n / (1024 ** 3):.3f} GiB"


def tensor_bytes_sha256(tensor: Any) -> str:
    """Compute bitwise SHA-256 of raw tensor memory without precision or dtype conversion.

    Works for float8_e4m3fn, uint8, int8, float32, bfloat16, float16, etc.

    # Errors: raises TypeError if tensor is not a torch.Tensor.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")
    if tensor.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def get_module_parent(model: Any, name: str) -> Tuple[Any, str]:
    """Retrieve parent module and attribute name for a dotted path."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _matches_pattern(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern)


def _extract_tied_tensors(
    manifest: Optional[Dict[str, Any]],
    sharded_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Extract tied tensor mapping from qtea-v1 and/or qtea-sharded-v2 manifests."""
    tied: Dict[str, str] = {}
    if isinstance(manifest, dict):
        top_tied = manifest.get("tied_tensors")
        if isinstance(top_tied, dict):
            tied.update({str(k): str(v) for k, v in top_tied.items()})
        pres = manifest.get("preserved")
        if isinstance(pres, dict) and isinstance(pres.get("tied_tensors"), dict):
            tied.update({str(k): str(v) for k, v in pres["tied_tensors"].items()})
    if isinstance(sharded_meta, dict) and isinstance(sharded_meta.get("source_tied_tensors"), dict):
        tied.update({str(k): str(v) for k, v in sharded_meta["source_tied_tensors"].items()})
    return tied


def _validate_projection_tensors(
    proj_name: str,
    entry: Dict[str, Any],
    tensors: Dict[str, Any],
    error_cls: Any = QTEASourceError,
) -> None:
    """Validate QTEA projection buffer presence, dtypes, and geometry against manifest entry."""
    import torch

    required_fields = {
        "ternary",
        "col_v",
        "group_alpha",
        "group_beta",
        "salient_cols",
        "residual_index",
        "residual_scale",
    }
    missing = required_fields - set(tensors.keys())
    if missing:
        raise error_cls(f"Projection {proj_name} is missing required QTEA buffers: {sorted(missing)}")

    rows = int(entry["rows"])
    cols = int(entry["cols"])
    group_size = int(entry["group_size"])
    residual_group = int(entry["residual_group"])
    has_bias = bool(entry.get("has_bias", False))

    if has_bias and "bias" not in tensors:
        raise error_cls(f"Projection {proj_name} declares has_bias=True but 'bias' tensor is missing")
    if not has_bias and "bias" in tensors:
        raise error_cls(f"Projection {proj_name} declares has_bias=False but 'bias' tensor is present")
    if has_bias and tuple(tensors["bias"].shape) != (rows,):
        raise error_cls(
            f"Projection {proj_name} bias shape mismatch: got {tuple(tensors['bias'].shape)}, expected {(rows,)}"
        )

    groups = (cols + group_size - 1) // group_size
    packed_per_group = (group_size + 4) // 5
    packed_cols = groups * packed_per_group

    expected_fixed = {
        "ternary": (rows, packed_cols),
        "col_v": (cols,),
        "group_alpha": (rows, groups),
        "group_beta": (rows, groups),
    }
    for fname, exp_shape in expected_fixed.items():
        actual_shape = tuple(tensors[fname].shape)
        if actual_shape != exp_shape:
            raise error_cls(
                f"Projection {proj_name}.{fname} shape mismatch: got {actual_shape}, expected {exp_shape}"
            )
    if tensors["ternary"].dtype != torch.uint8:
        raise error_cls(f"Projection {proj_name}.ternary must have dtype uint8, got {tensors['ternary'].dtype}")

    salient_cols = tensors["salient_cols"]
    if salient_cols.ndim != 1:
        raise error_cls(f"Projection {proj_name}.salient_cols must be 1D, got shape {tuple(salient_cols.shape)}")
    k = int(salient_cols.numel())
    if tuple(tensors["residual_scale"].shape) != (k,):
        raise error_cls(
            f"Projection {proj_name}.residual_scale shape mismatch: got {tuple(tensors['residual_scale'].shape)}, expected {(k,)}"
        )

    if k > 0:
        num_r_groups = rows // residual_group
        packed_idx_cols = (num_r_groups + 3) // 4
        if tuple(tensors["residual_index"].shape) != (k, packed_idx_cols):
            raise error_cls(
                f"Projection {proj_name}.residual_index shape mismatch: "
                f"got {tuple(tensors['residual_index'].shape)}, expected {(k, packed_idx_cols)}"
            )
        if "residual_value" not in tensors:
            raise error_cls(f"Projection {proj_name} has {k} salient columns but 'residual_value' is missing")
        if tuple(tensors["residual_value"].shape) != (k, num_r_groups):
            raise error_cls(
                f"Projection {proj_name}.residual_value shape mismatch: "
                f"got {tuple(tensors['residual_value'].shape)}, expected {(k, num_r_groups)}"
            )
    else:
        if "residual_value" in tensors and tensors["residual_value"].numel() != 0:
            raise error_cls(
                f"Projection {proj_name} has 0 salient columns but non-empty residual_value {tuple(tensors['residual_value'].shape)}"
            )


# ==============================================================================
# 1. Remote Hugging Face Auditing
# ==============================================================================


def get_hf_api(token: Optional[str] = None) -> Any:
    """Instantiate HfApi with lazy import.

    # Errors: raises QTEASourceError if huggingface_hub is unavailable.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token)
    except ImportError as e:
        raise QTEASourceError("huggingface_hub package is required for HF Hub operations") from e


def fetch_remote_inventory(api: Any, repo: str, revision: str) -> Tuple[Any, Dict[str, int]]:
    """Fetch file inventory and sizes for a remote model repo.

    # Errors: raises QTEASourceError if repo or revision is not accessible.
    """
    try:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = {x.rfilename: int(x.size or 0) for x in info.siblings}
        return info, files
    except Exception as e:
        raise QTEASourceError(f"Failed to query remote inventory for {repo}@{revision}: {e}") from e


def fetch_manifest_from_hub(
    repo: str,
    revision: str,
    token: Optional[str] = None,
    filename: str = "manifest.json",
) -> Tuple[Dict[str, Any], Path]:
    """Download a JSON manifest file from Hub into local cache and parse it.

    # Errors: raises QTEASourceError if download or parsing fails.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                repo_type="model",
                token=token,
            )
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, path
    except Exception as e:
        raise QTEASourceError(f"Failed to fetch {filename} from {repo}@{revision}: {e}") from e


def audit_remote_qtea(
    repo: str = DEFAULT_REPO,
    revision: str = DEFAULT_SOURCE_BRANCH,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit remote Hugging Face QTEA repository without downloading weight files.

    Supports both unsharded `qtea-v1` (e.g. `main`) and sharded `qtea-sharded-v2`
    revisions. Inspects file counts, byte sizes by category, projection completeness,
    preserved tensor assets, duplicate conversion checkpoint storage, and
    expected download payload.

    Args:
        repo: Hugging Face model repository ID.
        revision: branch, tag, or commit SHA.
        token: optional Hugging Face API token.

    Returns:
        Structured audit report dictionary.

    # Errors: raises QTEASourceError if required QTEA payload files are missing.
    """
    api = get_hf_api(token)
    info, files = fetch_remote_inventory(api, repo, revision)
    total_bytes = sum(files.values())

    if "qtea-sharded-manifest.json" in files and "manifest.json" not in files:
        sharded_meta, _ = fetch_manifest_from_hub(
            repo, info.sha, token, filename="qtea-sharded-manifest.json"
        )
        src_manifest, _ = fetch_manifest_from_hub(
            repo, info.sha, token, filename="source-manifest.json"
        )
        encoded = sharded_meta.get("source_encoded") or src_manifest.get("encoded") or {}
        completeness = src_manifest.get("completeness") or {}
        expected_projections = completeness.get("expected_count", 248)
        shards = sharded_meta.get("shards") or []

        required_files = [
            "qtea-sharded-manifest.json",
            "source-manifest.json",
            "ternary_runtime.py",
            "load_qtea_package.py",
            "assets/config.json",
        ] + [s["file"] for s in shards if "file" in s]
        missing_files = [p for p in required_files if p not in files]
        stale_remnants = [
            p
            for p in files
            if p.startswith(("encoded/", "checkpoints/", "preserved/"))
            or p == "manifest.json"
            or p == "assets/model.safetensors.index.json"
        ]
        if missing_files or stale_remnants:
            raise QTEASourceError(
                f"Remote sharded repository {repo}@{revision} failed audit: "
                f"missing={missing_files[:10]}, stale_remnants={stale_remnants[:10]}"
            )
        shard_bytes = sum(files.get(s["file"], 0) for s in shards if "file" in s)
        return {
            "repo": repo,
            "format": "qtea-sharded-v2",
            "source_revision": info.sha,
            "requested_revision": revision,
            "visibility": getattr(info, "private", None),
            "file_count": len(files),
            "total_bytes": total_bytes,
            "total_size": format_bytes(total_bytes),
            "shard_count": len(shards),
            "shard_bytes": shard_bytes,
            "shard_size": format_bytes(shard_bytes),
            "tensor_count": sharded_meta.get("tensor_count"),
            "projection_count": len(encoded),
            "expected_projection_count": expected_projections,
            "conversion_complete": completeness.get("conversion_complete", False),
            "quality": sharded_meta.get("source_quality") or src_manifest.get("quality", {}),
            "outcome": sharded_meta.get("source_outcome") or src_manifest.get("outcome"),
            "missing_required_files": [],
        }

    manifest, _ = fetch_manifest_from_hub(repo, info.sha, token)

    encoded = manifest.get("encoded") or {}
    completeness = manifest.get("completeness") or {}
    expected_projections = completeness.get("expected_count", 248)

    required_files = [
        "manifest.json",
        "preserved/preserved.safetensors",
        "ternary_runtime.py",
        "load_qtea_package.py",
        "assets/config.json",
    ]
    required_files += [f"encoded/{v['file']}" for v in encoded.values() if "file" in v]
    missing_files = [p for p in required_files if p not in files]

    groups: Dict[str, Dict[str, int]] = {
        k: {"files": 0, "bytes": 0} for k in ("encoded", "checkpoints", "preserved", "assets", "other")
    }
    for name, size in files.items():
        head = name.split("/", 1)[0]
        group = head if head in groups else "other"
        groups[group]["files"] += 1
        groups[group]["bytes"] += size

    selected = [p for p in files if any(_matches_pattern(p, pat) for pat in ALLOW_PATTERNS)]
    selected_bytes = sum(files[p] for p in selected)

    unclassified_large = [
        {"path": p, "bytes": s, "size": format_bytes(s)}
        for p, s in sorted(files.items(), key=lambda z: -z[1])
        if s > 32 * 1024 * 1024
        and not any(_matches_pattern(p, pat) for pat in ALLOW_PATTERNS)
        and not any(_matches_pattern(p, pat) for pat in DELETE_OLD_PATTERNS)
    ]

    report = {
        "repo": repo,
        "format": manifest.get("format", "qtea-v1"),
        "source_revision": info.sha,
        "requested_revision": revision,
        "visibility": getattr(info, "private", None),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_size": format_bytes(total_bytes),
        "groups": groups,
        "download_allow_patterns": ALLOW_PATTERNS,
        "selected_file_count": len(selected),
        "selected_download_bytes": selected_bytes,
        "selected_download_size": format_bytes(selected_bytes),
        "projection_count": len(encoded),
        "expected_projection_count": expected_projections,
        "conversion_complete": completeness.get("conversion_complete", False),
        "quality": manifest.get("quality", {}),
        "outcome": manifest.get("outcome"),
        "missing_required_files": missing_files,
        "large_unclassified_files": unclassified_large[:50],
    }

    if missing_files:
        raise QTEASourceError(
            f"Remote repository {repo}@{revision} is missing {len(missing_files)} required QTEA files: "
            f"{missing_files[:10]}"
        )

    return report


# ==============================================================================
# 2. Source Validation & Lossless Sharded Repacking
# ==============================================================================


def validate_qtea_source(root: Union[Path, str]) -> Dict[str, Any]:
    """Validate a downloaded QTEA candidate directory before repacking.

    Verifies format, projection count, completeness flag, checksums of
    preserved tensors, individual projection files, and configuration assets.

    Args:
        root: path to downloaded source directory.

    Returns:
        Loaded source manifest dictionary.

    # Errors: raises QTEASourceError if any required component is missing or corrupt.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise QTEASourceError(f"Downloaded source manifest missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise QTEASourceError(f"Corrupt manifest JSON at {manifest_path}: {e}") from e

    if manifest.get("format") != "qtea-v1":
        raise QTEASourceError(f"Unsupported source format: {manifest.get('format')!r}; expected 'qtea-v1'")

    encoded = manifest.get("encoded") or {}
    completeness = manifest.get("completeness") or {}
    if not completeness.get("conversion_complete"):
        raise QTEASourceError("Source manifest does not declare a complete conversion")

    expected_count = completeness.get("expected_count")
    if expected_count is not None and len(encoded) != expected_count:
        raise QTEASourceError(
            f"Projection count mismatch: manifest has {len(encoded)} entries, expected {expected_count}"
        )

    preserved_file = root / "preserved" / "preserved.safetensors"
    expected_pres_sha = (manifest.get("preserved") or {}).get("preserved.safetensors", {}).get("sha256")
    if not preserved_file.is_file():
        raise QTEASourceError(f"Preserved tensor file missing: {preserved_file}")
    if expected_pres_sha and sha256_file(preserved_file) != expected_pres_sha:
        raise QTEASourceError(
            f"Preserved tensor SHA-256 mismatch: {sha256_file(preserved_file)} != {expected_pres_sha}"
        )

    for module_name, entry in encoded.items():
        fname = entry.get("file")
        if not fname:
            raise QTEASourceError(f"Malformed encoded entry for module {module_name}: missing file")
        proj_path = root / "encoded" / fname
        if not proj_path.is_file():
            raise QTEASourceError(f"Encoded projection file missing for {module_name}: {proj_path}")
        expected_sha = entry.get("sha256")
        if expected_sha and sha256_file(proj_path) != expected_sha:
            raise QTEASourceError(f"Encoded projection SHA-256 mismatch for {module_name}: {proj_path}")

    for required_name in ("ternary_runtime.py", "load_qtea_package.py"):
        if not (root / required_name).is_file():
            raise QTEASourceError(f"Required runtime file missing: {root / required_name}")

    if not (root / "assets" / "config.json").is_file():
        raise QTEASourceError(f"assets/config.json missing; cannot instantiate model skeleton at {root}")

    return manifest


def pack_qtea_shards(
    src: Union[Path, str],
    out: Union[Path, str],
    shard_mib: int = DEFAULT_SHARD_MIB,
    repo: str = DEFAULT_REPO,
) -> Dict[str, Any]:
    """Losslessly repack individual QTEA projections into indexed SafeTensors shards.

    Consolidates 248 individual projection files and preserved tensors into
    a clean set of indexed SafeTensors shards. Verifies bitwise tensor values
    and dtypes (including float8_e4m3fn residuals). Staged through an atomic
    temporary directory.

    Args:
        src: validated source directory containing qtea-v1 layout.
        out: destination directory for the sharded qtea-sharded-v2 package.
        shard_mib: target shard size in MiB (64 to 8192).
        repo: repository identifier for provenance metadata.

    Returns:
        Structured packaging report dictionary.

    # Errors: raises QTEAPackError on parameter conflicts, duplicate tensor keys,
    corrupted tensors, or bitwise verification mismatch.
    """
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as e:
        raise QTEAPackError("torch and safetensors are required for QTEA shard packing") from e

    if shard_mib < 64 or shard_mib > 8192:
        raise QTEAPackError(f"shard_mib must be between 64 and 8192; got {shard_mib}")

    src = Path(src).resolve()
    out = Path(out).resolve()

    manifest = validate_qtea_source(src)

    if out.exists():
        raise QTEAPackError(f"Output directory already exists; refusing overwrite: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".incomplete")
    if tmp.exists():
        raise QTEAPackError(f"Incomplete staging directory exists; inspect before deleting: {tmp}")
    tmp.mkdir(parents=True)

    target_bytes = shard_mib * 1024 * 1024
    shards: List[Dict[str, Any]] = []
    weight_map: Dict[str, str] = {}
    tensor_hashes: Dict[str, str] = {}
    current_shard: Dict[str, torch.Tensor] = {}
    current_size = 0
    shard_no = 0
    seen_keys: Set[str] = set()

    def flush_shard() -> None:
        nonlocal current_shard, current_size, shard_no
        if not current_shard:
            return
        shard_no += 1
        shard_name = f"qtea-{shard_no:04d}.safetensors"
        shard_path = tmp / shard_name
        save_file(current_shard, str(shard_path), metadata={"format": "qtea-shard-v2"})
        for k in current_shard:
            weight_map[k] = shard_name
        shard_bytes = shard_path.stat().st_size
        shards.append(
            {
                "file": shard_name,
                "sha256": sha256_file(shard_path),
                "bytes": shard_bytes,
                "tensor_count": len(current_shard),
            }
        )
        current_shard = {}
        current_size = 0

    def add_tensor(key: str, t: torch.Tensor) -> None:
        nonlocal current_size
        if key in seen_keys:
            raise QTEAPackError(f"Duplicate tensor key encountered: {key}")
        t = t.detach().contiguous().cpu()
        nbytes = t.numel() * t.element_size()
        if current_shard and (current_size + nbytes > target_bytes):
            flush_shard()
        current_shard[key] = t
        current_size += nbytes
        seen_keys.add(key)
        tensor_hashes[key] = tensor_bytes_sha256(t)

    # 1. Pack individual projection files with module-prefixed tensor keys.
    for module_name, entry in sorted(manifest["encoded"].items()):
        path = src / "encoded" / entry["file"]
        with safe_open(str(path), framework="pt", device="cpu") as f:
            proj_tensors = {field_name: f.get_tensor(field_name) for field_name in sorted(f.keys())}
        _validate_projection_tensors(module_name, entry, proj_tensors, error_cls=QTEAPackError)
        for field_name, t in proj_tensors.items():
            add_tensor(f"{module_name}.{field_name}", t)

    # 2. Pack preserved tensors.
    preserved_path = src / "preserved" / "preserved.safetensors"
    with safe_open(str(preserved_path), framework="pt", device="cpu") as f:
        for preserved_key in sorted(f.keys()):
            add_tensor(preserved_key, f.get_tensor(preserved_key))

    flush_shard()

    # 3. Copy inference-time assets; exclude stale BF16 shard index & shards.
    assets_src = src / "assets"
    shutil.copytree(
        assets_src,
        tmp / "assets",
        ignore=shutil.ignore_patterns("model.safetensors.index.json", "model-*.safetensors"),
    )

    # 4. Copy runtime files, provenance lock, and license; upgrade standalone loader for qtea-sharded-v2.
    for name in ("ternary_runtime.py", "provenance_lock.json", "LICENSE"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, tmp / name)

    from .export import _STANDALONE_LOADER_SOURCE, _TERNARY_RUNTIME_SOURCE

    if not (tmp / "ternary_runtime.py").is_file():
        (tmp / "ternary_runtime.py").write_text(_TERNARY_RUNTIME_SOURCE, encoding="utf-8")
    (tmp / "load_qtea_package.py").write_text(_STANDALONE_LOADER_SOURCE, encoding="utf-8")

    source_manifest_path = src / "manifest.json"
    shutil.copy2(source_manifest_path, tmp / "source-manifest.json")

    tied_tensors = _extract_tied_tensors(manifest)
    sharded_meta = {
        "format": "qtea-sharded-v2",
        "source_repo": repo,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_outcome": manifest.get("outcome"),
        "source_quality": manifest.get("quality"),
        "source_projection_count": len(manifest["encoded"]),
        "source_encoded": manifest["encoded"],
        "source_tied_tensors": tied_tensors,
        "weight_map": weight_map,
        "tensor_sha256": tensor_hashes,
        "shards": shards,
        "tensor_count": len(weight_map),
        "excluded_from_inference": [
            "checkpoints/*",
            "original per-projection file duplication",
            "assets/model.safetensors.index.json",
        ],
        "full_model_reload_verified": False,
        "teacher_logit_parity_verified": False,
        "packed_edge_kernel_verified": False,
    }
    sharded_manifest_file = tmp / "qtea-sharded-manifest.json"
    sharded_manifest_file.write_text(
        json.dumps(sharded_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp / "COMMITTED").write_text(sha256_file(sharded_manifest_file) + "\n", encoding="utf-8")

    # 5. Model card notice preservation.
    old_readme = src / "README.md"
    card = (
        old_readme.read_text(encoding="utf-8", errors="replace")
        if old_readme.is_file()
        else "---\nlicense: mit\n---\n# Azrael QTEA 9B\n"
    )
    card += "\n\n## Sharded inference package (qtea-sharded-v2)\n\n"
    card += (
        "This revision uses indexed `qtea-sharded-v2` SafeTensors shards. "
        "The original QTEA per-projection payloads and redundant conversion checkpoints are excluded from this branch.\n\n"
        "**Status:** experimental; quality outcome remains `QUALITY_REJECTED_RECOVERY_REQUIRED`. "
        "Projection reconstruction was verified at export; full-model reload, teacher-logit parity, "
        "and optimized packed-kernel inference must be tested independently.\n\n"
        "Load with `orn_g4.qtea_hf.load_qtea_full_model`; this is not a standard Hugging Face `from_pretrained()` checkpoint.\n"
    )
    (tmp / "README.md").write_text(card, encoding="utf-8")

    # 6. Bitwise verify every written tensor across shards.
    expected_keys = set(weight_map.keys())
    checked_keys: Set[str] = set()
    for shard in shards:
        path = tmp / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise QTEAPackError(f"Shard file hash mismatch after save: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in expected_keys or weight_map[key] != shard["file"]:
                    raise QTEAPackError(f"Unexpected or misrouted tensor in shard {shard['file']}: {key}")
                t = f.get_tensor(key)
                if tensor_bytes_sha256(t) != tensor_hashes[key]:
                    raise QTEAPackError(f"Bitwise tensor hash mismatch after repack for {key}")
                checked_keys.add(key)

    if checked_keys != expected_keys:
        missing = expected_keys - checked_keys
        extra = checked_keys - expected_keys
        raise QTEAPackError(
            f"Tensor inventory mismatch after repack: missing={len(missing)}, extra={len(extra)}"
        )

    # Atomic move
    os.replace(tmp, out)

    total_payload = sum(s["bytes"] for s in shards)
    return {
        "output": str(out),
        "shards": len(shards),
        "tensor_count": len(weight_map),
        "payload_bytes": total_payload,
        "payload_size": format_bytes(total_payload),
        "source_manifest_sha256": sharded_meta["source_manifest_sha256"],
        "quality_outcome": sharded_meta["source_outcome"],
        "full_model_reload_verified": False,
    }


def verify_qtea_shards(
    out: Union[Path, str],
    source_manifest: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Verify integrity of a packed qtea-sharded-v2 package against its manifests.

    Args:
        out: path to sharded package directory.
        source_manifest: optional path to original source manifest.json for comparison.

    Returns:
        Structured verification report dictionary.

    # Errors: raises QTEAVerifyError on manifest corruption, hash mismatch, or missing files.
    """
    out = Path(out).resolve()
    sharded_manifest_path = out / "qtea-sharded-manifest.json"
    bundled_source_manifest = out / "source-manifest.json"

    if not sharded_manifest_path.is_file():
        raise QTEAVerifyError(f"Sharded manifest missing: {sharded_manifest_path}")
    if not bundled_source_manifest.is_file():
        raise QTEAVerifyError(f"Bundled source manifest missing: {bundled_source_manifest}")

    try:
        meta = json.loads(sharded_manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise QTEAVerifyError(f"Corrupt sharded manifest JSON: {e}") from e

    if meta.get("format") != "qtea-sharded-v2":
        raise QTEAVerifyError(f"Unsupported format: {meta.get('format')!r}; expected 'qtea-sharded-v2'")

    bundled_src_sha = sha256_file(bundled_source_manifest)
    if meta.get("source_manifest_sha256") != bundled_src_sha:
        raise QTEAVerifyError("Bundled source-manifest.json hash does not match qtea-sharded-manifest record")

    if source_manifest is not None:
        source_manifest_path = Path(source_manifest).resolve()
        if not source_manifest_path.is_file():
            raise QTEAVerifyError(f"Provided external source manifest missing: {source_manifest_path}")
        if sha256_file(source_manifest_path) != bundled_src_sha:
            raise QTEAVerifyError("External source manifest does not match packed package source manifest")

    try:
        from safetensors import safe_open
    except ImportError as e:
        raise QTEAVerifyError("safetensors is required for QTEA shard verification") from e

    try:
        src_manifest_data = json.loads(bundled_source_manifest.read_text(encoding="utf-8"))
    except Exception as e:
        raise QTEAVerifyError(f"Corrupt bundled source-manifest.json: {e}") from e

    committed_file = out / "COMMITTED"
    if committed_file.is_file():
        if committed_file.read_text(encoding="utf-8").strip() != sha256_file(sharded_manifest_path):
            raise QTEAVerifyError("COMMITTED marker hash does not match qtea-sharded-manifest.json")

    shards = meta.get("shards") or []
    if not shards:
        raise QTEAVerifyError("Sharded manifest lists no shards")

    weight_map = meta.get("weight_map") or {}
    tensor_hashes = meta.get("tensor_sha256") or {}
    expected_keys = set(weight_map.keys())
    if len(expected_keys) != meta.get("tensor_count"):
        raise QTEAVerifyError(
            f"weight_map count ({len(expected_keys)}) does not match tensor_count ({meta.get('tensor_count')})"
        )

    total_bytes = 0
    seen_keys: Set[str] = set()
    all_tensors: Dict[str, Any] = {}

    for shard in shards:
        fname = shard.get("file")
        if not fname:
            raise QTEAVerifyError("Malformed shard record without 'file' field")
        shard_path = out / fname
        if not shard_path.is_file():
            raise QTEAVerifyError(f"Packed shard file missing: {shard_path}")
        real_sha = sha256_file(shard_path)
        if real_sha != shard.get("sha256"):
            raise QTEAVerifyError(f"Corrupt shard file {fname}: hash {real_sha} != {shard.get('sha256')}")
        total_bytes += shard_path.stat().st_size

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            shard_keys = list(f.keys())
            if shard.get("tensor_count") is not None and len(shard_keys) != shard["tensor_count"]:
                raise QTEAVerifyError(
                    f"Shard {fname} tensor count mismatch: got {len(shard_keys)}, expected {shard['tensor_count']}"
                )
            for key in shard_keys:
                if key in seen_keys:
                    raise QTEAVerifyError(f"Duplicate tensor key across shards: {key}")
                if weight_map.get(key) != fname:
                    raise QTEAVerifyError(
                        f"Shard index mismatch for {key}: found in {fname}, weight_map says {weight_map.get(key)!r}"
                    )
                t = f.get_tensor(key)
                expected_thash = tensor_hashes.get(key)
                if not expected_thash or tensor_bytes_sha256(t) != expected_thash:
                    raise QTEAVerifyError(f"Bitwise tensor SHA-256 mismatch in shard {fname} for {key}")
                seen_keys.add(key)
                all_tensors[key] = t

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)[:10]
        extra = sorted(seen_keys - expected_keys)[:10]
        raise QTEAVerifyError(f"Tensor inventory mismatch against weight_map: missing={missing}, extra={extra}")

    # Verify full encoded projection inventory and preserved tensor inventory against source-manifest.json
    encoded_map = src_manifest_data.get("encoded") or {}
    proj_prefixes: Set[str] = set()
    for proj_name, entry in encoded_map.items():
        prefix = f"{proj_name}."
        proj_prefixes.add(prefix)
        proj_tensors = {
            k[len(prefix):]: v
            for k, v in all_tensors.items()
            if k.startswith(prefix)
        }
        if not proj_tensors:
            raise QTEAVerifyError(f"Missing all tensors for encoded projection {proj_name}")
        _validate_projection_tensors(proj_name, entry, proj_tensors, error_cls=QTEAVerifyError)

    preserved_keys = [
        k for k in all_tensors if not any(k.startswith(p) for p in proj_prefixes)
    ]
    expected_pres_count = (
        (src_manifest_data.get("preserved") or {})
        .get("preserved.safetensors", {})
        .get("count")
    )
    if expected_pres_count is not None and len(preserved_keys) != int(expected_pres_count):
        raise QTEAVerifyError(
            f"Preserved tensor count mismatch: found {len(preserved_keys)}, expected {expected_pres_count}"
        )
    if not preserved_keys:
        raise QTEAVerifyError("No preserved non-projection tensors found in sharded package")

    # Verify assets/config.json exists and stale BF16 index is excluded
    if not (out / "assets" / "config.json").is_file():
        raise QTEAVerifyError("assets/config.json missing from packed package")
    if (out / "assets" / "model.safetensors.index.json").exists():
        raise QTEAVerifyError("Stale BF16 assets/model.safetensors.index.json present in packed package")

    return {
        "output": str(out),
        "shards": len(shards),
        "tensor_count": len(seen_keys),
        "projection_count": len(encoded_map),
        "preserved_tensor_count": len(preserved_keys),
        "payload_bytes": total_bytes,
        "payload_size": format_bytes(total_bytes),
        "source_manifest_sha256": meta["source_manifest_sha256"],
        "quality_outcome": meta.get("source_outcome"),
        "full_model_reload_verified": meta.get("full_model_reload_verified", False),
        "verified": True,
    }


# ==============================================================================
# 3. Direct Download & Staged Hub Publication
# ==============================================================================


def publish_qtea_sharded(
    repo: str = DEFAULT_REPO,
    source_revision: str = DEFAULT_SOURCE_BRANCH,
    branch: str = DEFAULT_STAGING_BRANCH,
    workdir: Optional[Union[Path, str]] = None,
    shard_mib: int = DEFAULT_SHARD_MIB,
    download_workers: int = 8,
    resume: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Download from Hugging Face, repack locally, and upload to a staging branch.

    Safety invariants:
    - Never modifies the source 'main' branch.
    - Fails closed if token is missing or authenticated user is not repo owner.
    - Verifies available ephemeral disk space before downloading.
    - Deletes per-projection files and duplicate checkpoints on the staging branch.
    - Performs remote inventory and manifest verification after upload.

    # Errors: raises QTEAPublishError or QTEASourceError on failure.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise QTEAPublishError("huggingface_hub is required for publication") from e

    # 0. Reject publishing to main or source revision
    if branch == "main" or branch == source_revision:
        raise QTEAPublishError(f"Refusing to publish to protected source branch {branch!r}")

    api = get_hf_api(token)

    # 1. Authenticate identity
    try:
        who = api.whoami()
    except Exception as e:
        raise QTEAPublishError(f"Failed to authenticate with Hugging Face Hub (token missing or invalid): {e}") from e

    username = (who or {}).get("name") or (who or {}).get("user")
    owner = repo.split("/", 1)[0]
    if not username or username.lower() != owner.lower():
        raise QTEAPublishError(
            f"Authenticated Hugging Face identity {username!r} does not match repo owner {owner!r}"
        )

    # 2. Pin source commit SHA
    info, remote_files = fetch_remote_inventory(api, repo, source_revision)
    source_sha = info.sha
    remote_manifest, _ = fetch_manifest_from_hub(repo, source_sha, token)

    # 3. Disk space check
    selected = [p for p in remote_files if any(_matches_pattern(p, pat) for pat in ALLOW_PATTERNS)]
    download_bytes = sum(remote_files[p] for p in selected)
    reserve = 2 * 1024 ** 3
    if workdir is None:
        workdir = Path(tempfile.gettempdir()) / "qtea-hf-work"
    try:
        work = Path(workdir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise QTEAPublishError(f"Failed to create scratch workdir {workdir!r}: {e}") from e
    free = shutil.disk_usage(work).free
    needed = download_bytes * 2 + reserve
    if free < needed:
        raise QTEAPublishError(
            f"Insufficient local scratch space: free={format_bytes(free)}, required={format_bytes(needed)}"
        )

    source_dir = work / f"source-{source_sha[:12]}"
    packed_dir = work / f"packed-{branch}"

    # 4. Download inference payload
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        revision=source_sha,
        local_dir=str(source_dir),
        allow_patterns=ALLOW_PATTERNS,
        token=token,
        max_workers=download_workers,
    )
    validate_qtea_source(source_dir)

    # 5. Repack or reuse
    if packed_dir.exists():
        if not resume:
            raise QTEAPackError(
                f"Packed output already exists at {packed_dir}. Use resume=True to verify and reuse it."
            )
        pack_report = verify_qtea_shards(packed_dir, source_dir / "manifest.json")
    else:
        pack_report = pack_qtea_shards(source_dir, packed_dir, shard_mib=shard_mib, repo=repo)

    # 6. Manage staging branch (NEVER touch main)
    if branch == "main" or branch == source_revision:
        raise QTEAPublishError(f"Refusing to publish to protected source branch {branch!r}")

    refs = api.list_repo_refs(repo, repo_type="model", token=token)
    branch_exists = any(ref.name == branch for ref in refs.branches)

    if branch_exists:
        if not resume:
            raise QTEAPublishError(f"Branch {branch!r} already exists. Pass resume=True only if it belongs to this run.")
        branch_info, branch_files = fetch_remote_inventory(api, repo, branch)
        if "source-manifest.json" in branch_files:
            remote_src_manifest = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename="source-manifest.json",
                    revision=branch,
                    repo_type="model",
                    token=token,
                )
            )
            if sha256_file(remote_src_manifest) != sha256_file(source_dir / "manifest.json"):
                raise QTEAPublishError(
                    "Existing staging branch belongs to a different source manifest; refusing resume"
                )
        elif branch_info.sha != source_sha:
            raise QTEAPublishError(
                "Existing staging branch advanced but lacks source-manifest.json; refusing resume"
            )
        branch_parent = branch_info.sha
    else:
        api.create_branch(repo_id=repo, branch=branch, revision=source_sha, repo_type="model", token=token)
        branch_parent = source_sha

    # 7. Upload sharded package & clean old per-projection files
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(packed_dir),
        revision=branch,
        token=token,
        commit_message="Stage losslessly sharded QTEA inference package (qtea-sharded-v2)",
        commit_description=(
            "Replace per-projection files with indexed SafeTensors shards. "
            "Exclude redundant conversion checkpoints. Main branch remains unchanged."
        ),
        delete_patterns=DELETE_OLD_PATTERNS,
        parent_commit=branch_parent,
    )

    # 8. Remote verification of published branch
    branch_info, published_remote_files = fetch_remote_inventory(api, repo, branch)
    expected_remote = {
        p.relative_to(packed_dir).as_posix(): p.stat().st_size for p in packed_dir.rglob("*") if p.is_file()
    }
    missing_remote = sorted(set(expected_remote) - set(published_remote_files))
    size_mismatches = {
        p: {"local": n, "remote": published_remote_files.get(p)}
        for p, n in expected_remote.items()
        if p in published_remote_files and published_remote_files[p] != n
    }
    old_remnants = [
        p
        for p in published_remote_files
        if p.startswith(("encoded/", "checkpoints/", "preserved/"))
        or p == "manifest.json"
        or p.endswith(".tar.gz")
    ]
    if missing_remote or size_mismatches or old_remnants:
        raise QTEAPublishError(
            f"Remote staging branch failed verification: missing={missing_remote[:10]}, "
            f"size_mismatch={size_mismatches}, old_remnants={old_remnants[:10]}"
        )

    remote_sharded_manifest = Path(
        hf_hub_download(
            repo_id=repo,
            filename="qtea-sharded-manifest.json",
            revision=branch,
            repo_type="model",
            token=token,
        )
    )
    if sha256_file(remote_sharded_manifest) != sha256_file(packed_dir / "qtea-sharded-manifest.json"):
        raise QTEAPublishError("Remote sharded manifest hash does not match local manifest hash")

    report = {
        "repo": repo,
        "branch": branch,
        "source_revision": source_sha,
        "published_revision": branch_info.sha,
        "commit_url": getattr(commit, "commit_url", None),
        "local_pack": pack_report,
        "remote_files": len(published_remote_files),
        "remote_inventory_matches": True,
        "remote_manifest_sha256_matches": True,
        "remote_shard_sizes_match": True,
        "main_branch_modified": False,
    }
    (packed_dir / "publication-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# ==============================================================================
# 4. Complete-Model Loading (qtea-v1 or qtea-sharded-v2)
# ==============================================================================


def load_qtea_full_model(
    root: Union[Path, str],
    device: Union[str, Any] = "cpu",
    trust_remote_code: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load and reconstruct complete Ornith multimodal model from QTEA package.

    Instantiates the model skeleton on meta device using config.json (avoiding
    any BF16 model download), replaces 248 projections with QTEALinear, assigns
    weights, handles tied aliases and empty residual buffers, and restores
    Qwen3.5 non-persistent RoPE buffers.

    Args:
        root: path to local QTEA candidate or sharded directory.
        device: target device ('cpu', 'cuda', 'cuda:0', etc.).
        trust_remote_code: Hugging Face trust_remote_code flag.

    Returns:
        tuple of ``(model, tokenizer, metadata)``.

    # Errors: raises QTEALoadError if model skeleton or weights fail reconstruction.
    """
    try:
        import contextlib
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer
    except ImportError as e:
        raise QTEALoadError("torch, transformers, and safetensors are required to load QTEA model") from e

    @contextlib.contextmanager
    def _init_empty():
        try:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=True):
                yield
        except ImportError:
            with torch.device("meta"):
                yield

    from .loader import restore_qwen3_5_nonpersistent_buffers
    from .qtea import QTEALinear

    root = Path(root).resolve()
    sharded_manifest_path = root / "qtea-sharded-manifest.json"
    is_sharded = sharded_manifest_path.is_file()

    if is_sharded:
        sharded_meta = json.loads(sharded_manifest_path.read_text(encoding="utf-8"))
        if sharded_meta.get("format") != "qtea-sharded-v2":
            raise QTEALoadError(f"Unsupported sharded format: {sharded_meta.get('format')!r}")
        src_manifest_path = root / "source-manifest.json"
        if not src_manifest_path.is_file():
            raise QTEALoadError(f"source-manifest.json missing in sharded package: {root}")
        if sha256_file(src_manifest_path) != sharded_meta["source_manifest_sha256"]:
            raise QTEALoadError("source-manifest.json hash mismatch against qtea-sharded-manifest")
        original_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
        encoded = sharded_meta["source_encoded"]
    else:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise QTEALoadError(f"manifest.json missing in QTEA package: {root}")
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if original_manifest.get("format") != "qtea-v1":
            raise QTEALoadError(f"Unsupported format: {original_manifest.get('format')!r}")
        encoded = original_manifest.get("encoded") or {}
        sharded_meta = None

    if not original_manifest.get("completeness", {}).get("conversion_complete"):
        raise QTEALoadError("Package does not declare a complete conversion")

    expected_count = original_manifest.get("completeness", {}).get("expected_count")
    if expected_count is not None and len(encoded) != expected_count:
        raise QTEALoadError(f"Projection count mismatch: {len(encoded)} vs expected {expected_count}")

    assets_dir = root / "assets"
    config_path = assets_dir / "config.json"
    if not config_path.is_file():
        raise QTEALoadError(f"assets/config.json missing at {assets_dir}")

    cfg = AutoConfig.from_pretrained(str(assets_dir), trust_remote_code=trust_remote_code)

    # 1. Instantiate skeleton with empty weights in bfloat16
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with _init_empty():
            model = AutoModelForMultimodalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    finally:
        torch.set_default_dtype(prev_dtype)

    # 2. Replace projections with QTEALinear on meta device
    for mod_name, entry in encoded.items():
        try:
            parent, attr = get_module_parent(model, mod_name)
        except AttributeError as e:
            raise QTEALoadError(f"Target module path does not exist in model architecture: {mod_name}") from e

        if not hasattr(parent, attr):
            raise QTEALoadError(f"Missing target projection in model skeleton: {mod_name}")

        q_unit = QTEALinear(
            in_features=int(entry["cols"]),
            out_features=int(entry["rows"]),
            group_size=int(entry["group_size"]),
            residual_group=int(entry["residual_group"]),
            bias=bool(entry.get("has_bias", False)),
            device="meta",
        )
        setattr(parent, attr, q_unit)

    # 3. Assign weights into state_dict with strict shape validation
    expected_keys = set(model.state_dict().keys())
    loaded_keys: Set[str] = set()
    dynamic_buffer_names = {"salient_cols", "residual_index", "residual_value", "residual_scale"}

    def assign_weights(weights: Dict[str, torch.Tensor]) -> None:
        nonlocal loaded_keys
        if not weights:
            return
        extra = set(weights.keys()) - expected_keys
        if extra:
            raise QTEALoadError(f"Unexpected weights in checkpoint: {sorted(extra)[:10]}")
        repeated = set(weights.keys()) & loaded_keys
        if repeated:
            raise QTEALoadError(f"Duplicate checkpoint keys encountered: {sorted(repeated)[:10]}")

        for k, v in weights.items():
            parent, attr = get_module_parent(model, k)
            if attr in parent._parameters:
                existing_param = parent._parameters[attr]
                if existing_param is not None and tuple(v.shape) != tuple(existing_param.shape):
                    raise QTEALoadError(
                        f"Parameter shape mismatch for {k}: checkpoint has {tuple(v.shape)}, "
                        f"model expects {tuple(existing_param.shape)}"
                    )
                req = existing_param.requires_grad if existing_param is not None else False
                parent._parameters[attr] = torch.nn.Parameter(v, requires_grad=req)
            elif attr in parent._buffers:
                existing_buf = parent._buffers[attr]
                if (
                    existing_buf is not None
                    and attr not in dynamic_buffer_names
                    and tuple(v.shape) != tuple(existing_buf.shape)
                ):
                    raise QTEALoadError(
                        f"Buffer shape mismatch for {k}: checkpoint has {tuple(v.shape)}, "
                        f"model expects {tuple(existing_buf.shape)}"
                    )
                parent._buffers[attr] = v
            else:
                setattr(parent, attr, v)
        loaded_keys |= set(weights.keys())

    if is_sharded:
        for shard_entry in sharded_meta["shards"]:
            p = root / shard_entry["file"]
            if sha256_file(p) != shard_entry["sha256"]:
                raise QTEALoadError(f"Corrupt shard file: {p}")
            assign_weights(load_file(str(p), device="cpu"))
    else:
        pres_p = root / "preserved" / "preserved.safetensors"
        pres_expected_sha = original_manifest["preserved"]["preserved.safetensors"]["sha256"]
        if sha256_file(pres_p) != pres_expected_sha:
            raise QTEALoadError(f"Corrupt preserved safetensors: {pres_p}")
        assign_weights(load_file(str(pres_p), device="cpu"))
        for mod_name, entry in encoded.items():
            proj_p = root / "encoded" / entry["file"]
            if sha256_file(proj_p) != entry["sha256"]:
                raise QTEALoadError(f"Corrupt encoded file: {proj_p}")
            data = load_file(str(proj_p), device="cpu")
            assign_weights({f"{mod_name}.{k}": v for k, v in data.items()})

    # 4. Handle legitimate empty residual buffers and tied tensors
    tied_tensors: Dict[str, str] = _extract_tied_tensors(original_manifest, sharded_meta)
    missing_keys = expected_keys - loaded_keys - set(tied_tensors.keys())

    for k in list(missing_keys):
        if k.endswith(".residual_value"):
            try:
                parent, attr = get_module_parent(model, k)
                buf = getattr(parent, attr, None)
                salient = getattr(parent, "salient_cols", None)
                if (
                    buf is not None
                    and getattr(buf, "is_meta", False)
                    and buf.numel() == 0
                    and salient is not None
                    and salient.numel() == 0
                ):
                    parent._buffers[attr] = torch.empty(buf.shape, dtype=buf.dtype, device="cpu")
                    missing_keys.remove(k)
            except Exception:
                pass

    # Support implicit lm_head <-> embed_tokens weight tying when not explicitly listed in tied_tensors
    if "lm_head.weight" in missing_keys:
        for candidate_embed in (
            "model.language_model.embed_tokens.weight",
            "model.language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ):
            if candidate_embed in loaded_keys:
                tied_tensors["lm_head.weight"] = candidate_embed
                missing_keys.remove("lm_head.weight")
                break

    if missing_keys:
        raise QTEALoadError(
            f"Model state incomplete, {len(missing_keys)} missing keys: {sorted(missing_keys)[:10]}"
        )

    # Validate each QTEALinear projection's loaded buffers for internal consistency
    for mod_name, entry in encoded.items():
        parent, attr = get_module_parent(model, mod_name)
        q_mod = getattr(parent, attr)
        proj_dict = {
            bname: getattr(q_mod, bname)
            for bname in (
                "ternary",
                "col_v",
                "group_alpha",
                "group_beta",
                "salient_cols",
                "residual_index",
                "residual_value",
                "residual_scale",
            )
            if getattr(q_mod, bname, None) is not None
        }
        if getattr(q_mod, "bias", None) is not None:
            proj_dict["bias"] = q_mod.bias
        _validate_projection_tensors(mod_name, entry, proj_dict, error_cls=QTEALoadError)

    # 5. Restore tied tensor aliases
    for alias, canonical in tied_tensors.items():
        if canonical not in loaded_keys:
            raise QTEALoadError(f"Cannot restore tied tensor {alias}: canonical tensor {canonical} was not loaded")
        try:
            p1, a1 = get_module_parent(model, alias)
            p2, a2 = get_module_parent(model, canonical)
            src_tensor = getattr(p2, a2)
            dst_tensor = getattr(p1, a1, None)
            if dst_tensor is not None and tuple(dst_tensor.shape) != tuple(src_tensor.shape):
                raise QTEALoadError(
                    f"Tied tensor shape mismatch between {alias} {tuple(dst_tensor.shape)} "
                    f"and {canonical} {tuple(src_tensor.shape)}"
                )
            setattr(p1, a1, src_tensor)
        except QTEALoadError:
            raise
        except Exception as e:
            raise QTEALoadError(f"Failed to restore tied tensor alias {alias} -> {canonical}: {e}") from e

    # 6. Reconstruct CPU buffers for non-persistent RoPE buffers
    for _, m in model.named_modules():
        for name in getattr(m, "_non_persistent_buffers_set", set()):
            b = m._buffers.get(name)
            if b is not None and getattr(b, "is_meta", False):
                m._buffers[name] = torch.empty(b.shape, dtype=b.dtype, device="cpu")

    restored_rotary = restore_qwen3_5_nonpersistent_buffers(model)

    # 7. Assert no meta tensors remain
    still_meta_params = [k for k, v in model.named_parameters() if getattr(v, "is_meta", False)]
    still_meta_buffers = [k for k, v in model.named_buffers() if getattr(v, "is_meta", False)]
    if still_meta_params or still_meta_buffers:
        raise QTEALoadError(
            f"Meta tensors remain after loading: params={still_meta_params[:5]}, buffers={still_meta_buffers[:5]}"
        )

    # 8. Device placement & eval
    target_device = torch.device(device)
    model.eval()
    model.to(target_device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(assets_dir), trust_remote_code=trust_remote_code)
    except Exception as e:
        raise QTEALoadError(f"Failed to load tokenizer from {assets_dir}: {e}") from e

    metadata = {
        "projections": len(encoded),
        "restored_rotary_buffers": len(restored_rotary),
        "outcome": original_manifest.get("outcome"),
        "sharded": is_sharded,
    }
    return model, tokenizer, metadata


# ==============================================================================
# 5. Inference Benchmarking & Memory Accounting
# ==============================================================================


def benchmark_qtea_inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Explain briefly why the sky is blue.",
    device: Union[str, Any] = "cpu",
    max_new_tokens: int = 8,
    details: Optional[Dict[str, Any]] = None,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    load_seconds: Optional[float] = None,
    load_peak_memory_gib: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute smoke forward pass and generation benchmark with memory accounting.

    Records load peak memory separately from inference peak memory. Explicitly
    labels that the benchmark runs under QTEALinear's reference forward path
    (materializing effective dense weights) and does not measure packed-kernel
    acceleration.

    Args:
        model: instantiated Ornith QTEA model.
        tokenizer: loaded tokenizer.
        prompt: benchmark input prompt.
        device: execution device.
        max_new_tokens: number of new tokens to generate.
        details: optional loading metadata dictionary.
        repo: repository identifier.
        revision: repository revision.
        load_seconds: recorded model load duration in seconds.
        load_peak_memory_gib: recorded peak memory during loading.

    Returns:
        Structured benchmark report dictionary.

    # Errors: raises QTEABenchmarkError if forward pass or generation fails.
    """
    try:
        import torch
    except ImportError as e:
        raise QTEABenchmarkError("torch is required for running benchmarks") from e

    dev_str = str(device)
    is_cuda = dev_str.startswith("cuda") and torch.cuda.is_available()

    resident_gib = round(torch.cuda.memory_allocated() / (1024 ** 3), 3) if is_cuda else None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    try:
        raw_inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(raw_inputs, "to"):
            inputs = raw_inputs.to(dev_str)
        elif isinstance(raw_inputs, dict):
            inputs = {k: v.to(dev_str) if hasattr(v, "to") else v for k, v in raw_inputs.items()}
        else:
            inputs = raw_inputs
    except Exception as e:
        raise QTEABenchmarkError(f"Tokenization failed: {e}") from e

    # Forward pass
    with torch.inference_mode():
        if is_cuda:
            torch.cuda.synchronize()
        start_fwd = time.perf_counter()
        try:
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits
        except Exception as e:
            raise QTEABenchmarkError(f"Forward pass failed: {e}") from e

        if is_cuda:
            torch.cuda.synchronize()
        fwd_seconds = time.perf_counter() - start_fwd

        if logits.ndim != 3 or not torch.isfinite(logits).all():
            raise QTEABenchmarkError("Invalid logits produced by forward pass (non-finite or non-3D)")
        logits_shape = list(logits.shape)
        del logits
        del outputs

        # Generation
        if is_cuda:
            torch.cuda.synchronize()
        start_gen = time.perf_counter()
        try:
            pad_id = getattr(tokenizer, "eos_token_id", None)
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        except Exception as e:
            raise QTEABenchmarkError(f"Generation failed: {e}") from e

        if is_cuda:
            torch.cuda.synchronize()
        gen_seconds = time.perf_counter() - start_gen

    num_generated = int(gen_out.shape[-1] - inputs["input_ids"].shape[-1])
    tokens_per_sec = round(num_generated / gen_seconds, 3) if gen_seconds > 0 else None
    decoded_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)

    inference_peak_gib = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if is_cuda else None

    report = {
        "status": "FULL_MODEL_GENERATION_COMPLETED",
        "source_repo": repo,
        "source_revision": revision,
        **(details or {}),
        "device": dev_str,
        "precision": "as serialized (FP8 residuals + continuous scales + ternary)",
        "load_seconds": round(load_seconds, 3) if load_seconds is not None else None,
        "load_peak_memory_gib": load_peak_memory_gib,
        "resident_vram_gib": resident_gib,
        "physical_peak_memory_gib": inference_peak_gib,
        "inference_peak_memory_gib": inference_peak_gib,
        "inference_peak_gib": inference_peak_gib,
        "forward_seconds": round(fwd_seconds, 3),
        "generation_seconds": round(gen_seconds, 3),
        "generated_tokens": num_generated,
        "generation_tokens_per_second": tokens_per_sec,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "logits_shape": logits_shape,
        "output_text": decoded_text,
        "full_model_reload_verified": True,
        "teacher_logit_parity_verified": False,
        "packed_kernel_benchmark": False,
        "inference_notes": (
            "QTEALinear reference forward path materializes effective dense weights during execution; "
            "this benchmark does not measure packed-kernel acceleration. "
            "Single-prompt generation smoke benchmark tests model executable completeness, "
            "not statistical teacher logit parity."
        ),
    }
    return report


def run_qtea_benchmark(
    source: Optional[Union[Path, str]] = None,
    repo: Optional[str] = None,
    revision: str = DEFAULT_STAGING_BRANCH,
    cache_dir: Optional[Union[Path, str]] = None,
    device: str = "cuda:0",
    prompt: str = "Explain briefly why the sky is blue.",
    max_new_tokens: int = 8,
    output: Optional[Union[Path, str]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end benchmark from either local directory or Hugging Face.

    # Errors: raises QTEABenchmarkError or QTEALoadError on failure.
    """
    try:
        import torch
    except ImportError as e:
        raise QTEABenchmarkError("PyTorch is required for benchmarking") from e

    if bool(source) == bool(repo):
        raise QTEABenchmarkError("Provide exactly one of 'source' (local dir) or 'repo' (Hub model ID)")

    if repo:
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "qtea-hf-benchmark"
        try:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    repo_id=repo,
                    repo_type="model",
                    revision=revision,
                    local_dir=str(cache_dir),
                    allow_patterns=[
                        "qtea-*.safetensors",
                        "qtea-sharded-manifest.json",
                        "source-manifest.json",
                        "manifest.json",
                        "encoded/*.safetensors",
                        "preserved/preserved.safetensors",
                        "assets/**",
                        "ternary_runtime.py",
                        "load_qtea_package.py",
                        "provenance_lock.json",
                        "README.md",
                    ],
                    token=token,
                    max_workers=8,
                )
            )
        except Exception as e:
            raise QTEASourceError(f"Failed downloading benchmark package from {repo}@{revision}: {e}") from e
    else:
        root = Path(source)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QTEABenchmarkError(f"CUDA device requested ({device}) but CUDA is not available on this host")

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model, tokenizer, details = load_qtea_full_model(root, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0
    load_peak_gib = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if device.startswith("cuda") else None

    report = benchmark_qtea_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        details=details,
        repo=repo,
        revision=revision,
        load_seconds=load_seconds,
        load_peak_memory_gib=load_peak_gib,
    )

    if output:
        out_path = Path(output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


__all__ = [
    "DEFAULT_REPO",
    "DEFAULT_SOURCE_BRANCH",
    "DEFAULT_STAGING_BRANCH",
    "DEFAULT_SHARD_MIB",
    "ALLOW_PATTERNS",
    "DELETE_OLD_PATTERNS",
    "format_bytes",
    "tensor_bytes_sha256",
    "get_module_parent",
    "get_hf_api",
    "fetch_remote_inventory",
    "fetch_manifest_from_hub",
    "audit_remote_qtea",
    "validate_qtea_source",
    "pack_qtea_shards",
    "verify_qtea_shards",
    "publish_qtea_sharded",
    "load_qtea_full_model",
    "benchmark_qtea_inference",
    "run_qtea_benchmark",
]
