"""Inspect the immutable Azrael MVP training subset without changing dataset files.

Archive-wide counts include the separate persona shard; frozen MVP files under
train_ready/mvp/ are the canonical behavioral training mixture.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from .data import DatasetLayout, ShardRef, _discover_arrays_in_shard, load_manifest, locate_dataset
from .errors import DatasetError
from .utils import sha256_file

_HASH = re.compile(r"(?:sha256:)?([a-fA-F0-9]{64})$")


def _digest(value: Any) -> str | None:
    match = _HASH.fullmatch(value.strip()) if isinstance(value, str) else None
    return match.group(1).lower() if match else None


def _declared_checksums(root: Path) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = {}

    def put(path: str, digest: Any) -> None:
        digest = _digest(digest)
        path = str(path).replace(chr(92), "/").removeprefix("./").lstrip("/")
        if digest is not None and path:
            hashes.setdefault(path, set()).add(digest)

    def walk(obj: Any) -> None:
        if isinstance(obj, list):
            for element in obj:
                walk(element)
        elif isinstance(obj, dict):
            path = next((obj[k] for k in ("path", "file", "filename", "name", "relative_path")
                         if isinstance(obj.get(k), str)), None)
            digest = next((_digest(obj.get(k)) for k in
                           ("sha256", "checksum", "hash", "file_sha256")
                           if _digest(obj.get(k)) is not None), None)
            if path and digest:
                put(path, digest)
            for k, value in obj.items():
                if isinstance(value, (list, dict)):
                    walk(value)
                elif isinstance(value, str) and _digest(value) and k not in (
                    "sha256", "checksum", "hash", "file_sha256"
                ):
                    put(k, value)

    for name in ("DATASET_RELEASE_MANIFEST.json", "manifests/manifest.json",
                 "train_ready/mvp/frozen_manifest.json"):
        path = root / name
        if path.is_file():
            walk(json.loads(path.read_text(encoding="utf-8")))

    sidecar = root / "manifests" / "shard_hashes.sha256"
    if sidecar.is_file():
        for number, line in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^(?:sha256:)?([a-fA-F0-9]{64})\s+\*?(.+)$", line)
            if match:
                put(match.group(2).strip(), match.group(1))
                continue
            match = re.match(r"^(.+?)\s+(?:sha256:)?([a-fA-F0-9]{64})$", line)
            if match:
                put(match.group(1).strip(), match.group(2))
                continue
            raise DatasetError(f"invalid checksum line {sidecar}:{number}: {line[:100]}")
    return hashes


def _lookup_sha(file: Path, root: Path, hashes: dict[str, set[str]]) -> str:
    rel = file.relative_to(root).as_posix()
    candidates = set(hashes.get(rel, set()))
    for path, values in hashes.items():
        if path == file.name or path.endswith("/" + rel):
            candidates.update(values)
    if not candidates:
        raise DatasetError(f"missing frozen shard checksum in existing manifests: {rel}")
    if len(candidates) != 1:
        raise DatasetError(f"conflicting checksum records for {rel}: {sorted(candidates)}")
    return next(iter(candidates))


def build_frozen_mvp_layout(
    source: str | Path,
    *,
    revision: str | None = None,
    local_dir: str | Path = "/tmp/ornith-dataset",
    exclude_dirs: list[str] | None = None,
) -> DatasetLayout:
    """Load the seven-shard frozen MVP directory, excluding persona and eval."""
    root = Path(locate_dataset(source, local_dir=local_dir, revision=revision,
                               allow_drive_fallback=False)).resolve()
    if not root.is_dir():
        raise DatasetError(f"expected unpacked frozen dataset, got {root}")
    if not (root / "DATASET_RELEASE_MANIFEST.json").is_file():
        nested = root / "azrael-dataset"
        if (nested / "DATASET_RELEASE_MANIFEST.json").is_file():
            root = nested
    manifest = load_manifest(root)
    directory = root / "train_ready" / "mvp"
    if not directory.is_dir():
        raise DatasetError(f"frozen training subset missing: {directory}")
    files = sorted(directory.glob("*_frozen.parquet"))
    if not files:
        raise DatasetError(f"no *_frozen.parquet files found under {directory}")
    if sorted(directory.glob("*.parquet")) != files:
        raise DatasetError("unexpected non-frozen Parquet files in MVP training directory")
    hashes = _declared_checksums(root)
    if not hashes:
        raise DatasetError("no frozen shard SHA-256 proofs available")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError("pyarrow is required to validate frozen shards") from exc

    required = {"id", "source", "input_ids", "attention_mask", "labels",
                "token_count", "supervised_count"}
    shards: list[ShardRef] = []
    for path in files:
        expected = _lookup_sha(path, root, hashes)
        observed = sha256_file(path).removeprefix("sha256:").lower()
        if observed != expected:
            raise DatasetError(f"frozen shard SHA-256 mismatch: {path.name} "
                               f"expected={expected} actual={observed}")
        columns = _discover_arrays_in_shard(path)
        missing = required.difference(columns)
        if missing:
            raise DatasetError(f"frozen shard {path.name} missing columns {sorted(missing)}")
        rows = pq.ParquetFile(str(path)).metadata.num_rows
        shards.append(ShardRef(name=path.name, path=path,
                               checksum="sha256:" + expected, arrays=columns,
                               num_examples=rows, checksum_ok=True))
    return DatasetLayout(
        zip_path=None, extract_dir=root, dataset_root=root, manifest=manifest,
        train_ready_dir=root / "train_ready", shards=shards,
        eval_firewall_dir=(root / "eval_firewall") if (root / "eval_firewall").is_dir() else None,
        excluded_dirs=sorted(set(exclude_dirs or []) | {"eval_firewall", "persona"}),
    )


def verify_frozen_template(dataset_root: str | Path, model_template_sha256: str) -> str:
    """Check frozen Jinja bytes against model and template.sha256 when present."""
    root = Path(dataset_root)
    template = root / "template" / "chat_template.jinja"
    if not template.is_file():
        raise DatasetError(f"frozen dataset template missing: {template}")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    expected = model_template_sha256.removeprefix("sha256:").lower()
    if digest != expected:
        raise DatasetError(f"model/dataset chat template mismatch: {digest} != {expected}")
    sidecar = root / "template" / "template.sha256"
    if sidecar.is_file():
        content = sidecar.read_text(encoding="utf-8").strip().splitlines()
        declared = _digest(content[0].split()[0]) if content else None
        if declared is None or declared != digest:
            raise DatasetError(f"frozen template checksum mismatch in {sidecar}")
    return digest
