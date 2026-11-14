"""Independent conventional 4-bit (Q4) fallback from the merged BF16 SFT teacher.

Provides a direct fallback quantization, export, and standalone loading path
that does NOT depend on QTEA ternary conversion or QTEA recovery. Converts all
discovered Ornith language projections (expected 248) to ``Protected4BitLinear``
(``act_sym_int4_g128``) while preserving embeddings, norms, vision modules, MTP,
and output head.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .errors import ExportError, QTEAError
from .export import (
    export_mixed_precision_package,
    load_mixed_precision_package,
    verify_mixed_precision_package,
)
from .quant4 import Protected4BitLinear
from .qtea import (
    MixedPrecisionPlan,
    _set_submodule_by_name,
    discover_mixed_precision_targets,
)

Q4_FALLBACK_FORMAT = "azrael-q4-fallback-v1"


def build_q4_fallback_model(
    teacher_model: Any,
    *,
    group_size: int = 128,
    expected_qtea: int = 112,
    expected_int4: int = 88,
    expected_total: int = 248,
    strict: bool = True,
    inplace: bool = False,
) -> Tuple[Any, Dict[str, Dict[str, Any]], MixedPrecisionPlan]:
    """Convert all discovered language projections directly to 4-bit from BF16 teacher.

    Independent of QTEA conversion and recovery: every target projection in
    ``plan.targets`` (248 for full Ornith-1.5-9B) is quantized with
    :class:`Protected4BitLinear` while non-target modules remain untouched.

    # Errors: raises QTEAError if PyTorch is unavailable or architecture
    validation fails under ``strict=True``.
    """
    if not _TORCH_AVAILABLE:
        raise QTEAError("PyTorch is required for build_q4_fallback_model")

    work_model = teacher_model if inplace else copy.deepcopy(teacher_model)
    plan = discover_mixed_precision_targets(
        work_model,
        expected_qtea=expected_qtea,
        expected_int4=expected_int4,
        expected_total=expected_total,
        strict=strict,
    )

    q4_packed: Dict[str, Dict[str, Any]] = {}
    for target in plan.targets:
        parent = work_model
        for p in target.name.split(".")[:-1]:
            parent = getattr(parent, p)
        src_mod = getattr(parent, target.name.split(".")[-1])
        if not isinstance(src_mod, nn.Linear):
            raise QTEAError(
                f"Fallback target {target.name!r} is {type(src_mod).__name__}, expected nn.Linear"
            )
        q4_mod = Protected4BitLinear.from_linear(src_mod, group_size=group_size)
        q4_packed[target.name] = q4_mod.to_layer_dict()
        _set_submodule_by_name(work_model, target.name, q4_mod)

    return work_model, q4_packed, plan


def export_q4_fallback_package(
    q4_model: Any,
    output_dir: Path | str,
    *,
    plan: Optional[MixedPrecisionPlan] = None,
    assets_dir: Optional[Path | str] = None,
    shard_max_bytes: Optional[int] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export standalone Q4 fallback package without BF16 teacher dependency.

    # Errors: raises ExportError if serialization fails.
    """
    return export_mixed_precision_package(
        q4_model,
        output_dir,
        plan=plan,
        assets_dir=assets_dir,
        shard_max_bytes=shard_max_bytes,
        provenance=provenance,
        recovery_status="q4_fallback_no_qtea_recovery",
        format_name=Q4_FALLBACK_FORMAT,
    )


def verify_q4_fallback_package(package_dir: Path | str) -> Dict[str, Any]:
    """Cryptographically verify an exported Q4 fallback package.

    # Errors: raises ExportError if verification fails.
    """
    return verify_mixed_precision_package(package_dir)


def load_q4_fallback_package(
    package_dir: Path | str,
    *,
    model_factory: Optional[Any] = None,
    trust_remote_code: bool = False,
    device: str = "cpu",
    verify: bool = True,
) -> Any:
    """Load an exported Q4 fallback package standalone without BF16 teacher.

    # Errors: raises ExportError if loading or verification fails.
    """
    return load_mixed_precision_package(
        package_dir,
        model_factory=model_factory,
        trust_remote_code=trust_remote_code,
        device=device,
        verify=verify,
    )


GGUF_FALLBACK_FORMAT = "q4_k_m-gdn_f16-v1"
LLAMA_CPP_PINNED_COMMIT = "eaced9a156dcfc8133231a0a292cda094b9fdc78"
GGUF_PRESERVED_F16_PATTERNS: tuple[str, ...] = ("ssm_alpha", "ssm_beta")


def verify_llama_cpp_commit(
    llama_cpp_dir: Path | str,
    *,
    expected_commit: str = LLAMA_CPP_PINNED_COMMIT,
) -> str:
    """Verify that ``llama_cpp_dir`` matches the pinned upstream commit SHA.

    # Errors: raises :class:`ExportError` when the directory is missing, not a
    git checkout, or checked out at a different commit.
    """
    import subprocess

    repo = Path(llama_cpp_dir).resolve()
    if not repo.is_dir():
        raise ExportError(f"llama.cpp directory does not exist: {repo}")
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ExportError(
            f"failed to read git HEAD from llama.cpp directory {repo}: {proc.stderr.strip()}"
        )
    actual = proc.stdout.strip().lower()
    expected = str(expected_commit).strip().lower()
    if actual != expected:
        raise ExportError(
            f"llama.cpp commit mismatch at {repo}: got {actual!r}, expected pinned {expected!r}"
        )
    return actual


def _resolve_llama_cpp_tools(
    llama_cpp_dir: Path,
    *,
    convert_script: Path | str | None = None,
    llama_quantize_bin: Path | str | None = None,
) -> tuple[Path, Path]:
    import os
    import shutil

    conv = (
        Path(convert_script).resolve()
        if convert_script is not None
        else (llama_cpp_dir / "convert_hf_to_gguf.py").resolve()
    )
    if not conv.is_file():
        raise ExportError(f"convert_hf_to_gguf.py not found at {conv}")

    quant: Path | None = None
    if llama_quantize_bin is not None:
        cand = Path(llama_quantize_bin).resolve()
        if cand.is_file() and os.access(cand, os.X_OK):
            quant = cand
        else:
            raise ExportError(f"llama-quantize binary missing or not executable: {cand}")
    else:
        for cand in (
            llama_cpp_dir / "build" / "bin" / "llama-quantize",
            llama_cpp_dir / "llama-quantize",
        ):
            if cand.is_file() and os.access(cand, os.X_OK):
                quant = cand.resolve()
                break
        if quant is None:
            which_q = shutil.which("llama-quantize")
            if which_q:
                quant = Path(which_q).resolve()
    if quant is None:
        raise ExportError(
            f"llama-quantize binary not found under {llama_cpp_dir} or PATH; pass llama_quantize_bin="
        )
    return conv, quant


def verify_gguf_fallback(
    gguf_path: Path | str,
    *,
    expected_arch: str | None = "qwen35",
    expected_gdn_layers: int | None = None,
    require_quantized_weights: bool = True,
    require_ssm_gates: bool = True,
) -> dict[str, Any]:
    """Verify GGUF fallback architecture, MTP/vision exclusion, F16 GDN gates, and Q4_K weights.

    # Errors: raises :class:`ExportError` if any GDN gate (``ssm_alpha`` /
    ``ssm_beta``) was quantized away from F16, MTP/vision tensors leaked,
    architecture mismatches, or no quantized weights exist.
    """
    try:
        import gguf
    except ImportError as exc:
        raise ExportError(f"python gguf package is required for GGUF verification: {exc}") from exc

    from .utils import sha256_file

    resolved = Path(gguf_path).resolve()
    if not resolved.is_file():
        raise ExportError(f"GGUF file does not exist: {resolved}")

    try:
        reader = gguf.GGUFReader(str(resolved))
    except Exception as exc:
        raise ExportError(f"invalid GGUF file {resolved}: {type(exc).__name__}: {exc}") from exc

    arch_field = reader.get_field("general.architecture")
    if arch_field is None or not arch_field.data:
        raise ExportError(f"GGUF file {resolved} missing required metadata field 'general.architecture'")
    arch_val = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8", errors="replace").strip()
    if expected_arch is not None and arch_val != expected_arch:
        raise ExportError(
            f"GGUF architecture mismatch in {resolved}: got {arch_val!r}, expected {expected_arch!r}"
        )

    ssm_alpha: list[str] = []
    ssm_beta: list[str] = []
    non_f16_ssm: list[tuple[str, str]] = []
    leaked_non_text: list[str] = []
    type_counts: dict[str, int] = {}

    for tensor in reader.tensors:
        t_name = str(tensor.name)
        t_type_name = str(tensor.tensor_type.name)
        type_counts[t_type_name] = type_counts.get(t_type_name, 0) + 1

        lower = t_name.lower()
        if (
            lower.startswith(("mtp.", "nextn.", "v.", "visual.", "vision."))
            or ".mtp." in lower
            or ".visual." in lower
            or ".vision." in lower
        ):
            leaked_non_text.append(t_name)

        if ".ssm_alpha." in t_name or t_name.endswith("ssm_alpha.weight"):
            ssm_alpha.append(t_name)
            if tensor.tensor_type != gguf.GGMLQuantizationType.F16:
                non_f16_ssm.append((t_name, t_type_name))
        elif ".ssm_beta." in t_name or t_name.endswith("ssm_beta.weight"):
            ssm_beta.append(t_name)
            if tensor.tensor_type != gguf.GGMLQuantizationType.F16:
                non_f16_ssm.append((t_name, t_type_name))

    if leaked_non_text:
        raise ExportError(
            f"GGUF fallback must exclude MTP/vision tensors; found leaked keys: {leaked_non_text[:5]}"
        )
    if non_f16_ssm:
        raise ExportError(
            f"ADV-12.1 requires ssm_alpha/ssm_beta GDN gates preserved in F16; found non-F16: {non_f16_ssm[:5]}"
        )
    if require_ssm_gates and (not ssm_alpha or not ssm_beta):
        raise ExportError(
            f"GGUF fallback expected F16 ssm_alpha and ssm_beta GDN gates, got alpha={len(ssm_alpha)}, beta={len(ssm_beta)}"
        )
    if len(ssm_alpha) != len(ssm_beta):
        raise ExportError(
            f"GGUF ssm_alpha ({len(ssm_alpha)}) and ssm_beta ({len(ssm_beta)}) tensor counts differ"
        )
    if expected_gdn_layers is not None and len(ssm_alpha) != int(expected_gdn_layers):
        raise ExportError(
            f"GGUF expected {expected_gdn_layers} GDN layers, found alpha={len(ssm_alpha)}, beta={len(ssm_beta)}"
        )

    q4_k_count = sum(cnt for name, cnt in type_counts.items() if name.startswith("Q4_K"))
    if require_quantized_weights and q4_k_count == 0:
        raise ExportError(
            f"GGUF fallback has zero Q4_K quantized tensors (found types: {sorted(type_counts)})"
        )

    return {
        "format": GGUF_FALLBACK_FORMAT,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "architecture": arch_val,
        "tensor_count": len(reader.tensors),
        "ssm_alpha_count": len(ssm_alpha),
        "ssm_beta_count": len(ssm_beta),
        "ssm_f16_verified": True,
        "q4_k_tensor_count": q4_k_count,
        "type_counts": type_counts,
    }


def export_text_q4_k_m_gguf(
    source_model_dir: Path | str,
    output_dir: Path | str,
    *,
    llama_cpp_dir: Path | str,
    llama_quantize_bin: Path | str | None = None,
    convert_script: Path | str | None = None,
    output_filename: str | None = None,
    quant_type: str = "Q4_K_M",
    outtype: str = "f16",
    expected_commit: str = LLAMA_CPP_PINNED_COMMIT,
    expected_arch: str | None = "qwen35",
    expected_gdn_layers: int | None = None,
    keep_intermediate: bool = False,
    overwrite: bool = False,
    python_exe: str | None = None,
    run_cmd: Any = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Export causal text BF16 checkpoint to Q4_K_M GGUF via pinned llama.cpp with F16 GDN gates.

    Extracts standalone causal text checkpoint automatically when given a
    multimodal checkpoint, invokes ``convert_hf_to_gguf.py --no-mtp``, and
    quantizes with explicit ``--tensor-type ssm_alpha=f16`` and
    ``--tensor-type ssm_beta=f16`` so all 48 GDN gate projections stay F16.

    # Errors: raises :class:`ExportError` if llama.cpp commit/tools mismatch,
    subprocess conversion fails, or post-export GGUF verification fails.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys

    from .modeling import extract_text_causal_lm_checkpoint, load_config
    from .utils import write_json_atomic

    src_dir = Path(source_model_dir).resolve()
    out_dir = Path(output_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ExportError(f"output directory already exists and overwrite=False: {out_dir}")

    commit_sha = verify_llama_cpp_commit(llama_cpp_dir, expected_commit=expected_commit)
    conv_path, quant_path = _resolve_llama_cpp_tools(
        Path(llama_cpp_dir).resolve(),
        convert_script=convert_script,
        llama_quantize_bin=llama_quantize_bin,
    )

    cfg = load_config(src_dir)
    is_mm = (
        "vision_config" in cfg
        or isinstance(cfg.get("text_config"), dict)
        or str(cfg.get("model_type", "")) == "qwen3_5"
    )
    text_cfg = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg
    inferred_gdn = expected_gdn_layers
    if inferred_gdn is None and isinstance(text_cfg.get("layer_types"), list):
        linear_count = sum(1 for lt in text_cfg["layer_types"] if lt == "linear_attention")
        if linear_count > 0:
            inferred_gdn = linear_count

    qtype_clean = str(quant_type).strip().upper()
    if not qtype_clean:
        raise ExportError("quant_type must not be empty")

    def _exec(cmd: list[str]) -> None:
        if run_cmd is not None:
            proc = run_cmd(cmd)
            if proc is not None and getattr(proc, "returncode", 0) != 0:
                stderr = getattr(proc, "stderr", "")
                raise ExportError(f"GGUF command failed ({proc.returncode}): {cmd!r}: {stderr}")
            return
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise ExportError(f"GGUF command failed ({proc.returncode}): {cmd!r}\n{tail}")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = out_dir.with_name(out_dir.name + ".incomplete")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    stem = src_dir.name or "azrael-text"
    q4_name = output_filename or f"{stem}-{qtype_clean}.gguf"
    f16_name = f"{stem}-{str(outtype).lower()}.gguf"
    q4_stage = stage / q4_name
    f16_stage = stage / f16_name
    extract_stage = stage / "_extracted"

    try:
        if is_mm:
            extract_text_causal_lm_checkpoint(src_dir, extract_stage)
            text_dir = extract_stage / "teacher_text_bf16"
        else:
            text_dir = src_dir

        convert_cmd = [
            str(python_exe or sys.executable),
            str(conv_path),
            str(text_dir),
            "--outfile",
            str(f16_stage),
            "--outtype",
            str(outtype),
            "--no-mtp",
        ]
        _exec(convert_cmd)
        if not f16_stage.is_file():
            raise ExportError(f"convert_hf_to_gguf.py did not produce expected GGUF: {f16_stage}")

        quant_cmd = [
            str(quant_path),
            "--tensor-type",
            "ssm_alpha=f16",
            "--tensor-type",
            "ssm_beta=f16",
            str(f16_stage),
            str(q4_stage),
            qtype_clean,
        ]
        _exec(quant_cmd)
        if not q4_stage.is_file():
            raise ExportError(f"llama-quantize did not produce expected GGUF: {q4_stage}")

        verification: dict[str, Any] | None = None
        if verify:
            verification = verify_gguf_fallback(
                q4_stage,
                expected_arch=expected_arch,
                expected_gdn_layers=inferred_gdn,
            )

        if not keep_intermediate:
            if f16_stage.is_file():
                f16_stage.unlink()
            if extract_stage.exists():
                shutil.rmtree(extract_stage)

        manifest = {
            "format": GGUF_FALLBACK_FORMAT,
            "llama_cpp_commit": commit_sha,
            "quant_type": qtype_clean,
            "outtype": str(outtype),
            "gguf_file": q4_name,
            "preserved_f16_patterns": list(GGUF_PRESERVED_F16_PATTERNS),
            "convert_cmd": convert_cmd,
            "quantize_cmd": quant_cmd,
            "verification": verification,
        }
        write_json_atomic(stage / "gguf_fallback_manifest.json", manifest)

        if out_dir.exists() and overwrite:
            shutil.rmtree(out_dir)
        os.replace(stage, out_dir)
        manifest["output_dir"] = str(out_dir)
        manifest["gguf_path"] = str(out_dir / q4_name)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "Q4_FALLBACK_FORMAT",
    "GGUF_FALLBACK_FORMAT",
    "LLAMA_CPP_PINNED_COMMIT",
    "GGUF_PRESERVED_F16_PATTERNS",
    "verify_llama_cpp_commit",
    "verify_gguf_fallback",
    "export_text_q4_k_m_gguf",
    "build_q4_fallback_model",
    "export_q4_fallback_package",
    "verify_q4_fallback_package",
    "load_q4_fallback_package",
]
