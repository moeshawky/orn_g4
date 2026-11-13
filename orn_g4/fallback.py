"""Independent conventional 4-bit (Q4) fallback from the merged BF16 SFT teacher.

Provides a direct fallback quantization, export, and standalone loading path
that does NOT depend on QTEA ternary conversion or QTEA recovery. Converts all
discovered Ornith language projections (expected 248) to ``Protected4BitLinear``
(``affine_uint4``) while preserving embeddings, norms, vision modules, MTP,
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
    expected_qtea: int = 144,
    expected_int4: int = 104,
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


__all__ = [
    "Q4_FALLBACK_FORMAT",
    "build_q4_fallback_model",
    "export_q4_fallback_package",
    "verify_q4_fallback_package",
    "load_q4_fallback_package",
]
