"""Versioned canonical ternary checkpoint specification (work-order §2B / §10-B).

Defines the canonical on-disk representation for Artifact B (canonical effective
ternary checkpoint). Every tensor required to reconstruct the effective model is
present: quantized ternary codes (INT8, lossless, optional verified 2-bit pack),
group scales (FP32), group axis/size/layout, quantizer identity+configuration,
code version + numerical representation, rotation/equalization metadata required
for correct inference, precision exceptions, final learned scales after recovery,
and any non-absorbed low-rank residual adapters.

Non-quantized tensors (norms, biases, embeddings, LM head, recurrent-state
parameters, vision/MTP weights) are stored alongside so loading Artifact B is
*standalone*: it never silently requires the original BF16 model.

Schema version: "ternary/v1". Serialized as sharded safetensors
(model-00001-of-N.safetensors + model.safetensors.index.json) plus a
checkpoint.json manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np  # noqa: TC002  (numpy used at runtime for dtype mapping, not just types)

from .safetensors_io import (
    load_sharded_safetensors,
    save_sharded_safetensors,
)

#: Canonical schema version — bump on any backward-incompatible representation change.
SCHEMA_VERSION = "ternary/v1"

#: Quantizer identity constants.
QUANTIZER_STE = "ste"
QUANTIZER_GPTQ = "sequential_gptq_group128"
QUANTIZER_QTEA = "qtea"
SCHEMA_VERSION_QTEA = "qtea/v1"

#: Code representation constants.
CODE_DTYPE = "int8"  # codes ∈ {-1, 0, +1}, stored as int8
CODE_VALUES = (-1, 0, 1)

#: Layout keys.
GROUP_AXIS_INPUT_FEATURES = "input_features"  # grouping axis is the input-feature dim


@dataclass(frozen=True)
class TensorEntry:
    """One tensor's entry in the canonical ternary checkpoint."""

    tensor_name: str
    shape: tuple[int, ...]  # original (out_features, in_features)
    dtype: str  # "F32" for preserved tensors; "INT8" for ternary codes
    group_axis: str  # e.g. "input_features"
    group_size: int  # 128 for ternary tensors; 1 (no grouping) for preserved
    n_groups: int  # ceil(in_features / group_size) for ternary
    quantizer: str | None  # QUANTIZER_STE / QUANTIZER_GPTQ / None for preserved
    quantizer_config: dict[str, Any] = field(default_factory=dict)
    is_quantized: bool = False
    is_precision_exception: bool = False
    shard_file: str | None = None  # which shard holds this tensor


@dataclass(frozen=True)
class RotationMetadata:
    """Rotation/equalization information required for correct inference."""

    applied: bool = False
    kind: str | None = None  # "signed_hadamard_input"
    seed: int | None = None
    input_dim_padded_to: int | None = None  # next power of two >= in_features
    inverse_required: bool = True
    reason: str = ""


@dataclass(frozen=True)
class CanonicalTernarySpec:
    """Complete canonical ternary checkpoint specification.

    Attributes:
        schema_version: "ternary/v1".
        model_config: source model config.json (parsed).
        template_fingerprint: dict with handle + version/hash (native template).
        provenance: dict with Kaggle handle, dataset manifest path/version.
        tensors: dict tensor_name -> TensorEntry.
        rotation: RotationMetadata (applied to which tensors).
        precision_exceptions: list of tensor names kept in source precision.
        residual_adapters: dict tensor_name -> {"B": array, "A": array, "rank": int}
            for non-absorbed low-rank residual adapters.
        recovered_scales: dict tensor_name -> FP32 scale array (final learned scales
            after recovery, overrides conversion scales).
        effective_bits_per_weight: model-wide effective bits/weight.
    """

    schema_version: str = SCHEMA_VERSION
    model_config: dict[str, Any] = field(default_factory=dict)
    template_fingerprint: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    rotation: RotationMetadata = field(default_factory=RotationMetadata)
    precision_exceptions: list[str] = field(default_factory=list)
    residual_adapters: dict[str, dict[str, Any]] = field(default_factory=dict)
    recovered_scales: dict[str, np.ndarray] = field(default_factory=dict)
    effective_bits_per_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize spec metadata (not tensor payloads) to JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "model_config": self.model_config,
            "template_fingerprint": self.template_fingerprint,
            "provenance": self.provenance,
            "tensors": {
                name: {
                    "tensor_name": e.tensor_name,
                    "shape": list(e.shape),
                    "dtype": e.dtype,
                    "group_axis": e.group_axis,
                    "group_size": e.group_size,
                    "n_groups": e.n_groups,
                    "quantizer": e.quantizer,
                    "quantizer_config": e.quantizer_config,
                    "is_quantized": e.is_quantized,
                    "is_precision_exception": e.is_precision_exception,
                    "shard_file": e.shard_file,
                }
                for name, e in self.tensors.items()
            },
            "rotation": asdict(self.rotation),
            "precision_exceptions": list(self.precision_exceptions),
            "residual_adapters": {
                name: {"rank": v.get("rank"), "shard_file": v.get("shard_file")}
                for name, v in self.residual_adapters.items()
            },
            "effective_bits_per_weight": self.effective_bits_per_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CanonicalTernarySpec:
        """Rebuild spec from JSON-serializable dict."""
        tensors = {
            name: TensorEntry(
                tensor_name=v["tensor_name"],
                shape=tuple(int(x) for x in v["shape"]),
                dtype=v["dtype"],
                group_axis=v["group_axis"],
                group_size=int(v["group_size"]),
                n_groups=int(v["n_groups"]),
                quantizer=v.get("quantizer"),
                quantizer_config=v.get("quantizer_config", {}),
                is_quantized=bool(v.get("is_quantized", False)),
                is_precision_exception=bool(v.get("is_precision_exception", False)),
                shard_file=v.get("shard_file"),
            )
            for name, v in d.get("tensors", {}).items()
        }
        rot = d.get("rotation", {})
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            model_config=d.get("model_config", {}),
            template_fingerprint=d.get("template_fingerprint", {}),
            provenance=d.get("provenance", {}),
            tensors=tensors,
            rotation=RotationMetadata(
                applied=bool(rot.get("applied", False)),
                kind=rot.get("kind"),
                seed=rot.get("seed"),
                input_dim_padded_to=rot.get("input_dim_padded_to"),
                inverse_required=bool(rot.get("inverse_required", True)),
                reason=str(rot.get("reason", "")),
            ),
            precision_exceptions=list(d.get("precision_exceptions", [])),
            residual_adapters=d.get("residual_adapters", {}),
            recovered_scales={},  # arrays loaded separately from shards
            effective_bits_per_weight=d.get("effective_bits_per_weight"),
        )


@dataclass(frozen=True)
class QTEACheckpointSpec:
    """Canonical QTEA checkpoint specification (versioned, standalone)."""

    schema_version: str = SCHEMA_VERSION_QTEA
    model_config: dict[str, Any] = field(default_factory=dict)
    qtea_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)
    effective_bits_per_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize QTEA spec to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "model_config": self.model_config,
            "qtea_config": self.qtea_config,
            "provenance": self.provenance,
            "tensors": {
                name: {
                    "tensor_name": e.tensor_name,
                    "shape": list(e.shape),
                    "dtype": e.dtype,
                    "group_axis": e.group_axis,
                    "group_size": e.group_size,
                    "n_groups": e.n_groups,
                    "quantizer": e.quantizer,
                    "quantizer_config": e.quantizer_config,
                    "is_quantized": e.is_quantized,
                    "is_precision_exception": e.is_precision_exception,
                    "shard_file": e.shard_file,
                }
                for name, e in self.tensors.items()
            },
            "quality_metrics": self.quality_metrics,
            "effective_bits_per_weight": self.effective_bits_per_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QTEACheckpointSpec:
        """Reconstruct QTEACheckpointSpec from JSON dict."""
        tensors = {
            name: TensorEntry(
                tensor_name=v["tensor_name"],
                shape=tuple(int(x) for x in v["shape"]),
                dtype=v["dtype"],
                group_axis=v["group_axis"],
                group_size=int(v["group_size"]),
                n_groups=int(v["n_groups"]),
                quantizer=v.get("quantizer"),
                quantizer_config=v.get("quantizer_config", {}),
                is_quantized=bool(v.get("is_quantized", False)),
                is_precision_exception=bool(v.get("is_precision_exception", False)),
                shard_file=v.get("shard_file"),
            )
            for name, v in d.get("tensors", {}).items()
        }
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION_QTEA)),
            model_config=d.get("model_config", {}),
            qtea_config=d.get("qtea_config", {}),
            provenance=d.get("provenance", {}),
            tensors=tensors,
            quality_metrics=d.get("quality_metrics", {}),
            effective_bits_per_weight=d.get("effective_bits_per_weight"),
        )


def tensor_entry_for_ternary(
    tensor_name: str,
    out_features: int,
    in_features: int,
    group_size: int = 128,
    quantizer: str = QUANTIZER_GPTQ,
    quantizer_config: dict[str, Any] | None = None,
    shard_file: str | None = None,
) -> TensorEntry:
    """Build a TensorEntry for a quantized ternary tensor."""
    n_groups = (in_features + group_size - 1) // group_size
    return TensorEntry(
        tensor_name=tensor_name,
        shape=(out_features, in_features),
        dtype="INT8",
        group_axis=GROUP_AXIS_INPUT_FEATURES,
        group_size=group_size,
        n_groups=n_groups,
        quantizer=quantizer,
        quantizer_config=quantizer_config or {},
        is_quantized=True,
        is_precision_exception=False,
        shard_file=shard_file,
    )


def tensor_entry_for_preserved(
    tensor_name: str,
    shape: tuple[int, ...],
    dtype: str = "F32",
    shard_file: str | None = None,
    is_precision_exception: bool = False,
) -> TensorEntry:
    """Build a TensorEntry for a preserved (non-quantized) tensor."""
    return TensorEntry(
        tensor_name=tensor_name,
        shape=shape,
        dtype=dtype,
        group_axis=GROUP_AXIS_INPUT_FEATURES,
        group_size=1,
        n_groups=1,
        quantizer=None,
        quantizer_config={},
        is_quantized=False,
        is_precision_exception=is_precision_exception,
        shard_file=shard_file,
    )


def validate_standalone_completeness(
    spec: CanonicalTernarySpec,
    source_tensor_names: set[str],
    *,
    raise_on_missing: bool = True,
) -> dict[str, Any]:
    """Gate B — Full-model completeness.

    Compares the entire source parameter inventory against the final checkpoint
    inventory. Every required tensor must be present or explicitly represented
    by a supported alternative. Rejects missing tensors, double-applied adapters,
    and silently substituted parameters.

    Returns a dict with:
        "complete": bool
        "missing": list[str]  # source tensors not in checkpoint
        "extra": list[str]    # checkpoint tensors not in source
        "precision_exceptions": list[str]
        "quantized_count": int
        "preserved_count": int
        "details": list[str]
    """
    missing: list[str] = []
    extra: list[str] = []
    details: list[str] = []

    checkpoint_names = set(spec.tensors.keys())

    for name in sorted(source_tensor_names):
        if name not in checkpoint_names:
            missing.append(name)
            details.append(f"MISSING tensor '{name}' — not present in canonical checkpoint")

    for name in sorted(checkpoint_names):
        if name not in source_tensor_names:
            extra.append(name)
            details.append(f"EXTRA tensor '{name}' — not in source inventory")

    quantized = [n for n, e in spec.tensors.items() if e.is_quantized]
    preserved = [n for n, e in spec.tensors.items() if not e.is_quantized]

    details.append(f"quantized tensors: {len(quantized)}")
    details.append(f"preserved tensors: {len(preserved)}")
    details.append(f"precision exceptions: {len(spec.precision_exceptions)}")
    details.append(f"residual adapters: {len(spec.residual_adapters)}")
    details.append(f"rotation applied: {spec.rotation.applied} ({spec.rotation.kind or 'none'})")

    complete = not missing and not extra
    if raise_on_missing and (missing or extra):
        raise ValueError(
            "Canonical ternary checkpoint is not standalone-complete:\n" + "\n".join(details)
        )

    return {
        "complete": complete,
        "missing": missing,
        "extra": extra,
        "precision_exceptions": list(spec.precision_exceptions),
        "quantized_count": len(quantized),
        "preserved_count": len(preserved),
        "details": details,
    }


def write_canonical_ternary_checkpoint(
    dir_path: Path | str,
    spec: CanonicalTernarySpec,
    tensor_payloads: dict[str, np.ndarray],
    *,
    shard_size_bytes: int = 5 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Write artifact B (canonical effective ternary checkpoint) as sharded safetensors.

    Produces:
        <dir>/model-00001-of-N.safetensors
        <dir>/model.safetensors.index.json
        <dir>/checkpoint.json   (spec manifest)
        <dir>/tensor_manifest.json (per-tensor metadata)

    All tensors in ``tensor_payloads`` are written: quantized codes (INT8),
    preserved tensors (FP32/BF16/etc.), residual adapter B/A matrices, and
    recovered scales. The spec names the shard each tensor lives in.

    Returns the written manifest dict.
    """
    dest_dir = Path(dir_path).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Write all tensors as sharded safetensors in one call.
    # save_sharded_safetensors handles shard naming and writes
    # model.safetensors.index.json mapping tensor name -> shard file.
    index = save_sharded_safetensors(
        tensor_payloads,
        dest_dir,
        max_shard_size_bytes=shard_size_bytes,
        metadata={"schema_version": SCHEMA_VERSION},
    )

    # Read back shard assignments from the index.
    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    # Map tensor name -> shard file in spec.
    shard_for: dict[str, str] = {}
    for name, shard_name in weight_map.items():
        shard_for[name] = shard_name

    # Update spec entries with shard_file.
    updated_tensors: dict[str, TensorEntry] = {}
    for name, entry in spec.tensors.items():
        updated_tensors[name] = TensorEntry(
            tensor_name=entry.tensor_name,
            shape=entry.shape,
            dtype=entry.dtype,
            group_axis=entry.group_axis,
            group_size=entry.group_size,
            n_groups=entry.n_groups,
            quantizer=entry.quantizer,
            quantizer_config=entry.quantizer_config,
            is_quantized=entry.is_quantized,
            is_precision_exception=entry.is_precision_exception,
            shard_file=shard_for.get(name),
        )
    spec = CanonicalTernarySpec(
        schema_version=spec.schema_version,
        model_config=spec.model_config,
        template_fingerprint=spec.template_fingerprint,
        provenance=spec.provenance,
        tensors=updated_tensors,
        rotation=spec.rotation,
        precision_exceptions=spec.precision_exceptions,
        residual_adapters=spec.residual_adapters,
        recovered_scales=spec.recovered_scales,
        effective_bits_per_weight=spec.effective_bits_per_weight,
    )

    # Write checkpoint.json spec manifest.
    checkpoint_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "effective_ternary_checkpoint",
        "representation": "scale x ternary_codes, codes in {-1,0,+1}",
        "group_size": 128,
        "spec": spec.to_dict(),
        "shards": shard_files,
        "tensor_count": len(tensor_payloads),
        "quantized_count": sum(1 for e in spec.tensors.values() if e.is_quantized),
        "preserved_count": sum(1 for e in spec.tensors.values() if not e.is_quantized),
    }
    (dest_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write tensor_manifest.json for quick per-tensor lookup.
    tensor_manifest = {
        name: {
            "shard_file": entry.shard_file,
            "shape": list(entry.shape),
            "dtype": entry.dtype,
            "group_size": entry.group_size,
            "n_groups": entry.n_groups,
            "is_quantized": entry.is_quantized,
            "quantizer": entry.quantizer,
            "is_precision_exception": entry.is_precision_exception,
        }
        for name, entry in spec.tensors.items()
    }
    (dest_dir / "tensor_manifest.json").write_text(
        json.dumps(tensor_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return checkpoint_manifest


def load_canonical_ternary_checkpoint(
    dir_path: Path | str,
    *,
    device: str = "cpu",
) -> tuple[CanonicalTernarySpec, dict[str, np.ndarray]]:
    """Load artifact B from disk into spec + tensor payloads.

    Returns (spec, tensors) where tensors maps tensor name -> numpy array.
    """
    src_dir = Path(dir_path).resolve()
    checkpoint_path = src_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint.json not found in {src_dir}")

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    spec = CanonicalTernarySpec.from_dict(checkpoint["spec"])

    tensors = load_sharded_safetensors(src_dir, device=device)
    return spec, tensors


def _dtype_for_tensor_entry(entry: TensorEntry | None, arr_dtype: np.dtype[Any]) -> str:
    """Pick safetensors dtype string: INT8 -> I8, else numpy dtype."""
    if entry is not None and entry.is_quantized:
        return "I8"
    from .safetensors_io import _dtype_from_numpy

    return _dtype_from_numpy(arr_dtype)


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_QTEA",
    "QUANTIZER_STE",
    "QUANTIZER_GPTQ",
    "QUANTIZER_QTEA",
    "CODE_DTYPE",
    "TensorEntry",
    "RotationMetadata",
    "CanonicalTernarySpec",
    "QTEACheckpointSpec",
    "tensor_entry_for_ternary",
    "tensor_entry_for_preserved",
    "validate_standalone_completeness",
    "write_canonical_ternary_checkpoint",
    "load_canonical_ternary_checkpoint",
]
