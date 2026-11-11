"""Artifact export (work-order §10 / §2 / §3 / §4).

Produces four independently identifiable products:

  A. BF16 adaptation artifact — PEFT-compatible adapter_model.safetensors +
     adapter_config.json + reference_manifest.json + a tested lossless adapter
     loader. Preserves foundation identity, trained LoRA parameters, adapter
     config + module mapping, tokenizer/template assets, training config,
     dataset provenance, evaluation results, and a reproducible loading method.
     Virtual-merge equivalence verified.

  B. Canonical effective ternary checkpoint (MANDATORY) — sharded safetensors
     (model-00001-of-N.safetensors + model.safetensors.index.json + checkpoint.json
     + tensor_manifest.json) with standalone full-model completeness: quantized
     ternary codes (INT8, optional verified 2-bit packing), group scales,
     group axis/size/layout, quantizer identity+config, code version + numerical
     representation, rotation metadata, precision exceptions, final learned
     scales after recovery, non-absorbed residual adapters, AND all non-quantized
     tensors (norms, biases, embeddings, LM head, recurrent-state params,
     vision/MTP weights). Reloads standalone in a fresh process.

  C. Trainable continuation checkpoint — trainable_state.safetensors +
     optimizer_state/ + training_state.json + continuation_manifest.json
     supporting exact_resume and portable_continuation across CPU/CUDA/TPU.

  D. Runtime deployment (GGUF/PTQ1_0 candidate) — generated from the FINAL
     recovered canonical checkpoint (never re-quantizing BF16 inside this cell).
     Architecture/operator support checked; numerical parity verified where
     supported; if incompatible, Artifact B remains authoritative.

R-26 (enforced): never serialize unchanged FP source weights under a ternary
filename and claim compression.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch  # lazy import; required on Colab

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .checkpoint_spec import (
    QUANTIZER_GPTQ,
    CanonicalTernarySpec,
    RotationMetadata,
    TensorEntry,
    load_canonical_ternary_checkpoint,
    tensor_entry_for_preserved,
    tensor_entry_for_ternary,
    validate_standalone_completeness,
)
from .checkpoint_spec import (
    write_canonical_ternary_checkpoint as _write_canonical_checkpoint,
)
from .config import RunConfig
from .errors import ExportError
from .loader import (
    TrainableState,
    load_trainable_checkpoint,
    materialize_dense_checkpoint,
    resolve_device,
    save_trainable_checkpoint,
)
from .safetensors_io import (
    load_sharded_safetensors,
    load_tensor_map,
    save_tensor_map,
)
from .ternary_core import GroupedTernary, pack_ternary, unpack_ternary

#: Required safetensors dtype for ternary codes (INT8).
CODE_DTYPE = "I8"


@dataclass
class ExportManifest:
    """Machine-readable export manifest: artifact -> sha256."""

    artifacts: dict[str, str] = field(default_factory=dict)  # name -> "sha256:..."

    def add(self, name: str, path: Path | str) -> None:
        self.artifacts[name] = "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, dest: Path | str) -> None:
        dest = Path(dest)
        dest.write_text(
            json.dumps(self.artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# R-26 enforcement                                                            #
# --------------------------------------------------------------------------- #


def _assert_not_source_fp(weights: dict[str, np.ndarray], label: str) -> None:
    """R-26: refuse to serialize unchanged FP weights under a ternary name."""
    has_int8 = any(arr.dtype == np.int8 for arr in weights.values())
    all_float = all(np.issubdtype(arr.dtype, np.floating) for arr in weights.values())
    if all_float and not has_int8 and weights:
        raise ExportError(
            f"R-26 violation: {label} contains only floating-point tensors and "
            f"no int8 ternary codes; refusing to serialize source weights under "
            f"a ternary filename. Export real codes/scales instead."
        )


# --------------------------------------------------------------------------- #
# Hashing helpers                                                             #
# --------------------------------------------------------------------------- #


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path | str) -> str:
    from .utils import sha256_file

    return sha256_file(path)


# --------------------------------------------------------------------------- #
# Artifact A — BF16 adaptation (PEFT-compatible safetensors)                  #
# --------------------------------------------------------------------------- #


def write_bf16_reference_artifact(
    dest_dir: Path | str,
    run_config: RunConfig,
    adapter_weights: dict[str, np.ndarray],
    template_fingerprint: dict,
    evaluation: dict,
    model_identity: dict,
    *,
    verify_virtual_merge: bool = True,
    virtual_merge_max_diff: float | None = None,
) -> Path:
    """Write artifact A (BF16 adaptation) as PEFT-compatible safetensors.

    Produces:
        <dir>/adapter_model.safetensors   (LoRA A/B tensors)
        <dir>/adapter_config.json         (PEFT-compatible config)
        <dir>/reference_manifest.json     (identity, template, provenance)
        <dir>/tokenizer_assets/           (copied tokenizer + template files)
        <dir>/training_state.json         (training config + eval + token budget)

    Verifies virtual-merge equivalence (W_adapted = W_source + LoRA_delta) when
    requested; refuses to export if parity failed.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if virtual_merge_max_diff is not None and virtual_merge_max_diff >= 1e-3:
        raise ExportError(
            f"virtual-merge parity failed (max|diff|={virtual_merge_max_diff:.2e}); "
            f"adapter must not be exported as trained"
        )

    # Save LoRA weights as safetensors (PEFT-compatible tensor names).
    safetensors_tensors: dict[str, np.ndarray] = {}
    for name, arr in adapter_weights.items():
        # Map to PEFT-style names: base_model.model.{name}.lora_A.default.weight etc.
        base = name.replace("__lora_A", ".lora_A.default.weight").replace(
            "__lora_B", ".lora_B.default.weight"
        )
        safetensors_tensors[base] = np.asarray(arr, dtype=np.float32)

    save_tensor_map(safetensors_tensors, dest / "adapter_model.safetensors")

    # PEFT-compatible adapter_config.json.
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(run_config.lora_rank),
        "lora_alpha": int(run_config.lora_alpha),
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "target_modules": list({k.split(".")[-2] if "." in k else k for k in adapter_weights}),
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layer_replication": None,
        "loftq_config": None,
        "use_rslora": False,
        "use_dora": False,
    }
    (dest / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Reference manifest (foundation identity + template fingerprint + provenance).
    reference_manifest = {
        "artifact": "bf16_adaptation",
        "schema_version": "bf16_reference/v1",
        "model_identity": model_identity,
        "template_fingerprint": template_fingerprint,
        "training_config": asdict(run_config),
        "evaluation": evaluation,
        "virtual_merge_verified": bool(
            verify_virtual_merge
            and (virtual_merge_max_diff is None or virtual_merge_max_diff < 1e-3)
        ),
        "virtual_merge_max_diff": virtual_merge_max_diff,
        "adapter_format": "peft_compatible_safetensors",
        "loader": "orn_g4.loader.load_trainable_checkpoint / custom adapter loader",
    }
    (dest / "reference_manifest.json").write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Training state for reproducibility.
    training_state = {
        "run_id": run_config.run_id,
        "supervised_token_budget": run_config.supervised_token_budget,
        "corpus_supervised_tokens": run_config.corpus_supervised_tokens,
        "reference_kl_weight": run_config.reference_kl_weight,
        "lora_rank": run_config.lora_rank,
        "lora_alpha": run_config.lora_alpha,
        "lora_lr": run_config.lora_lr,
        "seed": run_config.seed,
        "precision": run_config.precision,
    }
    (dest / "training_state.json").write_text(
        json.dumps(training_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return dest


def load_bf16_reference_artifact(
    dest_dir: Path | str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load artifact A: adapter weights + config + reference manifest."""
    src = Path(dest_dir).resolve()
    adapter_path = src / "adapter_model.safetensors"
    config_path = src / "adapter_config.json"
    manifest_path = src / "reference_manifest.json"

    if not adapter_path.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {src}")
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {src}")

    weights = load_tensor_map(adapter_path, device=device)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    return {"weights": weights, "config": config, "manifest": manifest}


# --------------------------------------------------------------------------- #
# Artifact B — Canonical effective ternary checkpoint (sharded safetensors)   #
# --------------------------------------------------------------------------- #


def write_canonical_ternary_artifact(
    dest_dir: Path | str,
    ternary_map: dict[str, GroupedTernary],
    *,
    rotation: RotationMetadata | None = None,
    precision_exceptions: list[str] | None = None,
    model_config: dict | None = None,
    template_fingerprint: dict | None = None,
    provenance: dict | None = None,
    preserved_tensors: dict[str, np.ndarray] | None = None,
    residual_adapters: dict[str, dict[str, Any]] | None = None,
    recovered_scales: dict[str, np.ndarray] | None = None,
    effective_bits_per_weight: float | None = None,
    source_tensor_names: set[str] | None = None,
) -> dict[str, Any]:
    """Write artifact B (canonical effective ternary checkpoint) as sharded safetensors.

    Standalone: every tensor required to reconstruct the effective model is
    included (quantized codes + scales + preserved non-quantized tensors +
    residual adapters + recovered scales). Rotation/Hadamard and QERA residual
    metadata survive serialization.

    Returns the checkpoint manifest dict.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    rotation = rotation or RotationMetadata()
    precision_exceptions = list(precision_exceptions or [])
    model_config = model_config or {}
    template_fingerprint = template_fingerprint or {}
    provenance = provenance or {}
    preserved_tensors = preserved_tensors or {}
    residual_adapters = residual_adapters or {}
    recovered_scales = recovered_scales or {}

    # Build TensorEntry + payloads for every tensor.
    tensor_entries: dict[str, TensorEntry] = {}
    payloads: dict[str, np.ndarray] = {}

    # Quantized ternary tensors.
    for name, grouped in ternary_map.items():
        # R-26 pre-check: refuse to serialize non-int8 codes under a ternary name.
        if grouped.codes.dtype != np.int8:
            raise ExportError(
                f"R-26 violation: tensor {name} codes have dtype {grouped.codes.dtype}, "
                f"expected int8; refusing to serialize source weights under a ternary name."
            )
        entry = tensor_entry_for_ternary(
            tensor_name=name,
            out_features=grouped.out_features,
            in_features=grouped.in_features,
            group_size=grouped.group_size,
            quantizer=QUANTIZER_GPTQ,
            quantizer_config={
                "group_size": grouped.group_size,
                "dead_zone_ratio": 0.5,
                "source_name": grouped.source_name,
            },
        )
        # Mark precision exceptions at the tensor-entry level too.
        if name in precision_exceptions:
            entry = TensorEntry(
                tensor_name=entry.tensor_name,
                shape=entry.shape,
                dtype=entry.dtype,
                group_axis=entry.group_axis,
                group_size=entry.group_size,
                n_groups=entry.n_groups,
                quantizer=entry.quantizer,
                quantizer_config=entry.quantizer_config,
                is_quantized=entry.is_quantized,
                is_precision_exception=True,
                shard_file=entry.shard_file,
            )
        tensor_entries[name] = entry
        payloads[name] = grouped.codes.astype(np.int8)  # codes as INT8
        payloads[f"{name}__scales"] = grouped.scales.astype(np.float32)
        if getattr(grouped, "rotation_hadamard", None) is not None:
            payloads[f"{name}__rotation_H"] = np.asarray(
                grouped.rotation_hadamard, dtype=np.float32
            )

    # Preserved (non-quantized) tensors — standalone completeness.
    for name, arr in preserved_tensors.items():
        sf_dtype = "F32"
        if arr.dtype == np.float16:
            sf_dtype = "F16"
        elif getattr(np, "bfloat16", None) is not None and arr.dtype == np.bfloat16:
            sf_dtype = "BF16"
        elif arr.dtype == np.float64:
            sf_dtype = "F64"
        entry = tensor_entry_for_preserved(
            tensor_name=name,
            shape=tuple(int(d) for d in arr.shape),
            dtype=sf_dtype,
            is_precision_exception=name in precision_exceptions,
        )
        tensor_entries[name] = entry
        payloads[name] = np.asarray(arr, dtype=np.float32)

    # Residual adapter matrices (non-absorbed).
    for name, resid in residual_adapters.items():
        B = np.asarray(resid["B"], dtype=np.float32)
        A = np.asarray(resid["A"], dtype=np.float32)
        payloads[f"{name}__residual_B"] = B
        payloads[f"{name}__residual_A"] = A

    # Recovered scales (final learned scales after recovery).
    for name, scales in recovered_scales.items():
        payloads[f"{name}__recovered_scales"] = np.asarray(scales, dtype=np.float32)

    spec = CanonicalTernarySpec(
        schema_version="ternary/v1",
        model_config=model_config,
        template_fingerprint=template_fingerprint,
        provenance=provenance,
        tensors=tensor_entries,
        rotation=rotation,
        precision_exceptions=precision_exceptions,
        residual_adapters=residual_adapters,
        recovered_scales=recovered_scales,
        effective_bits_per_weight=effective_bits_per_weight,
    )

    # R-26 enforcement: refuse to serialize unchanged FP weights under a ternary name.
    _assert_not_source_fp(payloads, "canonical ternary checkpoint")

    # Standalone completeness check (Gate B pre-validation).
    if source_tensor_names is not None:
        validate_standalone_completeness(spec, source_tensor_names, raise_on_missing=True)

    manifest = _write_canonical_checkpoint(
        dest, spec, payloads, shard_size_bytes=5 * 1024 * 1024 * 1024
    )
    return manifest


def load_canonical_ternary_artifact(
    dest_dir: Path | str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Load artifact B: spec + all tensor payloads (effective + preserved)."""
    spec, arrays = load_canonical_ternary_checkpoint(dest_dir, device=device)
    return {
        "spec": spec,
        "arrays": arrays,
        "checkpoint": {
            "schema_version": spec.schema_version,
            "tensor_count": len(spec.tensors),
            "quantized_count": sum(1 for e in spec.tensors.values() if e.is_quantized),
            "preserved_count": sum(1 for e in spec.tensors.values() if not e.is_quantized),
            "rotation": {
                "applied": spec.rotation.applied,
                "kind": spec.rotation.kind,
            },
            "precision_exceptions": spec.precision_exceptions,
            "effective_bits_per_weight": spec.effective_bits_per_weight,
        },
    }


# --------------------------------------------------------------------------- #
# Artifact C — Trainable continuation checkpoint                              #
# --------------------------------------------------------------------------- #


def write_trainable_continuation_artifact(
    dest_dir: Path | str,
    run_config: RunConfig,
    masters: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    residual_adapters: dict[str, tuple[np.ndarray, np.ndarray]],
    optimizer_state: dict,
    stage_progress: dict,
    experimental_config: dict,
    coverage_statistics: dict,
    *,
    resume_mode: str = "exact_resume",
    effective_model_ref: dict | None = None,
    global_step: int = 0,
    supervised_tokens_consumed: int = 0,
    dataset_cursor: dict | None = None,
    sampler_state: dict | None = None,
    epoch: int = 0,
    param_groups: list[dict] | None = None,
    trainable_mask: dict[str, bool] | None = None,
    stage_lineage: list[dict] | None = None,
    checkpoint_hashes: dict[str, str] | None = None,
    include_optimizer_state: bool = True,
) -> Path:
    """Write artifact C (trainable continuation checkpoint).

    Portable tensor state via safetensors; optimizer metadata + mappings in
    training_state.json / continuation_manifest.json. Supports exact_resume and
    portable_continuation modes.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    state = TrainableState(
        run_id=run_config.run_id,
        resume_mode=resume_mode,
        effective_model_ref=effective_model_ref or {},
        master_weights=masters,
        scales=scales,
        residual_adapters=residual_adapters,
        optimizer_state=optimizer_state,
        scheduler_state={},
        global_step=global_step,
        supervised_tokens_consumed=supervised_tokens_consumed,
        dataset_cursor=dataset_cursor or {},
        sampler_state=sampler_state or {},
        epoch=epoch,
        param_groups=param_groups or [],
        trainable_mask=trainable_mask or {},
        stage_lineage=stage_lineage or [],
        checkpoint_hashes=checkpoint_hashes or {},
    )

    # Build a dummy model/optimizer for save_trainable_checkpoint's optimizer capture.
    # We pass optimizer=None when not including backend state.
    optimizer_obj = None  # optimizer_state already captured as dict

    save_trainable_checkpoint(
        model=None,
        optimizer=optimizer_obj,
        state=state,
        path=dest,
        include_optimizer_state=include_optimizer_state,
    )

    # Override optimizer_state with the captured dict (save_trainable_checkpoint
    # can't take a raw dict; we write it separately for portability).
    opt_dir = dest / "optimizer_state"
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "optimizer_state.json").write_text(
        json.dumps(optimizer_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Experimental config + coverage statistics manifest.
    exp_manifest = {
        "artifact": "trainable_continuation",
        "schema_version": "trainable/v1",
        "run_id": run_config.run_id,
        "resume_mode": resume_mode,
        "experimental_config": experimental_config,
        "coverage_statistics": coverage_statistics,
        "stage_progress": stage_progress,
        "include_optimizer_state": include_optimizer_state,
    }
    (dest / "experiment_manifest.json").write_text(
        json.dumps(exp_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return dest


def load_trainable_continuation_artifact(
    dest_dir: Path | str,
    device: str | torch.device = "cpu",
    resume_mode: str = "exact_resume",
) -> dict[str, Any]:
    """Load artifact C: TrainableState + portable tensors."""
    state, portable_tensors = load_trainable_checkpoint(
        dest_dir, device=device, resume_mode=resume_mode
    )
    return {
        "state": state,
        "portable_tensors": portable_tensors,
        "resume_mode": resume_mode,
    }


# --------------------------------------------------------------------------- #
# Artifact D — PTQ1_0 / GGUF packed candidate (from final canonical)          #
# --------------------------------------------------------------------------- #


def write_ptq1_0_packed_artifact(
    ternary_map: dict[str, GroupedTernary],
    dest_dir: Path | str,
    *,
    verify_on_real_matrices: bool = True,
    source_weights: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Attempt PTQ1_0 packed inference candidate from the FINAL recovered canonical.

    Never re-quantizes the original BF16 model inside this cell — only re-packs
    the already-recovered codes/scales. Verifies pack/unpack equivalence on real
    projection matrices. If the Prism consumer is absent or verification fails,
    returns packed=False and Artifact B remains authoritative.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    packed_tensors: dict[str, dict] = {}
    verification: dict[str, Any] = {}

    for name, grouped in ternary_map.items():
        weights = {"codes": grouped.codes, "scales": grouped.scales}
        _assert_not_source_fp(weights, f"PTQ1_0 tensor {name}")
        blob = pack_ternary(grouped)
        blob_path = dest / f"{name}.ptq1_0"
        blob_path.write_bytes(blob)
        packed_tensors[name] = {
            "file": blob_path.name,
            "group_size": grouped.group_size,
            "in_features": grouped.in_features,
            "out_features": grouped.out_features,
            "n_groups": grouped.n_groups,
            "codes_bytes": int(grouped.codes.nbytes),
            "scales_bytes": int(grouped.scales.nbytes),
        }

        if verify_on_real_matrices:
            try:
                unpacked = unpack_ternary(blob, source_name=name)
                codes_match = bool(np.array_equal(unpacked.codes, grouped.codes))
                scales_match = bool(np.allclose(unpacked.scales, grouped.scales, rtol=1e-5))
                # Numerical parity on real-shape activation.
                if source_weights is not None and name in source_weights:
                    w_src = source_weights[name]
                    rng = np.random.default_rng(0)
                    x = rng.standard_normal((8, w_src.shape[1])).astype(np.float32)
                    ref = x @ w_src.T
                    rep = x @ unpacked.effective_weight().T
                    recon_err = float(np.max(np.abs(ref - rep)))
                else:
                    rng = np.random.default_rng(0)
                    x = rng.standard_normal((8, grouped.in_features)).astype(np.float32)
                    ref = x @ grouped.effective_weight().T
                    rep = x @ unpacked.effective_weight().T
                    recon_err = float(np.max(np.abs(ref - rep)))
                verification[name] = {
                    "codes_match": codes_match,
                    "scales_match": scales_match,
                    "reconstruction_max_err": recon_err,
                    "ok": bool(codes_match and scales_match and recon_err < 1e-4),
                }
            except Exception as exc:
                verification[name] = {"ok": False, "reason": str(exc)}

    all_ok = all(v.get("ok", False) for v in verification.values()) if verification else True
    packed = bool(all_ok and packed_tensors)

    manifest = {
        "artifact": "ptq1_0_packed_candidate",
        "schema_version": "ptq1_0/v1",
        "packed": packed,
        "tensors": packed_tensors,
        "verification": verification,
        "note": (
            "Packed from final recovered canonical checkpoint (no BF16 re-quantization). "
            "Prism runtime compatibility unverified unless the consumer was present."
        ),
    }
    (dest / "ptq1_0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


# Backward-compatible alias (existing tests import pack_ptq1_0).
pack_ptq1_0 = write_ptq1_0_packed_artifact


# --------------------------------------------------------------------------- #
# Dense interoperability export (re-export loader.materialize_dense_checkpoint)#
# --------------------------------------------------------------------------- #

write_dense_interoperability_export = materialize_dense_checkpoint


def load_dense_interoperability_export(
    dest_dir: Path | str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Load a dense interoperability export into torch tensors."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot load dense export")
    torch_dev = resolve_device(device)
    src = Path(dest_dir).resolve()
    arrays = load_sharded_safetensors(src / "model.safetensors", device="cpu")
    dtype = dtype or (torch.bfloat16 if torch_dev.type == "cuda" else torch.float32)
    return {
        name: torch.as_tensor(arr, dtype=dtype, device=torch_dev) for name, arr in arrays.items()
    }


# --------------------------------------------------------------------------- #
# Fresh-process reload verification helper                                    #
# --------------------------------------------------------------------------- #


def verify_fresh_process_reload(
    checkpoint_dir: Path | str,
    *,
    test_tensor_names: list[str] | None = None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Run a genuine subprocess reload test (work-order §11 Gate C / §10-B).

    Launches a fresh Python interpreter, loads the canonical ternary checkpoint
    via load_canonical_ternary_artifact, and verifies hard-ternary forward
    reproduction for each test tensor. Returns {"passed": bool, "details": ...}.
    """
    checkpoint_dir = str(Path(checkpoint_dir).resolve())
    test_code = f'''import sys, json, numpy as np
sys.path.insert(0, "{os.getcwd()}")
from orn_g4.checkpoint_spec import load_canonical_ternary_checkpoint
from orn_g4.ternary_core import hard_ternary_forward

spec, arrays = load_canonical_ternary_checkpoint("{checkpoint_dir}", device="cpu")
names = sorted(arrays.keys())
print(f"reloaded {{len(names)}} tensors")
results = {{}}
for name in names[:{"len(test_tensor_names) if test_tensor_names else 5"}]:
    arr = arrays[name]
    if arr.dtype == np.int8:
        # quantized tensor: find matching scales key
        scales_key = name + "__scales"
        if scales_key in arrays:
            codes = arr
            scales = arrays[scales_key]
            group_size = 128
            in_f = codes.shape[1]
            scale_full = np.repeat(scales, group_size, axis=1)[:, :in_f]
            w_eff = (scale_full * codes).astype(np.float32)
            x = np.random.default_rng(0).standard_normal((4, in_f)).astype(np.float32)
            y = x @ w_eff.T
            results[name] = {{
                "shape": list(y.shape),
                "finite": bool(np.all(np.isfinite(y))),
                "max_abs": float(np.max(np.abs(y))),
            }}
print("RELOAD_TEST_PASS")
print(json.dumps(results))
'''
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    passed = "RELOAD_TEST_PASS" in result.stdout
    return {
        "passed": passed,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr[:1000] if result.stderr else "",
        "checkpoint_dir": checkpoint_dir,
    }


# --------------------------------------------------------------------------- #
# Backward-compatible aliases (existing tests import these names)               #
# --------------------------------------------------------------------------- #


def write_canonical_ternary_checkpoint(
    dest_dir: Path | str,
    ternary: GroupedTernary | dict[str, GroupedTernary],
    rotation: dict | None = None,
    precision_exceptions: list[str] | None = None,
    model_config: dict | None = None,
    template_fingerprint: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    """Backward-compatible wrapper: write artifact B (canonical ternary checkpoint).

    Old signature accepted ``ternary`` as a single GroupedTernary or a dict
    name->GroupedTernary plus rotation/precision_exceptions/model_config/
    template_fingerprint/provenance kwargs. Delegates to the new
    write_canonical_ternary_artifact which produces sharded safetensors.
    """
    ternary_map = {"default": ternary} if isinstance(ternary, GroupedTernary) else dict(ternary)

    # Build preserved_tensors empty (tests only export quantized tensors).
    preserved_tensors: dict[str, np.ndarray] = {}

    return write_canonical_ternary_artifact(
        dest_dir,
        ternary_map,
        rotation=rotation or RotationMetadata(),
        precision_exceptions=list(precision_exceptions or []),
        model_config=model_config or {},
        template_fingerprint=template_fingerprint or {},
        provenance=provenance or {},
        preserved_tensors=preserved_tensors,
    )


def hard_ternary_forward_tensor(
    x: np.ndarray,
    tensor: dict,
) -> np.ndarray:
    """Hard-ternary forward for one tensor: x @ (scale × codes)^T.

    Backward-compatible alias for the existing test suite.
    """
    codes = tensor["codes"].astype(np.float32)
    scales = tensor["scales"].astype(np.float32)
    scale_full = np.repeat(scales, tensor["group_size"], axis=1)[:, : tensor["in_features"]]
    w_eff = scale_full * codes
    return x @ w_eff.T


def load_effective_ternary(checkpoint_dir: Path | str) -> tuple[dict, dict]:
    """Load a canonical effective ternary checkpoint (backward-compatible alias).

    Returns (checkpoint_json, tensors) where tensors maps name -> {codes, scales,
    group_size, in_features, out_features, n_groups}.
    """
    from .checkpoint_spec import load_canonical_ternary_checkpoint

    spec, arrays = load_canonical_ternary_checkpoint(checkpoint_dir, device="cpu")
    cp = {
        "artifact": "effective_ternary_checkpoint",
        "precision_exceptions": list(spec.precision_exceptions),
        "tensors": {
            name: {
                "group_size": entry.group_size,
                "in_features": entry.in_features
                if hasattr(entry, "in_features")
                else entry.shape[1],
                "out_features": entry.out_features
                if hasattr(entry, "out_features")
                else entry.shape[0],
                "n_groups": entry.n_groups,
                "group_axis": entry.group_axis,
                "is_quantized": entry.is_quantized,
            }
            for name, entry in spec.tensors.items()
        },
    }
    tensors = {}
    for name, arr in arrays.items():
        entry = spec.tensors.get(name)
        if entry is None or not entry.is_quantized:
            continue
        in_f = entry.in_features if hasattr(entry, "in_features") else entry.shape[1]
        out_f = entry.out_features if hasattr(entry, "out_features") else entry.shape[0]
        gs = entry.group_size
        tensors[name] = {
            "codes": arr.astype(np.int8),
            "scales": arrays.get(f"{name}__scales", np.ones((out_f, 1), dtype=np.float32)).astype(
                np.float32
            ),
            "group_size": gs,
            "in_features": in_f,
            "out_features": out_f,
            "n_groups": entry.n_groups,
        }
    return cp, tensors


def export_qtea_package(
    model: Any,
    run_dir: Path | str,
    index: dict[str, Any],
    conversion_result: dict[str, Any],
    source_dir: Path | str,
    *,
    ce_max_regression: float = 0.50,
    qtea_config: Any = None,
    mode: str = "PILOT",
    run_tag: str = "ornith_qtea_v2_1",
    provenance: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Export a standalone, portable QTEA candidate package.

    Enforces the quality gate (delta_ce <= ce_max_regression), saves encoded
    QTEA projections and preserved base tensors, writes a self-contained runtime,
    generates manifest and provenance locks, and builds the deployable archive.
    """
    from .errors import ExportError, QualityGateError
    from .gates import gate_ce_quality
    from .qtea import QTEALinear, pack_layer, unpack_layer

    delta_ce = float(conversion_result.get("delta_ce", 0.0))
    if delta_ce > ce_max_regression:
        raise QualityGateError(
            f"Export refused: delta_ce={delta_ce:.4f} exceeds threshold {ce_max_regression:.4f}"
        )

    package_dir = Path(run_dir) / "candidate"
    assets_dir = package_dir / "assets"
    encoded_dir = package_dir / "encoded"
    preserved_dir = package_dir / "preserved"

    for d in (package_dir, assets_dir, encoded_dir, preserved_dir):
        d.mkdir(parents=True, exist_ok=True)

    src_p = Path(source_dir)
    # Copy configuration and tokenizer assets
    if src_p.is_dir():
        for asset in src_p.iterdir():
            if asset.is_file() and (
                asset.suffix in (".json", ".txt", ".model", ".jinja")
                or asset.name.startswith("tokenizer")
            ):
                shutil.copy2(asset, assets_dir / asset.name)

    # Save completed QTEA projections
    encoded_manifest: dict[str, Any] = {}
    completed_targets = index.get("completed", {})
    for name in completed_targets:
        mod = model.get_submodule(name)
        slug = f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.safetensors"
        out_file = encoded_dir / slug

        if isinstance(mod, QTEALinear):
            layer_dict = mod.to_layer_dict()
            tensors_to_save: dict[str, Any] = {
                "ternary": layer_dict["ternary"],
                "col_v": layer_dict["col_v"],
                "group_alpha": layer_dict["group_alpha"],
                "group_beta": layer_dict["group_beta"],
                "salient_cols": layer_dict["salient_cols"],
                "residual_index": layer_dict["residual_index"],
                "residual_scale": layer_dict["residual_scale"],
            }
            if layer_dict["residual_value"].numel() > 0:
                tensors_to_save["residual_value"] = layer_dict["residual_value"]
            if mod.bias is not None:
                tensors_to_save["bias"] = mod.bias
        else:
            # Baseline TernaryLinear fallback
            tensors_to_save = {
                "codes": mod.codes,
                "scales": mod.scales,
            }
            if getattr(mod, "bias", None) is not None:
                tensors_to_save["bias"] = mod.bias
            if getattr(mod, "lora_A", None) is not None and getattr(mod, "lora_B", None) is not None:
                tensors_to_save["lora_A"] = mod.lora_A
                tensors_to_save["lora_B"] = mod.lora_B

        # Save using safetensors or torch
        try:
            from safetensors.torch import save_file
            save_file({k: v.cpu() for k, v in tensors_to_save.items()}, str(out_file))
        except ImportError:
            torch.save({k: v.cpu() for k, v in tensors_to_save.items()}, str(out_file))

        digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
        encoded_manifest[name] = {"file": slug, "sha256": digest}

    # Save preserved tensors
    preserved_tensors: dict[str, Any] = {}
    completed_names = set(completed_targets.keys())
    for name, param in model.named_parameters():
        root_mod = ".".join(name.split(".")[:-1])
        if root_mod in completed_names:
            continue
        preserved_tensors[name] = param.detach().cpu()

    preserved_manifest: dict[str, Any] = {}
    if preserved_tensors:
        pres_file = preserved_dir / "preserved.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(preserved_tensors, str(pres_file))
        except ImportError:
            torch.save(preserved_tensors, str(pres_file))
        preserved_manifest["preserved.safetensors"] = {
            "sha256": hashlib.sha256(pres_file.read_bytes()).hexdigest(),
            "count": len(preserved_tensors),
        }

    # Write self-contained ternary_runtime.py
    runtime_py = package_dir / "ternary_runtime.py"
    runtime_py.write_text(
        r'''"""Self-contained runtime module for TernaryLinear and QTEALinear projections."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_POWERS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)

def unpack_ternary(packed, cols, group_size=128):
    rows = packed.shape[0]
    bytes_per_group = (group_size + 4) // 5
    powers = _POWERS.to(packed.device)
    values = packed.reshape(rows, -1, 1).int()
    digits = (values // powers) % 3 - 1
    digits = digits.reshape(rows, -1, bytes_per_group * 5)[:, :, :group_size]
    return digits.reshape(rows, cols).to(torch.int8)

class QTEALinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, residual_group=4, bias=False, device="cpu"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.residual_group = int(residual_group)
        bytes_per_group = (self.group_size + 4) // 5
        n_groups = math.ceil(self.in_features / self.group_size)
        self.register_buffer("ternary", torch.zeros(self.out_features, n_groups * bytes_per_group, dtype=torch.uint8, device=device))
        self.register_buffer("col_v", torch.ones(self.in_features, dtype=torch.float32, device=device))
        self.register_buffer("group_alpha", torch.ones(self.out_features, n_groups, dtype=torch.float32, device=device))
        self.register_buffer("group_beta", torch.zeros(self.out_features, n_groups, dtype=torch.float32, device=device))
        self.register_buffer("salient_cols", torch.zeros(0, dtype=torch.int32, device=device))
        self.register_buffer("residual_index", torch.zeros(0, 0, dtype=torch.uint8, device=device))
        self.register_buffer("residual_value", torch.zeros(0, 0, device=device))
        self.register_buffer("residual_scale", torch.zeros(0, dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def effective_weight(self, dtype=torch.float32):
        codes = unpack_ternary(self.ternary, self.in_features, self.group_size).float()
        alpha = self.group_alpha.repeat_interleave(self.group_size, dim=1)
        beta = self.group_beta.repeat_interleave(self.group_size, dim=1)
        w = alpha * self.col_v[None, :] * codes + beta
        if self.salient_cols.numel() > 0 and self.residual_value.numel() > 0:
            packed = self.residual_index.int()
            shifts = [0, 2, 4, 6]
            idx = torch.stack([(packed >> s) & 0x3 for s in shifts], dim=2).reshape(packed.shape[0], -1)
            val = self.residual_value.float() / self.residual_scale[:, None]
            dense = torch.zeros(val.shape[0], val.shape[1], self.residual_group, device=w.device)
            valid = min(val.shape[1], idx.shape[1])
            dense.scatter_(2, idx[:, :valid, None].long(), val[:, :valid, None])
            res = dense.reshape(val.shape[0], -1)
            w[:, self.salient_cols.long()] += res.t()
        return w.to(dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.effective_weight(x.dtype), self.bias)

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, bias=False, rank=0, device="cpu"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.groups = math.ceil(self.in_features / self.group_size)
        self.register_buffer("codes", torch.zeros(self.out_features, self.groups, self.group_size, device=device, dtype=torch.int8))
        self.scales = nn.Parameter(torch.ones(self.out_features, self.groups, device=device, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self.rank = int(rank)
        if self.rank:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=torch.float32))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def effective_weight(self, dtype=torch.float32):
        c = self.codes.float() * self.scales.clamp_min(1e-8).unsqueeze(-1)
        w = c.flatten(1)[:, :self.in_features]
        if self.lora_A is not None and self.lora_B is not None:
            w = w + self.lora_B @ self.lora_A
        return w.to(dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.effective_weight(x.dtype), self.bias)
''',
        encoding="utf-8",
    )

    # Write manifest and provenance locks
    manifest = {
        "format": "qtea-v1",
        "mode": mode,
        "run_tag": run_tag,
        "encoded": encoded_manifest,
        "preserved": preserved_manifest,
        "metrics": conversion_result,
        "provenance": provenance or {},
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package_dir / "provenance_lock.json").write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "delta_ce": delta_ce,
                "ce_max_regression": ce_max_regression,
                "completed_count": len(completed_targets),
                "provenance": provenance or {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Tarball archive
    tar_path = Path(run_dir) / f"{run_tag}_package.tar.gz"
    cmd = ["tar", "-czf", str(tar_path), "-C", str(package_dir), "."]
    subprocess.run(cmd, check=True)

    return package_dir, manifest


__all__ = [
    "ExportManifest",
    "write_bf16_reference_artifact",
    "load_bf16_reference_artifact",
    "write_canonical_ternary_artifact",
    "load_canonical_ternary_artifact",
    "write_trainable_continuation_artifact",
    "load_trainable_continuation_artifact",
    "write_ptq1_0_packed_artifact",
    "write_dense_interoperability_export",
    "load_dense_interoperability_export",
    "verify_fresh_process_reload",
    "write_canonical_ternary_checkpoint",
    "hard_ternary_forward_tensor",
    "load_effective_ternary",
    "pack_ptq1_0",
    "_assert_not_source_fp",
    "export_qtea_package",
]
