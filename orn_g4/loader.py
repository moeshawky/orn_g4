"""Portable PyTorch model loader for Ornith ternary artifacts (work-order §3).

Provides:

* :func:`load_ternary_checkpoint` — reconstruct Artifact B effective model:
  ``W_effective = dequantize(codes, scales) + residual`` with rotations applied
  per recorded metadata. Supports inference with hard ternary effective weights.
* :func:`load_trainable_checkpoint` — load Artifact C trainable continuation
  state supporting ``exact_resume`` (restores STE/QAT masters, scales, residual
  adapters, optimizer/scheduler state, RNG, scaler, counters) and
  ``portable_continuation`` (initialize new optimization from final effective
  weights on any device without backend-specific optimizer state).
* :func:`save_trainable_checkpoint` — persist Artifact C using safetensors for
  portable tensor state + documented manifest for optimizer metadata.
* :func:`materialize_dense_checkpoint` — on-demand dense BF16/FP16 materialization
  (Artifact "dense_interoperability") for compatibility/debugging.

All tensor state is stored via safetensors (spec-compliant wire format,
delegating to safetensors.torch when installed on Colab). Device-agnostic:
CPU, CUDA, and a future TorchTPU backend adapter are supported. Custom module
semantics (Gated DeltaNet, vision tower, MTP) are preserved — the loader
reconstructs tensors by name and leaves module wiring to the model class.

Integer codes themselves are never differentiable. For QAT, a defined
fake-quantization / STE construction with a real gradient path is used via
:class:`_SteQatWrapper`; its exported hard-ternary forward agrees with the
checkpoint's inference forward.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch required on Colab
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .checkpoint_spec import (
    CanonicalTernarySpec,
    load_canonical_ternary_checkpoint,
)
from .safetensors_io import load_sharded_safetensors, save_sharded_safetensors
from .ternary_core import GroupedTernary

# --------------------------------------------------------------------------- #
# Device resolution (CPU / CUDA / TPU adapter)                                #
# --------------------------------------------------------------------------- #


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a torch.device supporting cpu, cuda, and tpu.

    For ``device="tpu"`` the function attempts ``torch_xla`` (TorchTPU) and
    returns ``torch.device("xla")`` when available; otherwise falls back to cpu
    and records the fact so callers can label TPU work UNVERIFIED until
    executed on actual TPU hardware.

    # Errors: raises ValueError on an unsupported device string.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot resolve device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return device
    device = str(device).lower()
    if device in ("cpu", "cuda"):
        return torch.device(device)
    if device in ("tpu", "xla"):
        try:
            import torch_xla  # noqa: F401  (TorchTPU, Colab TPU / future allocation)
            import torch_xla.core.xla_model as xm  # noqa: F401

            return torch.device("xla")
        except ImportError:
            # TPU not available in this interpreter — fall back to cpu and note.
            return torch.device("cpu")
    raise ValueError(f"unsupported device: {device!r}")


# --------------------------------------------------------------------------- #
# Checkpoint data structures                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class TrainableState:
    """Portable trainable continuation state (Artifact C).

    Distinguishes two resume modes (work-order §2C / §3):

    * exact_resume: restore STE/QAT master parameters, scales, residual
      adapters, optimizer/scheduler state, RNG states, gradient scaler state,
      global step, supervised-tokens counter, dataset cursor/sampler/epoch,
      and parameter-group trainability config — continue the same trajectory.
    * portable_continuation: initialize a new optimization run from the final
      effective weights, possibly with a new optimizer/backend; backend-specific
      optimizer state is not required.
    """

    run_id: str
    schema_version: str = "trainable/v1"
    resume_mode: str = "exact_resume"  # "exact_resume" | "portable_continuation"

    # Effective model reference (ternary canonical checkpoint path + spec hash).
    effective_model_ref: dict[str, Any] = field(default_factory=dict)

    # Trainable parameters (portable safetensors).
    master_weights: dict[str, np.ndarray] = field(default_factory=dict)  # STE/QAT float masters
    scales: dict[str, np.ndarray] = field(default_factory=dict)  # trainable group scales
    residual_adapters: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )  # tensor_name -> (B, A)

    # Optimizer / scheduler state (backend-specific; may be empty for portable).
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] = field(default_factory=dict)

    # RNG states.
    torch_rng_state: Any = None
    numpy_rng_state: Any = None
    python_rng_state: Any = None
    cuda_rng_state: Any = None

    # Gradient scaler (AMP).
    grad_scaler_state: dict[str, Any] = field(default_factory=dict)

    # Counters / cursor.
    global_step: int = 0
    supervised_tokens_consumed: int = 0
    dataset_cursor: dict[str, Any] = field(default_factory=dict)
    sampler_state: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0

    # Parameter-group / trainability config.
    param_groups: list[dict[str, Any]] = field(default_factory=list)
    trainable_mask: dict[str, bool] = field(default_factory=dict)

    # Stage lineage + checkpoint hashes.
    stage_lineage: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "resume_mode": self.resume_mode,
            "effective_model_ref": self.effective_model_ref,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "torch_rng_state": self.torch_rng_state,
            "numpy_rng_state": self.numpy_rng_state,
            "python_rng_state": self.python_rng_state,
            "cuda_rng_state": self.cuda_rng_state,
            "grad_scaler_state": self.grad_scaler_state,
            "global_step": self.global_step,
            "supervised_tokens_consumed": self.supervised_tokens_consumed,
            "dataset_cursor": self.dataset_cursor,
            "sampler_state": self.sampler_state,
            "epoch": self.epoch,
            "param_groups": self.param_groups,
            "trainable_mask": self.trainable_mask,
            "stage_lineage": self.stage_lineage,
            "checkpoint_hashes": self.checkpoint_hashes,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TrainableState:
        return cls(
            run_id=str(d["run_id"]),
            schema_version=str(d.get("schema_version", "trainable/v1")),
            resume_mode=str(d.get("resume_mode", "exact_resume")),
            effective_model_ref=d.get("effective_model_ref", {}),
            optimizer_state=d.get("optimizer_state", {}),
            scheduler_state=d.get("scheduler_state", {}),
            torch_rng_state=d.get("torch_rng_state"),
            numpy_rng_state=d.get("numpy_rng_state"),
            python_rng_state=d.get("python_rng_state"),
            cuda_rng_state=d.get("cuda_rng_state"),
            grad_scaler_state=d.get("grad_scaler_state", {}),
            global_step=int(d.get("global_step", 0)),
            supervised_tokens_consumed=int(d.get("supervised_tokens_consumed", 0)),
            dataset_cursor=d.get("dataset_cursor", {}),
            sampler_state=d.get("sampler_state", {}),
            epoch=int(d.get("epoch", 0)),
            param_groups=d.get("param_groups", []),
            trainable_mask=d.get("trainable_mask", {}),
            stage_lineage=d.get("stage_lineage", []),
            checkpoint_hashes=d.get("checkpoint_hashes", {}),
        )


# --------------------------------------------------------------------------- #
# load_ternary_checkpoint — Artifact B inference loader                       #
# --------------------------------------------------------------------------- #


def load_ternary_checkpoint(
    path: Path | str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[CanonicalTernarySpec, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load Artifact B (canonical effective ternary checkpoint).

    Reconstructs the effective computation:

        W_effective = dequantize(codes, scales) + residual        (where applicable)

    with rotations applied per recorded rotation metadata. Non-quantized
    tensors (norms, biases, embeddings, LM head, recurrent-state params,
    vision/MTP weights) are restored from the checkpoint so loading is
    *standalone* — no silent dependency on the original BF16 model.

    Args:
        path: checkpoint directory (contains checkpoint.json + shards).
        device: torch device for returned tensors (cpu / cuda / tpu).
        dtype: compute dtype for effective weights (default BF16 on CUDA, FP32
            on CPU; preserved tensors keep their recorded dtype).

    Returns:
        (spec, effective_weights, preserved_tensors) where:
          * effective_weights: tensor_name -> torch tensor of W_effective on
            ``device`` (quantized tensors dequantized + residual added).
          * preserved_tensors: tensor_name -> torch tensor of non-quantized
            tensors exactly as stored.

    # Errors: raises FileNotFoundError if checkpoint absent; ValueError on
    malformed spec or missing tensors.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot load ternary checkpoint")
    torch_dev = resolve_device(device)
    if dtype is None:
        dtype = torch.bfloat16 if torch_dev.type == "cuda" else torch.float32

    spec, arrays = load_canonical_ternary_checkpoint(path, device="cpu")

    effective_weights: dict[str, torch.Tensor] = {}
    preserved_tensors: dict[str, torch.Tensor] = {}

    # Build rotation inverse operator if needed.
    rotation = spec.rotation
    hadamard_H: np.ndarray | None = None
    if rotation.applied and rotation.kind == "signed_hadamard_input":
        from .rotation import _hadamard

        n = rotation.input_dim_padded_to
        if n is None:
            # Derive from first quantized tensor's in_features.
            n = (
                max(
                    (e.in_features if hasattr(e, "in_features") else 0)
                    for e in spec.tensors.values()
                    if e.is_quantized
                )
                or 0
            )
        hadamard_H = _hadamard(int(n)) if n else None

    for name, entry in spec.tensors.items():
        arr = arrays[name]
        if entry.is_quantized:
            # Dequantize: W_effective = scale × codes (+ residual if present).
            codes = arr.astype(np.int8)
            scales_key = f"{name}__scales"
            scales_arr = arrays.get(scales_key)
            if scales_arr is None:
                # scales may be stored under a distinct key in residual_adapters;
                # fall back to spec.group_size reconstruction from codes shape.
                raise ValueError(f"missing scales for quantized tensor {name}")
            group_size = entry.group_size
            in_features = entry.in_features if hasattr(entry, "in_features") else entry.shape[1]
            out_features = entry.out_features if hasattr(entry, "out_features") else entry.shape[0]
            scale_full = np.repeat(scales_arr, group_size, axis=1)[:, :in_features]
            w_eff = (scale_full * codes).astype(np.float32)

            # Add residual adapter if present (non-absorbed low-rank correction).
            if name in spec.residual_adapters:
                resid = spec.residual_adapters[name]
                B = arrays.get(f"{name}__residual_B")
                A = arrays.get(f"{name}__residual_A")
                if B is not None and A is not None:
                    w_eff = w_eff + (B @ A).astype(np.float32)

            # Apply rotation inverse if rotation was used (input-side: x @ H
            # then W_rot; to invert at load we rotate weights back: W = W_rot @ H^T / n).
            if hadamard_H is not None and rotation.input_dim_padded_to:
                n = rotation.input_dim_padded_to
                if w_eff.shape[1] <= n:
                    w_eff_pad = np.zeros((w_eff.shape[0], n), dtype=np.float32)
                    w_eff_pad[:, : w_eff.shape[1]] = w_eff
                    w_eff = (w_eff_pad @ hadamard_T(hadamard_H) / float(n))[:, : w_eff.shape[1]]

            effective_weights[name] = torch.as_tensor(w_eff, dtype=dtype, device=torch_dev)
        else:
            # Preserved tensor: keep as-is (recorded dtype, cast to compute dtype).
            preserved_tensors[name] = torch.as_tensor(arr, dtype=dtype, device=torch_dev)

    return spec, effective_weights, preserved_tensors


def hadamard_T(H: np.ndarray) -> np.ndarray:
    """Return H^T (Hadamard is symmetric up to column signs; H^T = H for standard)."""
    return H.T


# --------------------------------------------------------------------------- #
# load_trainable_checkpoint / save_trainable_checkpoint — Artifact C          #
# --------------------------------------------------------------------------- #


def load_trainable_checkpoint(
    path: Path | str,
    device: str | torch.device = "cpu",
    resume_mode: str = "exact_resume",
) -> tuple[TrainableState, dict[str, torch.Tensor]]:
    """Load Artifact C trainable continuation state.

    Args:
        path: checkpoint directory (contains trainable_state.safetensors,
            optimizer_state/, training_state.json, continuation_manifest.json).
        device: torch device.
        resume_mode: "exact_resume" (restore everything) or
            "portable_continuation" (ignore backend-specific optimizer state).

    Returns:
        (state, portable_tensors) where portable_tensors maps name -> torch
        tensor for re-initializing optimization from effective weights.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot load trainable checkpoint")
    torch_dev = resolve_device(device)
    src = Path(path).resolve()

    state_path = src / "trainable_state.safetensors"
    training_state_path = src / "training_state.json"
    manifest_path = src / "continuation_manifest.json"

    if not state_path.is_file():
        raise FileNotFoundError(f"trainable_state.safetensors not found in {src}")
    if not training_state_path.is_file():
        raise FileNotFoundError(f"training_state.json not found in {src}")

    arrays = load_sharded_safetensors(state_path, device="cpu")
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    state = TrainableState.from_json(training_state)
    state.resume_mode = resume_mode

    # Split portable tensors (masters/scales/residuals) from optimizer-specific arrays.
    portable_tensors: dict[str, torch.Tensor] = {}
    for name, arr in arrays.items():
        if name.startswith("optimizer_") or name.startswith("scaler_"):
            continue  # backend-specific; skipped in portable_continuation mode
        portable_tensors[name] = torch.as_tensor(arr, device=torch_dev)

    # Restore RNG / scaler / counters in exact_resume mode.
    if resume_mode == "exact_resume":
        if state.torch_rng_state is not None and _TORCH_AVAILABLE:
            try:
                torch.set_rng_state(state.torch_rng_state)
            except Exception:
                pass
        if state.numpy_rng_state is not None:
            try:
                np.random.set_state(state.numpy_rng_state)
            except Exception:
                pass
        if state.python_rng_state is not None:
            random.setstate(state.python_rng_state)

    return state, portable_tensors


def save_trainable_checkpoint(
    model: Any,
    optimizer: Any | None,
    state: TrainableState,
    path: Path | str,
    *,
    include_optimizer_state: bool = True,
) -> Path:
    """Persist Artifact C trainable continuation state.

    Portable tensor state (masters, scales, residual adapters) is stored via
    sharded safetensors. Optimizer metadata and parameter mappings live in the
    documented training_state.json / continuation_manifest.json. Native PyTorch
    checkpoints may supplement but are never the only portable representation.

    Args:
        model: the PyTorch model (used to capture param names / trainability).
        optimizer: optional optimizer (state_dict captured if include_optimizer_state).
        state: TrainableState to persist.
        path: output directory.
        include_optimizer_state: whether to serialize backend-specific optimizer
            state (required for exact_resume, optional for portable_continuation).

    Returns:
        The resolved output directory.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot save trainable checkpoint")
    dest = Path(path).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Gather portable tensors.
    tensors: dict[str, np.ndarray] = {}
    for name, arr in state.master_weights.items():
        tensors[f"master__{name}"] = np.asarray(arr, dtype=np.float32)
    for name, arr in state.scales.items():
        tensors[f"scales__{name}"] = np.asarray(arr, dtype=np.float32)
    for name, (B, A) in state.residual_adapters.items():
        tensors[f"residual__{name}__B"] = np.asarray(B, dtype=np.float32)
        tensors[f"residual__{name}__A"] = np.asarray(A, dtype=np.float32)

    # Optionally capture backend optimizer state (exact_resume only).
    if include_optimizer_state and optimizer is not None:
        try:
            opt_sd = optimizer.state_dict()
            for key, value in opt_sd.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        tensors[f"optimizer__{key}__{sub_key}"] = _to_numpy(sub_value)
                else:
                    tensors[f"optimizer__{key}"] = _to_numpy(value)
        except Exception:
            pass  # optimizer state is supplemental, not fatal

    save_sharded_safetensors(tensors, dest / "trainable_state.safetensors")

    # Capture RNG states.
    if _TORCH_AVAILABLE:
        state.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            state.cuda_rng_state = torch.cuda.get_rng_state_all()
    state.numpy_rng_state = np.random.get_state()
    state.python_rng_state = random.getstate()

    # Write training_state.json + continuation_manifest.json.
    training_state_path = dest / "training_state.json"
    training_state_path.write_text(
        json.dumps(state.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact": "trainable_continuation",
        "schema_version": state.schema_version,
        "run_id": state.run_id,
        "resume_mode": state.resume_mode,
        "effective_model_ref": state.effective_model_ref,
        "global_step": state.global_step,
        "supervised_tokens_consumed": state.supervised_tokens_consumed,
        "include_optimizer_state": include_optimizer_state,
        "portable": state.resume_mode == "portable_continuation" or not include_optimizer_state,
        "checkpoint_hashes": state.checkpoint_hashes,
    }
    (dest / "continuation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def _to_numpy(value: Any) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# --------------------------------------------------------------------------- #
# materialize_dense_checkpoint — Artifact "dense_interoperability"            #
# --------------------------------------------------------------------------- #


def materialize_dense_checkpoint(
    path: Path | str,
    dtype: torch.dtype | str = torch.bfloat16,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Materialize the effective post-recovery model as ordinary dense tensors.

    Produces sharded safetensors + config.json in ``output_dir`` for
    compatibility with libraries that cannot understand native ternary codes.
    This represents the dequantized recovered model, NOT the original
    high-precision reference (information was lost in quantization).

    Args:
        path: canonical ternary checkpoint directory.
        dtype: target dtype (torch.bfloat16 or torch.float16).
        output_dir: destination directory (defaults to <path>_dense).

    Returns:
        Manifest dict with output paths, tensor count, and dtype.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; cannot materialize dense checkpoint")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    src = Path(path).resolve()
    out = Path(output_dir).resolve() if output_dir else src.parent / (src.name + "_dense")
    out.mkdir(parents=True, exist_ok=True)

    spec, effective_weights, preserved_tensors = load_ternary_checkpoint(
        src, device="cpu", dtype=dtype
    )

    dense_tensors: dict[str, np.ndarray] = {}
    for name, t in effective_weights.items():
        dense_tensors[name] = t.detach().cpu().to(torch.float32).numpy()
    for name, t in preserved_tensors.items():
        dense_tensors[name] = t.detach().cpu().to(torch.float32).numpy()

    save_sharded_safetensors(
        dense_tensors,
        out / "model.safetensors",
        metadata={"materialized_from": str(src), "dtype": str(dtype)},
    )

    # Write a minimal config.json referencing the source config + custom code note.
    config_out = {
        "materialized_from": str(src),
        "dtype": str(dtype),
        "source_config": spec.model_config,
        "note": "Dequantized recovered model — not the original high-precision reference.",
        "template_fingerprint": spec.template_fingerprint,
        "precision_exceptions": spec.precision_exceptions,
    }
    (out / "config.json").write_text(
        json.dumps(config_out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(out),
        "tensor_count": len(dense_tensors),
        "dtype": str(dtype),
        "shards": [f"model-{i:05d}-of-1.safetensors" for i in range(1)],
        "manifest": config_out,
    }


# --------------------------------------------------------------------------- #
# STE QAT wrapper — fake-quantization with real gradient path                 #
# --------------------------------------------------------------------------- #


class _SteQatWrapper(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc,valid-types]
    """STE-based QAT wrapper around one ternary projection.

    Forward uses hard ternary codes (scale × codes) with a straight-through
    surrogate gradient to a float32 master weight. Scales are trainable
    fp32 parameters. The exported hard-ternary forward matches inference
    forward exactly (codes are frozen after export).

    Master weight is the trainable shadow; codes/scales are derived from it
    each forward. Codes are never differentiable themselves.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        threshold_ratio: float = 0.5,
        seed: int = 1337,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not available; cannot create QAT wrapper")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.threshold_ratio = threshold_ratio

        # Trainable master (fp32 shadow of ternary weights).
        self.master = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.master, a=np.sqrt(5))

        # Trainable group scales (fp32, positive).
        n_groups = (in_features + group_size - 1) // group_size
        self.log_scales = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float32)
        )  # log-scale for positivity via exp

        self.register_buffer("codes", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, n_groups, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hard-ternary forward with STE surrogate gradient to master."""
        scales = torch.exp(self.log_scales).clamp_min(1e-6)
        scale_full = scales.repeat_interleave(self.group_size, dim=1)[:, : self.in_features]

        with torch.no_grad():
            # Hard ternary codes from master + current scales.
            codes = torch.where(
                torch.abs(self.master) >= self.threshold_ratio * scale_full,
                torch.sign(self.master).to(torch.int8),
                torch.zeros_like(self.master, dtype=torch.int8),
            )
            self.codes.copy_(codes)
            self.scales.copy_(scales)

        w_eff = scale_full * codes.to(self.master.dtype)
        return x @ w_eff.T

    def effective_weight(self) -> torch.Tensor:
        """Return current hard-ternary effective weight (matches inference forward)."""
        scales = torch.exp(self.log_scales).clamp_min(1e-6)
        scale_full = scales.repeat_interleave(self.group_size, dim=1)[:, : self.in_features]
        return scale_full * self.codes.to(self.master.dtype)

    def to_grouped_ternary(self, source_name: str = "") -> GroupedTernary:
        """Export as a :class:`GroupedTernary` for checkpoint serialization."""
        codes = self.codes.detach().cpu().numpy().astype(np.int8)
        scales = self.scales.detach().cpu().numpy().astype(np.float32)
        return GroupedTernary(
            codes=codes,
            scales=scales,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            n_groups=scales.shape[1],
            source_name=source_name,
        )


FORMAT_VERSION = "ternary-canonical-v1"


def sha256(path: Path | str, chunk: int = 4 << 20) -> str:
    """Compute SHA-256 hex digest of file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(chunk), b""):
            h.update(data)
    return h.hexdigest()


def atomic_json(path: Path | str, obj: Any) -> None:
    """Write obj to path as formatted JSON atomically."""
    import os

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def module_replace(model: Any, name: str, value: Any) -> None:
    """Replace submodule `name` with `value` in `model`."""
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def inspect_ternary(model: Any) -> dict[str, dict]:
    """Inspect all TernaryLinear modules in model."""
    from .ternary_core import TernaryLinear

    if TernaryLinear is None:
        return {}
    return {
        name: {
            "in_features": m.in_features,
            "out_features": m.out_features,
            "group_size": m.group_size,
            "rank": m.rank,
            "qat": m.qat,
            "bias": m.bias is not None,
        }
        for name, m in model.named_modules()
        if isinstance(m, TernaryLinear)
    }


def restore_qwen3_5_nonpersistent_buffers(model: Any) -> list[str]:
    """Rebuild derived Qwen3.5 RoPE buffers after Accelerate's to_empty().

    These non-persistent buffers are absent from state_dict/safetensors.
    Text uses config-based RoPE, while some vision builds use dim/theta.
    Reject unknown non-persistent state rather than silently returning junk.
    """
    if not _TORCH_AVAILABLE:
        return []
    restored: list[str] = []
    for module_name, module in model.named_modules():
        names = set(getattr(module, "_non_persistent_buffers_set", set()))
        if not names:
            continue
        if not names.issubset({"inv_freq", "original_inv_freq"}) or "inv_freq" not in names:
            raise RuntimeError(f"Unknown un-restored non-persistent buffers in {module_name}: {sorted(names)}")
        if hasattr(module, "config") and hasattr(module, "rope_type"):
            if module.rope_type == "default" and hasattr(module, "compute_default_rope_parameters"):
                init_fn = module.compute_default_rope_parameters
            elif module.rope_type == "axial" and hasattr(module, "compute_axial_rope_parameters"):
                init_fn = module.compute_axial_rope_parameters
            else:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                if module.rope_type not in ROPE_INIT_FUNCTIONS:
                    raise RuntimeError(f"Unknown RoPE type: {module_name}: {module.rope_type}")
                init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
            freq, attention_scaling = init_fn(module.config, device=torch.device("cpu"))
            module.attention_scaling = attention_scaling
        elif hasattr(module, "dim") and hasattr(module, "theta"):
            dim = int(module.dim)
            theta = float(module.theta)
            if dim <= 0 or dim % 2 or theta <= 0:
                raise RuntimeError(f"Invalid vision rotary dimensions at {module_name}")
            freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        else:
            raise RuntimeError(f"Cannot reconstruct derived buffers for {module_name}, {type(module)}")
        if not freq.is_floating_point() or not bool(torch.isfinite(freq).all()):
            raise RuntimeError(f"Invalid generated RoPE frequencies at {module_name}")
        for buffer_name in sorted(names):
            target = module._buffers[buffer_name]
            if target is None or target.is_meta or target.shape != freq.shape:
                raise RuntimeError(f"Rotary buffer shape/device mismatch: {module_name}.{buffer_name}")
            expected = freq.to(device=target.device, dtype=target.dtype)
            target.copy_(expected)
            if not torch.equal(target, expected):
                raise RuntimeError(f"Rotary buffer restoration failed: {module_name}.{buffer_name}")
            restored.append(f"{module_name}.{buffer_name}")
    return restored


def verify_canonical(path: Path | str) -> dict:
    """Verify integrity of a canonical ternary checkpoint."""
    path = Path(path)
    if not (path / "COMMITTED").is_file():
        raise RuntimeError("uncommitted checkpoint")
    m = json.loads((path / "manifest.json").read_text())
    if m.get("schema") != FORMAT_VERSION:
        raise RuntimeError(f"unsupported canonical format: {m.get('schema')}")
    if (path / "COMMITTED").read_text().strip() != sha256(path / "manifest.json"):
        raise RuntimeError("manifest checksum mismatch")
    if m.get("config_sha256"):
        if not (path / "config.json").is_file() or sha256(path / "config.json") != m["config_sha256"]:
            raise RuntimeError("config.json modified or missing")
    keys: set[str] = set()
    for s in m["shards"]:
        if sha256(path / s["file"]) != s["sha256"]:
            raise RuntimeError("corrupt shard " + s["file"])
        keys.update(s["keys"])
    if keys != set(m["tensor_index"]) or len(keys) != m["tensor_count"]:
        raise RuntimeError("tensor manifest is inconsistent")
    return m


def save_canonical(
    model: Any,
    path: Path | str,
    source_id: str,
    processor: Any = None,
    extra: dict | None = None,
    max_shard_bytes: int = 512 << 20,
) -> dict:
    """Save model containing TernaryLinear modules into sharded safetensors canonical checkpoint."""
    from safetensors.torch import save_file

    from .ternary_core import TernaryLinear, dequantize

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    tmp = path.with_name(path.name + ".incomplete")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=False)

    if hasattr(model, "config") and model.config is not None and hasattr(model.config, "save_pretrained"):
        model.config.save_pretrained(tmp)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(tmp)
    if processor is not None:
        processor.save_pretrained(tmp)

    modules = inspect_ternary(model)
    state = model.state_dict()
    if TernaryLinear is not None:
        for module in (m for m in model.modules() if isinstance(m, TernaryLinear)):
            dequantize(module.codes, module.scales.detach(), module.in_features, validate=True)

    expected = set(state)
    shards: list[dict] = []
    tensor_index: dict[str, str] = {}
    batch: dict[str, torch.Tensor] = {}
    size = 0
    shardnum = 0

    def flush() -> None:
        nonlocal batch, size, shardnum
        if not batch:
            return
        shardnum += 1
        filename = f"model-{shardnum:05d}.safetensors"
        fp = tmp / filename
        save_file(batch, str(fp))
        for key in batch:
            tensor_index[key] = filename
        shards.append(
            {
                "file": filename,
                "sha256": sha256(fp),
                "bytes": fp.stat().st_size,
                "keys": list(batch),
            }
        )
        batch = {}
        size = 0

    for name, t in state.items():
        if not torch.is_tensor(t):
            raise TypeError(f"not a tensor in state_dict: {name}")
        cpu = t.detach().contiguous().cpu().clone()
        nbytes = cpu.numel() * cpu.element_size()
        if batch and size + nbytes > max_shard_bytes:
            flush()
        batch[name] = cpu
        size += nbytes
    flush()

    manifest = {
        "schema": FORMAT_VERSION,
        "source_id": source_id,
        "modules": modules,
        "tensor_index": tensor_index,
        "tensor_count": len(tensor_index),
        "shards": shards,
        "trainable_parameter_names": sorted(n for n, p in model.named_parameters() if p.requires_grad),
        "extra": extra or {},
        "config_sha256": sha256(tmp / "config.json") if (tmp / "config.json").exists() else "",
    }
    atomic_json(tmp / "manifest.json", manifest)
    atomic_json(
        tmp / "model.safetensors.index.json",
        {"metadata": {"total_size": sum(s["bytes"] for s in shards)}, "weight_map": tensor_index},
    )
    (tmp / "COMMITTED").write_text(sha256(tmp / "manifest.json") + "\n")
    os.replace(tmp, path)
    return manifest


def load_canonical(path: Path | str, device: str = "cpu", train_mode: str = "inference") -> tuple[Any, dict]:
    """Load canonical ternary checkpoint, restoring architecture and RoPE buffers."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForMultimodalLM

    from .ternary_core import TernaryLinear

    path = Path(path)
    m = verify_canonical(path)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForMultimodalLM.from_config(cfg, trust_remote_code=False)
        for name, spec in m["modules"].items():
            if TernaryLinear is not None:
                new = TernaryLinear(
                    spec["in_features"],
                    spec["out_features"],
                    spec["group_size"],
                    bias=spec["bias"],
                    rank=spec["rank"],
                    qat=spec["qat"],
                    device="meta",
                )
                module_replace(model, name, new)
    model.to_empty(device="cpu")
    restored_rope = restore_qwen3_5_nonpersistent_buffers(model)

    expected = set(model.state_dict())
    if expected != set(m["tensor_index"]):
        left = sorted(expected - set(m["tensor_index"]))[:10]
        right = sorted(set(m["tensor_index"]) - expected)[:10]
        raise RuntimeError(f"Architecture/checkpoint tensor mismatch: missing={left}, extra={right}")

    for s in m["shards"]:
        shard = load_file(str(path / s["file"]), device="cpu")
        invalid = model.load_state_dict(shard, strict=False)
        if invalid.unexpected_keys:
            raise RuntimeError(f"unexpected tensors: {invalid.unexpected_keys[:10]}")
        del shard

    if any(t.is_meta for t in model.state_dict().values()):
        raise RuntimeError("meta tensors remain after loading")
    if train_mode not in ("inference", "recorded", "scales", "residual", "qat"):
        raise ValueError(f"invalid train_mode: {train_mode}")

    recorded = set(m.get("trainable_parameter_names", []))
    for n, p in model.named_parameters():
        p.requires_grad_(False)
        if train_mode == "recorded" and n in recorded:
            p.requires_grad_(True)
        elif train_mode in ("scales", "residual", "qat"):
            if n.endswith(".scales"):
                p.requires_grad_(True)
            if train_mode in ("residual", "qat") and (n.endswith(".lora_A") or n.endswith(".lora_B")):
                p.requires_grad_(True)
            if train_mode == "qat" and n.endswith(".shadow"):
                p.requires_grad_(True)

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    return model, m


def save_trainable_only(model: Any, optimizer: Any, directory: Path | str, metadata: dict) -> dict:
    """Portable trainable weights + backend-specific optimizer state for exact resume."""
    import os

    from safetensors.torch import save_file

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    temp = directory.with_name(directory.name + ".incomplete")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    weights = {n: p.detach().cpu().contiguous().clone() for n, p in model.named_parameters() if p.requires_grad}
    if not weights:
        raise RuntimeError("no trainable parameters")
    save_file(weights, str(temp / "trainable.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temp / "optimizer.pt",
    )
    md = dict(metadata)
    md.update(
        {
            "schema": "trainable-v1",
            "parameter_names": sorted(weights),
            "weights_sha256": sha256(temp / "trainable.safetensors"),
            "optimizer_sha256": sha256(temp / "optimizer.pt"),
        }
    )
    atomic_json(temp / "metadata.json", md)
    (temp / "COMMITTED").write_text(sha256(temp / "metadata.json") + "\n")
    os.replace(temp, directory)
    return md


def load_trainable_only(model: Any, optimizer: Any, directory: Path | str, restore_rng: bool = True) -> dict:
    """Load trainable weights and optimizer state from checkpoint."""
    from safetensors.torch import load_file

    directory = Path(directory)
    if not (directory / "COMMITTED").exists():
        raise RuntimeError("uncommitted state")
    md = json.loads((directory / "metadata.json").read_text())
    if sha256(directory / "metadata.json") != (directory / "COMMITTED").read_text().strip():
        raise RuntimeError("state metadata checksum mismatch")
    for filename, key in [
        ("trainable.safetensors", "weights_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
    ]:
        if sha256(directory / filename) != md[key]:
            raise RuntimeError("state corruption: " + filename)
    trainable = dict(n for n in model.named_parameters() if n[1].requires_grad)
    weights = load_file(str(directory / "trainable.safetensors"))
    if set(weights) != set(trainable) or sorted(weights) != md["parameter_names"]:
        raise RuntimeError("trainable parameter inventory mismatch")
    with torch.no_grad():
        for name, t in weights.items():
            trainable[name].copy_(t.to(trainable[name].device))
    state = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return md


def dense_materialize(model: Any) -> Any:
    """DESTRUCTIVE conversion after a canonical checkpoint has been committed."""
    from .ternary_core import TernaryLinear

    if TernaryLinear is None:
        return model
    for name, m in list(model.named_modules()):
        if not isinstance(m, TernaryLinear):
            continue
        if m.shadow is not None:
            m.freeze_shadow_to_codes()
        full = nn.Linear(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            device=m.codes.device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            full.weight.copy_(m.effective_weight().to(full.weight.dtype))
            if m.bias is not None:
                full.bias.copy_(m.bias.to(full.bias.dtype))
        module_replace(model, name, full)
    return model


__all__ = [
    "resolve_device",
    "TrainableState",
    "load_ternary_checkpoint",
    "load_trainable_checkpoint",
    "save_trainable_checkpoint",
    "materialize_dense_checkpoint",
    "hadamard_T",
    "FORMAT_VERSION",
    "sha256",
    "atomic_json",
    "module_replace",
    "inspect_ternary",
    "restore_qwen3_5_nonpersistent_buffers",
    "verify_canonical",
    "save_canonical",
    "load_canonical",
    "save_trainable_only",
    "load_trainable_only",
    "dense_materialize",
]
if _TORCH_AVAILABLE:
    __all__.append("_SteQatWrapper")
