"""Behavioral SFT LoRA (work-order §5).

Applies genuine trainable LoRA ``nn.Module`` adapters to the discovered
projection modules of the Ornith checkpoint: rank 32, alpha 64, lr 2e-5,
microbatch 1, BF16, gradient checkpointing. Targets attention (q/k/v/o),
relevant Gated DeltaNet projections, and selected MLP projections. Freezes the
vision tower, MTP, embeddings and LM head.

Measures actual trainable parameters and projection coverage directly from the
PyTorch model. Tests virtual-merge numerical parity
W_adapted = W_source + LoRA_delta before virtual merging is trusted.

torch/transformers are imported lazily inside functions so this module imports
on any host (including this GPU-less one). The ``_LoRAWrapper`` nn.Module and
``apply_lora_to_model`` are only defined when torch is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch  # torch is required for real LoRA nn.Module adapters
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch required for training
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .errors import LoRAError


@dataclass(frozen=True)
class LoRAConfig:
    """Hyperparameters for the short behavioral SFT LoRA stage (work-order §5)."""

    rank: int = 32
    alpha: int = 64
    lr: float = 2e-5
    microbatch: int = 1
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    supervised_token_budget: int = 2_000_000
    target_module_patterns: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Gated DeltaNet projections (work-order §1: all 24 Gated DeltaNet layers)
        "delta",
        "gated",
        "linear_attn",
        "fc",
    )
    freeze_targets: tuple[str, ...] = (
        "vision_tower",
        "vision_encoder",
        "mtp",
        "embed_tokens",
        "embed_positions",
        "lm_head",
        "output_embeddings",
    )


@dataclass(frozen=True)
class LoRAResult:
    """Outcome of the short behavioral SFT stage."""

    adapter_path: str | None  # path to saved adapter weights
    trainable_params: int
    total_params: int
    trainable_fraction: float
    projection_coverage: dict[str, int]  # module class -> count of adapted projections
    virtual_merge_max_diff: float  # max |W_adapted - W_source - LoRA_delta| on test forward
    parity_passed: bool
    eval_source_metric: float | None
    eval_adapted_metric: float | None
    regression: bool  # True if adaptation caused material regression
    supervised_tokens_consumed: int = 0  # actual supervised tokens counted during training
    notes: str = ""


def _is_target(name: str, config: LoRAConfig) -> bool:
    """Return True when ``name`` matches any LoRA target pattern."""
    return any(pat in name for pat in config.target_module_patterns)


def _set_module(model: Any, module_name: str, new_module: Any) -> None:
    """Replace a submodule given its dotted name (e.g. ``model.layers.0.q_proj``)."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


if _TORCH_AVAILABLE:

    class _LoRAWrapper(nn.Module):
        """``nn.Module`` that wraps a frozen base Linear with trainable LoRA params.

        Forward: ``y = W_base x + (alpha / rank) * B @ (A x)``.

        ``W_base`` is frozen (requires_grad=False). ``lora_A`` (rank, in) and
        ``lora_B`` (out, rank) are ``nn.Parameter``s with requires_grad=True.
        A is initialized ~ N(0, 1/r); B is initialized to zero so the adapter is
        an identity delta at init (no behavior change before training).
        """

        def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, seed: int = 1337):
            super().__init__()
            self.base_layer = base_layer
            for p in self.base_layer.parameters():
                p.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

            in_features = base_layer.in_features
            out_features = base_layer.out_features

            self.rank = rank
            self.alpha = alpha
            self.scale = alpha / rank
            self.seed = seed

            # Trainable LoRA parameters.
            self.lora_A = nn.Parameter(
                torch.randn(rank, in_features, dtype=torch.float32) / (rank**0.5)
            )
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base = self.base_layer(x)
            delta = (x @ self.lora_A.T) @ self.lora_B.T  # (..., r) @ (r, out) -> (..., out)
            return base + self.scale * delta

        def delta_weight(self) -> torch.Tensor:
            """LoRA delta matrix (out, in) = scale * B @ A."""
            return self.scale * (self.lora_B @ self.lora_A)

        def adapted_weight(self) -> torch.Tensor:
            """W_adapted = W_source + LoRA_delta."""
            return self.base_layer.weight + self.delta_weight()

        def merge(self) -> None:
            """Merge the LoRA delta into the base weight in-place (zeroes B after)."""
            with torch.no_grad():
                self.base_layer.weight += self.delta_weight()
                self.lora_B.zero_()

        def trainable_param_count(self) -> int:
            return int(self.lora_A.numel() + self.lora_B.numel())

        def total_param_count(self) -> int:
            return int(self.base_layer.weight.numel())

        def state_dict(self) -> dict[str, torch.Tensor]:
            """Return adapter state (A, B) for save/reload."""
            return {
                "lora_A": self.lora_A.detach().clone(),
                "lora_B": self.lora_B.detach().clone(),
            }

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            """Load adapter state previously produced by :meth:`state_dict`."""
            self.lora_A.data = state["lora_A"].to(self.lora_A.dtype)
            self.lora_B.data = state["lora_B"].to(self.lora_B.dtype)


def apply_lora_to_model(
    model: Any,
    config: LoRAConfig | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Wrap selected ``nn.Linear`` modules of ``model`` with trainable LoRA.

    Walks ``model.named_modules()``; every ``nn.Linear`` whose dotted name
    matches any ``target_module_patterns`` is replaced in place by an
    ``_LoRAWrapper``. All other parameters are left frozen
    (requires_grad=False).

    Returns a mapping of adapted module name -> ``_LoRAWrapper``.

    # Errors: raises :class:`LoRAError` when no target modules are matched, or
    when torch is not available.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError(
            "torch is not available; cannot apply real LoRA nn.Module adapters. "
            "Install torch in the notebook (cell 01) before cell 07."
        )
    import torch.nn as nn  # local, guaranteed available

    config = config or LoRAConfig()
    adapted: dict[str, Any] = {}
    matched = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _is_target(name, config):
            continue
        wrapped = _LoRAWrapper(module, config.rank, config.alpha, seed=seed)
        _set_module(model, name, wrapped)
        adapted[name] = wrapped
        matched += 1

    if matched == 0:
        raise LoRAError(
            "no nn.Linear modules matched the LoRA target patterns; "
            f"patterns={config.target_module_patterns}"
        )
    return adapted


def freeze_targets_on_model(model: Any, targets: list[str]) -> dict[str, bool]:
    """Freeze (requires_grad=False) the named top-level submodules of ``model``.

    ``targets`` are attribute paths (e.g. ``["model.vision_tower"]``). Returns
    target -> whether it was found and frozen.

    # Errors: raises :class:`LoRAError` if ``model`` has no parameters.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot freeze model targets")
    if not any(True for _ in model.parameters()):
        raise LoRAError("model has no parameters; cannot freeze targets")
    results: dict[str, bool] = {}
    for target in targets:
        obj = model
        found = True
        for part in target.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and hasattr(obj, "parameters"):
            for p in obj.parameters():
                p.requires_grad = False
            results[target] = True
        else:
            results[target] = False
    return results


def count_trainable_params(model: Any) -> tuple[int, int]:
    """Return (trainable_params, total_params) directly from the PyTorch model."""
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot count model parameters")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def measure_coverage(adapted: dict[str, Any], config: LoRAConfig | None = None) -> dict[str, int]:
    """Count adapted projections per module-class keyword.

    Returns keyword -> number of blocks adapted with a LoRA wrapper.
    """
    config = config or LoRAConfig()
    coverage: dict[str, int] = {}
    for name in adapted:
        klass = "other"
        for pat in config.target_module_patterns:
            if pat in name:
                klass = pat
                break
        coverage[klass] = coverage.get(klass, 0) + 1
    return coverage


def save_adapter(adapted: dict[str, Any], path: Any) -> Any:
    """Save all adapted LoRA wrappers' state dicts to ``path`` (numpy .npz)."""
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot save adapter")
    save: dict[str, np.ndarray] = {}
    for name, wrapper in adapted.items():
        sd = wrapper.state_dict()
        save[f"{name}__lora_A"] = sd["lora_A"].cpu().numpy()
        save[f"{name}__lora_B"] = sd["lora_B"].cpu().numpy()
    np.savez(path, **save)
    return path


def load_adapter(adapted: dict[str, Any], path: Any) -> None:
    """Load adapter state from a numpy .npz written by :func:`save_adapter`."""
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot load adapter")
    data = np.load(path)
    for name, wrapper in adapted.items():
        a_key = f"{name}__lora_A"
        b_key = f"{name}__lora_B"
        if a_key not in data or b_key not in data:
            raise LoRAError(
                f"adapter file {path} missing keys for {name}: expected {a_key!r} and {b_key!r}"
            )
        wrapper.lora_A.data = torch.as_tensor(data[a_key], dtype=wrapper.lora_A.dtype)
        wrapper.lora_B.data = torch.as_tensor(data[b_key], dtype=wrapper.lora_B.dtype)


def virtual_merge_parity(
    adapted: dict[str, Any],
    test_inputs: list[Any],
    tol: float = 1e-3,
) -> tuple[float, bool]:
    """Test W_adapted = W_source + LoRA_delta numerical parity.

    For each adapted wrapper, compares forward(x @ W_adapted^T) against
    forward(x @ W_base^T + scale * (x @ A^T) @ B^T) — identical by construction.
    The real check is that adapted forward differs from source-only forward by
    exactly the LoRA contribution, and delta reconstruction matches scale*B@A.

    Returns (max_abs_diff, passed).
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot check virtual-merge parity")
    max_diff = 0.0
    for _, wrapper in adapted.items():
        delta = wrapper.delta_weight()
        expected = wrapper.scale * (wrapper.lora_B @ wrapper.lora_A)
        diff = float(torch.max(torch.abs(delta - expected)).item()) if delta.numel() else 0.0
        max_diff = max(max_diff, diff)

        for x in test_inputs:
            if x.shape[-1] != wrapper.base_layer.in_features:
                continue
            adapted_out = wrapper.forward(x)
            base_out = wrapper.base_layer(x)
            delta_out = adapted_out - base_out
            expected_delta_out = x @ delta.T
            d = (
                float(torch.max(torch.abs(delta_out - expected_delta_out)).item())
                if delta_out.numel()
                else 0.0
            )
            max_diff = max(max_diff, d)
    return max_diff, bool(max_diff < tol)


def make_training_step(
    model: Any,
    _adapted: dict[str, Any],
    input_ids: Any,
    labels: Any,
    attention_mask: Any | None,
    _config: LoRAConfig,
) -> dict[str, float]:
    """Run one BF16 training step with assistant-only masked CE.

    Performs forward + backward + gradient computation on the adapted model.
    Returns a dict of metrics (loss, grad_norm, supervised_tokens).

    The caller owns the optimizer and gradient accumulation; this function
    executes a single forward/backward pass and returns measured metrics so the
    notebook cell can implement the full token-budgeted loop.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot run training step")
    model.train()
    # Only LoRA parameters require grad; everything else is frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Forward in BF16.
    device_type = "cuda" if input_ids.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = outputs.loss

    # Backward.
    loss.backward()

    # Count supervised (non-ignored) tokens.
    supervised_tokens = int((labels != -100).sum().item())

    # Gradient norm over trainable params.
    total_norm = 0.0
    for p in trainable_params:
        if p.grad is not None:
            total_norm += float(p.grad.data.norm(2).item() ** 2)
    total_norm = total_norm**0.5

    return {
        "loss": float(loss.detach().item()),
        "grad_norm": total_norm,
        "supervised_tokens": supervised_tokens,
    }


def optimizer_step(
    model: Any,
    lr: float,
    *,
    weight_decay: float = 0.0,
) -> None:
    """Perform a single AdamW step over trainable (LoRA) parameters only.

    Zeroes gradients afterward. Avoids full-model FP32 optimizer state by
    construction: only LoRA A/B parameters are optimized.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot run optimizer step")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.step()
    optimizer.zero_grad()


def discover_lora_targets(model: Any) -> list[str]:
    """Discover linear projections in model eligible for LoRA SFT."""
    names = []
    for n, m in model.named_modules():
        if not hasattr(m, "weight") or not (isinstance(m, nn.Linear) if _TORCH_AVAILABLE else False):
            continue
        parts = n.split(".")
        layer = "layers" in parts and any(p.isdigit() for p in parts)
        language = any(v in parts for v in ("language_model", "model", "text_model"))
        disallowed = any(v in n.lower() for v in ("visual", "vision", "mtp", "embed", "lm_head"))
        if layer and language and not disallowed:
            names.append(n)
    return names


def setup_peft_lora(
    model: Any,
    rank: int = 32,
    alpha: int = 64,
    target_modules: list[str] | None = None,
    dropout: float = 0.0,
    gradient_checkpointing: bool = True,
) -> Any:
    """Setup canonical PEFT LoRA model with gradient checkpointing handshake."""
    from peft import LoraConfig, get_peft_model

    targets = target_modules or discover_lora_targets(model)
    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_cfg)
    if gradient_checkpointing:
        if hasattr(peft_model, "gradient_checkpointing_enable"):
            peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
    return peft_model


def checkpoint_sft(
    net: Any,
    optimizer: Any,
    step: int,
    tokens: int,
    cursor: int,
    sft_dir: Path | str,
    template_sha: str = "",
) -> Path:
    """Persist interruptible SFT checkpoint with optimizer and RNG state."""
    import os
    import shutil
    from pathlib import Path

    from .loader import atomic_json

    out_dir = Path(sft_dir)
    snap = out_dir / f"step-{step:08d}"
    tmp = snap.with_name(snap.name + ".incomplete")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if hasattr(net, "save_pretrained"):
        net.save_pretrained(str(tmp / "adapter"), safe_serialization=True)
    else:
        save_adapter(net, tmp / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        tmp / "optimizer.pt",
    )
    atomic_json(
        tmp / "state.json",
        {
            "step": step,
            "tokens": tokens,
            "cursor": cursor,
            "template_sha256": template_sha,
        },
    )
    (tmp / "COMMITTED").write_text("complete\n")
    os.replace(tmp, snap)
    atomic_json(out_dir / "latest.json", {"path": str(snap)})
    return snap


__all__ = [
    "LoRAConfig",
    "LoRAResult",
    "_LoRAWrapper",
    "apply_lora_to_model",
    "discover_lora_targets",
    "setup_peft_lora",
    "checkpoint_sft",
    "freeze_targets_on_model",
    "count_trainable_params",
    "measure_coverage",
    "save_adapter",
    "load_adapter",
    "virtual_merge_parity",
    "make_training_step",
    "optimizer_step",
]
