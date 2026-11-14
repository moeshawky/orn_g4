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
    """Hyperparameters for the behavioral SFT LoRA stage (Ticket 007 / ADV-12.1)."""

    rank: int = 32
    alpha: int = 64
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    lr_scheduler_type: str = "cosine"
    grad_accum_steps: int = 16
    max_seq_len: int = 8192
    chunk_size: int = 256
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
            x_f = x.to(dtype=self.lora_A.dtype)
            delta = (x_f @ self.lora_A.T) @ self.lora_B.T  # (..., r) @ (r, out) -> (..., out)
            return base + self.scale * delta.to(dtype=base.dtype)

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

        def state_dict(
            self,
            *args: Any,
            destination: dict[str, torch.Tensor] | None = None,
            prefix: str = "",
            keep_vars: bool = False,
        ) -> dict[str, torch.Tensor]:
            """Return PyTorch module state when ``destination`` is provided, else LoRA adapter state."""
            if destination is not None or args or prefix or keep_vars:
                return super().state_dict(
                    *args, destination=destination, prefix=prefix, keep_vars=keep_vars
                )
            return {
                "lora_A": self.lora_A.detach().clone(),
                "lora_B": self.lora_B.detach().clone(),
            }

        def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False) -> Any:
            """Load adapter state or delegate full-module state_dict loading to nn.Module."""
            if "lora_A" in state_dict and "lora_B" in state_dict and "base_layer.weight" not in state_dict:
                self.lora_A.data = state_dict["lora_A"].to(self.lora_A.dtype)
                self.lora_B.data = state_dict["lora_B"].to(self.lora_B.dtype)
                return None
            return super().load_state_dict(state_dict, strict=strict, assign=assign)


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
    for p in model.parameters():
        p.requires_grad = False

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


def create_sft_optimizer(
    model: Any,
    lr: float = 5e-5,
    *,
    weight_decay: float = 0.01,
) -> Any:
    """Construct stateful AdamW optimizer over FP32 trainable LoRA parameters.

    # Errors: raises :class:`LoRAError` if torch is unavailable, no parameters
    are trainable, or any trainable parameter is not ``torch.float32``.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot create SFT optimizer")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise LoRAError("model has no trainable parameters for SFT optimizer")
    non_fp32 = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if non_fp32:
        raise LoRAError(f"SFT trainable LoRA parameters must be torch.float32, got non-FP32: {non_fp32[:5]}")
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_sft_scheduler(
    optimizer: Any,
    total_steps: int,
    *,
    warmup_ratio: float = 0.10,
    lr_scheduler_type: str = "cosine",
) -> Any:
    """Create a resumable LR scheduler (linear warmup + cosine or constant decay)."""
    import math

    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot create LR scheduler")
    total = max(1, int(total_steps))
    warmup_steps = int(round(total * float(warmup_ratio)))
    kind = str(lr_scheduler_type).lower().strip()
    if kind not in {"cosine", "constant"}:
        raise LoRAError(f"unsupported lr_scheduler_type: {lr_scheduler_type!r}")

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if kind == "constant":
            return 1.0
        denom = max(1, total - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _make_ce_chunk_runner(lm_head: Any, ignore_index: int):
    import torch.nn.functional as F

    def _chunk_fn(h_c: Any, y_c: Any):
        w = getattr(lm_head, "weight", None)
        h_in = h_c.to(dtype=w.dtype) if w is not None and h_c.dtype != w.dtype else h_c
        logits_c = lm_head(h_in).float()
        return F.cross_entropy(logits_c, y_c, ignore_index=ignore_index, reduction="sum")

    return _chunk_fn


def chunked_causal_ce_loss(
    hidden_states: Any,
    lm_head: Any,
    labels: Any,
    *,
    chunk_size: int = 256,
    ignore_index: int = -100,
    shift_labels: bool = True,
    grad_checkpoint: bool = False,
) -> tuple[Any, int]:
    """Compute exact summed causal cross-entropy in token chunks without B*T*V peak VRAM.

    Args:
        hidden_states: decoder final hidden states ``(B, T, H)``.
        lm_head: output linear/quantized projection module ``(H -> V)``.
        labels: unshifted aligned token labels ``(B, T)`` with ``ignore_index``.
        chunk_size: sequence tokens per chunk (default ``256``).
        shift_labels: when True (default), aligns ``hidden_states[:, :-1]`` with
            ``labels[:, 1:]`` for next-token prediction.
        grad_checkpoint: recompute per-chunk ``lm_head`` projections on backward.

    Returns:
        ``(loss_sum, supervised_token_count)`` where ``loss_sum`` is a scalar
        ``float32`` tensor summed over all non-ignored tokens.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot compute chunked causal CE loss")
    if chunk_size <= 0:
        raise LoRAError(f"chunk_size must be positive, got {chunk_size}")
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise LoRAError(
            f"expected 3D hidden_states and 2D labels, got {tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )

    if shift_labels:
        h = hidden_states[:, :-1, :]
        y = labels[:, 1:]
    else:
        h = hidden_states
        y = labels

    if h.shape[:2] != y.shape:
        raise LoRAError(
            f"aligned hidden/label shape mismatch: {tuple(h.shape[:2])} vs {tuple(y.shape)}"
        )

    h_flat = h.reshape(-1, h.shape[-1])
    y_flat = y.reshape(-1)
    n_tokens = int(h_flat.shape[0])
    supervised_count = int((y_flat != ignore_index).sum().item())

    if supervised_count == 0 or n_tokens == 0:
        return hidden_states.sum().float() * 0.0, 0

    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as grad_cp

    loss_sum = None
    for start in range(0, n_tokens, int(chunk_size)):
        end = min(n_tokens, start + int(chunk_size))
        y_c = y_flat[start:end]
        if not bool((y_c != ignore_index).any().item()):
            continue
        h_c = h_flat[start:end]
        if grad_checkpoint and (h_c.requires_grad or any(p.requires_grad for p in lm_head.parameters())):
            runner = _make_ce_chunk_runner(lm_head, ignore_index)
            chunk_loss = grad_cp(runner, h_c, y_c, use_reentrant=False)
        else:
            logits_c = lm_head(h_c).float()
            chunk_loss = F.cross_entropy(logits_c, y_c, ignore_index=ignore_index, reduction="sum")
        loss_sum = chunk_loss if loss_sum is None else (loss_sum + chunk_loss)

    if loss_sum is None:
        return hidden_states.sum().float() * 0.0, 0
    return loss_sum, supervised_count


def _extract_backbone_and_head(model: Any) -> tuple[Any | None, Any | None]:
    # Supports raw Qwen3_5ForCausalLM, PeftModel wrapping Qwen3_5ForCausalLM,
    # or simple synthetic test models with .model/.lm_head attributes.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base, "model", None)
    if backbone is not None and hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    lm_head = getattr(base, "lm_head", None)
    if backbone is not None and lm_head is not None:
        return backbone, lm_head
    return None, None


def run_sft_gradient_accumulation_step(
    model: Any,
    microbatches: list[dict[str, Any]],
    optimizer: Any,
    *,
    scheduler: Any = None,
    config: LoRAConfig | None = None,
    chunk_size: int | None = None,
    max_seq_len: int | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Execute one full-trajectory SFT gradient accumulation window with exact token normalization.

    Computes ``sum_m(loss_sum_m) / max(1, sum_m N_m)`` across pre-grouped window
    microbatches without sliding-window sequence slicing.

    # Errors: raises :class:`LoRAError` if a batch exceeds ``max_seq_len`` or
    contains non-finite loss/gradients.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot run SFT accumulation step")
    cfg = config or LoRAConfig()
    eff_chunk = int(chunk_size if chunk_size is not None else cfg.chunk_size)
    eff_max_seq = int(max_seq_len if max_seq_len is not None else cfg.max_seq_len)
    if not microbatches:
        raise LoRAError("microbatches must not be empty")

    total_sup_tokens = 0
    for mb in microbatches:
        labels = mb["labels"]
        seq_len = int(mb["input_ids"].shape[1])
        if seq_len > eff_max_seq:
            raise LoRAError(
                f"trajectory length {seq_len} exceeds max_seq_len={eff_max_seq}; "
                "refusing silent sliding-window slicing"
            )
        total_sup_tokens += int((labels[:, 1:] != -100).sum().item())

    if total_sup_tokens == 0:
        raise LoRAError("accumulation window has 0 supervised tokens")

    backbone, lm_head = _extract_backbone_and_head(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    accum_loss_val = 0.0
    for mb in microbatches:
        input_ids = mb["input_ids"]
        labels = mb["labels"]
        attention_mask = mb.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if backbone is not None and lm_head is not None:
                out = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                loss_sum, _n_tok = chunked_causal_ce_loss(
                    hidden,
                    lm_head,
                    labels,
                    chunk_size=eff_chunk,
                    shift_labels=True,
                    grad_checkpoint=bool(cfg.gradient_checkpointing),
                )
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                n_tok = max(1, int((labels[:, 1:] != -100).sum().item()))
                loss_sum = out.loss.float() * float(n_tok)

        norm_loss = loss_sum / float(total_sup_tokens)
        if not bool(torch.isfinite(norm_loss).all().item()):
            raise LoRAError("non-finite SFT loss encountered during accumulation")
        norm_loss.backward()
        accum_loss_val += float(norm_loss.detach().item())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm).item())
    if not np.isfinite(total_norm):
        raise LoRAError("non-finite gradient norm in SFT accumulation step")

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": accum_loss_val,
        "grad_norm": total_norm,
        "supervised_tokens": float(total_sup_tokens),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def optimizer_step(
    model: Any,
    lr: float = 5e-5,
    *,
    optimizer: Any = None,
    scheduler: Any = None,
    weight_decay: float = 0.01,
    max_grad_norm: float | None = 1.0,
) -> Any:
    """Perform an optimizer step over trainable LoRA parameters only.

    Accepts a stateful ``optimizer`` (and optional ``scheduler``) so momentum/variance
    buffers persist across steps. Falls back to one-shot AdamW creation only when
    ``optimizer is None`` for backward compatibility. Zeroes gradients afterward.
    """
    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot run optimizer step")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return optimizer
    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    return optimizer


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
    """Setup canonical PEFT LoRA model with FP32 trainable adapters and grad checkpointing."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    targets = target_modules or discover_lora_targets(model)
    peft_cfg = PeftLoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_cfg)
    # Ticket 007: Keep trainable LoRA adapters in FP32 while base weights stay BF16.
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(dtype=torch.float32)
    if gradient_checkpointing:
        if hasattr(peft_model, "gradient_checkpointing_enable"):
            peft_model.gradient_checkpointing_enable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
    return peft_model


def export_sft_adapter_bf16(
    net: Any,
    output_dir: Any,
    *,
    adapted: dict[str, Any] | None = None,
) -> Any:
    """Export trainable SFT LoRA parameters serialized as BF16 safetensors without mutating live FP32 params."""
    import os
    from pathlib import Path

    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot export BF16 adapter")
    try:
        from safetensors.torch import save_file as st_save_file
    except ImportError as exc:
        raise LoRAError(f"safetensors.torch is required to export BF16 adapter: {exc}") from exc

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(net, "peft_config") and hasattr(net, "save_pretrained"):
        net.save_pretrained(str(out), safe_serialization=True)
        model_sf = out / "adapter_model.safetensors"
        if model_sf.is_file():
            from safetensors.torch import load_file as st_load_file

            raw = st_load_file(str(model_sf), device="cpu")
            bf16_sd = {k: v.to(dtype=torch.bfloat16).contiguous() for k, v in raw.items()}
            tmp = out / "adapter_model.safetensors.tmp"
            st_save_file(bf16_sd, str(tmp))
            os.replace(tmp, model_sf)
        return out

    # Custom _LoRAWrapper / plain nn.Module fallback: save trainable params in BF16.
    tensors: dict[str, Any] = {}
    if adapted is not None:
        for name, wrapper in adapted.items():
            tensors[f"{name}.lora_A"] = wrapper.lora_A.detach().cpu().to(dtype=torch.bfloat16).contiguous()
            tensors[f"{name}.lora_B"] = wrapper.lora_B.detach().cpu().to(dtype=torch.bfloat16).contiguous()
    else:
        for name, p in net.named_parameters():
            if p.requires_grad:
                tensors[name] = p.detach().cpu().to(dtype=torch.bfloat16).contiguous()
    if not tensors:
        raise LoRAError("no trainable LoRA parameters found to export")
    tmp = out / "adapter_model.safetensors.tmp"
    st_save_file(tensors, str(tmp))
    os.replace(tmp, out / "adapter_model.safetensors")
    return out


def checkpoint_sft(
    net: Any,
    optimizer: Any,
    step: int,
    tokens: int,
    cursor: int,
    sft_dir: Any,
    template_sha: str = "",
    *,
    scheduler: Any = None,
) -> Any:
    """Persist interruptible SFT checkpoint with optimizer, scheduler, and RNG state."""
    import os
    import shutil
    from pathlib import Path

    from .loader import atomic_json

    out_dir = Path(sft_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / f"step-{step:08d}"
    tmp = snap.with_name(snap.name + ".incomplete")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if hasattr(net, "peft_config") and hasattr(net, "save_pretrained"):
        net.save_pretrained(str(tmp / "adapter"), safe_serialization=True)
    else:
        trainable = {n: p.detach().cpu().contiguous() for n, p in net.named_parameters() if p.requires_grad}
        torch.save(trainable, tmp / "adapter.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
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


def restore_sft_checkpoint(
    net: Any,
    optimizer: Any,
    sft_dir: Any,
    *,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore interruptible SFT checkpoint (weights, optimizer, scheduler, RNG, cursor)."""
    import json
    from pathlib import Path

    if not _TORCH_AVAILABLE:
        raise LoRAError("torch is not available; cannot restore SFT checkpoint")

    root = Path(sft_dir)
    if (root / "COMMITTED").is_file():
        snap = root
    elif (root / "latest.json").is_file():
        snap = Path(json.loads((root / "latest.json").read_text(encoding="utf-8"))["path"])
    else:
        raise LoRAError(f"no committed SFT checkpoint found in {root}")

    if not (snap / "COMMITTED").is_file():
        raise LoRAError(f"uncommitted SFT checkpoint: {snap}")

    state = json.loads((snap / "state.json").read_text(encoding="utf-8"))
    if (snap / "adapter").is_dir() and hasattr(net, "load_adapter"):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as st_load_file

        sf_path = snap / "adapter" / "adapter_model.safetensors"
        if sf_path.is_file():
            sd = st_load_file(str(sf_path), device="cpu")
            set_peft_model_state_dict(net, sd)
    elif (snap / "adapter.pt").is_file():
        trainable_sd = torch.load(snap / "adapter.pt", map_location="cpu", weights_only=True)
        named = dict(net.named_parameters())
        with torch.no_grad():
            for name, tensor in trainable_sd.items():
                if name not in named:
                    raise LoRAError(f"SFT checkpoint parameter missing on model: {name!r}")
                named[name].copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))
    else:
        raise LoRAError(f"SFT checkpoint missing adapter state: {snap}")

    opt_bundle = torch.load(snap / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt_bundle["optimizer"])
    if scheduler is not None and opt_bundle.get("scheduler") is not None:
        scheduler.load_state_dict(opt_bundle["scheduler"])
    if restore_rng and opt_bundle.get("rng") is not None:
        torch.set_rng_state(opt_bundle["rng"])
        if torch.cuda.is_available() and opt_bundle.get("cuda_rng"):
            torch.cuda.set_rng_state_all(opt_bundle["cuda_rng"])
    return state


__all__ = [
    "LoRAConfig",
    "LoRAResult",
    "_LoRAWrapper",
    "apply_lora_to_model",
    "discover_lora_targets",
    "setup_peft_lora",
    "export_sft_adapter_bf16",
    "create_sft_optimizer",
    "build_sft_scheduler",
    "chunked_causal_ce_loss",
    "run_sft_gradient_accumulation_step",
    "checkpoint_sft",
    "restore_sft_checkpoint",
    "freeze_targets_on_model",
    "count_trainable_params",
    "measure_coverage",
    "save_adapter",
    "load_adapter",
    "virtual_merge_parity",
    "make_training_step",
    "optimizer_step",
]
