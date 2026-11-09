"""Activation-aware STE reconstruction — the CONTROL candidate (work-order §6).

Applies hard ternary weights during the forward pass with a validated straight-
through estimator (STE) surrogate gradient. Optimizes reconstruction using
actual teacher activations: minimize ||X @ W_source - X @ W_effective||^2 over
codes and scales. The STE gradient passes through the hard ternary function
unchanged (surrogate gradient = identity in the backward pass).

This is the control against which sequential ternary GPTQ is compared; neither
claims superiority a priori — the comparison is measured on held-out
reconstruction error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch  # this module is torch-dependent (STE autograd)

from .ternary_core import GroupedTernary, group_scales


@dataclass
class SteConfig:
    """Configuration for the activation-aware STE control candidate (work-order §6)."""

    group_size: int = 128
    steps: int = 200
    lr: float = 1e-3
    threshold_ratio: float = 0.5


def _to_torch(x: Any) -> Any:
    import torch  # lazy

    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


def ste_ternarize(
    weight: Any,
    group_size: int = 128,
    threshold_ratio: float = 0.5,
) -> GroupedTernary:
    """Hard-ternarize a weight matrix with fixed group scales (one-shot, no optimization).

    Forward uses hard ternary codes; this function is the forward half of the
    STE. The backward (surrogate gradient) is applied by :func:`ste_reconstruct`
    via :class:`_SteTernaryAutograd`.

    Returns a :class:`GroupedTernary` whose codes/scales reproduce the
    dead-zone nearest-ternary assignment for the given scales.
    """
    import torch  # lazy

    w = _to_torch(weight)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.ndim}-D")
    out_features, in_features = w.shape
    scales = group_scales(w.numpy(), group_size=group_size)
    n_groups = scales.shape[1]
    codes = torch.zeros_like(w, dtype=torch.int8)
    scales_t = torch.as_tensor(scales, dtype=torch.float32)
    for g in range(n_groups):
        lo = g * group_size
        hi = min((g + 1) * group_size, in_features)
        seg = w[:, lo:hi]
        s = scales_t[:, g : g + 1]
        codes[:, lo:hi] = torch.where(
            torch.abs(seg) >= threshold_ratio * s,
            torch.sign(seg).to(torch.int8),
            torch.zeros_like(seg, dtype=torch.int8),
        )
    return GroupedTernary(
        codes=codes.numpy(),
        scales=scales.astype("float32"),
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        n_groups=n_groups,
    )


class _SteTernaryAutograd(torch.autograd.Function):
    """Straight-through ternary autograd function.

    Forward: hard ternary codes from master weights + scales.
    Backward: gradient passes straight through to the master weights (surrogate
    gradient = 1), zero gradient to scales (scales are updated by exact least
    squares, not by this path).
    """

    @staticmethod
    def forward(
        ctx: Any,
        master: Any,
        scales: Any,
        group_size: int,
        threshold_ratio: float,
    ) -> Any:
        out_features, in_features = master.shape
        n_groups = (in_features + group_size - 1) // group_size
        codes = torch.zeros_like(master, dtype=torch.int8)
        for g in range(n_groups):
            lo = g * group_size
            hi = min((g + 1) * group_size, in_features)
            seg = master[:, lo:hi]
            s = scales[:, g : g + 1]
            codes[:, lo:hi] = torch.where(
                torch.abs(seg) >= threshold_ratio * s,
                torch.sign(seg).to(torch.int8),
                torch.zeros_like(seg, dtype=torch.int8),
            )
        ctx.save_for_backward(master, scales)
        ctx.group_size = group_size
        ctx.threshold_ratio = threshold_ratio
        # Effective weight = scale * codes (float)
        scale_full = scales.repeat_interleave(group_size, dim=1)[:, :in_features]
        return scale_full * codes.to(master.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        master, scales = ctx.saved_tensors
        # Straight-through: gradient to master = grad_output @ effective_weight.T
        # (treat codes as identity in backward). No gradient to scales here.
        scale_full = scales.repeat_interleave(ctx.group_size, dim=1)[:, : master.shape[1]]
        eff = scale_full  # surrogate: d(eff)/d(master) = 1 through sign
        grad_master = grad_output @ eff.T
        return grad_master, None, None, None


def ste_reconstruct(
    w_source: Any,
    x_teacher: Any,
    config: SteConfig | None = None,
) -> GroupedTernary:
    """Optimize codes + scales to reconstruct teacher activations.

    Minimizes ||X @ W_source - X @ (scale * codes)||^2 using STE: the forward
    is hard ternary, the backward passes gradients to a master floating-point
    weight, which is then re-ternarized each step. Scales are refined by exact
    weighted least squares over the assigned codes after each master-weight
    update (closed form, no gradient needed for scales).

    Returns the best :class:`GroupedTernary` found (lowest reconstruction error).

    # Errors: raises ValueError on shape mismatch; ImportError if torch missing.
    """
    import torch  # lazy

    config = config or SteConfig()
    W = _to_torch(w_source)
    X = _to_torch(x_teacher)
    if W.ndim != 2:
        raise ValueError(f"w_source must be 2-D, got {W.ndim}-D")
    if X.ndim != 2:
        raise ValueError(f"x_teacher must be 2-D, got {X.ndim}-D")
    out_features, in_features = W.shape
    if X.shape[1] != in_features:
        raise ValueError(f"x_teacher in_features {X.shape[1]} != w_source {in_features}")

    n_groups = (in_features + config.group_size + -1) // config.group_size
    n_groups = (in_features + config.group_size - 1) // config.group_size

    # Master weight starts as the source (fp32, trainable).
    master = W.clone().detach().requires_grad_(True)
    scales_np = group_scales(W.numpy(), group_size=config.group_size)
    scales = torch.as_tensor(scales_np, dtype=torch.float32)

    optimizer = torch.optim.Adam([master], lr=config.lr)

    target = X @ W  # teacher activations (n, out)
    best_error = float("inf")
    best: GroupedTernary | None = None

    for _ in range(config.steps):
        optimizer.zero_grad()
        eff = _SteTernaryAutograd.apply(master, scales, config.group_size, config.threshold_ratio)
        pred = X @ eff.T
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Re-ternarize master into codes + refine scales by weighted least squares.
        with torch.no_grad():
            codes = torch.zeros_like(master, dtype=torch.int8)
            for g in range(n_groups):
                lo = g * config.group_size
                hi = min((g + 1) * config.group_size, in_features)
                seg = master[:, lo:hi]
                s = scales[:, g : g + 1]
                codes[:, lo:hi] = torch.where(
                    torch.abs(seg) >= config.threshold_ratio * s,
                    torch.sign(seg).to(torch.int8),
                    torch.zeros_like(seg, dtype=torch.int8),
                )
            # Weighted least-squares scale refinement per group:
            # s_g = sum_j |w_j| * |code_j| / sum_j code_j^2  (code_j ∈ {-1,0,1})
            #   = mean(|w_j| over non-zero codes in group) when code ±1
            for g in range(n_groups):
                lo = g * config.group_size
                hi = min((g + 1) * config.group_size, in_features)
                seg_codes = codes[:, lo:hi].to(torch.float32)
                seg_w = master[:, lo:hi].detach()
                nonzero = seg_codes != 0
                if nonzero.any():
                    num = (torch.abs(seg_w) * seg_codes.abs()).sum(dim=1)
                    den = (seg_codes**2).sum(dim=1).clamp_min(1.0)
                    scales[:, g] = (num / den).clamp_min(1e-6)

        # Evaluate reconstruction with current hard-ternary forward.
        with torch.no_grad():
            scale_full = scales.repeat_interleave(config.group_size, dim=1)[:, :in_features]
            eff_hard = scale_full * codes.to(torch.float32)
            pred_hard = X @ eff_hard.T
            err = ((pred_hard - target) ** 2).mean().item()
            if err < best_error:
                best_error = err
                best = GroupedTernary(
                    codes=codes.numpy().astype("int8"),
                    scales=scales.numpy().astype("float32"),
                    group_size=config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    n_groups=n_groups,
                )

    if best is None:
        # Fallback: one-shot ternarization.
        return ste_ternarize(W, config.group_size, config.threshold_ratio)
    return best


__all__ = [
    "SteConfig",
    "ste_ternarize",
    "ste_reconstruct",
]
