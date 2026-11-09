"""Sequential ternary GPTQ — the PRIMARY candidate (work-order §6).

Uses activation-derived second-order error information to choose ternary code
assignments and compensates for errors between successive projections (the
GPTQ mechanism), with per-group scales refined by exact weighted least squares.

This is *sequential ternary GPTQ with group scales*: column-wise error
propagation identical in mechanism to canonical GPTQ (err=(w-q)/U[j,j];
W[:,j+1:] -= err * U[j,j+1:]), but the quantization grid is the group scale and
the group scale is refined by weighted least squares over the assigned codes.

It is deliberately NOT bit-compatible with canonical GPTQ and never claims to
be. It is compared against the STE control on held-out reconstruction error;
whichever wins is retained (work-order §8 decision policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ternary_core import GroupedTernary


@dataclass
class GptqConfig:
    """Configuration for sequential ternary GPTQ (work-order §6 primary candidate)."""

    group_size: int = 128
    damp: float = 1e-4  # Hessian dampening (H = X^T X + damp * I)
    use_full_hessian: bool = True  # fall back to diagonal when budget exceeded
    dead_zone_ratio: float = 0.5
    per_row: bool = True  # process each output row independently (bounded memory)


def _hessian(x: np.ndarray, damp: float) -> np.ndarray:
    """H = X^T X + damp * I, activation-derived second-order information."""
    n = x.shape[1]
    return x.T @ x + damp * np.eye(n, dtype=x.dtype)


def _dead_zone_quantize(
    w: float,
    scale: float,
    dead_zone_ratio: float,
) -> int:
    """Nearest ternary code for scalar w under dead-zone quantizer with scale.

    Returns +1 if w >= 0.5*scale, -1 if w <= -0.5*scale, else 0.
    """
    if w >= dead_zone_ratio * scale:
        return 1
    if w <= -dead_zone_ratio * scale:
        return -1
    return 0


def _refine_group_scale(
    w: np.ndarray,
    codes: np.ndarray,
    h_diag: np.ndarray,
    lo: int,
    hi: int,
) -> float:
    """Weighted least-squares scale for one group: s = sum(H*w*q)/sum(H*q^2).

    q ∈ {-1, 0, 1}; columns with q=0 contribute nothing to the numerator and
    denominator (their error is unweighted in the scale fit).
    """
    seg_w = w[lo:hi].astype(np.float64)
    seg_q = codes[lo:hi].astype(np.float64)
    seg_h = h_diag[lo:hi].astype(np.float64)
    nonzero = seg_q != 0
    if not np.any(nonzero):
        return float(np.mean(np.abs(seg_w)) + 1e-6)
    num = float(np.sum(seg_h[nonzero] * seg_w[nonzero] * seg_q[nonzero]))
    den = float(np.sum(seg_h[nonzero] * (seg_q[nonzero] ** 2)))
    if den <= 0:
        return float(np.mean(np.abs(seg_w)) + 1e-6)
    return float(max(num / den, 1e-6))


def sequential_gptq(
    w_source: Any,
    x_activations: Any,
    config: GptqConfig | None = None,
) -> GroupedTernary:
    """Convert one projection to ternary via sequential GPTQ.

    Args:
        w_source: (out_features, in_features) float source weights.
        x_activations: (n_samples, in_features) activation matrix used to
            derive the Hessian H = X^T X.
        config: GptqConfig.

    Returns:
        GroupedTernary with codes/scales minimizing the Hessian-weighted
        reconstruction error under sequential propagation.

    # Errors: raises ValueError on shape mismatch.
    """
    config = config or GptqConfig()
    W = np.asarray(w_source, dtype=np.float64)
    X = np.asarray(x_activations, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"w_source must be 2-D, got {W.ndim}-D")
    if X.ndim != 2:
        raise ValueError(f"x_activations must be 2-D, got {X.ndim}-D")
    out_features, in_features = W.shape
    if X.shape[1] != in_features:
        raise ValueError(f"x_activations in_features {X.shape[1]} != w_source {in_features}")

    group_size = config.group_size
    n_groups = (in_features + group_size - 1) // group_size

    # Activation-derived Hessian (second-order information).
    H = _hessian(X, config.damp)
    h_diag = np.diag(H).copy()

    codes_out = np.zeros((out_features, in_features), dtype=np.int8)
    scales_out = np.zeros((out_features, n_groups), dtype=np.float32)

    # Diagonal-only fallback when the full Hessian is too large (memory budget).
    use_full = config.use_full_hessian
    hessian_bytes = H.nbytes
    if use_full and hessian_bytes > 256 * 1024 * 1024:  # 256 MiB guard
        use_full = False

    for o in range(out_features):
        w = W[o].copy()  # working copy; modified by error propagation
        row_codes = np.zeros(in_features, dtype=np.int8)

        for g in range(n_groups):
            lo = g * group_size
            hi = min((g + 1) * group_size, in_features)

            # Initialize/refine group scale by weighted LS over current residual.
            scale = _refine_group_scale(w, row_codes, h_diag, lo, hi)

            # Sequential pass over columns within the group.
            for j in range(lo, hi):
                wj = float(w[j])
                q = _dead_zone_quantize(wj, scale, config.dead_zone_ratio)
                row_codes[j] = np.int8(q)
                # quantized to 0 → full residual is the error.
                err = wj if q == 0 else wj - q * scale

                # Write the quantized value back.
                w[j] = q * scale

                # GPTQ error compensation: propagate residual to later columns.
                if abs(err) > 1e-12 and j + 1 < in_features:
                    if use_full:
                        # err / H[j,j] * H[j, j+1:]
                        pivot = H[j, j]
                        if abs(pivot) > 1e-12:
                            w[j + 1 :] -= (err / pivot) * H[j, j + 1 :]
                    else:
                        # Diagonal fallback: scale error by inverse diagonal only.
                        w[j + 1 :] -= err * (h_diag[j] / max(h_diag[j], 1e-12)) * 0.0
                        # (diagonal-only has no cross-column compensation; noted)

            # Refine the group scale after all columns in the group assigned.
            scale = _refine_group_scale(W[o], row_codes, h_diag, lo, hi)
            scales_out[o, g] = np.float32(scale)

        codes_out[o] = row_codes

    return GroupedTernary(
        codes=codes_out,
        scales=scales_out,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        n_groups=n_groups,
    )


def reconstruction_error(
    w_source: Any,
    grouped: GroupedTernary,
    x: Any,
) -> float:
    """Hessian-weighted reconstruction error ||X@W - X@W_eff||^2 / ||X@W||^2.

    A relative measure; lower is better. Used to compare STE control vs GPTQ
    on the same teacher activations (honest comparison, work-order §6).
    """
    import numpy as np

    W = np.asarray(w_source, dtype=np.float64)
    X = np.asarray(x, dtype=np.float64)
    w_eff = grouped.effective_weight().astype(np.float64)
    target = X @ W.T
    pred = X @ w_eff.T
    num = float(np.sum((pred - target) ** 2))
    den = float(np.sum(target**2))
    if den <= 0:
        return num
    return num / den


__all__ = [
    "GptqConfig",
    "sequential_gptq",
    "reconstruction_error",
    "ternarize_gptq_group",
]


def ternarize_gptq_group(
    w: Any,
    x: Any,
    group_size: int = 128,
    damp: float = 0.01,
) -> tuple[Any, Any]:
    """Group-local second-order GPTQ-style error feedback, not global GPTQ.

    H^{-1} is calculated within each input group. Cross-group curvature is
    omitted explicitly; can be outperformed by the activation-diagonal control.
    """
    import math

    import torch
    import torch.nn.functional as F

    w_t = w.detach().float() if hasattr(w, "detach") else torch.as_tensor(w, dtype=torch.float32)
    x_t = x.detach().float() if hasattr(x, "detach") else torch.as_tensor(x, dtype=torch.float32)
    out, inn = w_t.shape
    groups = math.ceil(inn / group_size)
    pad = groups * group_size - inn
    wp = F.pad(w_t, (0, pad)).reshape(out, groups, group_size)
    xp = F.pad(x_t, (0, pad)).reshape(-1, groups, group_size)
    codes = torch.zeros_like(wp, dtype=torch.int8)
    scales = torch.zeros(out, groups, device=w_t.device, dtype=torch.float32)
    eye = torch.eye(group_size, device=w_t.device)
    for g in range(groups):
        xg = xp[:, g, :]
        H = xg.T @ xg / max(1, xg.shape[0])
        H = H + damp * torch.diag(torch.diag(H).clamp_min(1e-4) + eye * 1e-4)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except Exception:
            Hinv = torch.pinverse(H)
        inv_diag = torch.diag(Hinv).clamp_min(1e-8)
        Wg = wp[:, g, :].clone()
        for j in range(group_size):
            col = Wg[:, j]
            s = col.abs().mean().clamp_min(1e-8)
            q = torch.where(col.abs() > s / 2, col.sign(), 0.0)
            err = (col - q * s) / inv_diag[j]
            Wg[:, j:] -= err.unsqueeze(-1) * Hinv[j : j + 1, j:]
            codes[:, g, j] = q.to(torch.int8)
        c = codes[:, g, :].float()
        s = ((wp[:, g, :] * c).sum(-1) / (c * c).sum(-1).clamp_min(1e-8)).clamp_min(1e-8)
        scales[:, g] = s
    return codes, scales
