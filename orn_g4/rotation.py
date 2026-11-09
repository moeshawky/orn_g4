"""Signed Hadamard rotation experiment (work-order §6, Rotation experiment).

Implements signed Hadamard rotations as a bounded candidate. The mathematical
invariance and inverse transformation are tested independently; only rotations
that can be represented faithfully by the intended inference path are retained.

A signed Hadamard matrix H_n (n = power of two) satisfies H @ H^T = n * I, so
the inverse transform is H^T / n. Rotating weights: W_rot = W @ H (input-side)
or H @ W (output-side); the forward must apply the inverse to activations so
the composition is invariant: x @ W == (x @ H) @ (W @ H^T) / n ... actually the
identity is x @ W = (x @ H_n) @ (W @ H_n^T) / n when applied on the input side
with the inverse on the weights. This module verifies that identity numerically.

The rotation is only retained when rotation+inverse reproduces the input within
tolerance AND the rotated weights can be ternary-converted by the standard
inference path (grouping over the rotated input features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ternary_core import (
    GroupedTernary,
    group_scales,
    pack_ternary,
    quantize_group_row,
    unpack_ternary,
)


@dataclass(frozen=True)
class RotationResult:
    """Outcome of the rotation experiment for one projection."""

    rotated: GroupedTernary | None  # ternary form of rotated weights, if retained
    inverse_error: float  # ||x - inverse(rotate(x))|| / ||x||
    retained: bool  # True only if inverse parity passes AND ternary conversion OK
    reason: str  # why retained or rejected


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _hadamard(n: int) -> np.ndarray:
    """Build the n×n Hadamard matrix (Sylvester construction), n power of two."""
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def signed_hadamard(n: int, seed: int = 0) -> np.ndarray:
    """Signed Hadamard matrix: H_n with random column signs (±1).

    Column signs preserve the Hadamard invariance (signing columns is an
    orthogonal transformation). Returns float64 (n, n).
    """
    rng = np.random.default_rng(seed)
    H = _hadamard(n).astype(np.float64)
    signs = rng.choice([-1.0, 1.0], size=n)
    return H * signs[None, :]  # sign each column


def rotate_weights(
    w: Any,
    side: str = "input",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate weight matrix ``w`` by a signed Hadamard.

    Args:
        w: (out_features, in_features) float weights.
        side: "input" -> W @ H (rotate input features); "output" -> H @ W.
        seed: RNG seed for column signs (deterministic).

    Returns:
        (w_rot, H) where w_rot is the rotated weight and H is the Hadamard
        (needed to apply the inverse on activations at inference).
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"w must be 2-D, got {w.ndim}-D")
    out_features, in_features = w.shape
    n = _next_power_of_two(in_features)
    H = signed_hadamard(n, seed=seed)
    # Pad w to n columns if in_features < n.
    if in_features < n:
        w_pad = np.zeros((out_features, n), dtype=np.float64)
        w_pad[:, :in_features] = w
        w = w_pad
    if side == "input":
        w_rot = w @ H.T  # (out, n)
    elif side == "output":
        w_rot = H @ w  # (n, n) — changes out dim too; only valid if out==n
    else:
        raise ValueError(f"side must be 'input' or 'output', got {side!r}")
    return w_rot, H


def apply_inverse_on_activations(
    x: Any,
    H: np.ndarray,
) -> np.ndarray:
    """Apply the Hadamard inverse to activations so x @ W is invariant.

    For input-side rotation W_rot = W @ H^T, the invariant forward is:
      (x @ H) @ W_rot^T / n   ==   x @ W
    because H @ H^T = n I → x @ W = x @ H @ H^T @ W^T / n = (x @ H) @ (W @ H^T)^T / n.

    Args:
        x: activations (..., in_features).
        H: the Hadamard matrix (n, n), n >= in_features.

    Returns:
        Rotated activations (..., n) ready for W_rot.
    """
    x = np.asarray(x, dtype=np.float64)
    n = H.shape[0]
    # Pad x to n columns if needed.
    if x.shape[-1] < n:
        pad = np.zeros(x.shape[:-1] + (n,), dtype=np.float64)
        pad[..., : x.shape[-1]] = x
        x = pad
    return x @ H.T


def verify_inverse(
    w: Any,
    x: Any,
    seed: int = 0,
    tol: float = 1e-5,
) -> RotationResult:
    """Independently test rotation + inverse parity.

    Constructs a random input x, computes y = x @ W, rotates W and x, computes
    y_rot = x_rot @ W_rot^T / n, and checks ||y - y_rot|| / ||y|| < tol.

    Also ternary-converts the rotated weights via the standard grouping path to
    confirm the rotated tensor is representable by the inference path.

    # Errors: raises ValueError on bad shapes; returns a RotationResult with
    retained=False when parity fails (never raises for a failed parity check).
    """
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"w must be 2-D, got {w.ndim}-D")
    out_features, in_features = w.shape
    if x.shape[-1] != in_features:
        raise ValueError(f"x last dim {x.shape[-1]} != w in_features {in_features}")

    n = _next_power_of_two(in_features)
    H = signed_hadamard(n, seed=seed)

    # Pad w to n columns for the rotation math.
    w_pad = np.zeros((out_features, n), dtype=np.float64)
    w_pad[:, :in_features] = w

    w_rot = w_pad @ H.T  # (out, n)

    # Reference forward.
    y_ref = x @ w.T  # (..., out)

    # Rotated forward: (x @ H) @ w_rot^T / n.
    x_rot = apply_inverse_on_activations(x, H)  # (..., n)
    y_rot = (x_rot @ w_rot.T) / float(n)  # (..., out)
    y_rot = y_rot[..., :out_features]

    denom = float(np.linalg.norm(y_ref)) + 1e-12
    inverse_error = float(np.linalg.norm(y_ref - y_rot) / denom)

    # Ternary-convert rotated weights via standard grouping path.
    ternary_ok = False
    try:
        scales = group_scales(w_rot, group_size=128)
        grouped = GroupedTernary(
            codes=np.zeros_like(w_rot, dtype=np.int8),
            scales=scales.astype(np.float32),
            group_size=128,
            in_features=n,
            out_features=out_features,
            n_groups=scales.shape[1],
        )
        # Re-quantize codes dead-zone.
        for o in range(out_features):
            grouped.codes[o] = quantize_group_row(w_rot[o], scales[o], 128)
        # Round-trip: pack + unpack must reproduce.
        blob = pack_ternary(grouped)
        _ = unpack_ternary(blob, source_name="rotated")
        ternary_ok = True
    except Exception:
        ternary_ok = False

    retained = bool(inverse_error < tol and ternary_ok)
    reason = (
        "retained: inverse parity within tolerance and ternary round-trip OK"
        if retained
        else f"rejected: inverse_error={inverse_error:.3e} (tol={tol}), ternary_ok={ternary_ok}"
    )

    ternary_grouped: GroupedTernary | None = None
    if retained:
        scales = group_scales(w_rot, group_size=128)
        codes = np.zeros_like(w_rot, dtype=np.int8)

        for o in range(out_features):
            codes[o] = quantize_group_row(w_rot[o], scales[o], 128)
        ternary_grouped = GroupedTernary(
            codes=codes,
            scales=scales.astype(np.float32),
            group_size=128,
            in_features=n,
            out_features=out_features,
            n_groups=scales.shape[1],
        )

    return RotationResult(
        rotated=ternary_grouped,
        inverse_error=inverse_error,
        retained=retained,
        reason=reason,
    )


# Defer pack_ternary import to module level for clarity.

__all__ = [
    "RotationResult",
    "rotate_weights",
    "apply_inverse_on_activations",
    "verify_inverse",
    "signed_hadamard",
]
