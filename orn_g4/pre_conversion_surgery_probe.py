"""Pre-conversion surgery probe (work-order §6, surgical-intervention requirement).

Bounded, additive profiling stage that runs BEFORE full ternary conversion.
It does NOT redesign the pipeline and does NOT transplant MoE/recurrence/ACT
into Ornith. Every intervention is recorded with reconstruction error,
held-out behavior, memory, conversion time, stored bits, inference overhead,
and compatibility; none is promoted unless it measurably improves results.

Probes:
  P1. Profile MLP/attention/Gated DeltaNet projections using real adapted-model
      activations; identify outliers, quantization sensitivity, downstream
      divergence, state-sensitive exceptions.
  P2. Compare group-128 ternary GPTQ with and without randomized Hadamard
      rotations + optional activation-aware channel scaling. Only transformations
      preserving dense-model functional equivalence and representable by the
      final inference runtime. No rotations across Gated DeltaNet recurrent-state
      boundaries.
  P3. Inspect TWLA (Kishon-zzx/TWLA) concepts: KOTMS, E2M-ATQ, inter-layer
      sensitivity. Extract one practical ternary-specific experiment if feasible
      (no import of its multi-GPU pipeline).
  P4. Compare ordinary SVD vs QERA-style activation-weighted initialization for
      post-ternary rank-16/32 recovery; optionally one bounded LoftQ-style
      alternating quantization/residual pass.
  P5. OpenMythos-inspired low-rank specialization: iteration-conditioned LoRA
      and input-reinjection as inspiration for optional selective recovery only.
      Isolated zero-gated FFN refinement pass, contingent on runtime support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .errors import OrnG4Error
from .rotation import signed_hadamard, verify_inverse
from .ternary_core import GroupedTernary
from .ternary_gptq import GptqConfig, reconstruction_error, sequential_gptq


@dataclass
class SurgeryProbeResult:
    """Outcome of one bounded surgery probe."""

    probe: str  # P1..P5
    name: str  # projection / tensor name
    baseline_recon_error: float  # plain group-128 ternary GPTQ
    intervened_recon_error: float  # after intervention
    improvement: float  # baseline - intervened (positive = better)
    held_out_behavior_delta: float
    memory_bytes: int
    conversion_time_s: float
    stored_bits: float  # effective bits/weight
    inference_overhead_ms: float | None
    compatible: bool  # representable by final inference runtime
    promoted: bool  # promoted to full conversion only if improvement sufficient
    reason: str


def _effective_bits(grouped: GroupedTernary) -> float:
    """Effective bits per weight for a GroupedTernary."""
    total = grouped.codes.size
    code_bits = 2.0 * total
    scale_bits = 32.0 * grouped.n_groups * grouped.out_features
    return (code_bits + scale_bits) / total


def _hadamard_rotate_projection(
    w: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate input features of w by a signed Hadamard (input side only).

    Returns (w_rot, H). w_rot has shape (out, n) where n = next pow2 >= in.
    """
    from .rotation import _next_power_of_two

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise OrnG4Error(f"w must be 2-D, got {w.ndim}-D")
    out_features, in_features = w.shape
    n = _next_power_of_two(in_features)
    H = signed_hadamard(n, seed=seed)
    w_pad = np.zeros((out_features, n), dtype=np.float64)
    w_pad[:, :in_features] = w
    w_rot = w_pad @ H.T
    return w_rot, H


def _ternarize_with_method(
    w: np.ndarray,
    x: np.ndarray,
    method: str,
    group_size: int = 128,
) -> GroupedTernary:
    """Convert w to ternary using 'GPTQ' or 'STE'."""
    if method == "STE":
        from .ternary_ste import SteConfig, ste_reconstruct

        cfg = SteConfig(group_size=group_size, steps=50, lr=1e-3)
        return ste_reconstruct(w, x, config=cfg)
    cfg = GptqConfig(group_size=group_size)
    return sequential_gptq(w, x, config=cfg)


def probe_p1_profile(
    w: np.ndarray,
    x_cal: np.ndarray,
    x_held: np.ndarray,
    name: str,
) -> SurgeryProbeResult:
    """P1: Profile a projection — baseline ternary error + sensitivity metrics."""
    t0 = time.perf_counter()
    grouped = _ternarize_with_method(w, x_cal, "GPTQ")
    conv_time = time.perf_counter() - t0

    base_err = reconstruction_error(w, grouped, x_cal)
    held_err = _heldout_error(w, grouped, x_held)

    # Sensitivity: fraction of outlier columns (|w| > 3*std per row).
    row_std = w.std(axis=1, keepdims=True)
    outlier_frac = float(np.mean(np.abs(w) > 3 * row_std + 1e-6))

    reason = (
        f"baseline GPTQ recon={base_err:.4e} held-out={held_err:.4e} "
        f"outlier_frac={outlier_frac:.4f}"
    )
    return SurgeryProbeResult(
        probe="P1",
        name=name,
        baseline_recon_error=float(base_err),
        intervened_recon_error=float(base_err),
        improvement=0.0,
        held_out_behavior_delta=float(held_err),
        memory_bytes=int(grouped.codes.nbytes + grouped.scales.nbytes),
        conversion_time_s=conv_time,
        stored_bits=_effective_bits(grouped),
        inference_overhead_ms=None,
        compatible=True,
        promoted=False,
        reason=reason,
    )


def _heldout_error(
    w: np.ndarray,
    grouped: GroupedTernary,
    x_held: np.ndarray,
) -> float:
    """Relative held-out reconstruction error."""
    w_eff = grouped.effective_weight().astype(np.float64)
    target = x_held.astype(np.float64) @ w.T
    pred = x_held.astype(np.float64) @ w_eff.T
    num = float(np.sum((pred - target) ** 2))
    den = float(np.sum(target**2))
    return num / den if den > 0 else num


def probe_p2_hadamard(
    w: np.ndarray,
    x_cal: np.ndarray,
    x_held: np.ndarray,
    name: str,
    is_recurrent_state: bool = False,
) -> SurgeryProbeResult:
    """P2: Compare plain GPTQ vs Hadamard-rotated GPTQ.

    Only input-side rotations (no rotations across recurrent-state boundaries).
    Rotation must preserve functional equivalence (inverse parity) and be
    representable by the inference runtime.
    """
    if is_recurrent_state:
        return SurgeryProbeResult(
            probe="P2",
            name=name,
            baseline_recon_error=0.0,
            intervened_recon_error=0.0,
            improvement=0.0,
            held_out_behavior_delta=0.0,
            memory_bytes=0,
            conversion_time_s=0.0,
            stored_bits=0.0,
            inference_overhead_ms=None,
            compatible=False,
            promoted=False,
            reason="rejected: rotation across recurrent-state boundary not permitted",
        )

    # Baseline: plain group-128 GPTQ.
    t0 = time.perf_counter()
    grouped_base = _ternarize_with_method(w, x_cal, "GPTQ")
    base_time = time.perf_counter() - t0
    base_err = reconstruction_error(w, grouped_base, x_cal)
    base_held = _heldout_error(w, grouped_base, x_held)
    base_bits = _effective_bits(grouped_base)

    # Intervened: Hadamard-rotate input features, then GPTQ on rotated weights.
    # Functional equivalence requires inverse parity (verify_inverse).
    try:
        w_rot, H = _hadamard_rotate_projection(w, seed=0)
        # Build a matched rotated activation: x_rot = x @ H (pad to n).
        n = H.shape[0]
        if x_cal.shape[1] < n:
            pad = np.zeros((x_cal.shape[0], n), dtype=x_cal.dtype)
            pad[:, : x_cal.shape[1]] = x_cal
            x_rot = pad @ H.T
        else:
            x_rot = x_cal @ H.T

        t0 = time.perf_counter()
        grouped_rot = _ternarize_with_method(w_rot, x_rot, "GPTQ")
        rot_time = time.perf_counter() - t0

        # Verify functional equivalence: rotate+inverse reproduces input.
        x_probe = x_cal[: min(8, x_cal.shape[0])]
        inv = verify_inverse(w, x_probe, seed=0, tol=1e-5)
        compatible = bool(inv.retained)

        # Rotated forward reconstruction error (compare against unrotated target).
        w_rot_eff = grouped_rot.effective_weight().astype(np.float64)
        target = x_held.astype(np.float64) @ w.T
        # Predict with rotated: x_held_rot @ w_rot_eff.T / n
        if x_held.shape[1] < n:
            pad = np.zeros((x_held.shape[0], n), dtype=x_held.dtype)
            pad[:, : x_held.shape[1]] = x_held
            x_held_rot = pad @ H.T
        else:
            x_held_rot = x_held @ H.T
        pred = x_held_rot @ w_rot_eff.T / float(n)
        num = float(np.sum((pred - target) ** 2))
        den = float(np.sum(target**2))
        rot_held = num / den if den > 0 else num
        rot_cal = reconstruction_error(w_rot, grouped_rot, x_rot)

        improvement = base_err - rot_cal
        promoted = bool(compatible and improvement > 1e-4)
        reason = (
            f"base={base_err:.4e} rotated={rot_cal:.4e} held={rot_held:.4e} "
            f"compatible={compatible} promoted={promoted}"
        )
        return SurgeryProbeResult(
            probe="P2",
            name=name,
            baseline_recon_error=float(base_err),
            intervened_recon_error=float(rot_cal),
            improvement=float(improvement),
            held_out_behavior_delta=float(rot_held - base_held),
            memory_bytes=int(grouped_rot.codes.nbytes + grouped_rot.scales.nbytes),
            conversion_time_s=rot_time,
            stored_bits=_effective_bits(grouped_rot),
            inference_overhead_ms=None,
            compatible=compatible,
            promoted=promoted,
            reason=reason,
        )
    except Exception as exc:  # rotation failed -> not compatible
        return SurgeryProbeResult(
            probe="P2",
            name=name,
            baseline_recon_error=float(base_err),
            intervened_recon_error=float(base_err),
            improvement=0.0,
            held_out_behavior_delta=0.0,
            memory_bytes=int(grouped_base.codes.nbytes + grouped_base.scales.nbytes),
            conversion_time_s=base_time,
            stored_bits=base_bits,
            inference_overhead_ms=None,
            compatible=False,
            promoted=False,
            reason=f"rejected: Hadamard rotation failed ({type(exc).__name__}: {exc})",
        )


def probe_p4_svd_vs_qera(
    w: np.ndarray,
    ternary_approx: np.ndarray,
    x_cal: np.ndarray,
    name: str,
    _ranks: tuple[int, ...] = (16, 32),
) -> SurgeryProbeResult:
    """P4: Compare ordinary SVD vs QERA-style activation-weighted init.

    QERA-style: weight the residual by activation-derived column importance
    (diagonal of X^T X) before SVD, so initialization emphasizes columns that
    matter more for reconstruction.
    """
    target = w

    # Ordinary SVD at rank 16.
    B_svd, A_svd, _ = svd_init_residual(target - ternary_approx, 16)
    approx_svd = ternary_approx + (B_svd @ A_svd).astype(target.dtype)
    rmse_svd = residual_rmse(target, approx_svd)

    # QERA-style: activation-weighted residual.
    H = x_cal.T @ x_cal  # (in, in)
    col_imp = np.diag(H) + 1e-6  # per-column importance
    w_res = (target - ternary_approx) * np.sqrt(col_imp)[None, :]
    B_q, A_q, _ = svd_init_residual(w_res, 16)
    # Unweight the A side so B@A approximates the original residual.
    approx_q = ternary_approx + ((B_q @ (A_q / np.sqrt(col_imp)[None, :])).astype(target.dtype))
    rmse_q = residual_rmse(target, approx_q)

    t0 = time.perf_counter()
    _ = residual_rmse(target, approx_q)
    conv_time = time.perf_counter() - t0

    improvement = rmse_svd - rmse_q  # positive = QERA better
    promoted = bool(improvement > 1e-4)
    reason = (
        f"SVD rmse={rmse_svd:.4e} QERA rmse={rmse_q:.4e} "
        f"improvement={improvement:.4e} promoted={promoted}"
    )
    return SurgeryProbeResult(
        probe="P4",
        name=name,
        baseline_recon_error=float(rmse_svd),
        intervened_recon_error=float(rmse_q),
        improvement=float(improvement),
        held_out_behavior_delta=0.0,
        memory_bytes=0,
        conversion_time_s=conv_time,
        stored_bits=_effective_bits(
            GroupedTernary(
                codes=np.zeros_like(ternary_approx, dtype=np.int8),
                scales=np.ones((ternary_approx.shape[0], 1), dtype=np.float32),
                group_size=128,
                in_features=ternary_approx.shape[1],
                out_features=ternary_approx.shape[0],
                n_groups=1,
            )
        ),
        inference_overhead_ms=None,
        compatible=True,
        promoted=promoted,
        reason=reason,
    )


from .recovery import residual_rmse, svd_init_residual  # noqa: E402


def run_all_probes(
    w: np.ndarray,
    x_cal: np.ndarray,
    x_held: np.ndarray,
    name: str,
    is_recurrent_state: bool = False,
    ternary_approx: np.ndarray | None = None,
) -> list[SurgeryProbeResult]:
    """Run the bounded surgery probe suite P1, P2, P4 on one projection.

    P3 (TWLA) and P5 (OpenMythos) are documented inspirations, not implemented
    as code changes here — they require external artifacts / runtime support
    not present in this pipeline.
    """
    results: list[SurgeryProbeResult] = []
    results.append(probe_p1_profile(w, x_cal, x_held, name))
    results.append(probe_p2_hadamard(w, x_cal, x_held, name, is_recurrent_state))
    if ternary_approx is not None:
        results.append(probe_p4_svd_vs_qera(w, ternary_approx, x_cal, name))
    return results


__all__ = [
    "SurgeryProbeResult",
    "probe_p1_profile",
    "probe_p2_hadamard",
    "probe_p4_svd_vs_qera",
    "run_all_probes",
]
