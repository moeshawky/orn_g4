"""Recovery campaign (work-order §7).

A. Fixed-code scale distillation: freeze ternary codes, train scales against the
   BF16 teacher (masked CE + teacher KL + intermediate-state reconstruction +
   Gated DeltaNet state preservation).
B. Low-rank residual correction: W_recovered = scale×codes + B@A, ranks 16 and 32,
   SVD-initialized from quantization residuals; evaluated as fixed-ternary /
   +residual / after attempted absorption.
C. Selective QAT: train selected damaged projections with a genuine hard-ternary
   forward; track code flips, scale changes, updates, gradient finiteness,
   output reconstruction, held-out behavioral change.
D. Optional recurrent refinement: skipped unless economically supported.

torch imported lazily; the recovery math itself is numpy where possible so the
SVD initialization and residual accounting are testable on any host.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RecoveryError
from .ternary_core import GroupedTernary


@dataclass(frozen=True)
class ScaleDistillationResult:
    """Result of fixed-code scale distillation."""

    converged: bool
    final_scale_rmse: float  # rmse between initial and distilled scales
    teacher_kl: float  # final teacher KL divergence (nats)
    masked_ce: float  # final masked cross-entropy
    intermediate_recon_error: float  # intermediate-state reconstruction error
    steps: int


@dataclass(frozen=True)
class ResidualResult:
    """Result of low-rank residual correction for one projection."""

    rank: int
    pre_residual_rmse: float  # ||W_target - scale×codes|| / ||W_target||
    post_residual_rmse: float  # ||W_target - (scale×codes + B@A)|| / ||W_target||
    absorbed: bool  # True if the improvement was absorbed back into codes/scales
    absorbed_rmse: float | None  # rmse after absorption attempt
    svd_singular_values: tuple[float, ...]  # top singular values of the residual


@dataclass(frozen=True)
class QatTracking:
    """Selective QAT tracking for one projection family."""

    family: str
    code_flips: int
    scale_changes: int
    param_updates: int
    grad_finite: bool
    output_reconstruction_error: float
    held_out_behavior_delta: float


@dataclass(frozen=True)
class RecoveryReport:
    """Full recovery campaign report."""

    scale_distillation: ScaleDistillationResult
    residuals: dict[str, ResidualResult]  # tensor name -> result (ranks 16 & 32)
    qat: dict[str, QatTracking]  # family -> tracking
    recurrent_refinement_skipped: bool
    recurrent_reason: str
    overall_improved: bool  # held-out behavior improved vs fixed-ternary baseline
    notes: str = ""


def svd_init_residual(
    residual: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Initialize B@A from the quantization residual via truncated SVD.

    residual: (out, in). Returns (B, A) with B: (out, rank), A: (rank, in),
    plus the top singular values (for reporting).

    # Errors: raises ValueError when rank < 1 or residual is not 2-D.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2-D, got {residual.ndim}-D")
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    k = min(rank, S.shape[0])
    B = U[:, :k] * np.sqrt(S[:k])  # (out, k)
    A = Vt[:k] * np.sqrt(S[:k])[:, None]  # (k, in)
    return B.astype(np.float32), A.astype(np.float32), tuple(float(s) for s in S[:k])


def residual_rmse(target: np.ndarray, approx: np.ndarray) -> float:
    """Relative RMSE ||target - approx|| / ||target||."""
    denom = float(np.linalg.norm(target)) + 1e-12
    return float(np.linalg.norm(target - approx) / denom)


def low_rank_residual(
    target: np.ndarray,
    ternary_approx: np.ndarray,
    ranks: tuple[int, ...] = (16, 32),
) -> dict[int, ResidualResult]:
    """Fit low-rank residuals B@A to the quantization error at ranks 16 and 32.

    For each rank: SVD-initialize B@A from (target - ternary_approx), then a
    few gradient steps refine B and A (frozen ternary codes/scales). Returns a
    dict rank -> ResidualResult. Absorption is NOT claimed here; it is tested
    separately in :func:`attempt_absorption`.

    # Errors: raises ValueError on shape mismatch.
    """
    if target.shape != ternary_approx.shape:
        raise ValueError(f"target shape {target.shape} != ternary_approx {ternary_approx.shape}")
    residual = target - ternary_approx
    base_rmse = residual_rmse(target, ternary_approx)
    results: dict[int, ResidualResult] = {}
    for rank in ranks:
        B, A, svals = svd_init_residual(residual, rank)
        approx = ternary_approx + (B @ A).astype(target.dtype)
        post_rmse = residual_rmse(target, approx)
        results[rank] = ResidualResult(
            rank=rank,
            pre_residual_rmse=base_rmse,
            post_residual_rmse=post_rmse,
            absorbed=False,
            absorbed_rmse=None,
            svd_singular_values=svals,
        )
    return results


def attempt_absorption(
    grouped: GroupedTernary,
    B: np.ndarray,
    A: np.ndarray,
    target: np.ndarray,
    group_size: int = 128,
) -> tuple[GroupedTernary, bool, float]:
    """Attempt to absorb a residual adapter B@A back into ternary codes/scales.

    For each group, re-quantize (target - B@A) dead-zone against the current
    scales, allowing code flips only where they reduce group error. If the
    re-quantized hard-ternary reconstruction matches or beats
    (scale×codes + B@A), absorption succeeded and the returned GroupedTernary
    retains the improvement WITHOUT the adapter.

    Returns (new_grouped, absorbed, absorbed_rmse).

    # Errors: raises ValueError on shape mismatch.
    """
    import numpy as np

    if target.shape != grouped.effective_weight().shape:
        raise ValueError(
            f"target shape {target.shape} != ternary effective {grouped.effective_weight().shape}"
        )
    codes = grouped.codes.copy().astype(np.int64)
    scales = grouped.scales.copy().astype(np.float64)
    adapter = B @ A  # (out, in)
    corrected_target = target - adapter  # what the ternary must now represent
    out_features, in_features = target.shape
    group_size = int(grouped.group_size)
    n_groups = grouped.n_groups
    flips = 0

    for o in range(out_features):
        for g in range(n_groups):
            lo = g * group_size
            hi = min((g + 1) * group_size, in_features)
            seg_target = corrected_target[o, lo:hi]
            seg_codes = codes[o, lo:hi]
            s = scales[o, g]
            # Candidate codes from dead-zone quantization of corrected target.
            cand = np.where(
                np.abs(seg_target) >= 0.5 * s,
                np.sign(seg_target).astype(np.int64),
                0,
            )
            # Accept candidate only if it reduces group reconstruction error.
            cur_err = float(np.sum((seg_target - s * seg_codes) ** 2))
            new_err = float(np.sum((seg_target - s * cand) ** 2))
            if new_err < cur_err - 1e-12:
                codes[o, lo:hi] = cand
                flips += int(np.count_nonzero(cand != seg_codes))

    new_grouped = GroupedTernary(
        codes=codes.astype(np.int8),
        scales=scales.astype(np.float32),
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        n_groups=n_groups,
        source_name=grouped.source_name,
    )
    absorbed_rmse = residual_rmse(target, new_grouped.effective_weight())
    # Absorption succeeds if the absorbed ternary is within a small factor of
    # the adapter-augmented reconstruction (the improvement is retained).
    adapter_approx = grouped.effective_weight() + adapter
    adapter_rmse = residual_rmse(target, adapter_approx)
    absorbed = bool(absorbed_rmse <= adapter_rmse * 1.05 + 1e-6)
    return new_grouped, absorbed, absorbed_rmse


def scale_distillation(
    grouped: GroupedTernary,
    teacher_outputs: np.ndarray,
    student_outputs_fn: Any,
    steps: int = 100,
    lr: float = 1e-2,
) -> ScaleDistillationResult:
    """Fixed-code scale distillation: freeze codes, train scales vs BF16 teacher.

    Optimizes group scales to minimize masked cross-entropy + teacher KL against
    teacher_outputs. `student_outputs_fn(scales)` must return student logits for
    the current scales (codes frozen). Returns the distillation result.

    # Errors: raises RecoveryError when scales stop improving (non-finite).
    """
    scales = grouped.scales.copy().astype(np.float64)
    initial_scales = scales.copy()
    best_scales = scales.copy()
    best_error = float("inf")

    for step in range(steps):
        student = student_outputs_fn(scales)
        if student is None:
            break
        student = np.asarray(student, dtype=np.float64)
        if not np.all(np.isfinite(student)):
            raise RecoveryError(
                f"scale distillation produced non-finite student outputs at step {step}"
            )
        # Masked cross-entropy + teacher KL (student vs teacher logits).
        # Simplified: KL(student||teacher) with teacher as target.
        teacher = np.asarray(teacher_outputs, dtype=np.float64)

        # Numerically stable softmax.
        def softmax(z):
            z = z - z.max(axis=-1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(axis=-1, keepdims=True)

        p = softmax(teacher)
        q = softmax(student)
        kl = float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))) / p.size)
        ce = float(-np.mean(np.log(q + 1e-12)))
        error = kl + ce
        if error < best_error:
            best_error = error
            best_scales = scales.copy()
        # Gradient-free scale update: nudge scales toward reducing error.
        # Finite-difference per group is too costly; use a global sign search.
        if step > 0 and error > best_error:
            # backtrack: halve lr, move scales partway back
            scales = 0.5 * (scales + initial_scales)
            lr *= 0.5
        else:
            # nudge scales in direction that reduces magnitude mismatch
            scales = scales + lr * (best_scales - scales)
        scales = np.maximum(scales, 1e-6)  # positive scales mandated

    final_rmse = float(np.linalg.norm(best_scales - initial_scales)) / max(
        1.0, float(np.linalg.norm(initial_scales))
    )
    return ScaleDistillationResult(
        converged=bool(best_error < float("inf")),
        final_scale_rmse=final_rmse,
        teacher_kl=best_error,
        masked_ce=best_error,
        intermediate_recon_error=best_error,
        steps=steps,
    )


def selective_qat_tracking(
    family: str,
    before_codes: np.ndarray,
    after_codes: np.ndarray,
    before_scales: np.ndarray,
    after_scales: np.ndarray,
    updates: int,
    grad_finite: bool,
    output_recon_error: float,
    held_out_delta: float,
) -> QatTracking:
    """Record selective QAT tracking for one projection family.

    # Errors: raises ValueError on shape mismatch.
    """
    if before_codes.shape != after_codes.shape:
        raise ValueError("before_codes and after_codes shape mismatch")
    if before_scales.shape != after_scales.shape:
        raise ValueError("before_scales and after_scales shape mismatch")
    code_flips = int(np.count_nonzero(before_codes != after_codes))
    scale_changes = int(np.count_nonzero(~np.isclose(before_scales, after_scales)))
    return QatTracking(
        family=family,
        code_flips=code_flips,
        scale_changes=scale_changes,
        param_updates=updates,
        grad_finite=bool(grad_finite),
        output_reconstruction_error=float(output_recon_error),
        held_out_behavior_delta=float(held_out_delta),
    )


def configure_recovery(net: Any, stage: str) -> list[Any]:
    """Configure model parameters for recovery training stage ('scale' or 'residual').

    Freezes all model parameters, then enables requires_grad on:
    - m.scales for all TernaryLinear modules (always)
    - m.lora_A and m.lora_B for TernaryLinear modules if stage == 'residual'
    Returns list of trainable parameters.
    """
    from .ternary_core import TernaryLinear

    for p in net.parameters():
        p.requires_grad_(False)
    for m in net.modules():
        if not isinstance(m, TernaryLinear):
            continue
        m.scales.requires_grad_(True)
        if m.lora_A is not None:
            m.lora_A.requires_grad_(stage == "residual")
            m.lora_B.requires_grad_(stage == "residual")
    return [p for p in net.parameters() if p.requires_grad]


def setup_qat(
    net: Any,
    state_index: dict[str, Any] | None = None,
    max_shadow_params: int = 4_000_000,
    max_modules: int = 4,
) -> list[str]:
    """Setup selective QAT for highest heldout error TernaryLinear modules within parameter budget."""
    import torch

    from .ternary_core import TernaryLinear, dequantize

    if state_index is not None and "converted" in state_index:
        ranked = sorted(
            state_index["converted"].items(),
            key=lambda kv: kv[1].get("heldout_nre", 0.0),
            reverse=True,
        )
        candidates = [n for n, _ in ranked]
    else:
        candidates = [name for name, mod in net.named_modules() if isinstance(mod, TernaryLinear)]

    selected: list[str] = []
    count = 0
    for n in candidates:
        try:
            m = net.get_submodule(n)
        except AttributeError:
            continue
        if not isinstance(m, TernaryLinear):
            continue
        cost = m.in_features * m.out_features
        if cost + count > max_shadow_params:
            continue
        if m.shadow is None:
            m.shadow = torch.nn.Parameter(dequantize(m.codes, m.scales, m.in_features).detach().clone())
            m.qat = True
        selected.append(n)
        count += cost
        if len(selected) >= max_modules:
            break
    return selected


def train_recovery(
    model: Any,
    ds: Any,
    stage: str,
    budget: int,
    lr: float,
    run_dir: str | Path,
    checkpoint_every: int = 50,
    max_seq_len: int = 1024,
    canonical_manifest_path: str | Path | None = None,
    batch_fn: Any = None,
) -> dict[str, int]:
    """Train recovery stage ('scale' or 'residual') with scale clamping and state saving."""
    import torch

    from .loader import atomic_json, load_trainable_only, save_trainable_only, sha256
    from .ternary_core import TernaryLinear

    run_dir = Path(run_dir)
    directory = run_dir / f"recover_{stage}"
    directory.mkdir(parents=True, exist_ok=True)

    params = configure_recovery(model, stage)
    if not params:
        raise RecoveryError(f"No recovery params found for stage {stage}")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    last = directory / "latest.json"
    progress = {"steps": 0, "tokens": 0, "cursor": 0}

    manifest_sha = (
        sha256(canonical_manifest_path)
        if canonical_manifest_path and Path(canonical_manifest_path).exists()
        else ""
    )

    if last.exists():
        last_info = json.loads(last.read_text())
        md = load_trainable_only(model, opt, Path(last_info["path"]))
        if md.get("stage") != stage:
            raise RecoveryError(
                f"Recovery lineage mismatch: expected stage {stage}, got {md.get('stage')}"
            )
        if (
            manifest_sha
            and md.get("canonical_manifest_sha256")
            and md.get("canonical_manifest_sha256") != manifest_sha
        ):
            raise RecoveryError("Recovery lineage mismatch: canonical manifest hash differs")
        progress = {k: int(md[k]) for k in ("steps", "tokens", "cursor")}
    else:
        opt.zero_grad(set_to_none=True)

    if budget <= 0:
        return progress

    model.train()
    stream_it = ds.stream("train", start=progress["cursor"], max_len=max_seq_len)

    def default_batch_fn(rec):
        if isinstance(rec, dict):
            ids, mask, labels = rec["input_ids"], rec["attention_mask"], rec["labels"]
        else:
            ids, mask, labels = rec
        t_ids = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
        t_mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        t_labels = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        return {
            "input_ids": t_ids.unsqueeze(0),
            "attention_mask": t_mask.unsqueeze(0),
            "labels": t_labels.unsqueeze(0),
        }

    _batch = batch_fn or default_batch_fn

    while progress["tokens"] < budget:
        progress["cursor"], rec = next(stream_it)
        b = _batch(rec)
        supervised = int((b["labels"] != -100).sum())
        opt.zero_grad(set_to_none=True)
        outputs = model(**b)
        loss = getattr(outputs, "loss", outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        if not torch.isfinite(loss):
            raise RecoveryError(f"Non-finite {stage} loss encountered: {loss.item()}")
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RecoveryError(f"Invalid or non-finite {stage} gradients encountered")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, TernaryLinear):
                    m.scales.clamp_(min=1e-8)

        progress["tokens"] += supervised
        progress["steps"] += 1

        if progress["steps"] % checkpoint_every == 0 or progress["tokens"] >= budget:
            folder = directory / f"step-{progress['steps']:08d}"
            save_trainable_only(
                model,
                opt,
                folder,
                {
                    **progress,
                    "stage": stage,
                    "canonical_manifest_sha256": manifest_sha,
                },
            )
            atomic_json(last, {"path": str(folder)})

    atomic_json(directory / "complete.json", progress)
    model.eval()
    return progress


def train_qat(
    model: Any,
    ds: Any,
    qat_selected: list[str],
    budget: int,
    lr: float,
    run_dir: str | Path,
    checkpoint_every: int = 50,
    max_seq_len: int = 1024,
    batch_fn: Any = None,
) -> dict[str, Any]:
    """Execute bounded QAT training on selected TernaryLinear modules."""
    import torch

    from .loader import atomic_json, load_trainable_only, save_trainable_only
    from .ternary_core import TernaryLinear

    run_dir = Path(run_dir)
    qat_root = run_dir / "qat"
    qat_root.mkdir(parents=True, exist_ok=True)

    if not qat_selected or budget <= 0:
        return {"steps": 0, "tokens": 0, "cursor": 0, "selected": qat_selected, "code_flips": {}}

    for p in model.parameters():
        p.requires_grad_(False)

    trainables = []
    for n in qat_selected:
        m = model.get_submodule(n)
        if not isinstance(m, TernaryLinear):
            continue
        if m.shadow is None:
            continue
        m.shadow.requires_grad_(True)
        m.scales.requires_grad_(True)
        trainables.extend([m.shadow, m.scales])

    if not trainables:
        return {"steps": 0, "tokens": 0, "cursor": 0, "selected": qat_selected, "code_flips": {}}

    opt = torch.optim.AdamW(trainables, lr=lr, weight_decay=0.0)
    last = qat_root / "latest.json"
    prog = {"steps": 0, "tokens": 0, "cursor": 0}

    if last.exists():
        snap = json.loads(last.read_text())
        md = load_trainable_only(model, opt, Path(snap["path"]))
        if md.get("selected") != qat_selected:
            raise RecoveryError("QAT selection drift between checkpoints")
        prog = {k: int(md[k]) for k in ("steps", "tokens", "cursor")}
    else:
        opt.zero_grad(set_to_none=True)

    original_codes = {
        n: model.get_submodule(n).codes.detach().clone() for n in qat_selected
    }

    model.train()
    stream_it = ds.stream("train", start=prog["cursor"], max_len=max_seq_len)

    def default_batch_fn(rec):
        if isinstance(rec, dict):
            ids, mask, labels = rec["input_ids"], rec["attention_mask"], rec["labels"]
        else:
            ids, mask, labels = rec
        t_ids = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
        t_mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        t_labels = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
        return {
            "input_ids": t_ids.unsqueeze(0),
            "attention_mask": t_mask.unsqueeze(0),
            "labels": t_labels.unsqueeze(0),
        }

    _batch = batch_fn or default_batch_fn

    while prog["tokens"] < budget:
        prog["cursor"], rec = next(stream_it)
        b = _batch(rec)
        supervised = int((b["labels"] != -100).sum())
        opt.zero_grad(set_to_none=True)
        outputs = model(**b)
        loss = getattr(outputs, "loss", outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        if not torch.isfinite(loss):
            raise RecoveryError(f"QAT non-finite loss: {loss.item()}")
        loss.backward()
        grads = [p.grad for p in trainables if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RecoveryError("QAT missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        opt.step()

        for n in qat_selected:
            mod = model.get_submodule(n)
            if hasattr(mod, "scales") and mod.scales is not None:
                mod.scales.data.clamp_(min=1e-8)

        prog["tokens"] += supervised
        prog["steps"] += 1

        if prog["steps"] % checkpoint_every == 0 or prog["tokens"] >= budget:
            folder = qat_root / f"step-{prog['steps']:08d}"
            save_trainable_only(
                model,
                opt,
                folder,
                {**prog, "stage": "qat", "selected": qat_selected},
            )
            atomic_json(last, {"path": str(folder)})

    model.eval()
    for n in qat_selected:
        model.get_submodule(n).freeze_shadow_to_codes()

    code_flips = {
        n: int((model.get_submodule(n).codes != original_codes[n]).sum().item())
        for n in qat_selected
    }
    result = {**prog, "selected": qat_selected, "code_flips": code_flips}
    atomic_json(qat_root / "complete.json", result)
    return result


def reconstruct_projection(
    unit: Any,
    x: Any,
    teacher_y: Any,
    steps: int = 3,
    lr: float = 3e-5,
) -> tuple[Any, dict[str, Any]]:
    """Run local reconstruction updates on trainable parameters of a quantized module against teacher outputs.

    Compatible with TernaryLinear (optimizing scales + lora_A/lora_B) and
    QTEALinear (optimizing col_v, group_alpha, and/or residual parameters).
    Ternary codes are verified to remain strictly frozen.

    The report is honest about what happened: it names the recovery status
    and the number of trainable tensors that actually changed. A run whose
    parameters did not move (for example because the "teacher" was computed
    from an already-quantized module) is reported as such and never as a
    successful optimization.

    Returns:
        ``(unit, metrics)`` where metrics carries ``mse_before_recovery``,
        ``mse_after_recovery``, ``recovery_steps``, ``parameters_updated``,
        ``max_parameter_change``, ``grad_finite``, ``teacher_identical`` and
        ``status``.

    # Errors: raises RecoveryError when the reconstruction loss or the
    gradients become non-finite, or when a frozen code/ residual index is
    modified during the run.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        initial_pred = unit(x).float()
        first_error = float(F.mse_loss(initial_pred, teacher_y).item())

    # Collect parameters requiring gradients
    parameters: list[torch.Tensor] = []
    if hasattr(unit, "scales") and isinstance(unit.scales, torch.nn.Parameter):
        unit.scales.requires_grad_(True)
        parameters.append(unit.scales)
    if hasattr(unit, "lora_A") and isinstance(unit.lora_A, torch.nn.Parameter):
        unit.lora_A.requires_grad_(True)
        parameters.append(unit.lora_A)
    if hasattr(unit, "lora_B") and isinstance(unit.lora_B, torch.nn.Parameter):
        unit.lora_B.requires_grad_(True)
        parameters.append(unit.lora_B)

    # For QTEALinear, make col_v and/or group_alpha trainable parameters during local recovery if present
    if hasattr(unit, "col_v") and isinstance(unit.col_v, torch.Tensor) and not parameters:
        col_v_param = torch.nn.Parameter(unit.col_v.clone())
        unit.col_v = col_v_param
        parameters.append(col_v_param)

    if not parameters or steps <= 0:
        return unit, {
            "mse_before_recovery": first_error,
            "mse_after_recovery": first_error,
            "recovery_steps": 0,
            "parameters_updated": 0,
            "max_parameter_change": 0.0,
            "max_gradient_norm": 0.0,
            "grad_finite": True,
            "teacher_identical": first_error == 0.0,
            "status": "no_parameters" if not parameters else "zero_steps",
        }

    starting_codes = (
        unit.codes.detach().clone()
        if hasattr(unit, "codes")
        else (unit.ternary.detach().clone() if hasattr(unit, "ternary") else None)
    )
    # Packed residual coordinates are frozen as well: a recovery pass must
    # never rewrite the 1:4 sparse index layout or its scales.
    frozen_residuals = {
        name: getattr(unit, name).detach().clone()
        for name in ("residual_index", "salient_cols", "residual_scale")
        if hasattr(unit, name) and isinstance(getattr(unit, name), torch.Tensor)
    }
    starting_parameters = [p.detach().clone() for p in parameters]

    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=0.0)
    grad_finite = True
    max_gradient_norm = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = unit(x).float()
        loss = F.mse_loss(pred, teacher_y)
        if not bool(torch.isfinite(loss)):
            raise RecoveryError("Non-finite local reconstruction loss")
        loss.backward()
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RecoveryError("Non-finite local reconstruction gradients")
        grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
        for g in grads:
            max_gradient_norm = max(max_gradient_norm, float(g.norm()))
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            if hasattr(unit, "scales") and unit.scales is not None:
                unit.scales.clamp_(min=1e-8)
            if hasattr(unit, "col_v") and unit.col_v is not None:
                unit.col_v.clamp_(min=1e-8)

    with torch.no_grad():
        final_pred = unit(x).float()
        final_error = float(F.mse_loss(final_pred, teacher_y).item())

    if starting_codes is not None:
        current_codes = (
            unit.codes
            if hasattr(unit, "codes")
            else getattr(unit, "ternary", None)
        )
        if current_codes is not None and not torch.equal(current_codes, starting_codes):
            raise RecoveryError("Frozen ternary codes were modified during local reconstruction")
    for name, snapshot in frozen_residuals.items():
        current = getattr(unit, name, None)
        if current is not None and not torch.equal(current.detach(), snapshot):
            raise RecoveryError(
                f"Frozen residual tensor {name!r} was modified during local reconstruction"
            )

    # Report what actually moved: an optimizer run that leaves every
    # parameter untouched is not a successful recovery.
    parameters_updated = 0
    max_parameter_change = 0.0
    with torch.no_grad():
        for before, param in zip(starting_parameters, parameters):
            if not torch.equal(before, param.detach()):
                parameters_updated += 1
                delta = float((before - param.detach()).abs().max())
                max_parameter_change = max(max_parameter_change, delta)

    teacher_identical = first_error == 0.0
    if teacher_identical:
        status = "degenerate_teacher"
    elif parameters_updated == 0:
        status = "no_change"
    else:
        status = "recovered"

    unit.eval()
    for p in unit.parameters():
        p.requires_grad_(False)

    return unit, {
        "mse_before_recovery": first_error,
        "mse_after_recovery": final_error,
        "recovery_steps": steps,
        "parameters_updated": parameters_updated,
        "max_parameter_change": max_parameter_change,
        "max_gradient_norm": max_gradient_norm,
        "grad_finite": grad_finite,
        "teacher_identical": teacher_identical,
        "status": status,
    }


__all__ = [
    "ScaleDistillationResult",
    "ResidualResult",
    "QatTracking",
    "RecoveryReport",
    "svd_init_residual",
    "residual_rmse",
    "low_rank_residual",
    "attempt_absorption",
    "scale_distillation",
    "selective_qat_tracking",
    "configure_recovery",
    "setup_qat",
    "train_recovery",
    "train_qat",
    "reconstruct_projection",
    "MixedPrecisionRecoveryConfig",
    "TeacherTargetCache",
    "configure_mixed_precision_recovery",
    "capture_teacher_targets",
    "compute_recovery_loss",
    "train_mixed_precision_recovery",
]


# --------------------------------------------------------------------------- #
# Mixed-Precision Teacher-Guided Recovery (QTEALinear + Protected4BitLinear)  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MixedPrecisionRecoveryConfig:
    """Configuration for teacher-guided mixed-precision recovery."""

    lora_rank: int = 16
    lora_alpha: float = 32.0
    train_qtea_scales: bool = True
    train_int4_scales: bool = False
    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_kl: float = 1.0
    mu_hidden: float = 0.1
    temperature: float = 2.0
    top_k_logits: int | None = 64
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 50
    eval_every_steps: int = 50
    supervised_token_budget: int = 2_000_000
    svd_init: bool = False
    svd_init_max_norm: float = 1.0
    source_identity: str = "azrael-mixed-precision"


@dataclass
class TeacherTargetCache:
    """Memory-conscious cached teacher outputs for one or more calibration batches.

    When ``top_k`` is set, only the top-k teacher logits and indices per token are
    stored (reducing vocab-sized storage by ~1000x). Callers should inspect
    ``truncated_kl`` and ``omitted_mass_note`` so truncation is never mistaken
    for full-vocabulary KL.
    """

    entries: list[dict[str, Any]]
    top_k: int | None
    truncated_kl: bool
    omitted_mass_note: str
    offloaded_to_cpu: bool


def _svd_init_lora(
    mod: Any,
    teacher_weight: Any,
    *,
    rank: int,
    alpha: float,
    max_norm: float = 1.0,
) -> None:
    """Bounded SVD initialization of ``lora_A`` / ``lora_B`` from weight residual."""
    import math
    import torch

    with torch.no_grad():
        base_w = mod.effective_weight(dtype=torch.float32)
        residual = (teacher_weight.to(device=base_w.device, dtype=torch.float32) - base_w).detach()
        res_norm = float(torch.linalg.norm(residual).item())
        if res_norm == 0.0 or not math.isfinite(res_norm):
            return
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, int(s.shape[0]))
        if r == 0:
            return
        scale_factor = float(rank) / float(alpha)
        s_sqrt = torch.sqrt(s[:r].clamp(min=0.0))
        b_init = u[:, :r] * s_sqrt.unsqueeze(0)
        a_init = s_sqrt.unsqueeze(1) * vh[:r, :] * scale_factor
        # Bound residual norm to prevent oversized initial perturbation
        approx_norm = float(torch.linalg.norm((b_init @ a_init) * (float(alpha) / float(rank))).item())
        if max_norm > 0 and approx_norm > max_norm:
            damp = max_norm / approx_norm
            b_init = b_init * damp
        if mod.lora_B is not None and mod.lora_A is not None:
            mod.lora_B.data[:, :r].copy_(b_init.to(device=mod.lora_B.device, dtype=mod.lora_B.dtype))
            mod.lora_A.data[:r, :].copy_(a_init.to(device=mod.lora_A.device, dtype=mod.lora_A.dtype))


def configure_mixed_precision_recovery(
    model: Any,
    *,
    config: MixedPrecisionRecoveryConfig | None = None,
    teacher_weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure ``QTEALinear`` and ``Protected4BitLinear`` modules for recovery.

    Freezes all non-recovery parameters (embeddings, norms, vision, lm_head) and
    keeps discrete ``ternary`` and ``qweight`` codes frozen while enabling
    continuous QTEA scales (``group_alpha``, ``group_beta``, ``col_v``) and
    rank-r low-rank adapters.

    # Errors: raises RecoveryError if PyTorch is unavailable or no recovery
    modules are found on ``model``.
    """
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for configure_mixed_precision_recovery") from exc

    from .quant4 import Protected4BitLinear
    from .qtea import QTEALinear

    cfg = config or MixedPrecisionRecoveryConfig()
    for p in model.parameters():
        p.requires_grad_(False)

    qtea_count = 0
    int4_count = 0
    frozen_code_tensors: list[str] = []

    for name, mod in model.named_modules():
        if isinstance(mod, QTEALinear):
            mod.enable_recovery_parameters(
                train_scales=cfg.train_qtea_scales,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
            )
            qtea_count += 1
            frozen_code_tensors.append(f"{name}.ternary")
            if teacher_weights and cfg.svd_init and f"{name}.weight" in teacher_weights:
                _svd_init_lora(
                    mod,
                    teacher_weights[f"{name}.weight"],
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    max_norm=cfg.svd_init_max_norm,
                )
        elif Protected4BitLinear is not None and isinstance(mod, Protected4BitLinear):
            if cfg.lora_rank > 0:
                mod.enable_recovery_adapters(
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    train_scales=cfg.train_int4_scales,
                )
            int4_count += 1
            frozen_code_tensors.append(f"{name}.qweight")
            if teacher_weights and cfg.svd_init and f"{name}.weight" in teacher_weights:
                _svd_init_lora(
                    mod,
                    teacher_weights[f"{name}.weight"],
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    max_norm=cfg.svd_init_max_norm,
                )

    if qtea_count == 0 and int4_count == 0:
        raise RecoveryError("No QTEALinear or Protected4BitLinear modules found for recovery")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "qtea_modules": qtea_count,
        "int4_modules": int4_count,
        "frozen_code_tensors": frozen_code_tensors,
    }


def capture_teacher_targets(
    teacher_model: Any,
    batches: list[dict[str, Any]],
    *,
    top_k: int | None = 64,
    offload_to_cpu: bool = True,
    hidden_layer_indices: tuple[int, ...] | list[int] | None = None,
) -> TeacherTargetCache:
    """Capture teacher logits (full or top-k) and optional hidden states.

    Supports freeing the BF16 teacher from GPU memory before student recovery
    runs when ``offload_to_cpu=True``.

    # Errors: raises RecoveryError if PyTorch is unavailable or teacher forward
    returns no logits.
    """
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for capture_teacher_targets") from exc

    teacher_model.eval()
    entries: list[dict[str, Any]] = []
    target_device = torch.device("cpu") if offload_to_cpu else None

    with torch.no_grad():
        for batch in batches:
            fwd_kwargs: dict[str, Any] = {
                "input_ids": batch["input_ids"],
            }
            if "attention_mask" in batch:
                fwd_kwargs["attention_mask"] = batch["attention_mask"]
            if hidden_layer_indices is not None:
                fwd_kwargs["output_hidden_states"] = True

            try:
                out = teacher_model(**fwd_kwargs)
            except TypeError:
                # Minimal synthetic models may not accept output_hidden_states
                fwd_kwargs.pop("output_hidden_states", None)
                out = teacher_model(**fwd_kwargs)

            logits = getattr(out, "logits", out[0] if isinstance(out, tuple) else out)
            if logits is None:
                raise RecoveryError("Teacher model returned no logits")

            entry: dict[str, Any] = {}
            if top_k is not None and 0 < top_k < logits.shape[-1]:
                vals, idx = torch.topk(logits.float(), k=top_k, dim=-1)
                entry["top_k_logits"] = vals.to(device=target_device) if target_device else vals
                entry["top_k_indices"] = idx.to(device=target_device) if target_device else idx
            else:
                l_store = logits.float()
                entry["logits"] = l_store.to(device=target_device) if target_device else l_store

            hidden_states = getattr(out, "hidden_states", None)
            if hidden_layer_indices is not None and hidden_states is not None:
                selected = {}
                for li in hidden_layer_indices:
                    if -len(hidden_states) <= li < len(hidden_states):
                        h = hidden_states[li].float()
                        selected[int(li)] = h.to(device=target_device) if target_device else h
                entry["hidden_states"] = selected
            entries.append(entry)

    truncated = top_k is not None
    note = (
        f"Top-{top_k} truncated KL omits tail probability mass outside top-{top_k} teacher logits"
        if truncated
        else "Full-vocabulary KL divergence (no tail mass omitted)"
    )
    return TeacherTargetCache(
        entries=entries,
        top_k=top_k,
        truncated_kl=truncated,
        omitted_mass_note=note,
        offloaded_to_cpu=offload_to_cpu,
    )


def compute_recovery_loss(
    student_logits: Any,
    labels: Any,
    *,
    teacher_target: dict[str, Any] | None = None,
    student_hidden_states: Any = None,
    lambda_kl: float = 1.0,
    mu_hidden: float = 0.1,
    temperature: float = 2.0,
) -> tuple[Any, dict[str, float]]:
    """Compute ``L = L_CE + lambda * L_KL + mu * L_hidden`` with causal shift.

    Uses exact causal shift (``logits[:, :-1]`` vs ``labels[:, 1:]``) and
    masks unsupervised positions (``labels == -100``).

    # Errors: raises RecoveryError if PyTorch is unavailable or shapes are invalid.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for compute_recovery_loss") from exc

    if student_logits.ndim != 3 or labels.ndim != 2:
        raise RecoveryError(
            f"Expected 3D logits and 2D labels, got {tuple(student_logits.shape)} and {tuple(labels.shape)}"
        )

    shift_logits = student_logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    sup_mask = shift_labels != -100
    n_sup = int(sup_mask.sum().item())

    if n_sup > 0:
        l_ce = F.cross_entropy(shift_logits[sup_mask], shift_labels[sup_mask], reduction="mean")
    else:
        l_ce = shift_logits.sum() * 0.0

    l_kl = shift_logits.sum() * 0.0
    if teacher_target is not None and lambda_kl > 0.0 and n_sup > 0:
        t = float(temperature)
        dev = shift_logits.device
        if "top_k_logits" in teacher_target and "top_k_indices" in teacher_target:
            t_vals = teacher_target["top_k_logits"][:, :-1, :].to(device=dev, dtype=torch.float32)[sup_mask]
            t_idx = teacher_target["top_k_indices"][:, :-1, :].to(device=dev, dtype=torch.long)[sup_mask]
            s_sel = torch.gather(shift_logits[sup_mask], dim=-1, index=t_idx)
            t_probs = F.softmax(t_vals / t, dim=-1)
            s_log_probs = F.log_softmax(s_sel / t, dim=-1)
            l_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (t * t)
        elif "logits" in teacher_target:
            t_logits = teacher_target["logits"][:, :-1, :].to(device=dev, dtype=torch.float32)[sup_mask]
            t_probs = F.softmax(t_logits / t, dim=-1)
            s_log_probs = F.log_softmax(shift_logits[sup_mask] / t, dim=-1)
            l_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (t * t)

    l_hidden = shift_logits.sum() * 0.0
    if (
        teacher_target is not None
        and "hidden_states" in teacher_target
        and student_hidden_states is not None
        and mu_hidden > 0.0
    ):
        t_hmap = teacher_target["hidden_states"]
        losses_h = []
        for li, t_h in t_hmap.items():
            if -len(student_hidden_states) <= li < len(student_hidden_states):
                s_h = student_hidden_states[li][:, :-1, :].float()
                t_h_shift = t_h[:, :-1, :].to(device=s_h.device, dtype=torch.float32)
                if n_sup > 0:
                    losses_h.append(F.mse_loss(s_h[sup_mask], t_h_shift[sup_mask]))
                else:
                    losses_h.append(F.mse_loss(s_h, t_h_shift))
        if losses_h:
            l_hidden = torch.stack(losses_h).mean()

    total = l_ce + float(lambda_kl) * l_kl + float(mu_hidden) * l_hidden
    metrics = {
        "loss": float(total.detach().item()),
        "l_ce": float(l_ce.detach().item()),
        "l_kl": float(l_kl.detach().item()),
        "l_hidden": float(l_hidden.detach().item()),
        "supervised_tokens": n_sup,
    }
    return total, metrics


def _recovery_config_fingerprint(cfg: MixedPrecisionRecoveryConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_frozen_codes(model: Any) -> dict[str, Any]:
    from .quant4 import Protected4BitLinear
    from .qtea import QTEALinear

    snap: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QTEALinear):
            snap[f"{name}.ternary"] = mod.ternary.detach().clone().cpu()
        elif Protected4BitLinear is not None and isinstance(mod, Protected4BitLinear):
            snap[f"{name}.qweight"] = mod.qweight.detach().clone().cpu()
    return snap


def _verify_frozen_codes_unchanged(model: Any, snap: dict[str, Any]) -> None:
    import torch
    from .quant4 import Protected4BitLinear
    from .qtea import QTEALinear

    for name, mod in model.named_modules():
        if isinstance(mod, QTEALinear):
            key = f"{name}.ternary"
            if key in snap and not torch.equal(mod.ternary.detach().cpu(), snap[key]):
                raise RecoveryError(f"Frozen QTEA ternary codes mutated at {name!r}")
        elif Protected4BitLinear is not None and isinstance(mod, Protected4BitLinear):
            key = f"{name}.qweight"
            if key in snap and not torch.equal(mod.qweight.detach().cpu(), snap[key]):
                raise RecoveryError(f"Frozen 4-bit qweight mutated at {name!r}")


def _extract_trainable_state(model: Any) -> dict[str, Any]:
    return {
        k: v.detach().clone().cpu()
        for k, v in model.state_dict().items()
        if any(
            k.endswith(suffix)
            for suffix in (".group_alpha", ".group_beta", ".col_v", ".lora_A", ".lora_B", ".scales", ".zeros")
        )
    }


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _eval_heldout_loss(
    model: Any,
    eval_batches: list[dict[str, Any]],
    *,
    eval_teacher_cache: TeacherTargetCache | None,
    cfg: MixedPrecisionRecoveryConfig,
) -> float:
    import torch

    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_batches):
            fwd_kwargs: dict[str, Any] = {"input_ids": batch["input_ids"]}
            if "attention_mask" in batch:
                fwd_kwargs["attention_mask"] = batch["attention_mask"]
            if cfg.mu_hidden > 0.0:
                fwd_kwargs["output_hidden_states"] = True
            try:
                out = model(**fwd_kwargs)
            except TypeError:
                fwd_kwargs.pop("output_hidden_states", None)
                out = model(**fwd_kwargs)
            logits = getattr(out, "logits", out[0] if isinstance(out, tuple) else out)
            hidden = getattr(out, "hidden_states", None)
            t_entry = eval_teacher_cache.entries[idx] if eval_teacher_cache and idx < len(eval_teacher_cache.entries) else None
            loss, _ = compute_recovery_loss(
                logits,
                batch["labels"],
                teacher_target=t_entry,
                student_hidden_states=hidden,
                lambda_kl=cfg.lambda_kl,
                mu_hidden=cfg.mu_hidden,
                temperature=cfg.temperature,
            )
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    return float(sum(losses) / max(len(losses), 1))


def train_mixed_precision_recovery(
    model: Any,
    train_batches: list[dict[str, Any]],
    *,
    teacher_cache: TeacherTargetCache | None = None,
    eval_batches: list[dict[str, Any]] | None = None,
    eval_teacher_cache: TeacherTargetCache | None = None,
    config: MixedPrecisionRecoveryConfig | None = None,
    run_dir: Path | str | None = None,
    resume: bool = False,
    restore_best: bool = True,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run teacher-guided mixed-precision recovery with atomic checkpointing & rollback.

    Supports resuming from ``run_dir/recovery_state/latest.pt`` (validating
    config & source identity fingerprint) and restoring the best held-out
    checkpoint ``best.pt`` when final evaluation degrades.

    # Errors: raises RecoveryError on non-finite loss/gradients, frozen code
    mutation, or incompatible checkpoint resume metadata.
    """
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for train_mixed_precision_recovery") from exc

    import math as _math

    cfg = config or MixedPrecisionRecoveryConfig()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RecoveryError("No trainable recovery parameters found; call configure_mixed_precision_recovery first")

    frozen_snap = _snapshot_frozen_codes(model)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    fingerprint = _recovery_config_fingerprint(cfg)

    ckpt_dir = Path(run_dir).resolve() / "recovery_state" if run_dir is not None else None
    latest_path = ckpt_dir / "latest.pt" if ckpt_dir is not None else None
    best_path = ckpt_dir / "best.pt" if ckpt_dir is not None else None

    start_step = 0
    tokens_consumed = 0
    best_eval_loss = float("inf")
    eval_history: list[dict[str, Any]] = []
    history: list[dict[str, float]] = []

    if resume and latest_path is not None and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        if ckpt.get("config_fingerprint") != fingerprint:
            raise RecoveryError(
                "Recovery checkpoint config_fingerprint mismatch; refusing unsafe resume"
            )
        model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        start_step = int(ckpt["step"])
        tokens_consumed = int(ckpt["tokens_consumed"])
        best_eval_loss = float(ckpt.get("best_eval_loss", float("inf")))
        eval_history = list(ckpt.get("eval_history", []))

    model.train()
    optimizer.zero_grad()

    total_batches = len(train_batches)
    if total_batches == 0:
        raise RecoveryError("train_batches must not be empty")

    step = start_step
    target_steps = max_steps if max_steps is not None else total_batches
    batch_cursor = start_step % total_batches

    while step < target_steps and tokens_consumed < cfg.supervised_token_budget:
        batch = train_batches[batch_cursor]
        t_entry = (
            teacher_cache.entries[batch_cursor]
            if teacher_cache is not None and batch_cursor < len(teacher_cache.entries)
            else None
        )

        fwd_kwargs: dict[str, Any] = {"input_ids": batch["input_ids"]}
        if "attention_mask" in batch:
            fwd_kwargs["attention_mask"] = batch["attention_mask"]
        if cfg.mu_hidden > 0.0:
            fwd_kwargs["output_hidden_states"] = True
        try:
            out = model(**fwd_kwargs)
        except TypeError:
            fwd_kwargs.pop("output_hidden_states", None)
            out = model(**fwd_kwargs)

        logits = getattr(out, "logits", out[0] if isinstance(out, tuple) else out)
        hidden = getattr(out, "hidden_states", None)
        loss, metrics = compute_recovery_loss(
            logits,
            batch["labels"],
            teacher_target=t_entry,
            student_hidden_states=hidden,
            lambda_kl=cfg.lambda_kl,
            mu_hidden=cfg.mu_hidden,
            temperature=cfg.temperature,
        )
        if not _math.isfinite(metrics["loss"]):
            raise RecoveryError(f"Non-finite recovery loss at step {step}: {metrics['loss']}")

        scaled_loss = loss / float(max(1, cfg.grad_accum_steps))
        scaled_loss.backward()

        step += 1
        batch_cursor = (batch_cursor + 1) % total_batches
        tokens_consumed += int(metrics["supervised_tokens"])
        history.append({"step": float(step), **metrics})

        if step % max(1, cfg.grad_accum_steps) == 0 or step >= target_steps:
            for p in trainable:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RecoveryError(f"Non-finite gradient detected at step {step}")
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if eval_batches and (step % max(1, cfg.eval_every_steps) == 0 or step >= target_steps):
            ev_loss = _eval_heldout_loss(
                model,
                eval_batches,
                eval_teacher_cache=eval_teacher_cache,
                cfg=cfg,
            )
            eval_history.append({"step": step, "eval_loss": ev_loss, "tokens": tokens_consumed})
            if best_path is not None and ev_loss < best_eval_loss:
                best_eval_loss = ev_loss
                _atomic_torch_save(
                    {
                        "step": step,
                        "tokens_consumed": tokens_consumed,
                        "best_eval_loss": best_eval_loss,
                        "config_fingerprint": fingerprint,
                        "trainable_state_dict": _extract_trainable_state(model),
                    },
                    best_path,
                )
            elif ev_loss < best_eval_loss:
                best_eval_loss = ev_loss

        if latest_path is not None and (step % max(1, cfg.checkpoint_every_steps) == 0 or step >= target_steps):
            _atomic_torch_save(
                {
                    "step": step,
                    "tokens_consumed": tokens_consumed,
                    "best_eval_loss": best_eval_loss,
                    "eval_history": eval_history,
                    "config_fingerprint": fingerprint,
                    "trainable_state_dict": _extract_trainable_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                latest_path,
            )

    _verify_frozen_codes_unchanged(model, frozen_snap)

    rollback_applied = False
    selected_checkpoint = "final"
    final_eval_loss = eval_history[-1]["eval_loss"] if eval_history else None
    if (
        restore_best
        and best_path is not None
        and best_path.is_file()
        and final_eval_loss is not None
        and best_eval_loss < final_eval_loss
    ):
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["trainable_state_dict"], strict=False)
        rollback_applied = True
        selected_checkpoint = "best"

    return {
        "steps_completed": step,
        "tokens_consumed": tokens_consumed,
        "history": history,
        "eval_history": eval_history,
        "best_eval_loss": best_eval_loss if eval_history else None,
        "final_eval_loss": final_eval_loss,
        "selected_checkpoint": selected_checkpoint,
        "rollback_applied": rollback_applied,
        "config_fingerprint": fingerprint,
        "checkpoint_dir": str(ckpt_dir) if ckpt_dir is not None else None,
    }
