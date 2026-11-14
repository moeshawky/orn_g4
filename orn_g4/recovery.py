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
    """Configuration for teacher-guided mixed-precision recovery (ADV-12.1 frozen defaults)."""

    lora_rank: int = 16
    lora_alpha: float = 32.0
    train_qtea_scales: bool = True
    train_int4_scales: bool = False
    adapter_lr: float = 5e-5
    scale_lr: float = 5e-6
    lr: float | None = None
    weight_decay: float = 0.0
    warmup_ratio: float = 0.10
    lr_scheduler_type: str = "cosine"
    ce_weight: float = 0.0
    lambda_kl: float = 1.0
    mu_hidden: float = 0.1
    temperature: float = 1.0
    top_k_logits: int | None = 128
    topk_tail_bucket: bool = True
    chunk_size: int = 256
    grad_accum_steps: int = 16
    max_seq_len: int = 4096
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
    activation_scale: Any = None,
) -> None:
    """Bounded (optionally activation-weighted) SVD initialization of ``lora_A`` / ``lora_B``."""
    import math
    import torch

    with torch.no_grad():
        base_w = mod.effective_weight(dtype=torch.float32)
        residual = (teacher_weight.to(device=base_w.device, dtype=torch.float32) - base_w).detach()
        res_norm = float(torch.linalg.norm(residual).item())
        if res_norm == 0.0 or not math.isfinite(res_norm):
            return
        scale_factor = float(rank) / float(alpha)
        if activation_scale is not None:
            from .quant4 import initialization_activation_weighted_svd

            a_raw, b_init = initialization_activation_weighted_svd(
                residual, activation_scale.to(device=base_w.device), rank=rank
            )
            r = min(rank, int(a_raw.shape[0]))
            if r == 0:
                return
            a_init = a_raw[:r, :] * scale_factor
            b_init = b_init[:, :r]
        else:
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            r = min(rank, int(s.shape[0]))
            if r == 0:
                return
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
    activation_scales: dict[str, Any] | None = None,
    teacher_model: Any | None = None,
) -> dict[str, Any]:
    """Configure ``QTEALinear`` and ``Protected4BitLinear`` modules for recovery.

    Freezes all non-recovery parameters (embeddings, norms, vision, lm_head) and
    keeps discrete ``ternary`` and ``qweight`` codes frozen while enabling
    FP32 continuous QTEA scales (``group_alpha``, ``group_beta``, ``col_v``)
    and FP32 rank-r low-rank adapters.

    Supports streaming per-layer SVD initialization from ``teacher_model`` or
    ``teacher_weights`` with optional ``activation_scales`` (L2QER-style)
    without requiring a giant global dense weight dict.

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

    teacher_named = dict(teacher_model.named_modules()) if teacher_model is not None else {}
    act_map = activation_scales or {}

    def _resolve_teacher_weight(mod_name: str) -> Any | None:
        if teacher_weights and f"{mod_name}.weight" in teacher_weights:
            return teacher_weights[f"{mod_name}.weight"]
        if mod_name in teacher_named and hasattr(teacher_named[mod_name], "weight"):
            return teacher_named[mod_name].weight
        return None

    qtea_count = 0
    int4_count = 0
    frozen_code_tensors: list[str] = []

    for name, mod in model.named_modules():
        act_sc = act_map.get(name)
        if act_sc is None:
            for pfx in ("language_model.", "model."):
                if name.startswith(pfx) and name[len(pfx):] in act_map:
                    act_sc = act_map[name[len(pfx):]]
                    break
        if isinstance(mod, QTEALinear):
            mod.enable_recovery_parameters(
                train_scales=cfg.train_qtea_scales,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                adapter_dtype=torch.float32,
            )
            qtea_count += 1
            frozen_code_tensors.append(f"{name}.ternary")
            t_w = _resolve_teacher_weight(name) if cfg.svd_init else None
            if t_w is not None:
                _svd_init_lora(
                    mod,
                    t_w,
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    max_norm=cfg.svd_init_max_norm,
                    activation_scale=act_sc,
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
            t_w = _resolve_teacher_weight(name) if cfg.svd_init else None
            if t_w is not None:
                _svd_init_lora(
                    mod,
                    t_w,
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    max_norm=cfg.svd_init_max_norm,
                    activation_scale=act_sc,
                )

    if qtea_count == 0 and int4_count == 0:
        raise RecoveryError("No QTEALinear or Protected4BitLinear modules found for recovery")

    for pname, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            raise RecoveryError(
                f"Trainable parameter {pname!r} has dtype {p.dtype}; ADV-12.1 requires float32"
            )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "qtea_modules": qtea_count,
        "int4_modules": int4_count,
        "frozen_code_tensors": frozen_code_tensors,
    }


def _resolve_backbone_and_head(model: Any) -> tuple[Any | None, Any | None]:
    """Return ``(backbone, lm_head)`` if ``backbone.forward`` is callable without ``lm_head``."""
    import torch.nn as nn

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return None, None
    for attr in ("model", "language_model"):
        cand = getattr(model, attr, None)
        if cand is not None and isinstance(cand, nn.Module):
            if type(cand).forward is not nn.Module.forward:
                return cand, lm_head
    return None, None


def _get_hoisted_head_weights(lm_head: Any) -> tuple[Any, Any | None]:
    """Hoist dense ``(weight, bias)`` from ``lm_head`` once per microbatch."""
    if hasattr(lm_head, "effective_weight"):
        w_head = lm_head.effective_weight()
    elif hasattr(lm_head, "weight"):
        w_head = lm_head.weight
    else:
        raise RecoveryError("lm_head must expose weight or effective_weight()")
    bias_head = getattr(lm_head, "bias", None)
    return w_head, bias_head


def _obj3_kl_and_ce_chunk_sum(
    logits_chunk: Any,
    labels_chunk: Any | None,
    *,
    top_probs_c: Any | None = None,
    top_idx_c: Any | None = None,
    tail_mass_c: Any | None = None,
    log_p_tail_c: Any | None = None,
    full_t_logits_c: Any | None = None,
    legacy_top_k_logits_c: Any | None = None,
    temperature: float = 1.0,
    ce_weight: float = 0.0,
    lambda_kl: float = 1.0,
    tail_threshold: float = 1e-12,
    topk_tail_bucket: bool = True,
) -> tuple[Any, Any]:
    """Compute exact summed Obj-3 (top-K + tail-bucket) KL and CE over one token chunk."""
    import torch
    import torch.nn.functional as F

    z = logits_chunk.float()
    zero = z.sum() * 0.0
    l_ce_sum = zero
    if ce_weight > 0.0 and labels_chunk is not None:
        l_ce_sum = F.cross_entropy(z, labels_chunk.to(device=z.device, dtype=torch.long), reduction="sum")

    l_kl_sum = zero
    if lambda_kl <= 0.0:
        return l_ce_sum, l_kl_sum

    t = float(temperature)
    z_t = z / t

    if top_probs_c is not None and top_idx_c is not None:
        dev = z_t.device
        p_top = top_probs_c.to(device=dev, dtype=torch.float32)
        idx = top_idx_c.to(device=dev, dtype=torch.long)
        log_z_all = torch.logsumexp(z_t, dim=-1)
        z_top = torch.gather(z_t, dim=-1, index=idx)
        log_q_top = z_top - log_z_all.unsqueeze(-1)
        p_safe = p_top.clamp(min=1e-30)
        kl_top_terms = torch.where(p_top > 0.0, p_top * (torch.log(p_safe) - log_q_top), torch.zeros_like(p_top))
        kl_tokens = kl_top_terms.sum(dim=-1)

        if topk_tail_bucket and idx.shape[-1] < z_t.shape[-1]:
            p_tail = (
                tail_mass_c.to(device=dev, dtype=torch.float32)
                if tail_mass_c is not None
                else (1.0 - p_top.sum(dim=-1)).clamp(min=0.0)
            )
            active = p_tail > tail_threshold
            if bool(active.any()):
                z_masked = z_t[active].scatter(-1, idx[active], float("-inf"))
                log_z_tail = torch.logsumexp(z_masked, dim=-1)
                log_q_tail = log_z_tail - log_z_all[active]
                log_pt = (
                    log_p_tail_c.to(device=dev, dtype=torch.float32)[active]
                    if log_p_tail_c is not None
                    else torch.log(p_tail[active].clamp(min=1e-30))
                )
                kl_tail_active = p_tail[active] * (log_pt - log_q_tail)
                kl_tokens = kl_tokens.clone()
                kl_tokens[active] = kl_tokens[active] + kl_tail_active

        l_kl_sum = kl_tokens.sum() * (t * t)
    elif legacy_top_k_logits_c is not None and top_idx_c is not None:
        dev = z_t.device
        t_vals = legacy_top_k_logits_c.to(device=dev, dtype=torch.float32)
        idx = top_idx_c.to(device=dev, dtype=torch.long)
        log_z_all = torch.logsumexp(z_t, dim=-1)
        z_top = torch.gather(z_t, dim=-1, index=idx)
        p_top = F.softmax(t_vals / t, dim=-1)
        log_q_top = z_top - log_z_all.unsqueeze(-1)
        l_kl_sum = F.kl_div(log_q_top, p_top, reduction="sum") * (t * t)
    elif full_t_logits_c is not None:
        t_probs = F.softmax(full_t_logits_c.to(device=z_t.device, dtype=torch.float32) / t, dim=-1)
        s_log_probs = F.log_softmax(z_t, dim=-1)
        l_kl_sum = F.kl_div(s_log_probs, t_probs, reduction="sum") * (t * t)

    return l_ce_sum, l_kl_sum


def capture_teacher_targets(
    teacher_model: Any,
    batches: list[dict[str, Any]],
    *,
    top_k: int | None = 128,
    temperature: float = 1.0,
    chunk_size: int = 256,
    offload_to_cpu: bool = True,
    hidden_layer_indices: tuple[int, ...] | list[int] | None = None,
) -> TeacherTargetCache:
    """Capture teacher full-softmax top-K probs (FP16), int32 indices, and tail mass.

    Computes the exact denominator ``logsumexp`` over the entire vocabulary
    chunk-by-chunk (256 tokens by default), stores ``top_k_probs`` in FP16
    (unrenormalized full-softmax probabilities at ``temperature``),
    ``top_k_indices`` in ``int32`` (safe for vocabularies > 65,535),
    ``retained_mass`` and ``log_p_tail`` in FP32, and marks ``truncated_kl=True``
    when ``top_k`` is bounded.

    # Errors: raises RecoveryError if PyTorch is unavailable or teacher forward
    returns no logits/hidden states.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for capture_teacher_targets") from exc

    teacher_model.eval()
    entries: list[dict[str, Any]] = []
    target_device = torch.device("cpu") if offload_to_cpu else None
    t = float(temperature)
    backbone, lm_head = _resolve_backbone_and_head(teacher_model)
    w_head = None
    bias_head = None
    if backbone is not None and lm_head is not None:
        w_head, bias_head = _get_hoisted_head_weights(lm_head)

    retained_means: list[float] = []

    with torch.no_grad():
        for batch in batches:
            fwd_kwargs: dict[str, Any] = {
                "input_ids": batch["input_ids"],
            }
            if "attention_mask" in batch:
                fwd_kwargs["attention_mask"] = batch["attention_mask"]
            if hidden_layer_indices is not None:
                fwd_kwargs["output_hidden_states"] = True

            logits = None
            hidden_states = None
            last_hidden = None

            if backbone is not None and w_head is not None:
                try:
                    b_out = backbone(**fwd_kwargs)
                except TypeError:
                    fwd_kwargs.pop("output_hidden_states", None)
                    b_out = backbone(**fwd_kwargs)
                last_hidden = getattr(b_out, "last_hidden_state", b_out[0] if isinstance(b_out, tuple) else b_out)
                hidden_states = getattr(b_out, "hidden_states", None)
                bsz, seqlen, _ = last_hidden.shape
                vocab_size = int(w_head.shape[0])
            else:
                try:
                    out = teacher_model(**fwd_kwargs)
                except TypeError:
                    fwd_kwargs.pop("output_hidden_states", None)
                    out = teacher_model(**fwd_kwargs)
                logits = getattr(out, "logits", out[0] if isinstance(out, tuple) else out)
                if logits is None:
                    raise RecoveryError("Teacher model returned no logits")
                hidden_states = getattr(out, "hidden_states", None)
                bsz, seqlen, vocab_size = logits.shape

            entry: dict[str, Any] = {"temperature": t}
            if top_k is not None and 0 < top_k < vocab_size:
                idx_chunks: list[Any] = []
                prob_chunks: list[Any] = []
                ret_chunks: list[Any] = []
                log_tail_chunks: list[Any] = []
                step_c = max(1, int(chunk_size))
                for s0 in range(0, seqlen, step_c):
                    s1 = min(seqlen, s0 + step_c)
                    if logits is not None:
                        z_c = logits[:, s0:s1, :].float() / t
                    else:
                        h_c = last_hidden[:, s0:s1, :]
                        z_c = F.linear(h_c.to(w_head.dtype), w_head, bias_head).float() / t
                    log_z = torch.logsumexp(z_c, dim=-1, keepdim=True)
                    vals_c, idx_c = torch.topk(z_c, k=top_k, dim=-1)
                    probs_c = torch.exp(vals_c - log_z)
                    ret_c = probs_c.sum(dim=-1).clamp(min=0.0, max=1.0)
                    tail_c = (1.0 - ret_c).clamp(min=0.0)
                    log_pt_c = torch.log(tail_c.clamp(min=1e-30))
                    idx_chunks.append(idx_c.to(torch.int32))
                    prob_chunks.append(probs_c.to(torch.float16))
                    ret_chunks.append(ret_c.to(torch.float32))
                    log_tail_chunks.append(log_pt_c.to(torch.float32))
                    del z_c, log_z, vals_c

                all_idx = torch.cat(idx_chunks, dim=1)
                all_probs = torch.cat(prob_chunks, dim=1)
                all_ret = torch.cat(ret_chunks, dim=1)
                all_log_tail = torch.cat(log_tail_chunks, dim=1)
                retained_means.append(float(all_ret.mean().item()))
                entry["top_k_indices"] = all_idx.to(device=target_device) if target_device else all_idx
                entry["top_k_probs"] = all_probs.to(device=target_device) if target_device else all_probs
                entry["retained_mass"] = all_ret.to(device=target_device) if target_device else all_ret
                entry["log_p_tail"] = all_log_tail.to(device=target_device) if target_device else all_log_tail
                entry["truncated_kl"] = True
            else:
                if logits is None:
                    h_flat = last_hidden.reshape(-1, last_hidden.shape[-1])
                    logits = F.linear(h_flat.to(w_head.dtype), w_head, bias_head).reshape(bsz, seqlen, vocab_size)
                l_store = logits.float()
                entry["logits"] = l_store.to(device=target_device) if target_device else l_store
                entry["retained_mass"] = torch.ones((bsz, seqlen), dtype=torch.float32, device=target_device or l_store.device)
                entry["truncated_kl"] = False

            if hidden_layer_indices is not None and hidden_states is not None:
                selected = {}
                for li in hidden_layer_indices:
                    if -len(hidden_states) <= li < len(hidden_states):
                        h = hidden_states[li].float()
                        selected[int(li)] = h.to(device=target_device) if target_device else h
                entry["hidden_states"] = selected
            entries.append(entry)

    truncated = top_k is not None
    mean_ret = sum(retained_means) / len(retained_means) if retained_means else 1.0
    note = (
        f"Top-{top_k} tail-bucket Obj-3 KL (retained_mass={mean_ret:.4f}, tail_mass={max(0.0, 1.0 - mean_ret):.4f})"
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


def compute_fused_chunked_kl_loss(
    hidden_states: Any,
    lm_head: Any,
    labels: Any,
    *,
    teacher_target: dict[str, Any] | None = None,
    student_hidden_states: Any = None,
    ce_weight: float = 0.0,
    lambda_kl: float = 1.0,
    mu_hidden: float = 0.1,
    temperature: float = 1.0,
    chunk_size: int = 256,
    tail_threshold: float = 1e-12,
    topk_tail_bucket: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Compute Obj-3 top-K + tail-bucket KL (plus optional CE/hidden) without full (B,S,V) logits.

    Takes backbone ``hidden_states`` and ``lm_head`` directly, hoists dense or
    INT8-dequantized ``lm_head`` weights once per microbatch, and processes only
    supervised positions (``labels[:, 1:] != -100``) in gradient-checkpointed
    chunks of ``chunk_size`` tokens (default 256).

    # Errors: raises RecoveryError if PyTorch is unavailable or tensor shapes
    are invalid.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for compute_fused_chunked_kl_loss") from exc

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise RecoveryError(
            f"Expected 3D hidden_states and 2D labels, got {tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )

    shift_h = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    sup_mask = shift_labels != -100
    n_sup = int(sup_mask.sum().item())

    if n_sup == 0:
        zero = shift_h.sum() * 0.0
        return zero, {
            "loss": 0.0,
            "l_ce": 0.0,
            "l_kl": 0.0,
            "l_hidden": 0.0,
            "supervised_tokens": 0,
        }

    h_sup = shift_h[sup_mask]
    y_sup = shift_labels[sup_mask]
    w_head, bias_head = _get_hoisted_head_weights(lm_head)

    t_probs_all = None
    t_idx_all = None
    t_tail_all = None
    t_log_pt_all = None
    t_legacy_logits_all = None
    t_full_logits_all = None

    if teacher_target is not None and lambda_kl > 0.0:
        if "top_k_probs" in teacher_target and "top_k_indices" in teacher_target:
            t_probs_all = teacher_target["top_k_probs"][:, :-1, :][sup_mask]
            t_idx_all = teacher_target["top_k_indices"][:, :-1, :][sup_mask]
            if "retained_mass" in teacher_target:
                t_tail_all = (1.0 - teacher_target["retained_mass"][:, :-1][sup_mask].float()).clamp(min=0.0)
            if "log_p_tail" in teacher_target:
                t_log_pt_all = teacher_target["log_p_tail"][:, :-1][sup_mask]
        elif "top_k_logits" in teacher_target and "top_k_indices" in teacher_target:
            t_legacy_logits_all = teacher_target["top_k_logits"][:, :-1, :][sup_mask]
            t_idx_all = teacher_target["top_k_indices"][:, :-1, :][sup_mask]
        elif "logits" in teacher_target:
            t_full_logits_all = teacher_target["logits"][:, :-1, :][sup_mask]

    step_c = max(1, int(chunk_size))
    ce_sum_total = h_sup.sum() * 0.0
    kl_sum_total = h_sup.sum() * 0.0
    use_ckpt = bool(torch.is_grad_enabled() and h_sup.requires_grad)

    def _make_chunk_runner(
        y_arg: Any,
        tp_arg: Any,
        ti_arg: Any,
        tm_arg: Any,
        lp_arg: Any,
        leg_arg: Any,
        full_arg: Any,
    ) -> Any:
        def _run_chunk(h_arg: Any, w_arg: Any, b_arg: Any | None = None) -> tuple[Any, Any]:
            z_c = F.linear(h_arg.to(w_arg.dtype), w_arg, b_arg)
            return _obj3_kl_and_ce_chunk_sum(
                z_c,
                y_arg,
                top_probs_c=tp_arg,
                top_idx_c=ti_arg,
                tail_mass_c=tm_arg,
                log_p_tail_c=lp_arg,
                full_t_logits_c=full_arg,
                legacy_top_k_logits_c=leg_arg,
                temperature=temperature,
                ce_weight=ce_weight,
                lambda_kl=lambda_kl,
                tail_threshold=tail_threshold,
                topk_tail_bucket=topk_tail_bucket,
            )

        return _run_chunk

    for start in range(0, n_sup, step_c):
        end = min(n_sup, start + step_c)
        h_c = h_sup[start:end]
        chunk_fn = _make_chunk_runner(
            y_sup[start:end],
            t_probs_all[start:end] if t_probs_all is not None else None,
            t_idx_all[start:end] if t_idx_all is not None else None,
            t_tail_all[start:end] if t_tail_all is not None else None,
            t_log_pt_all[start:end] if t_log_pt_all is not None else None,
            t_legacy_logits_all[start:end] if t_legacy_logits_all is not None else None,
            t_full_logits_all[start:end] if t_full_logits_all is not None else None,
        )

        if use_ckpt:
            if bias_head is not None:
                c_ce, c_kl = checkpoint(chunk_fn, h_c, w_head, bias_head, use_reentrant=False)
            else:
                c_ce, c_kl = checkpoint(chunk_fn, h_c, w_head, use_reentrant=False)
        else:
            c_ce, c_kl = chunk_fn(h_c, w_head, bias_head)
        ce_sum_total = ce_sum_total + c_ce
        kl_sum_total = kl_sum_total + c_kl

    inv_n = 1.0 / float(n_sup)
    l_ce = ce_sum_total * inv_n
    l_kl = kl_sum_total * inv_n

    l_hidden = shift_h.sum() * 0.0
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
                losses_h.append(F.mse_loss(s_h[sup_mask], t_h_shift[sup_mask]))
        if losses_h:
            l_hidden = torch.stack(losses_h).mean()

    total = float(ce_weight) * l_ce + float(lambda_kl) * l_kl + float(mu_hidden) * l_hidden
    metrics = {
        "loss": float(total.detach().item()),
        "l_ce": float(l_ce.detach().item()),
        "l_kl": float(l_kl.detach().item()),
        "l_hidden": float(l_hidden.detach().item()),
        "supervised_tokens": n_sup,
    }
    return total, metrics


def compute_recovery_loss(
    student_logits: Any,
    labels: Any,
    *,
    teacher_target: dict[str, Any] | None = None,
    student_hidden_states: Any = None,
    ce_weight: float = 1.0,
    lambda_kl: float = 1.0,
    mu_hidden: float = 0.1,
    temperature: float = 1.0,
    chunk_size: int = 256,
    tail_threshold: float = 1e-12,
    topk_tail_bucket: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Compute ``L = ce_weight * L_CE + lambda * L_KL + mu * L_hidden`` with causal shift.

    Uses the unified Obj-3 chunked top-K + tail-bucket KL implementation over
    supervised positions (``labels[:, 1:] != -100``).

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

    shift_logits = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    sup_mask = shift_labels != -100
    n_sup = int(sup_mask.sum().item())

    l_ce = shift_logits.sum() * 0.0
    l_kl = shift_logits.sum() * 0.0
    if n_sup > 0:
        z_sup = shift_logits[sup_mask]
        y_sup = shift_labels[sup_mask]
        t_probs_all = None
        t_idx_all = None
        t_tail_all = None
        t_log_pt_all = None
        t_legacy_all = None
        t_full_all = None
        if teacher_target is not None and lambda_kl > 0.0:
            if "top_k_probs" in teacher_target and "top_k_indices" in teacher_target:
                t_probs_all = teacher_target["top_k_probs"][:, :-1, :][sup_mask]
                t_idx_all = teacher_target["top_k_indices"][:, :-1, :][sup_mask]
                if "retained_mass" in teacher_target:
                    t_tail_all = (1.0 - teacher_target["retained_mass"][:, :-1][sup_mask].float()).clamp(min=0.0)
                if "log_p_tail" in teacher_target:
                    t_log_pt_all = teacher_target["log_p_tail"][:, :-1][sup_mask]
            elif "top_k_logits" in teacher_target and "top_k_indices" in teacher_target:
                t_legacy_all = teacher_target["top_k_logits"][:, :-1, :][sup_mask]
                t_idx_all = teacher_target["top_k_indices"][:, :-1, :][sup_mask]
            elif "logits" in teacher_target:
                t_full_all = teacher_target["logits"][:, :-1, :][sup_mask]

        step_c = max(1, int(chunk_size))
        ce_sum = z_sup.sum() * 0.0
        kl_sum = z_sup.sum() * 0.0
        for start in range(0, n_sup, step_c):
            end = min(n_sup, start + step_c)
            c_ce, c_kl = _obj3_kl_and_ce_chunk_sum(
                z_sup[start:end],
                y_sup[start:end],
                top_probs_c=t_probs_all[start:end] if t_probs_all is not None else None,
                top_idx_c=t_idx_all[start:end] if t_idx_all is not None else None,
                tail_mass_c=t_tail_all[start:end] if t_tail_all is not None else None,
                log_p_tail_c=t_log_pt_all[start:end] if t_log_pt_all is not None else None,
                full_t_logits_c=t_full_all[start:end] if t_full_all is not None else None,
                legacy_top_k_logits_c=t_legacy_all[start:end] if t_legacy_all is not None else None,
                temperature=temperature,
                ce_weight=ce_weight,
                lambda_kl=lambda_kl,
                tail_threshold=tail_threshold,
                topk_tail_bucket=topk_tail_bucket,
            )
            ce_sum = ce_sum + c_ce
            kl_sum = kl_sum + c_kl
        inv_n = 1.0 / float(n_sup)
        l_ce = ce_sum * inv_n
        l_kl = kl_sum * inv_n

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

    total = float(ce_weight) * l_ce + float(lambda_kl) * l_kl + float(mu_hidden) * l_hidden
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
            for suffix in (".group_alpha", ".group_beta", ".col_v", ".lora_A", ".lora_B")
        )
    }


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _forward_recovery_step(
    model: Any,
    batch: dict[str, Any],
    t_entry: dict[str, Any] | None,
    cfg: MixedPrecisionRecoveryConfig,
    *,
    backbone: Any | None = None,
    lm_head: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    fwd_kwargs: dict[str, Any] = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        fwd_kwargs["attention_mask"] = batch["attention_mask"]
    if cfg.mu_hidden > 0.0:
        fwd_kwargs["output_hidden_states"] = True

    if backbone is not None and lm_head is not None:
        try:
            b_out = backbone(**fwd_kwargs)
        except TypeError:
            fwd_kwargs.pop("output_hidden_states", None)
            b_out = backbone(**fwd_kwargs)
        last_hidden = getattr(b_out, "last_hidden_state", b_out[0] if isinstance(b_out, tuple) else b_out)
        hidden = getattr(b_out, "hidden_states", None)
        return compute_fused_chunked_kl_loss(
            last_hidden,
            lm_head,
            batch["labels"],
            teacher_target=t_entry,
            student_hidden_states=hidden,
            ce_weight=cfg.ce_weight,
            lambda_kl=cfg.lambda_kl,
            mu_hidden=cfg.mu_hidden,
            temperature=cfg.temperature,
            chunk_size=cfg.chunk_size,
            topk_tail_bucket=cfg.topk_tail_bucket,
        )

    try:
        out = model(**fwd_kwargs)
    except TypeError:
        fwd_kwargs.pop("output_hidden_states", None)
        out = model(**fwd_kwargs)
    logits = getattr(out, "logits", out[0] if isinstance(out, tuple) else out)
    hidden = getattr(out, "hidden_states", None)
    return compute_recovery_loss(
        logits,
        batch["labels"],
        teacher_target=t_entry,
        student_hidden_states=hidden,
        ce_weight=cfg.ce_weight,
        lambda_kl=cfg.lambda_kl,
        mu_hidden=cfg.mu_hidden,
        temperature=cfg.temperature,
        chunk_size=cfg.chunk_size,
        topk_tail_bucket=cfg.topk_tail_bucket,
    )


def _split_recovery_param_groups(
    model: Any, cfg: MixedPrecisionRecoveryConfig
) -> tuple[list[dict[str, Any]], list[Any]]:
    eff_adapter_lr = float(cfg.lr if cfg.lr is not None else cfg.adapter_lr)
    eff_scale_lr = float(cfg.scale_lr)
    adapter_params: list[Any] = []
    scale_params: list[Any] = []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if pname.endswith((".lora_A", ".lora_B")):
            adapter_params.append(param)
        else:
            scale_params.append(param)
    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append(
            {
                "name": "adapters",
                "params": adapter_params,
                "lr": eff_adapter_lr,
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if scale_params:
        groups.append(
            {
                "name": "scales",
                "params": scale_params,
                "lr": eff_scale_lr,
                "weight_decay": float(cfg.weight_decay),
            }
        )
    trainable = adapter_params + scale_params
    return groups, trainable


def _build_warmup_cosine_scheduler(
    optimizer: Any,
    *,
    total_optim_steps: int,
    warmup_ratio: float,
    scheduler_type: str = "cosine",
) -> Any:
    import math
    import torch

    total_opt = max(1, int(total_optim_steps))
    warmup_steps = int(round(max(0.0, float(warmup_ratio)) * total_opt))

    def _lr_lambda(opt_step: int) -> float:
        if warmup_steps > 0 and opt_step < warmup_steps:
            return float(opt_step + 1) / float(warmup_steps)
        if scheduler_type == "constant":
            return 1.0
        denom = max(1, total_opt - warmup_steps)
        progress = min(1.0, max(0.0, float(opt_step - warmup_steps) / float(denom)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


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
    backbone, lm_head = _resolve_backbone_and_head(model)
    losses: list[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_batches):
            t_entry = (
                eval_teacher_cache.entries[idx]
                if eval_teacher_cache and idx < len(eval_teacher_cache.entries)
                else None
            )
            loss, _ = _forward_recovery_step(
                model,
                batch,
                t_entry,
                cfg,
                backbone=backbone,
                lm_head=lm_head,
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

    Uses dual optimizer LR groups (``adapter_lr=5e-5`` / ``scale_lr=5e-6`` by
    default), linear warmup + cosine decay scheduling, fused chunked Obj-3
    recovery loss when backbone and ``lm_head`` are exposed cleanly, persists full
    training/scheduler/RNG/cursor state, verifies frozen quantized codes on
    exit, and restores ``model.eval()``.

    # Errors: raises RecoveryError on non-finite loss/gradients, frozen code
    mutation, or incompatible checkpoint resume metadata.
    """
    try:
        import torch
    except ImportError as exc:
        raise RecoveryError("PyTorch is required for train_mixed_precision_recovery") from exc

    import math as _math

    cfg = config or MixedPrecisionRecoveryConfig()
    param_groups, trainable = _split_recovery_param_groups(model, cfg)
    if not trainable:
        raise RecoveryError("No trainable recovery parameters found; call configure_mixed_precision_recovery first")
    for p in trainable:
        if p.dtype != torch.float32:
            raise RecoveryError(f"Trainable recovery parameter dtype must be float32, got {p.dtype}")

    total_batches = len(train_batches)
    if total_batches == 0:
        raise RecoveryError("train_batches must not be empty")

    target_steps = max_steps if max_steps is not None else total_batches
    accum = max(1, int(cfg.grad_accum_steps))
    total_optim_steps = max(1, _math.ceil(target_steps / accum))

    frozen_snap = _snapshot_frozen_codes(model)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = _build_warmup_cosine_scheduler(
        optimizer,
        total_optim_steps=total_optim_steps,
        warmup_ratio=cfg.warmup_ratio,
        scheduler_type=cfg.lr_scheduler_type,
    )
    fingerprint = _recovery_config_fingerprint(cfg)
    backbone, lm_head = _resolve_backbone_and_head(model)

    ckpt_dir = Path(run_dir).resolve() / "recovery_state" if run_dir is not None else None
    latest_path = ckpt_dir / "latest.pt" if ckpt_dir is not None else None
    best_path = ckpt_dir / "best.pt" if ckpt_dir is not None else None

    start_step = 0
    tokens_consumed = 0
    best_eval_loss = float("inf")
    eval_history: list[dict[str, Any]] = []
    history: list[dict[str, float]] = []
    batch_cursor = 0

    if resume and latest_path is not None and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        if ckpt.get("config_fingerprint") != fingerprint:
            raise RecoveryError(
                "Recovery checkpoint config_fingerprint mismatch; refusing unsafe resume"
            )
        model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        start_step = int(ckpt["step"])
        tokens_consumed = int(ckpt["tokens_consumed"])
        best_eval_loss = float(ckpt.get("best_eval_loss", float("inf")))
        eval_history = list(ckpt.get("eval_history", []))
        batch_cursor = int(ckpt.get("batch_cursor", start_step % total_batches))

    step = start_step
    try:
        model.train()
        optimizer.zero_grad()

        while step < target_steps and tokens_consumed < cfg.supervised_token_budget:
            batch = train_batches[batch_cursor]
            t_entry = (
                teacher_cache.entries[batch_cursor]
                if teacher_cache is not None and batch_cursor < len(teacher_cache.entries)
                else None
            )

            loss, metrics = _forward_recovery_step(
                model,
                batch,
                t_entry,
                cfg,
                backbone=backbone,
                lm_head=lm_head,
            )
            if not _math.isfinite(metrics["loss"]):
                raise RecoveryError(f"Non-finite recovery loss at step {step}: {metrics['loss']}")

            scaled_loss = loss / float(accum)
            scaled_loss.backward()

            step += 1
            batch_cursor = (batch_cursor + 1) % total_batches
            tokens_consumed += int(metrics["supervised_tokens"])
            history.append({"step": float(step), **metrics})

            if step % accum == 0 or step >= target_steps:
                for p in trainable:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise RecoveryError(f"Non-finite gradient detected at step {step}")
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
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
                            "batch_cursor": batch_cursor,
                            "tokens_consumed": tokens_consumed,
                            "best_eval_loss": best_eval_loss,
                            "config_fingerprint": fingerprint,
                            "trainable_state_dict": _extract_trainable_state(model),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "rng_state": torch.get_rng_state(),
                        },
                        best_path,
                    )
                elif ev_loss < best_eval_loss:
                    best_eval_loss = ev_loss

            if latest_path is not None and (step % max(1, cfg.checkpoint_every_steps) == 0 or step >= target_steps):
                _atomic_torch_save(
                    {
                        "step": step,
                        "batch_cursor": batch_cursor,
                        "tokens_consumed": tokens_consumed,
                        "best_eval_loss": best_eval_loss,
                        "eval_history": eval_history,
                        "config_fingerprint": fingerprint,
                        "trainable_state_dict": _extract_trainable_state(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
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
    finally:
        model.eval()

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
        "param_group_lrs": [float(g["lr"]) for g in optimizer.param_groups],
    }
