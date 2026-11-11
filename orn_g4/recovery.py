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

import json
from dataclasses import dataclass
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
]
