"""CPU-only tests for Azrael mixed-precision QTEA rebuild, recovery, and Q4 fallback."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from orn_g4.data import (
    prepare_full_context_windows,
    resolve_latest_dataset_revision,
    select_stratified_calibration,
)
from orn_g4.errors import DataAlignmentError, ExportError, QTEAError, RecoveryError
from orn_g4.export import (
    export_mixed_precision_package,
    load_mixed_precision_package,
    verify_mixed_precision_package,
)
from orn_g4.fallback import (
    build_q4_fallback_model,
    export_q4_fallback_package,
    load_q4_fallback_package,
    verify_q4_fallback_package,
)
from orn_g4.quant4 import Protected4BitLinear, pack_uint4, unpack_uint4
from orn_g4.qtea import (
    QTEAConfig,
    QTEALinear,
    convert_model_mixed_precision,
    discover_mixed_precision_targets,
    pack_layer,
    unpack_layer,
)
from orn_g4.recovery import (
    MixedPrecisionRecoveryConfig,
    capture_teacher_targets,
    compute_recovery_loss,
    configure_mixed_precision_recovery,
    train_mixed_precision_recovery,
)


def _make_hybrid_ornith_skeleton(
    *,
    num_layers: int = 32,
    dim: int = 16,
    vocab_size: int = 64,
    include_in_proj_a: bool = True,
    extra_unexpected: bool = False,
    tied_lm_head: bool = True,
) -> nn.Module:
    """Build a tiny synthetic Ornith-1.5 hybrid model (24 linear_attn + 8 self_attn + 32 mlp)."""

    class LinearAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_qkv = nn.Linear(dim, dim, bias=False)
            if include_in_proj_a:
                self.in_proj_a = nn.Linear(dim, dim, bias=False)
            self.in_proj_b = nn.Linear(dim, dim, bias=False)
            self.in_proj_z = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
            if extra_unexpected:
                self.unexpected_proj = nn.Linear(dim, dim, bias=False)

    class SelfAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(dim, dim, bias=False)
            self.up_proj = nn.Linear(dim, dim, bias=False)
            self.down_proj = nn.Linear(dim, dim, bias=False)

    class HybridLayer(nn.Module):
        def __init__(self, idx: int) -> None:
            super().__init__()
            self.input_layernorm = nn.LayerNorm(dim)
            if (idx + 1) % 4 == 0:
                self.self_attn = SelfAttn()
            else:
                self.linear_attn = LinearAttn()
            self.mlp = MLP()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if hasattr(self, "self_attn"):
                h = self.self_attn.o_proj(
                    self.self_attn.q_proj(x)
                    + self.self_attn.k_proj(x)
                    + self.self_attn.v_proj(x)
                )
            else:
                h = self.linear_attn.out_proj(
                    self.linear_attn.in_proj_qkv(x)
                    + self.linear_attn.in_proj_a(x)
                    + self.linear_attn.in_proj_b(x)
                    + self.linear_attn.in_proj_z(x)
                )
            x = x + h
            m = self.mlp.down_proj(
                torch.relu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
            )
            return x + m

    class TinyOrnith(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(dim, dim, bias=False)
            self.mtp = nn.Linear(dim, dim, bias=False)
            self.language_model = nn.Module()
            self.language_model.embed_tokens = nn.Embedding(vocab_size, dim)
            self.language_model.layers = nn.ModuleList(
                [HybridLayer(i) for i in range(num_layers)]
            )
            self.language_model.norm = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            if tied_lm_head:
                self.lm_head.weight = self.language_model.embed_tokens.weight

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_hidden_states: bool = False,
        ) -> SimpleNamespace:
            h = self.language_model.embed_tokens(input_ids)
            hidden_states = [h] if output_hidden_states else None
            for layer in self.language_model.layers:
                h = layer(h)
                if output_hidden_states and hidden_states is not None:
                    hidden_states.append(h)
            h = self.language_model.norm(h)
            logits = self.lm_head(h)
            return SimpleNamespace(logits=logits, hidden_states=tuple(hidden_states) if hidden_states is not None else None)

    return TinyOrnith()


def test_mixed_precision_discovery_counts_248_and_fails_closed() -> None:
    model = _make_hybrid_ornith_skeleton(num_layers=32, dim=16)
    plan = discover_mixed_precision_targets(model)
    assert plan.layer_count == 32
    assert plan.linear_attn_layers == 24
    assert plan.self_attn_layers == 8
    assert plan.mlp_layers == 32
    assert len(plan.qtea_targets) == 144
    assert len(plan.int4_targets) == 104
    assert len(plan.targets) == 248
    assert any(t.leaf == "in_proj_a" and t.precision == "int4" for t in plan.int4_targets)
    assert "visual" not in plan.assignments
    assert "lm_head" not in plan.assignments
    assert "mtp" not in plan.assignments

    # Missing in_proj_a -> count mismatch -> fail closed
    broken = _make_hybrid_ornith_skeleton(num_layers=32, dim=16, include_in_proj_a=False)
    with pytest.raises(QTEAError, match="count mismatch"):
        discover_mixed_precision_targets(broken)

    # Unexpected linear projection -> fail closed
    unexpected = _make_hybrid_ornith_skeleton(num_layers=32, dim=16, extra_unexpected=True)
    with pytest.raises(QTEAError, match="Unexpected projection"):
        discover_mixed_precision_targets(unexpected)


def test_qtea_unpack_autograd_safe_and_device_consistent() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(128, 32, bias=True)
    cfg = QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125)
    from orn_g4.qtea import calibrate_and_quantize_projection

    packed, qmod = calibrate_and_quantize_projection(
        "test.proj", lin, [torch.randn(16, 128)], config=cfg
    )
    # Verify pack/unpack parity
    w_unpacked = unpack_layer(packed)
    assert w_unpacked.shape == (32, 128)
    assert torch.allclose(w_unpacked, qmod.effective_weight(), atol=1e-6)

    # Enable differentiable recovery parameters + rank-16 LoRA -> backward must succeed without in-place error
    qmod.enable_recovery_parameters(train_scales=True, lora_rank=16, lora_alpha=32.0)
    x = torch.randn(2, 128, requires_grad=True)
    out = qmod(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert qmod.group_alpha.grad is not None and torch.isfinite(qmod.group_alpha.grad).all()
    assert qmod.group_beta.grad is not None and torch.isfinite(qmod.group_beta.grad).all()
    assert qmod.col_v.grad is not None and torch.isfinite(qmod.col_v.grad).all()
    assert qmod.lora_A.grad is not None and qmod.lora_B.grad is not None


def test_protected_4bit_pack_unpack_and_recovery_adapters() -> None:
    torch.manual_seed(1)
    # Odd column count to verify padding/slicing
    q = torch.randint(0, 16, (8, 13), dtype=torch.uint8)
    packed = pack_uint4(q)
    unpacked = unpack_uint4(packed, cols=13)
    assert torch.equal(q, unpacked)

    lin = nn.Linear(32, 16, bias=True)
    q4 = Protected4BitLinear.from_linear(lin, group_size=16)
    x = torch.randn(3, 32)
    err = (lin(x) - q4(x)).abs().mean().item()
    assert err < 0.05

    q4.enable_recovery_adapters(rank=16, alpha=32.0)
    qweight_before = q4.qweight.clone()
    out = q4(x)
    out.sum().backward()
    assert q4.lora_A.grad is not None and q4.lora_B.grad is not None
    assert torch.equal(q4.qweight, qweight_before)


def test_teacher_guided_recovery_checkpoint_resume_and_best_rollback(tmp_path: Path) -> None:
    torch.manual_seed(2)
    teacher = _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32)
    cfg = QTEAConfig(group_size=16, residual_group=4, salient_ratio=0.125)
    student, _, _, _ = convert_model_mixed_precision(
        teacher,
        qtea_config=cfg,
        int4_group_size=16,
        expected_qtea=18,
        expected_int4=13,
        expected_total=31,
        strict=True,
    )

    rec_cfg = MixedPrecisionRecoveryConfig(
        lora_rank=16,
        lora_alpha=32.0,
        lr=1e-3,
        lambda_kl=0.5,
        mu_hidden=0.1,
        top_k_logits=16,
        checkpoint_every_steps=1,
        eval_every_steps=1,
        supervised_token_budget=10_000,
        svd_init=True,
    )
    teacher_weights = {k: v for k, v in teacher.state_dict().items()}
    summary = configure_mixed_precision_recovery(
        student, config=rec_cfg, teacher_weights=teacher_weights
    )
    assert summary["qtea_modules"] == 18
    assert summary["int4_modules"] == 13
    assert summary["trainable_params"] > 0

    batches = [
        {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.cat(
                [torch.full((2, 2), -100, dtype=torch.long), torch.randint(0, 32, (2, 8))],
                dim=1,
            ),
        }
        for _ in range(3)
    ]
    eval_batches = [batches[0]]

    t_cache = capture_teacher_targets(
        teacher, batches, top_k=16, offload_to_cpu=True, hidden_layer_indices=(1, 3)
    )
    assert t_cache.truncated_kl is True
    assert "Top-16" in t_cache.omitted_mass_note
    ev_cache = capture_teacher_targets(
        teacher, eval_batches, top_k=16, offload_to_cpu=True, hidden_layer_indices=(1, 3)
    )

    run_dir = tmp_path / "recovery_run"
    res1 = train_mixed_precision_recovery(
        student,
        batches,
        teacher_cache=t_cache,
        eval_batches=eval_batches,
        eval_teacher_cache=ev_cache,
        config=rec_cfg,
        run_dir=run_dir,
        max_steps=2,
    )
    assert res1["steps_completed"] == 2
    assert (run_dir / "recovery_state" / "latest.pt").is_file()

    # Resume for 1 more step
    res2 = train_mixed_precision_recovery(
        student,
        batches,
        teacher_cache=t_cache,
        eval_batches=eval_batches,
        eval_teacher_cache=ev_cache,
        config=rec_cfg,
        run_dir=run_dir,
        resume=True,
        max_steps=3,
    )
    assert res2["steps_completed"] == 3

    # Mismatched config fingerprint on resume must fail closed
    bad_cfg = MixedPrecisionRecoveryConfig(lora_rank=8)
    with pytest.raises(RecoveryError, match="config_fingerprint mismatch"):
        train_mixed_precision_recovery(
            student,
            batches,
            config=bad_cfg,
            run_dir=run_dir,
            resume=True,
            max_steps=4,
        )


def test_dynamic_dataset_revision_stratified_calibration_and_full_8192_context() -> None:
    fake_info = SimpleNamespace(sha="a" * 40)
    with patch("huggingface_hub.HfApi.dataset_info", return_value=fake_info):
        sha = resolve_latest_dataset_revision("moeshawky/azrael-dataset")
    assert sha == "a" * 40

    records = [
        {"id": "raw1", "source": "s1", "token_count": 500, "supervised_count": 100, "is_raw_capture": True},
        {"id": "nosup", "source": "s1", "token_count": 500, "supervised_count": 0},
        {"id": "s1_short", "source": "s1", "token_count": 500, "supervised_count": 200},
        {"id": "s2_med", "source": "s2", "token_count": 2500, "supervised_count": 800},
        {"id": "s3_long", "source": "s3", "token_count": 6000, "supervised_count": 1200},
    ]
    sel = select_stratified_calibration(records, target_count=3, seed=7)
    assert [r["id"] for r in sel] == ["s1_short", "s2_med", "s3_long"]

    # Full 8192 context preserved without silent 3072 truncation
    ids_8k = list(range(8192))
    lbl_8k = list(range(8192))
    full_win = prepare_full_context_windows(ids_8k, lbl_8k, max_seq_len=8192)
    assert full_win["windowed"] is False
    assert full_win["effective_context_len"] == 8192
    assert len(full_win["windows"][0]["input_ids"]) == 8192

    # Memory-bounded windowing masks overlap prefix labels to -100
    chunked = prepare_full_context_windows(
        list(range(100)), list(range(100)), max_seq_len=8192, window_size=40, stride=30
    )
    assert chunked["windowed"] is True
    assert chunked["windows"][1]["labels"][:10] == [-100] * 10
    assert chunked["windows"][1]["labels"][10] == 40

    with pytest.raises(DataAlignmentError):
        prepare_full_context_windows(list(range(9000)), list(range(9000)), max_seq_len=8192)


def test_mixed_precision_export_load_and_independent_q4_fallback(tmp_path: Path) -> None:
    torch.manual_seed(3)
    teacher = _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32, tied_lm_head=True)
    cfg = QTEAConfig(group_size=16, residual_group=4, salient_ratio=0.125)
    student, _, _, plan = convert_model_mixed_precision(
        teacher,
        qtea_config=cfg,
        int4_group_size=16,
        expected_qtea=18,
        expected_int4=13,
        expected_total=31,
        strict=True,
    )
    configure_mixed_precision_recovery(student, config=MixedPrecisionRecoveryConfig(lora_rank=16))

    pkg_dir = tmp_path / "mp_pkg"
    export_mixed_precision_package(
        student, pkg_dir, plan=plan, shard_max_bytes=4096, provenance={"dataset_sha": "b" * 40}
    )
    ver = verify_mixed_precision_package(pkg_dir)
    assert ver["verified"] is True
    assert ver["counts"] == {"qtea": 18, "int4": 13, "total": 31}

    reloaded = load_mixed_precision_package(
        pkg_dir,
        model_factory=lambda _: _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32, tied_lm_head=True),
    )
    x = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        assert torch.allclose(student(x).logits, reloaded(x).logits, atol=1e-5)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.language_model.embed_tokens.weight.data_ptr()

    # Corrupt shard -> verify must fail
    shard_file = next(pkg_dir.glob("model-*.safetensors"))
    data = bytearray(shard_file.read_bytes())
    data[-1] ^= 0xFF
    shard_file.write_bytes(bytes(data))
    with pytest.raises(ExportError, match="SHA-256 mismatch"):
        verify_mixed_precision_package(pkg_dir)

    # Independent Q4 fallback directly from teacher without QTEA
    q4_model, _, q4_plan = build_q4_fallback_model(
        teacher,
        group_size=16,
        expected_qtea=18,
        expected_int4=13,
        expected_total=31,
        strict=True,
    )
    q4_dir = tmp_path / "q4_pkg"
    export_q4_fallback_package(q4_model, q4_dir, plan=q4_plan)
    q4_ver = verify_q4_fallback_package(q4_dir)
    assert q4_ver["verified"] is True
    assert q4_ver["counts"] == {"qtea": 0, "int4": 31, "total": 31}
    q4_reloaded = load_q4_fallback_package(
        q4_dir,
        model_factory=lambda _: _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32, tied_lm_head=True),
    )
    with torch.no_grad():
        assert torch.allclose(q4_model(x).logits, q4_reloaded(x).logits, atol=1e-5)
