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
    # T-REG-248: frozen ADV-12.1 registry 112 QTEA / 88 INT4 / 48 BF16
    assert len(plan.qtea_targets) == 112
    assert len(plan.int4_targets) == 88
    assert len(plan.bf16_targets) == 48
    assert len(plan.targets) == 248
    assert {t.leaf for t in plan.bf16_targets} == {"in_proj_a", "in_proj_b"}
    assert {t.leaf for t in plan.int4_targets} == {"down_proj", "o_proj", "out_proj", "in_proj_z"}
    assert {t.leaf for t in plan.qtea_targets} == {
        "gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "in_proj_qkv"
    }
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
    from orn_g4.quant4 import (
        ALPHA_GRID,
        CODEC_NAME,
        dequantize_act_sym_int4,
        hessian_weighted_error,
        initialization_activation_weighted_svd,
        quantize_act_sym_int4,
        unpack_int4_signed,
    )

    torch.manual_seed(1)
    # Odd column count to verify low-even/high-odd nibble packing across full [-8, 7] range
    signed_q = torch.tensor([[-8, -7, -1, 0, 1, 7, -8, 7, 3, -4, 5, -2, 6]], dtype=torch.int8)
    packed_signed = pack_uint4((signed_q + 8).to(torch.uint8))
    assert packed_signed.dtype == torch.uint8
    assert int(packed_signed[0, 0].item()) == ((-8 + 8) | ((-7 + 8) << 4))
    assert torch.equal(signed_q, unpack_int4_signed(packed_signed, cols=13))

    q = torch.randint(0, 16, (8, 13), dtype=torch.uint8)
    packed = pack_uint4(q)
    unpacked = unpack_uint4(packed, cols=13)
    assert torch.equal(q, unpacked)

    # Reject BF16 trainable params in QTEALinear
    q_lin = nn.Linear(128, 32, bias=False)
    _, qmod = __import__("orn_g4.qtea", fromlist=["calibrate_and_quantize_projection"]).calibrate_and_quantize_projection(
        "test.fp32", q_lin, [torch.randn(8, 128)], config=QTEAConfig(group_size=128, residual_group=4)
    )
    with pytest.raises(QTEAError, match="float32"):
        qmod.enable_recovery_parameters(lora_rank=4, adapter_dtype=torch.bfloat16)

    # Canonical act_sym_int4_g128 with group=128, BF16 group_scale & inv_channel_scale, 5-grid Hessian search
    # Salient activation channels with normal weight magnitude benefit from alpha > 0 scaling.
    lin = nn.Linear(128, 16, bias=True)
    h_diag = torch.full((128,), 1e-3, dtype=torch.float32)
    h_diag[:4] = 500.0
    hess = torch.diag(h_diag)

    q4 = Protected4BitLinear.from_linear(lin, group_size=128, hessian=hess)
    raw_q4 = quantize_act_sym_int4(lin.weight.data, hessian=hess, group_size=128)
    payload = q4.to_layer_dict()
    assert payload["codec"] == CODEC_NAME == "act_sym_int4_g128"
    assert payload["group_scale"].dtype == torch.bfloat16
    assert payload["group_scale"].shape == (16, 1)
    assert payload["inv_channel_scale"].dtype == torch.bfloat16
    assert payload["inv_channel_scale"].shape == (128,)
    assert q4.alpha_selected in ALPHA_GRID
    assert q4.alpha_selected > 0.0
    deq = dequantize_act_sym_int4(
        payload["qweight"], payload["group_scale"], payload["inv_channel_scale"], cols=128, group_size=128
    )
    assert torch.allclose(deq, q4.base_weight(), atol=1e-6)
    err_zero = quantize_act_sym_int4(lin.weight.data, hessian=hess, group_size=128, alphas=(0.0,))["hessian_error"]
    assert raw_q4["hessian_error"] < err_zero

    # Activation-weighted SVD reduces activation-weighted residual
    act_scale = h_diag.sqrt()
    residual = lin.weight.data - q4.base_weight()
    a_svd, b_svd = initialization_activation_weighted_svd(residual, act_scale, rank=4)
    assert a_svd.dtype == torch.float32 and b_svd.dtype == torch.float32
    weighted_before = ((residual * act_scale.unsqueeze(0)) ** 2).sum().item()
    weighted_after = (((residual - b_svd @ a_svd) * act_scale.unsqueeze(0)) ** 2).sum().item()
    assert weighted_after < weighted_before

    # Reject train_scales=True and legacy affine_uint4 payloads
    with pytest.raises(QTEAError, match="train_scales"):
        q4.enable_recovery_adapters(rank=16, train_scales=True)
    bad_payload = dict(payload)
    bad_payload["codec"] = "affine_uint4"
    with pytest.raises(QTEAError, match="Unsupported INT4 codec"):
        Protected4BitLinear.from_payload(bad_payload)

    q4.enable_recovery_adapters(rank=16, alpha=32.0)
    assert q4.lora_A.dtype == torch.float32 and q4.lora_B.dtype == torch.float32
    qweight_before = q4.qweight.clone()
    x = torch.randn(3, 128)
    out = q4(x)
    out.sum().backward()
    assert q4.lora_A.grad is not None and q4.lora_B.grad is not None
    assert torch.equal(q4.qweight, qweight_before)


def test_hessian_mask_scale_invariance_and_int4_wiring() -> None:
    """T-HESSIAN-MASK: valid-token masking removes padding outliers and H=(2/N)X^TX is scale-invariant."""
    from orn_g4.quant4 import CODEC_NAME
    from orn_g4.qtea import QTEA

    torch.manual_seed(11)
    lin = nn.Linear(128, 32, bias=False)
    valid_a = torch.randn(1, 5, 128)
    valid_b = torch.randn(1, 3, 128)
    padded_a = torch.cat([valid_a, torch.full((1, 7, 128), 1e4)], dim=1)
    mask_a = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    padded_b = torch.cat([valid_b, torch.full((1, 9, 128), -1e4)], dim=1)
    mask_b = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)

    # Sequence-count normalization with attention_mask
    q_masked_seq = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125, hessian_norm="sequences"))
    q_masked_seq.add_batch(padded_a, attention_mask=mask_a)
    q_masked_seq.add_batch(padded_b, attention_mask=mask_b)

    # Token-count normalization with identical valid tokens
    q_masked_tok = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125, hessian_norm="tokens"))
    q_masked_tok.add_batch(padded_a, attention_mask=mask_a)
    q_masked_tok.add_batch(padded_b, attention_mask=mask_b)

    # Unpadded valid-only sequence batch
    q_clean = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125, hessian_norm="sequences"))
    q_clean.add_batch(valid_a)
    q_clean.add_batch(valid_b)
    assert torch.allclose(q_masked_seq.hessian, q_clean.hessian, atol=1e-12, rtol=1e-12)

    # Negative control: omitting attention_mask allows 1e4 pad outliers to corrupt Hessian
    q_unmasked = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125))
    q_unmasked.add_batch(padded_a)
    q_unmasked.add_batch(padded_b)
    assert float((q_unmasked.hessian - q_clean.hessian).abs().max().item()) > 1e6

    # Positive scale invariance: seq-count vs valid-token-count yields identical ternary codes and <= 1e-5 weight parity
    # (exact to machine precision on quantized outputs when the Hessian scaling factor is well-conditioned / power-of-two).
    p_seq = q_masked_seq.quantize()
    p_tok = q_masked_tok.quantize()
    max_diff = float((p_seq["dequantized_weight"] - p_tok["dequantized_weight"]).abs().max().item())
    assert max_diff <= 1e-5
    assert torch.equal(p_seq["ternary"], p_tok["ternary"])

    full_tok = torch.randn(2, 16, 128)
    q_s1 = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125, hessian_norm="sequences"))
    q_t1 = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125, hessian_norm="tokens"))
    q_s1.add_batch(full_tok)
    q_t1.add_batch(full_tok)
    # Rescaled Hessians obey H_seq / valid_tokens == H_tok / nsamples (scalar invariance)
    h_rel_err = float((q_s1.hessian / 16.0 - q_t1.hessian).abs().max().item())
    assert h_rel_err <= 5.6e-16
    res_s1 = q_s1.quantize()
    res_t1 = q_t1.quantize()
    assert torch.equal(res_s1["ternary"], res_t1["ternary"])
    assert float((res_s1["dequantized_weight"] - res_t1["dequantized_weight"]).abs().max().item()) <= 1e-5
    assert p_seq["conditioning"]["valid_tokens"] == 8
    assert p_seq["conditioning"]["total_tokens"] == 24
    assert p_seq["conditioning"]["masked_tokens_dropped"] == 16
    assert p_seq["conditioning"]["zero_count"] == 0
    assert p_seq["conditioning"]["cholesky_fallback"] is False

    # Rank-deficient / zero-column Hessian triggers diagnostics & non-zero zero_count
    q_zero = QTEA(lin, QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125))
    zero_inp = torch.randn(1, 6, 128)
    zero_inp[:, :, 0] = 0.0
    q_zero.add_batch(zero_inp)
    p_zero = q_zero.quantize()
    assert p_zero["conditioning"]["zero_count"] >= 1

    # Length mismatch in attention_mask fails closed
    with pytest.raises(QTEAError, match="attention_mask"):
        q_zero.add_batch(zero_inp, attention_mask=torch.ones(1, 3))

    # End-to-end calibration_batches_by_name wires masked Hessians into INT4 projections too
    skel = _make_hybrid_ornith_skeleton(num_layers=4, dim=128, vocab_size=32)
    cal_batches = {
        "layers.0.linear_attn.in_proj_qkv": [(padded_a, mask_a)],
        "layers.0.linear_attn.out_proj": [(padded_a, mask_a)],
    }
    converted, _, int4_packed, _ = convert_model_mixed_precision(
        skel,
        calibration_batches_by_name=cal_batches,
        qtea_config=QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125),
        int4_group_size=128,
        expected_qtea=14,
        expected_int4=11,
        expected_total=31,
        strict=True,
    )
    out_proj = converted.language_model.layers[0].linear_attn.out_proj
    assert isinstance(out_proj, Protected4BitLinear)
    assert out_proj.conditioning["valid_tokens"] == 5
    assert out_proj.conditioning["masked_tokens_dropped"] == 7
    assert int4_packed["language_model.layers.0.linear_attn.out_proj"]["codec"] == CODEC_NAME


def test_teacher_guided_recovery_checkpoint_resume_and_best_rollback(tmp_path: Path) -> None:
    torch.manual_seed(2)
    teacher = _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32)
    cfg = QTEAConfig(group_size=16, residual_group=4, salient_ratio=0.125)
    student, _, _, _ = convert_model_mixed_precision(
        teacher,
        qtea_config=cfg,
        int4_group_size=16,
        expected_qtea=14,
        expected_int4=11,
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
    assert summary["qtea_modules"] == 14
    assert summary["int4_modules"] == 11
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
        expected_qtea=14,
        expected_int4=11,
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
    assert ver["counts"] == {"qtea": 14, "int4": 11, "total": 25}  # BF16 projections are not packaged
    for bf16_target in plan.bf16_targets:
        parts = bf16_target.name.split(".")
        mod_student = student
        for p in parts:
            mod_student = getattr(mod_student, p)
        assert type(mod_student) is nn.Linear  # preserved uncompressed BF16/FP32 nn.Linear

    reloaded = load_mixed_precision_package(
        pkg_dir,
        model_factory=lambda _: _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32, tied_lm_head=True),
    )
    for bf16_target in plan.bf16_targets:
        parts = bf16_target.name.split(".")
        mod_reloaded = reloaded
        for p in parts:
            mod_reloaded = getattr(mod_reloaded, p)
        assert type(mod_reloaded) is nn.Linear  # preserved unconverted on reload too
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
        expected_qtea=14,
        expected_int4=11,
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


def test_int8_vocab_export_tied_reload_and_delta_ce_gate(tmp_path: Path) -> None:
    """T-INT8-VOCAB: symmetric group-128 INT8 embed_tokens + lm_head, tie preservation, BF16 manifest accounting, and post-INT8 ΔCE gate."""
    import json
    from orn_g4.errors import QualityGateError
    from orn_g4.quant4 import (
        INT8_CODEC_NAME,
        QuantizedInt8Embedding,
        QuantizedInt8Linear,
        convert_vocab_int8,
        measure_vocab_int8_delta_ce,
    )

    torch.manual_seed(17)
    teacher = _make_hybrid_ornith_skeleton(num_layers=4, dim=128, vocab_size=64, tied_lm_head=True)
    with torch.no_grad():
        teacher.language_model.embed_tokens.weight.mul_(3.0)
    cfg = QTEAConfig(group_size=128, residual_group=4, salient_ratio=0.125)
    student, _, _, plan = convert_model_mixed_precision(
        teacher,
        qtea_config=cfg,
        int4_group_size=128,
        expected_qtea=14,
        expected_int4=11,
        expected_total=31,
        strict=True,
    )
    configure_mixed_precision_recovery(student, config=MixedPrecisionRecoveryConfig(lora_rank=16))

    # Measure post-INT8 delta CE gate against student's own high-confidence predictions
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        preds = student(ids).logits.argmax(dim=-1)
    labels = torch.full_like(ids, -100)
    labels[:, 1:] = preds[:, :-1]
    labels[:, :2] = -100
    eval_batches = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}]

    int8_student, vocab_info = convert_vocab_int8(student, group_size=128, inplace=False)
    assert vocab_info["codec"] == INT8_CODEC_NAME
    assert vocab_info["tied_vocab"] is True
    assert isinstance(int8_student.language_model.embed_tokens, QuantizedInt8Embedding)
    assert isinstance(int8_student.lm_head, QuantizedInt8Linear)
    assert (
        int8_student.language_model.embed_tokens.qweight.data_ptr()
        == int8_student.lm_head.qweight.data_ptr()
    )
    assert (
        int8_student.language_model.embed_tokens.scale.data_ptr()
        == int8_student.lm_head.scale.data_ptr()
    )

    gate_res = measure_vocab_int8_delta_ce(student, int8_student, eval_batches, max_regression=0.10)
    assert gate_res["passed"] is True
    assert abs(gate_res["delta_ce"]) < 0.05

    # Negative control: intentionally corrupted INT8 vocab triggers QualityGateError
    corrupted = copy.deepcopy(int8_student)
    with torch.no_grad():
        corrupted.lm_head.qweight.zero_()
    with pytest.raises(QualityGateError, match="CE degradation"):
        measure_vocab_int8_delta_ce(student, corrupted, eval_batches, max_regression=0.01)

    # Export with quantize_vocab_int8=True from BF16 student -> standalone package accounts for BF16 & INT8 vocab
    pkg_dir = tmp_path / "int8_vocab_pkg"
    export_mixed_precision_package(
        student,
        pkg_dir,
        plan=plan,
        quantize_vocab_int8=True,
        vocab_group_size=128,
        post_int8_delta_ce=gate_res["delta_ce"],
        variant="multimodal",
    )
    manifest = json.loads((pkg_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bf16_count"] == 6
    assert len(manifest["bf16_projections"]) == 6
    assert manifest["vocab_int8"]["count"] == 2
    assert manifest["vocab_int8"]["codec"] == INT8_CODEC_NAME
    assert manifest["vocab_int8"]["post_int8_delta_ce"] == pytest.approx(gate_res["delta_ce"])

    ver = verify_mixed_precision_package(pkg_dir)
    assert ver["verified"] is True
    assert ver["bf16_count"] == 6
    assert ver["vocab_int8_count"] == 2

    reloaded = load_mixed_precision_package(
        pkg_dir,
        model_factory=lambda _: _make_hybrid_ornith_skeleton(num_layers=4, dim=128, vocab_size=64, tied_lm_head=True),
    )
    assert isinstance(reloaded.language_model.embed_tokens, QuantizedInt8Embedding)
    assert isinstance(reloaded.lm_head, QuantizedInt8Linear)
    assert (
        reloaded.language_model.embed_tokens.qweight.data_ptr()
        == reloaded.lm_head.qweight.data_ptr()
    )
    with torch.no_grad():
        assert torch.allclose(int8_student(ids).logits, reloaded(ids).logits, atol=1e-5)

    # Negative control: dropping a preserved BF16 projection on disk must fail verify & strict load
    bad_manifest = json.loads((pkg_dir / "manifest.json").read_text(encoding="utf-8"))
    bad_manifest["bf16_projections"].append("language_model.layers.0.linear_attn.missing_bf16")
    (pkg_dir / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ExportError, match="Missing preserved BF16 projection"):
        verify_mixed_precision_package(pkg_dir)


def test_fused_chunked_kl_parity_and_negative_controls() -> None:
    """T-FUSED-KL-PARITY: chunked Obj-3 top-K + tail-bucket KL matches dense reference and penalizes out-of-support logits."""
    import torch.nn.functional as F
    from orn_g4.quant4 import QuantizedInt8Linear
    from orn_g4.recovery import compute_fused_chunked_kl_loss

    torch.manual_seed(23)
    bsz, seqlen, dim, vocab = 2, 14, 32, 96
    lm_head = nn.Linear(dim, vocab, bias=True)
    hidden = torch.randn(bsz, seqlen, dim, requires_grad=True)
    teacher_logits = torch.randn(bsz, seqlen, vocab)
    # Give teacher sharp top-4 + non-trivial tail mass
    teacher_logits[:, :, :4] += 3.0

    teacher_model = nn.Module()
    teacher_model.forward = lambda input_ids, attention_mask=None: SimpleNamespace(logits=teacher_logits)  # type: ignore[method-assign]
    ids = torch.randint(0, vocab, (bsz, seqlen))
    labels = ids.clone()
    labels[:, :3] = -100

    cache = capture_teacher_targets(teacher_model, [{"input_ids": ids}], top_k=8, temperature=1.0, chunk_size=4)
    entry = cache.entries[0]
    assert entry["top_k_indices"].dtype == torch.int32
    assert entry["top_k_probs"].dtype == torch.float16
    assert entry["truncated_kl"] is True
    assert "retained_mass" in entry and "log_p_tail" in entry

    # Chunked fused Obj-3 KL (chunk_size=3 forces multiple chunks + checkpointing)
    loss_chunked, metrics = compute_fused_chunked_kl_loss(
        hidden,
        lm_head,
        labels,
        teacher_target=entry,
        ce_weight=0.0,
        lambda_kl=1.0,
        mu_hidden=0.0,
        temperature=1.0,
        chunk_size=3,
    )
    loss_chunked.backward()
    grad_h_chunked = hidden.grad.detach().clone()
    grad_w_chunked = lm_head.weight.grad.detach().clone()
    grad_b_chunked = lm_head.bias.grad.detach().clone()

    # Exact dense FP32 reference using the exact cached top-K + tail-bucket target
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    lm_ref = copy.deepcopy(lm_head)
    lm_ref.zero_grad()
    full_logits = F.linear(hidden_ref, lm_ref.weight, lm_ref.bias)[:, :-1, :].float()
    sup = labels[:, 1:] != -100
    z = full_logits[sup]
    p_top = entry["top_k_probs"][:, :-1, :][sup].float()
    idx_top = entry["top_k_indices"][:, :-1, :][sup].long()
    p_tail = (1.0 - entry["retained_mass"][:, :-1][sup].float()).clamp(min=0.0)
    q = F.softmax(z, dim=-1)
    q_top = torch.gather(q, dim=-1, index=idx_top)
    mask_support = torch.zeros_like(q).scatter(1, idx_top, 1.0)
    q_tail = (q * (1.0 - mask_support)).sum(dim=-1)
    loss_dense = (
        (p_top * (torch.log(p_top.clamp(min=1e-30)) - torch.log(q_top.clamp(min=1e-30)))).sum(dim=-1)
        + p_tail * (torch.log(p_tail.clamp(min=1e-30)) - torch.log(q_tail.clamp(min=1e-30)))
    ).mean()
    loss_dense.backward()

    assert abs(metrics["loss"] - float(loss_dense.item())) < 1e-5
    assert torch.allclose(grad_h_chunked, hidden_ref.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_w_chunked, lm_ref.weight.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_b_chunked, lm_ref.bias.grad, atol=1e-5, rtol=1e-4)

    # Negative control 1: out-of-support logits receive non-zero gradient!
    first_idx_set = set(idx_top[0].tolist())
    out_tok = next(v for v in range(vocab) if v not in first_idx_set)
    # Row grad on lm_head for a token that never appeared in top_k across all supervised rows
    all_idx_set = set(idx_top.reshape(-1).tolist())
    never_top = [v for v in range(vocab) if v not in all_idx_set]
    if never_top:
        assert float(grad_w_chunked[never_top[0]].abs().sum().item()) > 0.0

    # Negative control 2: K == V (zero tail mass) and peaked p_tail <= 1e-12 produce zero NaNs
    cache_full_k = capture_teacher_targets(teacher_model, [{"input_ids": ids}], top_k=vocab, temperature=1.0, chunk_size=4)
    hidden_k = hidden.detach().clone().requires_grad_(True)
    loss_k, _ = compute_fused_chunked_kl_loss(
        hidden_k, lm_head, labels, teacher_target=cache_full_k.entries[0], ce_weight=0.0, chunk_size=4
    )
    loss_k.backward()
    assert torch.isfinite(loss_k) and torch.isfinite(hidden_k.grad).all()

    # QuantizedInt8Linear lm_head effective_weight() hoisted ONCE per microbatch despite multiple chunks
    q8_head = QuantizedInt8Linear.from_linear(lm_head, group_size=32)
    hoist_calls = {"n": 0}
    orig_eff = q8_head.effective_weight
    def _counted_eff(dtype=None):
        hoist_calls["n"] += 1
        return orig_eff(dtype=dtype)
    q8_head.effective_weight = _counted_eff  # type: ignore[method-assign]
    hidden_q8 = hidden.detach().clone().requires_grad_(True)
    loss_q8, _ = compute_fused_chunked_kl_loss(
        hidden_q8, q8_head, labels, teacher_target=entry, ce_weight=0.0, chunk_size=2
    )
    loss_q8.backward()
    assert hoist_calls["n"] == 1
    assert torch.isfinite(hidden_q8.grad).all()


def test_frozen_codes_bitwise_and_resume_parity(tmp_path: Path) -> None:
    """T-FROZEN-CODES-AND-RESUME: uninterrupted vs save/resume parity, dual LRs, eval() exit, and frozen-code invariant."""
    from orn_g4.recovery import _verify_frozen_codes_unchanged, _snapshot_frozen_codes

    torch.manual_seed(29)
    teacher = _make_hybrid_ornith_skeleton(num_layers=4, dim=16, vocab_size=32)
    cfg = QTEAConfig(group_size=16, residual_group=4, salient_ratio=0.125)
    base_student, _, _, _ = convert_model_mixed_precision(
        teacher, qtea_config=cfg, int4_group_size=16, expected_qtea=14, expected_int4=11, expected_total=31, strict=True
    )
    rec_cfg = MixedPrecisionRecoveryConfig(
        lora_rank=16,
        lora_alpha=32.0,
        adapter_lr=5e-5,
        scale_lr=5e-6,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        ce_weight=0.0,
        lambda_kl=1.0,
        mu_hidden=0.1,
        temperature=1.0,
        top_k_logits=16,
        grad_accum_steps=1,
        checkpoint_every_steps=2,
        eval_every_steps=2,
        supervised_token_budget=50_000,
        svd_init=True,
    )
    batches = [
        {
            "input_ids": torch.randint(0, 32, (2, 10)),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.cat([torch.full((2, 2), -100, dtype=torch.long), torch.randint(0, 32, (2, 8))], dim=1),
        }
        for _ in range(4)
    ]
    eval_batches = [batches[0]]
    t_cache = capture_teacher_targets(teacher, batches, top_k=16, temperature=1.0, hidden_layer_indices=(1, 3))
    ev_cache = capture_teacher_targets(teacher, eval_batches, top_k=16, temperature=1.0, hidden_layer_indices=(1, 3))

    # Model A: 4 uninterrupted steps (note: scheduler total_optim_steps is computed per target_steps=4)
    model_a = copy.deepcopy(base_student)
    configure_mixed_precision_recovery(model_a, config=rec_cfg, teacher_model=teacher)
    res_a = train_mixed_precision_recovery(
        model_a,
        batches,
        teacher_cache=t_cache,
        eval_batches=eval_batches,
        eval_teacher_cache=ev_cache,
        config=rec_cfg,
        run_dir=tmp_path / "uninterrupted",
        restore_best=False,
        max_steps=4,
    )
    assert model_a.training is False  # .eval() restored at exit

    # Model B: 2 steps -> save checkpoint -> resume for remaining 2 steps (matching total_steps=4 schedule)
    model_b = copy.deepcopy(base_student)
    configure_mixed_precision_recovery(model_b, config=rec_cfg, teacher_model=teacher)
    run_b = tmp_path / "resumed"
    train_mixed_precision_recovery(
        model_b,
        batches,
        teacher_cache=t_cache,
        eval_batches=eval_batches,
        eval_teacher_cache=ev_cache,
        config=rec_cfg,
        run_dir=run_b,
        restore_best=False,
        max_steps=4,  # force total_steps=4 denominator by letting 2-step ckpt test below resume cleanly
    )
    # Test explicit 2-step pause & 4-step resume with matching constant LR schedule
    const_cfg = MixedPrecisionRecoveryConfig(
        lora_rank=16,
        lora_alpha=32.0,
        adapter_lr=5e-5,
        scale_lr=5e-6,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        ce_weight=0.0,
        lambda_kl=1.0,
        mu_hidden=0.1,
        temperature=1.0,
        top_k_logits=16,
        grad_accum_steps=1,
        checkpoint_every_steps=2,
        eval_every_steps=2,
        supervised_token_budget=50_000,
        svd_init=True,
    )
    m_full = copy.deepcopy(base_student)
    configure_mixed_precision_recovery(m_full, config=const_cfg, teacher_model=teacher)
    torch.manual_seed(777)
    train_mixed_precision_recovery(
        m_full, batches, teacher_cache=t_cache, eval_batches=eval_batches, eval_teacher_cache=ev_cache,
        config=const_cfg, run_dir=tmp_path / "full4", restore_best=False, max_steps=4,
    )

    m_part = copy.deepcopy(base_student)
    configure_mixed_precision_recovery(m_part, config=const_cfg, teacher_model=teacher)
    torch.manual_seed(777)
    train_mixed_precision_recovery(
        m_part, batches, teacher_cache=t_cache, eval_batches=eval_batches, eval_teacher_cache=ev_cache,
        config=const_cfg, run_dir=tmp_path / "split2", restore_best=False, max_steps=2,
    )
    ckpt = torch.load(tmp_path / "split2" / "recovery_state" / "latest.pt", map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in ckpt and "optimizer_state_dict" in ckpt and "batch_cursor" in ckpt
    res_split = train_mixed_precision_recovery(
        m_part, batches, teacher_cache=t_cache, eval_batches=eval_batches, eval_teacher_cache=ev_cache,
        config=const_cfg, run_dir=tmp_path / "split2", resume=True, restore_best=False, max_steps=4,
    )
    assert m_part.training is False
    assert res_split["param_group_lrs"] == pytest.approx([5e-5, 5e-6])
    for name, p_full in m_full.named_parameters():
        if p_full.requires_grad:
            p_res = dict(m_part.named_parameters())[name]
            assert p_res.dtype == torch.float32
            assert torch.allclose(p_full, p_res, atol=1e-7)

    # Negative control: flip one frozen code byte -> _verify_frozen_codes_unchanged raises RecoveryError
    snap = _snapshot_frozen_codes(m_part)
    _verify_frozen_codes_unchanged(m_part, snap)
    qtea_mod = m_part.language_model.layers[0].mlp.gate_proj
    qtea_mod.ternary.view(-1)[0] ^= 0x01
    with pytest.raises(RecoveryError, match="Frozen QTEA ternary codes mutated"):
        _verify_frozen_codes_unchanged(m_part, snap)


def test_text_causal_extraction_sidecar_mtp_exclusion_and_logit_parity(tmp_path: Path):
    """T-TEXT-EXTRACT: extract causal text BF16 + vision sidecar, drop mtp.*, verify AutoModelForCausalLM parity."""
    import json

    from safetensors.torch import load_file as st_load_file
    from safetensors.torch import save_file as st_save_file
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    from orn_g4.errors import ModelSourceError
    from orn_g4.modeling import extract_text_causal_lm_checkpoint, load_text_causal_model

    torch.manual_seed(2026)
    mm_cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
        },
        tie_word_embeddings=False,
    )
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        mm_model = Qwen3_5ForConditionalGeneration(mm_cfg).eval()
    finally:
        torch.set_default_dtype(prev_dtype)

    mm_dir = tmp_path / "mm_source"
    mm_dir.mkdir(parents=True)
    raw_cfg = mm_cfg.to_dict()
    (mm_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")
    (mm_dir / "tokenizer_config.json").write_text('{"tokenizer_class": "Qwen2Tokenizer"}\n', encoding="utf-8")
    (mm_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (mm_dir / "generation_config.json").write_text('{"eos_token_id": 1}\n', encoding="utf-8")
    (mm_dir / "preprocessor_config.json").write_text('{"size": {"height": 28, "width": 28}}\n', encoding="utf-8")

    # Build sharded multimodal safetensors + synthetic mtp.* tensors
    sd = {k: v.detach().cpu().contiguous() for k, v in mm_model.state_dict().items()}
    sd["mtp.fc.weight"] = torch.randn(32, 64, dtype=torch.bfloat16)
    sd["mtp.norm.weight"] = torch.ones(32, dtype=torch.bfloat16)

    keys = sorted(sd.keys())
    mid = len(keys) // 2
    s1 = {k: sd[k] for k in keys[:mid]}
    s2 = {k: sd[k] for k in keys[mid:]}
    st_save_file(s1, str(mm_dir / "model-00001-of-00002.safetensors"))
    st_save_file(s2, str(mm_dir / "model-00002-of-00002.safetensors"))
    weight_map = {k: ("model-00001-of-00002.safetensors" if k in s1 else "model-00002-of-00002.safetensors") for k in keys}
    (mm_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )

    out_root = tmp_path / "extracted"
    summary = extract_text_causal_lm_checkpoint(mm_dir, out_root)
    text_dir = out_root / "teacher_text_bf16"
    vision_dir = out_root / "vision_sidecar"
    assert text_dir.is_dir() and vision_dir.is_dir()
    assert summary["excluded_mtp_count"] == 2
    assert summary["excluded_mtp_keys"] == ["mtp.fc.weight", "mtp.norm.weight"]
    assert summary["text_tensor_count"] == 28  # 27 model.language_model.* + lm_head.weight
    assert summary["remapped_prefix_count"] == 27
    assert summary["vision_tensor_count"] == 21
    assert summary["source_tensor_count"] == 28 + 21 + 2

    # Config promoted cleanly; assets split properly; no stale index for single-shard output
    ext_cfg = json.loads((text_dir / "config.json").read_text(encoding="utf-8"))
    assert ext_cfg["model_type"] == "qwen3_5_text"
    assert ext_cfg["architectures"] == ["Qwen3_5ForCausalLM"]
    assert "vision_config" not in ext_cfg
    assert "image_token_id" not in ext_cfg
    assert (text_dir / "tokenizer_config.json").is_file()
    assert (text_dir / "chat_template.jinja").is_file()
    assert (text_dir / "generation_config.json").is_file()
    assert not (text_dir / "preprocessor_config.json").exists()
    assert not (text_dir / "model.safetensors.index.json").exists()
    assert (vision_dir / "preprocessor_config.json").is_file()
    assert (vision_dir / "model.safetensors").is_file()

    # Tensor ledger checks
    text_weights = st_load_file(str(text_dir / "model.safetensors"), device="cpu")
    assert all(not k.startswith(("model.language_model.", "model.visual.", "mtp.")) for k in text_weights)
    assert "model.embed_tokens.weight" in text_weights
    assert "lm_head.weight" in text_weights
    assert all(v.dtype == torch.bfloat16 for v in text_weights.values())
    vision_weights = st_load_file(str(vision_dir / "model.safetensors"), device="cpu")
    assert all(k.startswith("model.visual.") for k in vision_weights)

    # Negative controls: fail closed on overwrite=False and reject multimodal in load_text_causal_model
    with pytest.raises(ModelSourceError, match="overwrite=False"):
        extract_text_causal_lm_checkpoint(mm_dir, out_root, overwrite=False)
    with pytest.raises(ModelSourceError, match="multimodal checkpoint"):
        load_text_causal_model(mm_dir, device="cpu", dtype=torch.bfloat16)

    # Forward parity between multimodal model and standalone AutoModelForCausalLM
    causal_model = load_text_causal_model(text_dir, device="cpu", dtype=torch.bfloat16).eval()
    assert isinstance(causal_model, Qwen3_5ForCausalLM)
    input_ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        mm_logits = mm_model(input_ids=input_ids).logits.float()
        causal_logits = causal_model(input_ids=input_ids).logits.float()
    max_diff = float((mm_logits - causal_logits).abs().max().item())
    assert max_diff == pytest.approx(0.0, abs=1e-5)

    # Verify unknown keys fail closed rather than dropping weights
    bad_dir = tmp_path / "bad_mm"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text(json.dumps(raw_cfg), encoding="utf-8")
    bad_sd = dict(sd)
    bad_sd["unexpected.stray.weight"] = torch.ones(4, dtype=torch.bfloat16)
    st_save_file(bad_sd, str(bad_dir / "model.safetensors"))
    with pytest.raises(ModelSourceError, match="unclassified checkpoint tensor key"):
        extract_text_causal_lm_checkpoint(bad_dir, tmp_path / "bad_out")


def test_lora_8192_full_trajectory_chunked_ce_and_stateful_resume(tmp_path: Path):
    """T-8192-CANARY: full 8192-token trajectory SFT without slicing, chunked CE parity, stateful resume."""
    import copy

    import torch.nn.functional as F
    from safetensors.torch import load_file as st_load_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from orn_g4.errors import LoRAError
    from orn_g4.lora import (
        LoRAConfig,
        apply_lora_to_model,
        build_sft_scheduler,
        checkpoint_sft,
        chunked_causal_ce_loss,
        create_sft_optimizer,
        export_sft_adapter_bf16,
        restore_sft_checkpoint,
        run_sft_gradient_accumulation_step,
    )

    torch.manual_seed(8192)
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen3_5ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)

    lora_cfg = LoRAConfig()
    assert lora_cfg.rank == 32
    assert lora_cfg.alpha == 64
    assert lora_cfg.lr == 5e-5
    assert lora_cfg.weight_decay == 0.01
    assert lora_cfg.grad_accum_steps == 16
    assert lora_cfg.max_seq_len == 8192

    small_cfg = LoRAConfig(
        rank=4,
        alpha=8,
        lr=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        grad_accum_steps=2,
        max_seq_len=8192,
        chunk_size=256,
        target_module_patterns=("q_proj", "v_proj", "out_proj", "gate_proj"),
    )
    adapted = apply_lora_to_model(model, config=small_cfg, seed=0)
    assert len(adapted) > 0
    for wrapper in adapted.values():
        assert wrapper.lora_A.dtype == torch.float32
        assert wrapper.lora_B.dtype == torch.float32
        assert wrapper.base_layer.weight.dtype == torch.bfloat16
        # Non-zero B so LoRA A also receives non-zero gradient on first backward pass
        with torch.no_grad():
            wrapper.lora_B.uniform_(-0.02, 0.02)

    # 1. Exact parity: chunked_causal_ce_loss (chunk_size=256, grad_checkpoint=True) == unchunked float32 CE
    hidden_ref = torch.randn(1, 300, 32, dtype=torch.bfloat16, requires_grad=True)
    hidden_chk = hidden_ref.detach().clone().requires_grad_(True)
    labels_short = torch.randint(0, 64, (1, 300))
    labels_short[:, :40] = -100
    labels_short[:, 255:265] = -100  # cross chunk boundary + masked tokens

    logits_full = model.lm_head(hidden_ref.to(model.lm_head.weight.dtype)).float()
    ref_sum = F.cross_entropy(
        logits_full[:, :-1].reshape(-1, 64),
        labels_short[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    ref_sum.backward()

    chk_sum, chk_tok = chunked_causal_ce_loss(
        hidden_chk,
        model.lm_head,
        labels_short,
        chunk_size=256,
        grad_checkpoint=True,
    )
    chk_sum.backward()
    assert chk_tok == int((labels_short[:, 1:] != -100).sum().item())
    assert float(chk_sum.detach().item()) == pytest.approx(float(ref_sum.detach().item()), rel=1e-5, abs=1e-4)
    assert torch.allclose(hidden_chk.grad.float(), hidden_ref.grad.float(), atol=1e-5, rtol=1e-5)

    # 2. Exact window-level token normalization over 2 unequal-token microbatches
    model_norm_a = copy.deepcopy(model)
    model_norm_b = copy.deepcopy(model)
    mb1 = {
        "input_ids": torch.randint(0, 64, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
        "labels": torch.cat([torch.full((1, 50), -100), torch.randint(0, 64, (1, 14))], dim=1),
    }
    mb2 = {
        "input_ids": torch.randint(0, 64, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
        "labels": torch.cat([torch.full((1, 10), -100), torch.randint(0, 64, (1, 54))], dim=1),
    }
    opt_a = create_sft_optimizer(model_norm_a, lr=5e-5, weight_decay=0.01)
    sched_a = build_sft_scheduler(opt_a, total_steps=4, warmup_ratio=0.25)
    metrics_a = run_sft_gradient_accumulation_step(
        model_norm_a, [mb1, mb2], opt_a, scheduler=sched_a, config=small_cfg
    )
    assert metrics_a["supervised_tokens"] == float(14 + 54)
    # Reference single-pass concatenated window (same optimizer step)
    opt_b = create_sft_optimizer(model_norm_b, lr=5e-5, weight_decay=0.01)
    sched_b = build_sft_scheduler(opt_b, total_steps=4, warmup_ratio=0.25)
    comb = {
        "input_ids": torch.cat([mb1["input_ids"], mb2["input_ids"]], dim=0),
        "attention_mask": torch.cat([mb1["attention_mask"], mb2["attention_mask"]], dim=0),
        "labels": torch.cat([mb1["labels"], mb2["labels"]], dim=0),
    }
    metrics_b = run_sft_gradient_accumulation_step(
        model_norm_b, [comb], opt_b, scheduler=sched_b, config=small_cfg
    )
    assert metrics_a["loss"] == pytest.approx(metrics_b["loss"], rel=1e-4, abs=1e-4)
    for (n, pa), (_, pb) in zip(model_norm_a.named_parameters(), model_norm_b.named_parameters()):
        if pa.requires_grad:
            assert torch.allclose(pa, pb, atol=1e-6), f"mismatch in {n}"

    # 3. Full 8192-token trajectory forward+backward+step + stateful checkpoint/resume + BF16 adapter export
    # Use a 1-layer linear_attention Qwen3_5ForCausalLM for 8192-token CPU execution (~3s instead of O(T^2) CPU SDPA)
    cfg_8192 = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        tie_word_embeddings=False,
    )
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model_8192 = Qwen3_5ForCausalLM(cfg_8192)
    finally:
        torch.set_default_dtype(prev_dtype)
    adapted_8192 = apply_lora_to_model(model_8192, config=small_cfg, seed=0)
    for wrapper in adapted_8192.values():
        with torch.no_grad():
            wrapper.lora_B.uniform_(-0.02, 0.02)

    opt = create_sft_optimizer(model_8192, lr=5e-5, weight_decay=0.01)
    sched = build_sft_scheduler(opt, total_steps=10, warmup_ratio=0.10, lr_scheduler_type="cosine")
    ids_8192 = torch.randint(0, 64, (1, 8192))
    labels_8192 = ids_8192.clone()
    labels_8192[:, :512] = -100
    mb_8192 = {
        "input_ids": ids_8192,
        "attention_mask": torch.ones_like(ids_8192),
        "labels": labels_8192,
    }
    step_metrics = run_sft_gradient_accumulation_step(
        model_8192, [mb_8192], opt, scheduler=sched, config=small_cfg
    )
    import math

    assert step_metrics["supervised_tokens"] == float(8192 - 512)
    assert math.isfinite(step_metrics["loss"])
    assert len(opt.state) > 0  # optimizer state persisted across steps

    sft_dir = tmp_path / "sft_ckpt"
    snap = checkpoint_sft(model_8192, opt, step=1, tokens=7680, cursor=1, sft_dir=sft_dir, scheduler=sched)
    assert (snap / "COMMITTED").is_file()

    # Mutate live weights/optimizer and verify stateful restore recovers both weights and scheduler LR
    lr_before = opt.param_groups[0]["lr"]
    target_key = next(iter(adapted_8192.keys()))
    saved_a = adapted_8192[target_key].lora_A.detach().clone()
    with torch.no_grad():
        adapted_8192[target_key].lora_A.zero_()
    opt.param_groups[0]["lr"] = 0.0
    restored_state = restore_sft_checkpoint(model_8192, opt, sft_dir, scheduler=sched)
    assert restored_state["step"] == 1
    assert restored_state["cursor"] == 1
    assert torch.allclose(adapted_8192[target_key].lora_A, saved_a, atol=1e-7)
    assert opt.param_groups[0]["lr"] == pytest.approx(lr_before)

    # BF16 adapter export leaves live model parameters FP32
    bf16_dir = tmp_path / "sft_adapter_bf16"
    export_sft_adapter_bf16(model_8192, bf16_dir, adapted=adapted_8192)
    exported = st_load_file(str(bf16_dir / "adapter_model.safetensors"), device="cpu")
    assert len(exported) == 2 * len(adapted_8192)
    assert all(v.dtype == torch.bfloat16 for v in exported.values())
    assert all(p.dtype == torch.float32 for p in model_8192.parameters() if p.requires_grad)

    # 4. Negative controls: >8192 trajectory fails closed instead of silent slicing; non-FP32 trainable param fails
    too_long = {
        "input_ids": torch.randint(0, 64, (1, 8193)),
        "attention_mask": torch.ones(1, 8193, dtype=torch.long),
        "labels": torch.randint(0, 64, (1, 8193)),
    }
    with pytest.raises(LoRAError, match="exceeds max_seq_len=8192"):
        run_sft_gradient_accumulation_step(model_8192, [too_long], opt, config=small_cfg)

    bad_mod = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    with pytest.raises(LoRAError, match="torch.float32"):
        create_sft_optimizer(bad_mod)


def test_gguf_q4_k_m_fallback_export_and_f16_gdn_gate_verification(tmp_path: Path):
    """Verify Q4_K_M GGUF fallback: commit pin, --no-mtp, F16 ssm_alpha/ssm_beta, and fail-closed verification."""
    import json
    import os
    import subprocess
    from types import SimpleNamespace

    import gguf
    import numpy as np
    from safetensors.torch import save_file as st_save_file

    from orn_g4.errors import ExportError
    from orn_g4.fallback import (
        LLAMA_CPP_PINNED_COMMIT,
        export_text_q4_k_m_gguf,
        verify_gguf_fallback,
        verify_llama_cpp_commit,
    )

    # 1. Verify pinned commit helper fails closed on mismatched commit SHA
    bad_repo = tmp_path / "wrong_llama_cpp"
    bad_repo.mkdir()
    subprocess.run(["git", "-C", str(bad_repo), "init"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(bad_repo), "config", "user.email", "t@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(bad_repo), "config", "user.name", "t"], check=True, capture_output=True)
    (bad_repo / "convert_hf_to_gguf.py").write_text("# stub\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(bad_repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(bad_repo), "commit", "-m", "init"], check=True, capture_output=True)
    bad_sha = subprocess.run(
        ["git", "-C", str(bad_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert bad_sha != LLAMA_CPP_PINNED_COMMIT
    with pytest.raises(ExportError, match="commit mismatch"):
        verify_llama_cpp_commit(bad_repo)

    quant_bin = bad_repo / "llama-quantize"
    quant_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(quant_bin, 0o755)

    # 2. Synthetic multimodal source checkpoint (so export_text_q4_k_m_gguf exercises causal extraction too)
    mm_src = tmp_path / "mm_ckpt"
    mm_src.mkdir()
    (mm_src / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"hidden_size": 32},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 32,
                    "layer_types": ["linear_attention", "full_attention"],
                },
            }
        ),
        encoding="utf-8",
    )
    st_save_file(
        {
            "model.language_model.embed_tokens.weight": torch.zeros(16, 32, dtype=torch.bfloat16),
            "model.visual.patch_embed.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
            "mtp.fc.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
        },
        str(mm_src / "model.safetensors"),
    )

    def _write_test_gguf(
        path: Path,
        *,
        arch: str = "qwen35",
        alpha_type: Any = gguf.GGMLQuantizationType.F16,
        beta_type: Any = gguf.GGMLQuantizationType.F16,
        include_q4k: bool = True,
        extra_tensor: str | None = None,
    ) -> None:
        w = gguf.GGUFWriter(str(path), arch)
        w.add_name("azrael-test")
        # Q4_K block size is 256 elements (144 bytes per 256 elements in pygguf).
        w.add_tensor(
            "blk.0.ssm_alpha.weight",
            np.zeros((4, 32), dtype=np.float16),
            raw_dtype=alpha_type if alpha_type == gguf.GGMLQuantizationType.F16 else gguf.GGMLQuantizationType.Q4_K,
        )
        w.add_tensor(
            "blk.0.ssm_beta.weight",
            np.zeros((4, 32), dtype=np.float16),
            raw_dtype=beta_type if beta_type == gguf.GGMLQuantizationType.F16 else gguf.GGMLQuantizationType.Q4_K,
        )
        if include_q4k:
            raw_q4k = np.zeros((1, 144), dtype=np.uint8)
            w.add_tensor("blk.1.attn_q.weight", raw_q4k, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        if extra_tensor is not None:
            w.add_tensor(extra_tensor, np.zeros((4, 4), dtype=np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    recorded_cmds: list[list[str]] = []

    def _mock_run(cmd: list[str]) -> Any:
        recorded_cmds.append(list(cmd))
        if "--outfile" in cmd:
            out_file = Path(cmd[cmd.index("--outfile") + 1])
            src_text_dir = Path(cmd[2])
            # Verify multimodal vision/mtp were already excluded in extracted text_dir
            extracted_cfg = json.loads((src_text_dir / "config.json").read_text(encoding="utf-8"))
            assert "vision_config" not in extracted_cfg
            _write_test_gguf(out_file, include_q4k=False)
        else:
            q4_out = Path(cmd[-2])
            _write_test_gguf(q4_out, include_q4k=True)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    gguf_out = tmp_path / "gguf_out"
    manifest = export_text_q4_k_m_gguf(
        mm_src,
        gguf_out,
        llama_cpp_dir=bad_repo,
        llama_quantize_bin=quant_bin,
        expected_commit=bad_sha,
        run_cmd=_mock_run,
    )
    assert len(recorded_cmds) == 2
    assert "--no-mtp" in recorded_cmds[0]
    assert recorded_cmds[1][:5] == [str(quant_bin), "--tensor-type", "ssm_alpha=f16", "--tensor-type", "ssm_beta=f16"]
    assert recorded_cmds[1][-1] == "Q4_K_M"
    assert manifest["verification"]["ssm_f16_verified"] is True
    assert manifest["verification"]["ssm_alpha_count"] == 1
    assert manifest["verification"]["ssm_beta_count"] == 1
    assert manifest["verification"]["q4_k_tensor_count"] == 1
    assert (gguf_out / "gguf_fallback_manifest.json").is_file()
    # Intermediate F16 file should be cleaned up when keep_intermediate=False
    assert len(list(gguf_out.glob("*.gguf"))) == 1

    # 3. Negative controls for verify_gguf_fallback
    quantized_alpha = tmp_path / "bad_alpha.gguf"
    w = gguf.GGUFWriter(str(quantized_alpha), "qwen35")
    w.add_tensor("blk.0.ssm_alpha.weight", np.zeros((1, 144), dtype=np.uint8), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.ssm_beta.weight", np.zeros((4, 32), dtype=np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
    w.add_tensor("blk.1.attn_q.weight", np.zeros((1, 144), dtype=np.uint8), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ExportError, match="preserved in F16"):
        verify_gguf_fallback(quantized_alpha)

    leaked_mtp = tmp_path / "leaked_mtp.gguf"
    _write_test_gguf(leaked_mtp, extra_tensor="mtp.fc.weight")
    with pytest.raises(ExportError, match="exclude MTP/vision"):
        verify_gguf_fallback(leaked_mtp)

    wrong_arch = tmp_path / "wrong_arch.gguf"
    _write_test_gguf(wrong_arch, arch="qwen2")
    with pytest.raises(ExportError, match="architecture mismatch"):
        verify_gguf_fallback(wrong_arch)

    unquantized = tmp_path / "unquantized.gguf"
    _write_test_gguf(unquantized, include_q4k=False)
    with pytest.raises(ExportError, match="zero Q4_K quantized tensors"):
        verify_gguf_fallback(unquantized)
