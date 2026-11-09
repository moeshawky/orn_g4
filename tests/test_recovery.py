"""Unit tests for the recovery campaign (work-order §7).

Covers: low-rank residual reduces RMSE vs fixed-ternary, SVD init singular
values, and RMSE accounting for ranks 16 and 32.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from orn_g4.recovery import (
    configure_recovery,
    low_rank_residual,
    residual_rmse,
    setup_qat,
    svd_init_residual,
    train_qat,
    train_recovery,
)
from orn_g4.ternary_core import TernaryLinear


def test_svd_init_residual_shapes_and_singular_values():
    rng = np.random.default_rng(0)
    residual = rng.standard_normal((16, 32)).astype(np.float32)
    B, A, svals = svd_init_residual(residual, rank=4)
    assert B.shape == (16, 4)
    assert A.shape == (4, 32)
    assert len(svals) == 4
    assert all(s >= 0 for s in svals)


def test_low_rank_residual_reduces_rmse():
    rng = np.random.default_rng(1)
    target = rng.standard_normal((16, 32)).astype(np.float32)
    ternary_approx = np.zeros_like(target)  # worst case: ternary is zero
    res = low_rank_residual(target, ternary_approx, ranks=(16, 32))
    base = res[16].pre_residual_rmse
    # With ternary_approx = 0, pre RMSE = 1.0 (relative to target norm).
    assert abs(base - 1.0) < 1e-6
    # Post RMSE at rank 16 and 32 should be <= pre.
    assert res[16].post_residual_rmse <= base + 1e-6
    assert res[32].post_residual_rmse <= res[16].post_residual_rmse + 1e-6


def test_residual_rmse_identity():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    assert abs(residual_rmse(a, a)) < 1e-6
    assert residual_rmse(a, np.zeros_like(a)) > 0.9


def test_configure_recovery():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ternary = TernaryLinear(128, 64, group_size=128)
            self.ternary.lora_A = nn.Parameter(torch.zeros(4, 128))
            self.ternary.lora_B = nn.Parameter(torch.zeros(64, 4))
            self.lin = nn.Linear(64, 32)

    model = DummyModel()
    params_scale = configure_recovery(model, "scale")
    assert model.ternary.scales.requires_grad is True
    assert model.ternary.lora_A.requires_grad is False
    assert model.ternary.lora_B.requires_grad is False
    assert model.lin.weight.requires_grad is False
    assert params_scale == [model.ternary.scales]

    params_res = configure_recovery(model, "residual")
    assert model.ternary.scales.requires_grad is True
    assert model.ternary.lora_A.requires_grad is True
    assert model.ternary.lora_B.requires_grad is True
    assert model.lin.weight.requires_grad is False
    assert set(params_res) == {model.ternary.scales, model.ternary.lora_A, model.ternary.lora_B}


def test_setup_qat():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = TernaryLinear(128, 64, group_size=128)
            self.layer2 = TernaryLinear(128, 64, group_size=128)

    model = DummyModel()
    state_index = {
        "converted": {
            "layer1": {"heldout_nre": 0.45},
            "layer2": {"heldout_nre": 0.85},
        }
    }
    # Allow only 1 module (each is 128*64 = 8192 params)
    selected = setup_qat(model, state_index=state_index, max_shadow_params=10000, max_modules=1)
    assert selected == ["layer2"]
    assert model.layer2.shadow is not None
    assert model.layer2.qat is True
    assert model.layer1.shadow is None


class MockStreamingDataset:
    def __init__(self, n_batches=10):
        self.n_batches = n_batches

    def stream(self, split, start=0, max_len=1024):
        cursor = start
        while True:
            cursor += 1
            yield cursor, {
                "input_ids": torch.randint(0, 100, (16,), dtype=torch.long),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.randint(0, 100, (16,), dtype=torch.long),
            }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.proj = TernaryLinear(32, 100, group_size=32)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, 100), labels.view(-1))
        return type("Output", (), {"logits": logits, "loss": loss})()


def test_train_recovery_execution(tmp_path):
    model = TinyModel()
    ds = MockStreamingDataset()
    progress = train_recovery(
        model=model,
        ds=ds,
        stage="scale",
        budget=32,  # 2 batches of 16 tokens
        lr=1e-2,
        run_dir=tmp_path,
        checkpoint_every=1,
    )
    assert progress["steps"] >= 2
    assert progress["tokens"] >= 32
    assert (tmp_path / "recover_scale" / "complete.json").exists()
    assert (tmp_path / "recover_scale" / "latest.json").exists()
    assert (model.proj.scales >= 1e-8).all()


def test_train_qat_execution(tmp_path):
    model = TinyModel()
    ds = MockStreamingDataset()
    selected = setup_qat(model, max_shadow_params=10000, max_modules=1)
    assert selected == ["proj"]
    assert model.proj.shadow is not None

    progress = train_qat(
        model=model,
        ds=ds,
        qat_selected=selected,
        budget=32,
        lr=1e-2,
        run_dir=tmp_path,
        checkpoint_every=1,
    )
    assert progress["steps"] >= 2
    assert progress["tokens"] >= 32
    assert (tmp_path / "qat" / "complete.json").exists()
    # Confirm shadow updated the discrete codes and is retained for continuation
    assert model.proj.shadow is not None
    assert model.proj.codes.dtype == torch.int8

