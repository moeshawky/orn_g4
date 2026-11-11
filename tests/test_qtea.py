"""Unit tests for orn_g4.qtea, recovery extensions, gates, and packaging."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from orn_g4.checkpoint_spec import (
    QUANTIZER_QTEA,
    SCHEMA_VERSION_QTEA,
    QTEACheckpointSpec,
    TensorEntry,
)
from orn_g4.errors import ExportError, QualityGateError, QTEAError
from orn_g4.export import export_qtea_package
from orn_g4.gates import gate_ce_quality, gate_logits_parity, gate_top1_parity
from orn_g4.qtea import (
    QTEA,
    QTEAConfig,
    QTEALinear,
    TernaryGroup,
    calibrate_and_quantize_projection,
    find_ornith_projections,
    pack_layer,
    pack_ternary,
    unpack_layer,
    unpack_ternary,
)
from orn_g4.recovery import reconstruct_projection


def test_qtea_config_defaults():
    cfg = QTEAConfig()
    assert cfg.group_size == 128
    assert cfg.salient_ratio == 0.05
    assert cfg.residual_group == 4
    assert cfg.delta_coef == 0.96
    assert cfg.itf_iters == 1
    assert cfg.rescale_iters == 2
    assert cfg.decay == 1.0
    assert cfg.damp == 0.007


def test_pack_unpack_ternary_roundtrip():
    # Codes in {-1, 0, 1}
    torch.manual_seed(42)
    rows, cols = 16, 256
    raw_codes = torch.randint(-1, 2, (rows, cols), dtype=torch.int8)

    packed = pack_ternary(raw_codes, group_size=128)
    assert packed.shape == (rows, 2 * 26)  # 2 groups of 128 cols, 26 bytes each
    assert packed.dtype == torch.uint8

    unpacked = unpack_ternary(packed, cols, group_size=128)
    assert unpacked.shape == (rows, cols)
    assert unpacked.dtype == torch.int8
    assert torch.equal(raw_codes, unpacked)


def test_ternary_group_fit():
    torch.manual_seed(42)
    rows, group_size = 32, 128
    W = torch.randn(rows, group_size)

    group = TernaryGroup(delta_coef=0.96, itf_iters=2).fit(W)
    assert group.alpha is not None and group.alpha.shape == (rows, 1)
    assert group.beta is not None and group.beta.shape == (rows, 1)
    assert group.delta is not None and group.delta.shape == (rows, 1)
    assert (group.alpha > 0).all()


def test_qtea_quantize_and_unpack():
    torch.manual_seed(42)
    in_features, out_features = 256, 128
    linear = nn.Linear(in_features, out_features, bias=True)
    orig_w = linear.weight.data.clone()

    cfg = QTEAConfig(group_size=128, salient_ratio=0.1, itf_iters=1, rescale_iters=2)
    quantizer = QTEA(linear, cfg)

    # Add calibration batches
    for _ in range(4):
        x = torch.randn(2, 8, in_features)
        quantizer.add_batch(x)

    payload = quantizer.quantize()
    assert payload["rows"] == out_features
    assert payload["cols"] == in_features
    assert payload["salient_cols"].numel() > 0

    # Test pack_layer and unpack_layer
    packed = pack_layer(payload)
    assert packed["ternary"].dtype == torch.uint8
    reconstructed = unpack_layer(packed)
    assert reconstructed.shape == (out_features, in_features)

    # Ensure reconstructed weight is non-trivial and correlates with original
    cos_sim = torch.cosine_similarity(orig_w.flatten(), reconstructed.flatten(), dim=0)
    assert cos_sim.item() > 0.80


def test_qtea_linear_module_parity():
    torch.manual_seed(42)
    in_features, out_features = 128, 64
    linear = nn.Linear(in_features, out_features, bias=True)

    calibration_batches = [torch.randn(2, 4, in_features) for _ in range(2)]
    packed, qtea_mod = calibrate_and_quantize_projection(
        "test_proj", linear, calibration_batches, QTEAConfig(group_size=128, salient_ratio=0.0)
    )

    assert isinstance(qtea_mod, QTEALinear)
    x = torch.randn(1, 4, in_features)
    y = qtea_mod(x)
    assert y.shape == (1, 4, out_features)

    w_eff = qtea_mod.effective_weight()
    assert w_eff.shape == (out_features, in_features)


def test_find_ornith_projections():
    class MockLinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)

    class MockSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 128)
            self.down_proj = nn.Linear(128, 64)

    class MockBlock(nn.Module):
        def __init__(self, idx):
            super().__init__()
            if idx == 3:
                self.self_attn = MockSelfAttn()
            else:
                self.linear_attn = MockLinearAttn()
            self.mlp = MockMLP()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([MockBlock(i) for i in range(4)])
            self.visual = nn.Linear(32, 32)  # Should NOT be picked

    model = MockModel()
    pilot_projs = find_ornith_projections(model, mode="PILOT")
    assert set(pilot_projs.keys()) == {
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.mlp.down_proj",
        "layers.3.self_attn.q_proj",
    }
    assert "visual" not in str(pilot_projs.keys())

    full_projs = find_ornith_projections(model, mode="FULL")
    assert len(full_projs) > len(pilot_projs)
    assert all("layers." in k for k in full_projs)


def test_reconstruct_projection_local_optimization():
    torch.manual_seed(42)
    in_features, out_features = 128, 64
    linear = nn.Linear(in_features, out_features)

    calib = [torch.randn(1, 4, in_features)]
    _, unit = calibrate_and_quantize_projection("test_proj", linear, calib)

    x = torch.randn(2, 4, in_features)
    teacher_y = linear(x).detach()

    unit_opt, metrics = reconstruct_projection(unit, x, teacher_y, steps=2, lr=1e-4)
    assert metrics["recovery_steps"] == 2
    assert math.isfinite(metrics["mse_before_recovery"])
    assert math.isfinite(metrics["mse_after_recovery"])


def test_gates_enforcement():
    # CE quality gate
    delta = gate_ce_quality(teacher_ce=1.20, student_ce=1.45, max_regression=0.50)
    assert abs(delta - 0.25) < 1e-5

    with pytest.raises(QualityGateError, match="Quality gate breach"):
        gate_ce_quality(teacher_ce=1.00, student_ce=1.65, max_regression=0.50)

    # Logits parity gate
    t1 = torch.tensor([[1.0, 2.0, 3.0]])
    t2 = torch.tensor([[1.001, 2.002, 3.001]])
    max_d = gate_logits_parity(t1, t2, tolerance=1e-2)
    assert max_d <= 1e-2

    with pytest.raises(QualityGateError, match="Logits parity breach"):
        gate_logits_parity(t1, torch.tensor([[1.5, 2.0, 3.0]]), tolerance=1e-2)

    assert gate_top1_parity(t1, t2) is True


def test_qtea_checkpoint_spec():
    spec = QTEACheckpointSpec(
        schema_version=SCHEMA_VERSION_QTEA,
        qtea_config={"group_size": 128, "salient_ratio": 0.05},
        quality_metrics={"delta_ce": 0.12},
    )
    d = spec.to_dict()
    assert d["schema_version"] == "qtea/v1"
    assert d["qtea_config"]["group_size"] == 128

    spec_back = QTEACheckpointSpec.from_dict(d)
    assert spec_back.schema_version == "qtea/v1"
    assert spec_back.quality_metrics["delta_ce"] == 0.12


def test_export_qtea_package(tmp_path):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 64)
            self.norm = nn.LayerNorm(128)

    model = SimpleModel()
    _, q_mod = calibrate_and_quantize_projection(
        "proj", model.proj, [torch.randn(1, 4, 128)]
    )
    model.proj = q_mod

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "config.json").write_text('{"model_type": "ornith"}')

    run_dir = tmp_path / "run"
    index = {"completed": {"proj": {"file": "proj.safetensors"}}}
    conv_res = {"delta_ce": 0.20, "teacher_ce": 1.0, "student_ce": 1.2}

    pkg_dir, manifest = export_qtea_package(
        model,
        run_dir,
        index,
        conv_res,
        src_dir,
        ce_max_regression=0.50,
        mode="PILOT",
        run_tag="test_qtea_run",
    )

    assert (pkg_dir / "manifest.json").is_file()
    assert (pkg_dir / "provenance_lock.json").is_file()
    assert (pkg_dir / "ternary_runtime.py").is_file()
    assert (pkg_dir / "load_qtea_package.py").is_file()
    assert (pkg_dir / "assets" / "config.json").is_file()
    assert (run_dir / "test_qtea_run_package.tar.gz").is_file()
    assert manifest["outcome"] == "QUALITY_ACCEPTED"

    # Verify quality gate rejects excessive regression. The artifact is still
    # preserved (preservation is separate from acceptance), but the export
    # refuses to approve the package.
    bad_res = {"delta_ce": 0.65, "teacher_ce": 1.0, "student_ce": 1.65}
    with pytest.raises(QualityGateError, match="Export refused"):
        export_qtea_package(
            model,
            run_dir,
            index,
            bad_res,
            src_dir,
            ce_max_regression=0.50,
        )

    # A quality-rejected artifact is preserved for the recovery phase.
    rejected_dir = run_dir / "candidate"
    rejected_manifest = json.loads((rejected_dir / "manifest.json").read_text())
    assert rejected_manifest["outcome"] == "QUALITY_REJECTED_RECOVERY_REQUIRED"
    assert rejected_manifest["quality"]["delta_ce"] == pytest.approx(0.65)
    assert rejected_manifest["quality"]["ce_max_regression"] == pytest.approx(0.50)

    # Missing quality measurements are rejected outright: a missing
    # measurement must never read as "no degradation".
    with pytest.raises(ExportError, match="quality measurements missing"):
        export_qtea_package(
            model,
            run_dir,
            index,
            {"delta_ce": 0.65},
            src_dir,
            ce_max_regression=0.50,
        )
