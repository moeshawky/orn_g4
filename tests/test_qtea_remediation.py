"""Regression tests for the QTEA pre-execution remediation.

Covers the concrete defects that could invalidate a FULL Ornith-1.5-9B
quantization run:

  * P0-2  device-correct residual reconstruction / decode path.
  * P0-3  non-mutating quantization (the BF16 module stays a teacher).
  * P0-4  local recovery that actually optimizes and reports honestly.
  * P0-5  artifact preservation separated from quality acceptance.
  * P1-6  standalone runtime numerically equivalent to orn_g4.qtea.
  * P1-7  conversion completeness / resumability accounting.

GPU-only assertions are skipped when no CUDA runtime is present; a skipped
test is UNVERIFIED, never PASS.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from orn_g4.errors import ExportError, QualityGateError
from orn_g4.export import (
    QTEA_OUTCOME_CONVERTED,
    QTEA_OUTCOME_QUALITY_ACCEPTED,
    QTEA_OUTCOME_QUALITY_REJECTED,
    audit_qtea_conversion,
    export_qtea_package,
    verify_qtea_package,
)
from orn_g4.qtea import (
    QTEA,
    QTEAConfig,
    QTEALinear,
    calibrate_and_quantize_projection,
    pack_layer,
    pack_ternary,
    residual_columns,
    unpack_layer,
    unpack_ternary,
)
from orn_g4.recovery import reconstruct_projection

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA runtime unavailable on this host — this assertion is UNVERIFIED",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_package_all_names_resolve():
    """Every name in orn_g4.__all__ must exist (RotationError used to dangle)."""
    import orn_g4

    missing = [name for name in orn_g4.__all__ if not hasattr(orn_g4, name)]
    assert missing == []


def _probe_device() -> torch.device:
    """A non-CPU device usable for device-placement assertions on any host.

    ``meta`` carries real device identity without allocating memory, so it
    discriminates a CPU-allocating decode path from a device-following one
    even on a CPU-only host.
    """
    return torch.device("cuda" if CUDA_AVAILABLE else "meta")


def _make_projection(
    in_features: int = 256,
    out_features: int = 128,
    salient_ratio: float = 0.1,
    seed: int = 0,
):
    """Return (source_linear, packed_payload, qtea_module, calibration_batches)."""
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=True)
    batches = [torch.randn(2, 8, in_features) for _ in range(3)]
    packed, module = calibrate_and_quantize_projection(
        "proj", linear, batches, QTEAConfig(group_size=128, salient_ratio=salient_ratio)
    )
    return linear, packed, module, batches


def _read_safetensors(path: Path):
    try:
        from safetensors.torch import load_file

        return load_file(str(path))
    except ImportError:  # pragma: no cover
        return torch.load(str(path), map_location="cpu", weights_only=True)


def _tiny_model(seed: int = 0):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 64)
            self.norm = nn.LayerNorm(128)
            self.register_buffer("scale_hint", torch.ones(4))

    torch.manual_seed(seed)
    model = Tiny()
    _, qtea_mod = calibrate_and_quantize_projection(
        "proj", model.proj, [torch.randn(1, 4, 128)]
    )
    model.proj = qtea_mod
    return model


# --------------------------------------------------------------------------
# P0-2  device-correct QTEA residual reconstruction
# --------------------------------------------------------------------------


def test_residual_columns_follows_the_residual_device():
    """residual_columns must allocate on the payload's device, not on CPU."""
    _, packed, _, _ = _make_projection()
    device = _probe_device()
    moved = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in packed.items()
    }
    dense = residual_columns(moved)
    assert dense.device == moved["residual_value"].device
    assert dense.shape == (packed["salient_cols"].numel(), packed["rows"])


def test_unpack_layer_keeps_the_layer_device():
    """unpack_layer must reconstruct on the layer's own device."""
    _, packed, _, _ = _make_projection()
    device = _probe_device()
    moved = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in packed.items()
    }
    weight = unpack_layer(moved)
    assert weight.device == device
    assert tuple(weight.shape) == (packed["rows"], packed["cols"])


def test_unpack_layer_value_parity_on_cpu():
    """CPU decode parity: the packed layer rebuilds the quantized weight exactly."""
    _, packed, module, _ = _make_projection()
    rebuilt = unpack_layer(packed)
    assert torch.equal(rebuilt, module.effective_weight())


def _independent_reference_weight(packed):
    """Rebuild the effective weight WITHOUT using orn_g4.qtea's decode path.

    Base-3 digits and the 2-bit sparse indices are decoded by hand so the
    comparison below is against an independent implementation, not against the
    same helper the production code uses.
    """
    rows = int(packed["rows"])
    cols = int(packed["cols"])
    group = int(packed["group_size"])
    residual_group = int(packed["residual_group"])

    codes = torch.zeros(rows, cols, dtype=torch.int8)
    raw = packed["ternary"].int()
    powers = [1, 3, 9, 27, 81]
    bytes_per_group = (group + 4) // 5
    n_groups = (cols + group - 1) // group
    for group_index in range(n_groups):
        first_column = group_index * group
        group_columns = min(group, cols - first_column)
        for local_byte in range(bytes_per_group):
            for row in range(rows):
                value = int(raw[row, group_index * bytes_per_group + local_byte])
                for k in range(5):
                    position = local_byte * 5 + k
                    if position >= group_columns:
                        break
                    codes[row, first_column + position] = (
                        (value // powers[k]) % 3 - 1
                    )

    alpha = packed["group_alpha"].repeat_interleave(group, dim=1)
    beta = packed["group_beta"].repeat_interleave(group, dim=1)
    weight = alpha * packed["col_v"][None, :] * codes.float() + beta

    salient = packed["salient_cols"].tolist()
    if salient and packed["residual_value"].numel() > 0:
        index_packed = packed["residual_index"].int()
        values = packed["residual_value"].float()
        scales = packed["residual_scale"].float()
        for salient_index, column in enumerate(salient):
            n_values = values.shape[1]
            for value_index in range(n_values):
                byte_index = value_index // 4
                shift = 2 * (value_index % 4)
                slot = int((index_packed[salient_index, byte_index] >> shift) & 0x3)
                residual = float(values[salient_index, value_index] / scales[salient_index])
                # Each residual entry covers one group of 4 output rows.
                weight[value_index * residual_group + slot, column] += residual
    return weight


def test_cpu_end_to_end_pack_reconstruct_and_forward():
    """Quantize -> pack -> load -> reconstruct -> forward, against a reference.

    The reference weight is rebuilt by an independent decoder, so a bug in the
    production decode path cannot hide behind a shared helper.
    """
    linear, packed, module, batches = _make_projection(salient_ratio=0.1)
    assert packed["salient_cols"].numel() > 0, "the fixture must carry salient residuals"

    reference = _independent_reference_weight(packed)

    reloaded = QTEALinear.from_payload(packed, bias=linear.bias.data)
    assert torch.equal(reloaded.effective_weight(), reference)

    x = batches[0]
    expected = torch.nn.functional.linear(x, reference, linear.bias.data)
    assert torch.allclose(reloaded(x), expected, atol=1e-6)


def test_qtea_from_payload_places_every_buffer_on_one_device():
    """from_payload must not mix devices between buffers."""
    _, packed, _, _ = _make_projection()
    device = _probe_device()
    module = QTEALinear.from_payload(packed, device=device)
    devices = {buf.device for _, buf in module.named_buffers()}
    assert devices == {device}
    assert module.effective_weight().device == device


def test_qtea_from_payload_infers_device_from_payload():
    """A payload that already lives on a device places the module there."""
    _, packed, _, _ = _make_projection()
    device = _probe_device()
    moved = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in packed.items()
    }
    module = QTEALinear.from_payload(moved)
    assert {buf.device for _, buf in module.named_buffers()} == {device}


@requires_cuda
def test_qtea_cuda_pack_unpack_reconstruct_and_forward_parity():
    """CUDA: pack -> load on device -> reconstruct -> forward == CPU reference."""
    linear, packed, module, _ = _make_projection()
    device = torch.device("cuda")

    cuda_module = QTEALinear.from_payload(packed, device=device)
    assert {buf.device for _, buf in cuda_module.named_buffers()} == {device}

    cpu_weight = module.effective_weight()
    cuda_weight = cuda_module.effective_weight()
    assert cuda_weight.device == device
    assert torch.equal(cuda_weight.cpu(), cpu_weight)

    x = torch.randn(2, 8, packed["cols"], device=device)
    cuda_out = cuda_module(x)
    assert cuda_out.device == device
    assert torch.allclose(cuda_out.cpu(), module(x.cpu()), atol=1e-5)


@requires_cuda
def test_qtea_cuda_residual_reconstruction_has_no_host_round_trip():
    """CUDA: the decode path must not allocate CPU temporaries."""
    _, packed, _, _ = _make_projection()
    device = torch.device("cuda")
    moved = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in packed.items()
    }
    dense = residual_columns(moved)
    assert dense.device == device
    assert dense.numel() > 0


# --------------------------------------------------------------------------
# P0-3  non-mutating source-weight conversion
# --------------------------------------------------------------------------


def test_quantize_does_not_mutate_the_source_module():
    linear, _, _, _ = _make_projection()
    weight_before = linear.weight.data.clone()
    bias_before = linear.bias.data.clone()

    quantizer = QTEA(linear, QTEAConfig(group_size=128, salient_ratio=0.1))
    for _ in range(3):
        quantizer.add_batch(torch.randn(2, 8, 256))
    quantizer.quantize()

    assert torch.equal(linear.weight.data, weight_before)
    assert torch.equal(linear.bias.data, bias_before)


def test_teacher_activations_are_identical_before_and_after_conversion():
    linear, _, module, batches = _make_projection()
    x = batches[0]
    with torch.no_grad():
        teacher_before = linear(x).clone()
    assert teacher_before.shape == (2, 8, 128)

    # The quantized module must differ from the teacher (otherwise the test
    # would not discriminate a poisoned teacher).
    with torch.no_grad():
        student = module(x)
    assert not torch.allclose(student, teacher_before, atol=1e-6)

    # A second conversion of the same source must not perturb the teacher.
    _, second = calibrate_and_quantize_projection(
        "proj2", linear, batches, QTEAConfig(group_size=128, salient_ratio=0.1)
    )
    with torch.no_grad():
        teacher_after = linear(x).clone()
    assert torch.equal(teacher_before, teacher_after)


def test_source_module_is_unchanged_after_a_failed_conversion():
    """A quantization exception must leave the source module untouched."""
    linear, _, _, batches = _make_projection()
    weight_before = linear.weight.data.clone()
    bias_before = linear.bias.data.clone()

    quantizer = QTEA(linear, QTEAConfig(group_size=128, salient_ratio=0.1))
    for batch in batches:
        quantizer.add_batch(batch)

    def explode(*_args, **_kwargs):
        raise RuntimeError("injected quantization failure")

    original = torch.Tensor.scatter_ if False else None  # noqa: F841
    monkey = quantizer
    monkey._select_salient = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected quantization failure"):
        monkey.quantize()

    assert torch.equal(linear.weight.data, weight_before)
    assert torch.equal(linear.bias.data, bias_before)


def test_source_and_quantized_module_do_not_share_weight_storage():
    linear, _, module, _ = _make_projection()
    assert module.effective_weight().data_ptr() != linear.weight.data_ptr()
    # Mutating the source afterwards must not move the quantized module.
    snapshot = module.effective_weight().clone()
    with torch.no_grad():
        linear.weight.add_(1.0)
    assert torch.equal(module.effective_weight(), snapshot)


def test_mutate_source_is_an_explicit_opt_in():
    linear = nn.Linear(256, 128, bias=False)
    weight_before = linear.weight.data.clone()
    quantizer = QTEA(linear, QTEAConfig(group_size=128, salient_ratio=0.1))
    for _ in range(2):
        quantizer.add_batch(torch.randn(2, 8, 256))
    quantizer.quantize(mutate_source=True)
    assert not torch.equal(linear.weight.data, weight_before)


# --------------------------------------------------------------------------
# P0-4  local recovery
# --------------------------------------------------------------------------


def test_recovery_updates_parameters_and_reports_success():
    linear, _, module, _ = _make_projection()
    x = torch.randn(2, 8, 256)
    with torch.no_grad():
        teacher_y = linear(x).clone()

    col_v_before = module.col_v.detach().clone()
    unit, metrics = reconstruct_projection(module, x, teacher_y, steps=5, lr=1e-2)

    assert metrics["recovery_steps"] == 5
    assert metrics["status"] == "recovered"
    assert metrics["parameters_updated"] >= 1
    assert metrics["max_parameter_change"] > 0.0
    assert metrics["max_gradient_norm"] > 0.0
    assert metrics["grad_finite"] is True
    assert metrics["teacher_identical"] is False
    assert metrics["mse_after_recovery"] < metrics["mse_before_recovery"]
    assert not torch.equal(unit.col_v.detach(), col_v_before)


def test_recovery_reports_no_change_instead_of_claiming_success():
    """A teacher identical to the student leaves no gradient — report it."""
    linear, _, module, batches = _make_projection()
    x = batches[0]
    with torch.no_grad():
        teacher_y = module(x).clone()  # teacher == student: no signal

    unit, metrics = reconstruct_projection(module, x, teacher_y, steps=3, lr=1e-2)
    assert metrics["teacher_identical"] is True
    assert metrics["parameters_updated"] == 0
    assert metrics["status"] == "degenerate_teacher"
    assert metrics["max_parameter_change"] == 0.0


def test_recovery_reports_zero_steps():
    linear, _, module, _ = _make_projection()
    x = torch.randn(2, 8, 256)
    with torch.no_grad():
        teacher_y = linear(x).clone()
    _, metrics = reconstruct_projection(module, x, teacher_y, steps=0, lr=1e-2)
    assert metrics["recovery_steps"] == 0
    assert metrics["parameters_updated"] == 0
    assert metrics["status"] == "zero_steps"
    assert metrics["mse_after_recovery"] == metrics["mse_before_recovery"]


def test_recovery_keeps_codes_and_residual_indices_frozen():
    linear, _, module, _ = _make_projection()
    x = torch.randn(2, 8, 256)
    with torch.no_grad():
        teacher_y = linear(x).clone()

    codes_before = module.ternary.detach().clone()
    index_before = module.residual_index.detach().clone()
    salient_before = module.salient_cols.detach().clone()

    unit, _ = reconstruct_projection(module, x, teacher_y, steps=3, lr=1e-2)

    assert torch.equal(unit.ternary, codes_before)
    assert torch.equal(unit.residual_index, index_before)
    assert torch.equal(unit.salient_cols, salient_before)


def test_recovery_state_survives_serialization():
    linear, _, module, _ = _make_projection()
    x = torch.randn(2, 8, 256)
    with torch.no_grad():
        teacher_y = linear(x).clone()
    unit, _ = reconstruct_projection(module, x, teacher_y, steps=3, lr=1e-2)

    state = unit.state_dict()
    assert "col_v" in state

    # Reload into a fresh module built from the same packed payload.
    fresh = QTEALinear.from_payload(
        {
            "cols": unit.in_features,
            "rows": unit.out_features,
            "group_size": unit.group_size,
            "residual_group": unit.residual_group,
            "ternary": unit.ternary,
            "col_v": unit.col_v.detach(),
            "group_alpha": unit.group_alpha,
            "group_beta": unit.group_beta,
            "salient_cols": unit.salient_cols,
            "residual_index": unit.residual_index,
            "residual_value": unit.residual_value,
            "residual_scale": unit.residual_scale,
        },
        bias=unit.bias.detach() if unit.bias is not None else None,
    )
    fresh.load_state_dict(state)
    assert torch.equal(fresh.effective_weight(), unit.effective_weight())


def test_recovery_col_v_is_a_parameter_after_the_buffer_transition():
    linear, _, module, _ = _make_projection()
    x = torch.randn(2, 8, 256)
    with torch.no_grad():
        teacher_y = linear(x).clone()
    unit, _ = reconstruct_projection(module, x, teacher_y, steps=1, lr=1e-2)

    assert isinstance(unit.col_v, torch.nn.Parameter)
    assert "col_v" not in unit._buffers
    assert "col_v" in unit._parameters
    # The exported layer dict must still expose the trained rescale.
    layer_dict = unit.to_layer_dict()
    assert torch.equal(layer_dict["col_v"].detach(), unit.col_v.detach())


# --------------------------------------------------------------------------
# P0-5  artifact preservation vs quality acceptance
# --------------------------------------------------------------------------


def _export_kwargs(tmp_path):
    model = _tiny_model()
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "config.json").write_text('{"model_type": "ornith"}')
    index = {"completed": {"proj": {"file": "proj.safetensors"}}}
    return model, src_dir, index


def test_export_quality_accepted_outcome(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    pkg, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
    )
    assert manifest["outcome"] == QTEA_OUTCOME_QUALITY_ACCEPTED
    assert manifest["quality"]["quality_status"] == "ACCEPTED"
    assert manifest["quality"]["teacher_ce"] == pytest.approx(1.0)
    assert manifest["quality"]["student_ce"] == pytest.approx(1.2)
    assert manifest["quality"]["delta_ce"] == pytest.approx(0.2)
    assert manifest["quality"]["ce_max_regression"] == pytest.approx(0.5)
    assert (pkg / "manifest.json").is_file()
    assert (tmp_path / "run" / "ornith_qtea_v2_1_package.tar.gz").is_file()


def test_export_quality_rejected_preserves_the_artifact(tmp_path):
    """A rejected conversion is valuable output: it must survive the refusal."""
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.9, "delta_ce": 0.9}
    with pytest.raises(QualityGateError, match="Export refused"):
        export_qtea_package(
            model,
            tmp_path / "run",
            index,
            conv,
            src_dir,
            ce_max_regression=0.5,
        )

    pkg = tmp_path / "run" / "candidate"
    assert pkg.is_dir()
    assert (pkg / "manifest.json").is_file()
    assert (pkg / "encoded").is_dir()
    manifest = json.loads((pkg / "manifest.json").read_text())
    assert manifest["outcome"] == QTEA_OUTCOME_QUALITY_REJECTED
    assert manifest["quality"]["delta_ce"] == pytest.approx(0.9)
    assert manifest["quality"]["ce_max_regression"] == pytest.approx(0.5)


def test_export_quality_rejected_opt_in_returns_without_approval(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.9, "delta_ce": 0.9}
    pkg, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        allow_quality_rejected=True,
    )
    assert manifest["outcome"] == QTEA_OUTCOME_QUALITY_REJECTED
    assert manifest["quality"]["quality_status"] == "REJECTED"
    assert pkg.is_dir()


def test_export_rejects_missing_measurements(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    with pytest.raises(ExportError, match="quality measurements missing"):
        export_qtea_package(
            model, tmp_path / "run", index, {"delta_ce": 0.1}, src_dir
        )


def test_export_rejects_non_finite_measurements(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": float("nan"), "delta_ce": 0.1}
    with pytest.raises(ExportError, match="not finite"):
        export_qtea_package(
            model, tmp_path / "run", index, conv, src_dir
        )


def test_export_rejects_inconsistent_measurements(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.05}
    with pytest.raises(ExportError, match="internally inconsistent"):
        export_qtea_package(
            model, tmp_path / "run", index, conv, src_dir
        )


def test_export_manifest_records_the_required_fields(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    _, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        mode="PILOT",
        provenance={"foundation_model": "moeshawky/Ornith-1.5-9B"},
        recovery_status={"stage": "local", "status": "not_run"},
    )
    assert manifest["outcome"] == QTEA_OUTCOME_QUALITY_ACCEPTED
    assert manifest["recipe"]["quantizer"] == "qtea"
    assert "expected_projection_count" in manifest["recipe"]
    assert "completed_projection_count" in manifest["recipe"]
    assert manifest["revisions"]["foundation_model"] == "moeshawky/Ornith-1.5-9B"
    assert manifest["completeness"]["expected_count"] == 1
    assert manifest["completeness"]["completed_count"] == 1
    assert manifest["recovery_status"]["status"] == "not_run"
    # Integrity hashes for every encoded projection.
    for name, entry in manifest["encoded"].items():
        assert len(entry["sha256"]) == 64
        assert len(entry["effective_weight_sha256"]) == 64


def test_export_preserves_non_quantized_buffers(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    pkg, manifest = export_qtea_package(
        model, tmp_path / "run", index, conv, src_dir, ce_max_regression=0.5
    )
    preserved = _read_safetensors(pkg / "preserved" / "preserved.safetensors")
    assert "scale_hint" in preserved  # a persistent buffer, not a parameter
    assert "norm.weight" in preserved
    assert manifest["preserved"]["preserved.safetensors"]["count"] == len(preserved)


def test_export_handles_tied_weights(tmp_path):
    """A tied weight must be stored once and the tie must be recorded."""
    model, src_dir, index = _export_kwargs(tmp_path)
    model.tied = model.norm.weight  # same storage under a second name
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    pkg, manifest = export_qtea_package(
        model, tmp_path / "run", index, conv, src_dir, ce_max_regression=0.5
    )
    preserved = _read_safetensors(pkg / "preserved" / "preserved.safetensors")
    # Exactly one copy is stored (safetensors refuses shared storage).
    assert ("norm.weight" in preserved) != ("tied" in preserved)
    ties = manifest["preserved"]["tied_tensors"]
    kept = next(name for name in ("norm.weight", "tied") if name in preserved)
    dropped = "norm.weight" if kept == "tied" else "tied"
    assert ties == {dropped: kept}


def test_export_does_not_write_into_the_source_directory(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    before = sorted(p.name for p in src_dir.iterdir())
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    export_qtea_package(
        model, tmp_path / "run", index, conv, src_dir, ce_max_regression=0.5
    )
    assert sorted(p.name for p in src_dir.iterdir()) == before


# --------------------------------------------------------------------------
# P1-6  standalone runtime
# --------------------------------------------------------------------------


def _exported_package(tmp_path, **kwargs):
    model, src_dir, index = _export_kwargs(tmp_path)
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    pkg, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        **kwargs,
    )
    return pkg, manifest


def _import_runtime(pkg: Path):
    spec = importlib.util.spec_from_file_location(
        "ternary_runtime_under_test", pkg / "ternary_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_runtime_loads_a_real_projection(tmp_path):
    """The shipped runtime must load the exported tensors without a size error."""
    pkg, manifest = _exported_package(tmp_path)
    runtime = _import_runtime(pkg)
    entry = next(iter(manifest["encoded"].values()))
    tensors = _read_safetensors(pkg / "encoded" / entry["file"])

    module = runtime.QTEALinear(
        in_features=entry["cols"],
        out_features=entry["rows"],
        group_size=entry["group_size"],
        residual_group=entry["residual_group"],
        bias=entry["has_bias"],
    )
    module.load_state_dict(tensors, strict=False)
    assert module.salient_cols.numel() == tensors["salient_cols"].numel()
    assert module.residual_index.numel() == tensors["residual_index"].numel()


def test_embedded_runtime_effective_weight_matches_the_library(tmp_path):
    pkg, manifest = _exported_package(tmp_path)
    runtime = _import_runtime(pkg)
    entry = next(iter(manifest["encoded"].values()))
    tensors = _read_safetensors(pkg / "encoded" / entry["file"])

    runtime_module = runtime.QTEALinear(
        in_features=entry["cols"],
        out_features=entry["rows"],
        group_size=entry["group_size"],
        residual_group=entry["residual_group"],
        bias=entry["has_bias"],
    )
    runtime_module.load_state_dict(tensors, strict=False)

    library = QTEALinear.from_payload(
        {
            "cols": entry["cols"],
            "rows": entry["rows"],
            "group_size": entry["group_size"],
            "residual_group": entry["residual_group"],
            "ternary": tensors["ternary"],
            "col_v": tensors["col_v"],
            "group_alpha": tensors["group_alpha"],
            "group_beta": tensors["group_beta"],
            "salient_cols": tensors["salient_cols"],
            "residual_index": tensors["residual_index"],
            "residual_value": tensors.get(
                "residual_value",
                torch.zeros(0, 0, dtype=torch.float8_e4m3fn),
            ),
            "residual_scale": tensors["residual_scale"],
        },
        bias=tensors["bias"] if "bias" in tensors else None,
    )

    assert torch.equal(runtime_module.effective_weight(), library.effective_weight())


def test_embedded_runtime_forward_matches_the_library(tmp_path):
    pkg, manifest = _exported_package(tmp_path)
    runtime = _import_runtime(pkg)
    entry = next(iter(manifest["encoded"].values()))
    tensors = _read_safetensors(pkg / "encoded" / entry["file"])

    runtime_module = runtime.QTEALinear(
        in_features=entry["cols"],
        out_features=entry["rows"],
        group_size=entry["group_size"],
        residual_group=entry["residual_group"],
        bias=entry["has_bias"],
    )
    runtime_module.load_state_dict(tensors, strict=False)

    library = QTEALinear.from_payload(
        {
            "cols": entry["cols"],
            "rows": entry["rows"],
            "group_size": entry["group_size"],
            "residual_group": entry["residual_group"],
            "ternary": tensors["ternary"],
            "col_v": tensors["col_v"],
            "group_alpha": tensors["group_alpha"],
            "group_beta": tensors["group_beta"],
            "salient_cols": tensors["salient_cols"],
            "residual_index": tensors["residual_index"],
            "residual_value": tensors.get(
                "residual_value", torch.zeros(0, 0, dtype=torch.float8_e4m3fn)
            ),
            "residual_scale": tensors["residual_scale"],
        },
        bias=tensors["bias"] if "bias" in tensors else None,
    )

    x = torch.randn(2, 4, entry["cols"])
    assert torch.allclose(runtime_module(x), library(x), atol=1e-6)


def test_embedded_runtime_base3_packing_roundtrip():
    """The shipped runtime decodes base-3 codes exactly like the library."""
    codes = torch.randint(-1, 2, (8, 256), dtype=torch.int8)
    packed = pack_ternary(codes, group_size=128)
    assert torch.equal(unpack_ternary(packed, 256, group_size=128), codes)

    source = (
        Path(__file__).resolve().parents[1] / "orn_g4" / "export.py"
    ).read_text(encoding="utf-8")
    # The runtime is generated from the module source: decode its unpack_ternary
    # by executing the shipped file in isolation.
    namespace: dict = {}
    exec(  # noqa: S102 - executing the generated runtime under test
        compile(source.split("_TERNARY_RUNTIME_SOURCE = r'''")[1].split("'''")[0],
                "ternary_runtime.py", "exec"),
        namespace,
    )
    runtime_codes = namespace["unpack_ternary"](packed, 256, 128)
    assert torch.equal(runtime_codes, codes)


def test_embedded_runtime_handles_a_layer_without_residuals(tmp_path):
    """Empty residual representation: both runtimes must agree."""
    torch.manual_seed(3)
    linear = nn.Linear(256, 64, bias=True)
    _, module = calibrate_and_quantize_projection(
        "proj", linear, [torch.randn(2, 4, 256)],
        QTEAConfig(group_size=128, salient_ratio=0.0),
    )
    assert module.salient_cols.numel() == 0

    pkg, manifest = _exported_package(tmp_path)
    runtime = _import_runtime(pkg)

    layer = module.to_layer_dict()
    runtime_module = runtime.QTEALinear(
        in_features=module.in_features,
        out_features=module.out_features,
        group_size=module.group_size,
        residual_group=module.residual_group,
        bias=module.bias is not None,
    )
    runtime_module.load_state_dict(
        {
            "ternary": layer["ternary"],
            "col_v": layer["col_v"],
            "group_alpha": layer["group_alpha"],
            "group_beta": layer["group_beta"],
            "salient_cols": layer["salient_cols"],
            "residual_index": layer["residual_index"],
            "residual_value": layer["residual_value"],
            "residual_scale": layer["residual_scale"],
            "bias": module.bias.detach(),
        },
        strict=False,
    )
    assert torch.equal(runtime_module.effective_weight(), module.effective_weight())


def test_verify_qtea_package_accepts_an_intact_package(tmp_path):
    pkg, _ = _exported_package(tmp_path)
    report = verify_qtea_package(pkg)
    assert report["ok"] is True, report["failures"]
    assert report["failures"] == []


def test_verify_qtea_package_detects_a_corrupted_projection(tmp_path):
    pkg, manifest = _exported_package(tmp_path)
    entry = next(iter(manifest["encoded"].values()))
    target = pkg / "encoded" / entry["file"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))

    report = verify_qtea_package(pkg)
    assert report["ok"] is False
    assert any("hash mismatch" in failure for failure in report["failures"])


def test_standalone_loader_does_not_need_orn_g4(tmp_path):
    """The archive must be reconstructible without an orn_g4 checkout."""
    pkg, manifest = _exported_package(tmp_path)
    loader_path = pkg / "load_qtea_package.py"
    assert loader_path.is_file()
    source = loader_path.read_text(encoding="utf-8")
    assert "import orn_g4" not in source
    assert "from orn_g4" not in source

    spec = importlib.util.spec_from_file_location("standalone_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    package = loader.load_package(str(pkg))
    assert set(package["projections"]) == set(manifest["encoded"])
    assert "scale_hint" in package["preserved_tensors"]


def test_package_declares_its_runtime_and_loader(tmp_path):
    pkg, manifest = _exported_package(tmp_path)
    assert manifest["runtime"]["module"] == "ternary_runtime.py"
    assert manifest["runtime"]["loader"] == "load_qtea_package.py"
    assert (pkg / manifest["runtime"]["module"]).is_file()
    assert (pkg / manifest["runtime"]["loader"]).is_file()


# --------------------------------------------------------------------------
# P1-7  completeness and resumability
# --------------------------------------------------------------------------


def test_audit_accounts_for_every_projection_class():
    index = {
        "expected": ["a", "b", "c", "d"],
        "completed": {"a": {}, "b": {}},
        "failed": {"c": "out of memory"},
    }
    audit = audit_qtea_conversion(index)
    assert audit["expected_count"] == 4
    assert audit["completed_count"] == 2
    assert audit["failed_count"] == 1
    assert audit["unprocessed_count"] == 1
    assert audit["unprocessed"] == ["d"]
    assert audit["expected_known"] is True
    assert audit["conversion_complete"] is False


def test_audit_detects_a_missing_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    index = {"expected": ["a"], "completed": {"a": {"file": "a.safetensors"}}}
    audit = audit_qtea_conversion(index, checkpoint_dir=checkpoint_dir)
    assert audit["missing_checkpoints"] == ["a"]
    assert audit["conversion_complete"] is False


def test_audit_detects_a_corrupted_checkpoint(tmp_path):
    import hashlib

    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    payload = checkpoint_dir / "a.safetensors"
    payload.write_bytes(b"not-the-original-bytes")
    digest = hashlib.sha256(b"original-bytes").hexdigest()
    index = {
        "expected": ["a"],
        "completed": {"a": {"file": "a.safetensors", "sha256": digest}},
    }
    audit = audit_qtea_conversion(index, checkpoint_dir=checkpoint_dir)
    assert audit["corrupted_checkpoints"], audit
    assert audit["corrupted_checkpoints"][0]["name"] == "a"
    assert audit["conversion_complete"] is False


def test_audit_detects_a_revision_mismatch():
    index = {"expected": ["a"], "completed": {"a": {}}, "revisions": {"dataset": "v1"}}
    audit = audit_qtea_conversion(index, required_revisions={"dataset": "v2"})
    assert audit["revision_mismatches"], audit
    assert audit["conversion_complete"] is False


def test_audit_detects_a_recipe_mismatch():
    """A resumed conversion must use the same quantizer recipe."""
    index = {
        "expected": ["a"],
        "completed": {"a": {}},
        "recipe": {"group_size": 128, "salient_ratio": 0.05},
    }
    audit = audit_qtea_conversion(
        index, required_recipe={"group_size": 128, "salient_ratio": 0.1}
    )
    assert audit["recipe_declared"] is True
    assert audit["recipe_mismatches"], audit
    assert audit["recipe_mismatches"][0]["key"] == "salient_ratio"
    assert audit["conversion_complete"] is False


def test_audit_does_not_invent_a_recipe_mismatch():
    """An index that declares no recipe is not a mismatch."""
    index = {"expected": ["a"], "completed": {"a": {}}}
    audit = audit_qtea_conversion(index, required_recipe={"group_size": 128})
    assert audit["recipe_declared"] is False
    assert audit["recipe_mismatches"] == []


def test_full_mode_refuses_an_incomplete_conversion(tmp_path):
    model, src_dir, _ = _export_kwargs(tmp_path)
    index = {
        "expected": ["proj", "proj2"],
        "completed": {"proj": {"file": "proj.safetensors"}},
    }
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    with pytest.raises(ExportError, match="incomplete conversion"):
        export_qtea_package(
            model,
            tmp_path / "run",
            index,
            conv,
            src_dir,
            ce_max_regression=0.5,
            mode="FULL",
        )


def test_full_mode_refuses_a_missing_expected_declaration(tmp_path):
    """A FULL package without a declared expected set cannot be verified."""
    model, src_dir, _ = _export_kwargs(tmp_path)
    index = {"completed": {"proj": {"file": "proj.safetensors"}}}
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    with pytest.raises(ExportError, match="incomplete conversion"):
        export_qtea_package(
            model,
            tmp_path / "run",
            index,
            conv,
            src_dir,
            ce_max_regression=0.5,
            mode="FULL",
        )


def test_full_mode_accepts_a_complete_conversion(tmp_path):
    model, src_dir, _ = _export_kwargs(tmp_path)
    index = {"expected": ["proj"], "completed": {"proj": {"file": "proj.safetensors"}}}
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    _, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        mode="FULL",
    )
    assert manifest["completeness"]["conversion_complete"] is True
    assert manifest["completeness"]["expected_known"] is True


def test_full_mode_rejects_a_mismatched_revision(tmp_path):
    model, src_dir, _ = _export_kwargs(tmp_path)
    index = {
        "expected": ["proj"],
        "completed": {"proj": {"file": "proj.safetensors"}},
        "revisions": {"dataset": "v1"},
    }
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    with pytest.raises(ExportError, match="incomplete conversion"):
        export_qtea_package(
            model,
            tmp_path / "run",
            index,
            conv,
            src_dir,
            ce_max_regression=0.5,
            mode="FULL",
            revisions={"dataset": "v2"},
        )


def test_partial_conversion_opt_in_records_the_incompleteness(tmp_path):
    model, src_dir, _ = _export_kwargs(tmp_path)
    index = {
        "expected": ["proj", "proj2"],
        "completed": {"proj": {"file": "proj.safetensors"}},
    }
    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    _, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        mode="FULL",
        allow_partial_conversion=True,
    )
    assert manifest["completeness"]["conversion_complete"] is False
    assert manifest["completeness"]["unprocessed_count"] == 1


def test_export_preserves_the_conversion_checkpoints(tmp_path):
    model, src_dir, index = _export_kwargs(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "proj.safetensors").write_bytes(b"checkpoint-bytes")
    index["completed"]["proj"]["file"] = "proj.safetensors"

    conv = {"teacher_ce": 1.0, "student_ce": 1.2, "delta_ce": 0.2}
    pkg, manifest = export_qtea_package(
        model,
        tmp_path / "run",
        index,
        conv,
        src_dir,
        ce_max_regression=0.5,
        checkpoint_dir=checkpoint_dir,
    )
    assert manifest["checkpoints"]["proj"]["file"].startswith("checkpoints/")
    assert (pkg / manifest["checkpoints"]["proj"]["file"]).is_file()
