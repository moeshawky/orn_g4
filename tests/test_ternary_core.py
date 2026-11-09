"""Unit tests for the ternary core representation (work-order §6).

Covers: group scales, dead-zone quantization, hard-ternary forward,
pack/unpack round-trip, sparsity and effective-bits accounting.
"""

import numpy as np
import pytest
from orn_g4.ternary_core import (
    GroupedTernary,
    group_scales,
    hard_ternary_forward,
    pack_ternary,
    quantize_group_row,
    unpack_ternary,
)


def test_group_scales_shape_and_positivity():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 260)).astype(np.float32)  # 260 = 2*128 + 4
    scales = group_scales(w, group_size=128)
    assert scales.shape == (8, 3)
    assert np.all(scales > 0), "group scales must be strictly positive"


def test_group_scales_zero_group_nudged():
    w = np.zeros((4, 128), dtype=np.float32)
    scales = group_scales(w, group_size=128)
    assert np.all(scales > 0)
    assert np.allclose(scales, 1e-6)


def test_quantize_group_row_codes_in_ternary_set():
    rng = np.random.default_rng(1)
    w = rng.standard_normal(260).astype(np.float32)
    scales = group_scales(w.reshape(1, -1), group_size=128)[0]
    codes = quantize_group_row(w, scales, group_size=128)
    assert codes.shape == (260,)
    assert set(np.unique(codes).tolist()) <= {-1, 0, 1}


def test_hard_ternary_forward_shape():
    grouped = GroupedTernary(
        codes=np.array([[1, 0, -1, 0], [0, 1, 1, 0]], dtype=np.int8),
        scales=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        group_size=2,
        in_features=4,
        out_features=2,
        n_groups=2,
    )
    x = np.ones(4, dtype=np.float32)
    out = hard_ternary_forward(x, grouped)
    assert out.shape == (2,)


def test_grouped_ternary_effective_weight_reconstruction():
    # W = scale * codes exactly by construction.
    grouped = GroupedTernary(
        codes=np.array([[1, -1, 1, -1], [1, 1, -1, -1]], dtype=np.int8),
        scales=np.array([[2.0, 0.5], [1.0, 3.0]], dtype=np.float32),
        group_size=2,
        in_features=4,
        out_features=2,
        n_groups=2,
    )
    w_eff = grouped.effective_weight()
    # row 0: scale group0=2 -> [2,-2], scale group1=0.5 -> [0.5,-0.5]
    # row 1: scale group0=1 -> [1,1], scale group1=3 -> [-3,-3]
    expected = np.array([[2.0, -2.0, 0.5, -0.5], [1.0, 1.0, -3.0, -3.0]], dtype=np.float32)
    assert np.allclose(w_eff, expected)


def test_pack_unpack_round_trip():
    rng = np.random.default_rng(2)
    codes = rng.integers(-1, 2, size=(4, 130)).astype(np.int8)
    scales = rng.standard_normal((4, 2)).astype(np.float32) + 1.0
    grouped = GroupedTernary(
        codes=codes,
        scales=scales,
        group_size=128,
        in_features=130,
        out_features=4,
        n_groups=2,
        source_name="probe",
    )
    blob = pack_ternary(grouped)
    unpacked = unpack_ternary(blob, source_name="probe")
    assert np.array_equal(unpacked.codes, codes)
    assert np.allclose(unpacked.scales, scales)
    assert unpacked.group_size == 128
    assert unpacked.in_features == 130
    assert unpacked.out_features == 4
    assert unpacked.n_groups == 2
    assert unpacked.source_name == "probe"


def test_pack_unpack_rejects_bad_magic():
    grouped = GroupedTernary(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones((2, 1), dtype=np.float32),
        group_size=128,
        in_features=2,
        out_features=2,
        n_groups=1,
    )
    blob = pack_ternary(grouped)
    bad = b"XXXX" + blob[4:]
    with pytest.raises(ValueError, match="bad ternary magic"):
        unpack_ternary(bad)


def test_pack_unpack_rejects_wrong_length():
    grouped = GroupedTernary(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones((2, 1), dtype=np.float32),
        group_size=128,
        in_features=2,
        out_features=2,
        n_groups=1,
    )
    blob = pack_ternary(grouped)
    with pytest.raises(ValueError, match="blob length"):
        unpack_ternary(blob[:-1])


def test_effective_bits_and_sparsity():
    codes = np.array([[1, 0, -1, 0]], dtype=np.int8)
    scales = np.ones((1, 2), dtype=np.float32)
    grouped = GroupedTernary(
        codes=codes,
        scales=scales,
        group_size=2,
        in_features=4,
        out_features=1,
        n_groups=2,
    )
    assert grouped.nonzero_fraction() == 0.5
    bits = grouped.effective_bits_per_weight()
    assert 0 < bits < 32
