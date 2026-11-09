"""Unit tests for sequential ternary GPTQ (work-order §6 primary candidate).

Covers: GPTQ produces ternary codes, runs on real-shaped matrices, and the
reconstruction-error function is sane. Deviation from canonical GPTQ is
documented in the module, not asserted here.
"""

import numpy as np
import pytest
from orn_g4.ternary_gptq import GptqConfig, reconstruction_error, sequential_gptq


def test_sequential_gptq_returns_ternary_codes():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 260)).astype(np.float64)
    x = rng.standard_normal((128, 260)).astype(np.float64)
    grouped = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    assert set(np.unique(grouped.codes).tolist()) <= {-1, 0, 1}
    assert grouped.group_size == 128
    assert grouped.in_features == 260
    assert grouped.out_features == 8
    assert grouped.n_groups == 3  # ceil(260/128)
    assert np.all(grouped.scales > 0)


def test_sequential_gptq_deterministic_given_same_input():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 128)).astype(np.float64)
    x = rng.standard_normal((64, 128)).astype(np.float64)
    g1 = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    g2 = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    assert np.array_equal(g1.codes, g2.codes)
    assert np.allclose(g1.scales, g2.scales)


def test_reconstruction_error_sane():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 64)).astype(np.float64)
    x = rng.standard_normal((32, 64)).astype(np.float64)
    grouped = sequential_gptq(w, x, config=GptqConfig(group_size=64))
    err = reconstruction_error(w, grouped, x)
    assert isinstance(err, float)
    assert err >= 0.0
    # Ternary reconstruction of a random matrix is imperfect: error > 0.
    assert err > 0.0


def test_gptq_handles_exact_group_size():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((2, 256)).astype(np.float64)  # exactly 2 groups of 128
    x = rng.standard_normal((32, 256)).astype(np.float64)
    grouped = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    assert grouped.n_groups == 2


def test_gptq_rejects_bad_shapes():
    w = np.zeros((4, 64))
    x = np.zeros((32, 32))  # wrong in_features
    with pytest.raises(ValueError, match="in_features"):
        sequential_gptq(w, x)
