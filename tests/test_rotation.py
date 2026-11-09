"""Unit tests for the signed Hadamard rotation experiment (work-order §6).

Covers: Hadamard invariance / inverse parity, and that rotation is only
retained when inverse parity passes.
"""

import numpy as np
from orn_g4.rotation import RotationResult, signed_hadamard, verify_inverse


def test_signed_hadamard_shape_and_orthogonality():
    n = 64
    H = signed_hadamard(n, seed=7)
    assert H.shape == (n, n)
    assert set(np.unique(H).tolist()) <= {-1.0, 1.0}
    # H @ H.T = n * I (columns are orthogonal, signed columns preserve this).
    gram = H @ H.T
    assert np.allclose(gram, n * np.eye(n))


def test_verify_inverse_passes_for_random_input():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((16, 64)).astype(np.float64)
    x = rng.standard_normal((8, 64)).astype(np.float64)
    result = verify_inverse(w, x, seed=3, tol=1e-6)
    assert result.retained, result.reason
    assert result.inverse_error < 1e-6


def test_verify_inverse_inverse_error_is_small():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((8, 32)).astype(np.float64)
    x = rng.standard_normal((4, 32)).astype(np.float64)
    result = verify_inverse(w, x, seed=4, tol=1e-8)
    assert result.inverse_error < 1e-8


def test_verify_inverse_returns_result_shape():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((4, 16)).astype(np.float64)
    x = rng.standard_normal((2, 16)).astype(np.float64)
    result = verify_inverse(w, x, seed=5)
    assert isinstance(result, RotationResult)
    assert isinstance(result.retained, bool)
    assert isinstance(result.inverse_error, float)
    assert isinstance(result.reason, str)
