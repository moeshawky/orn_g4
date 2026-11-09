"""Unit tests for pre_conversion_surgery_probe (work-order §6 surgical probes).

Covers: P1 profiling, P2 Hadamard rotation (accepted & recurrent-state rejected),
P4 SVD-vs-QERA comparison, and promotion logic.
"""

import numpy as np
from orn_g4.pre_conversion_surgery_probe import (
    probe_p1_profile,
    probe_p2_hadamard,
    probe_p4_svd_vs_qera,
    run_all_probes,
)
from orn_g4.ternary_gptq import GptqConfig, sequential_gptq


def _make_pair():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 128)).astype(np.float64)
    x = rng.standard_normal((64, 128)).astype(np.float64)
    return w, x


def test_probe_p1_profile_returns_baseline():
    w, x = _make_pair()
    r = probe_p1_profile(w, x, x, "mlp.down_proj")
    assert r.probe == "P1"
    assert r.baseline_recon_error > 0
    assert r.improvement == 0.0
    assert r.promoted is False
    assert r.compatible is True
    assert r.stored_bits > 0


def test_probe_p2_hadamard_accepted_when_equivalent():
    w, x = _make_pair()
    r = probe_p2_hadamard(w, x, x, "attn.q_proj", is_recurrent_state=False)
    assert r.probe == "P2"
    assert r.compatible is True  # Hadamard inverse parity passes
    assert r.baseline_recon_error > 0
    assert r.stored_bits > 0


def test_probe_p2_hadamard_rejected_recurrent_state():
    w, x = _make_pair()
    r = probe_p2_hadamard(w, x, x, "delta.state", is_recurrent_state=True)
    assert r.compatible is False
    assert r.promoted is False
    assert "recurrent-state" in r.reason


def test_probe_p4_svd_vs_qera():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 128)).astype(np.float64)
    x = rng.standard_normal((64, 128)).astype(np.float64)
    grouped = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    ternary_approx = grouped.effective_weight()
    r = probe_p4_svd_vs_qera(w, ternary_approx, x, "mlp.down_proj")
    assert r.probe == "P4"
    assert r.baseline_recon_error > 0
    assert r.compatible is True


def test_run_all_probes_returns_list():
    w, x = _make_pair()
    grouped = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    ternary_approx = grouped.effective_weight()
    results = run_all_probes(w, x, x, "mlp.down_proj", ternary_approx=ternary_approx)
    assert isinstance(results, list)
    assert len(results) == 3  # P1, P2, P4
    probes = {r.probe for r in results}
    assert probes == {"P1", "P2", "P4"}


def test_promotion_requires_improvement():
    # With identical inputs P2 improvement may be small; promotion gated.
    w, x = _make_pair()
    results = run_all_probes(w, x, x, "mlp.gate_proj")
    for r in results:
        if r.promoted:
            assert r.improvement > 0, f"{r.probe} promoted without improvement"
