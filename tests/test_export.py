"""Unit tests for export (work-order §10).

Covers: canonical ternary checkpoint write/reload, hard-ternary forward
reproduction, pack_ptq1_0 round-trip verification, and R-26 enforcement
(refusing to serialize pure-FP weights under a ternary name).

Artifact B now uses sharded safetensors (model-00001-of-N.safetensors +
model.safetensors.index.json + checkpoint.json + tensor_manifest.json)
instead of per-tensor .ternary blobs.
"""

import numpy as np
import pytest
from orn_g4.export import (
    ExportManifest,
    hard_ternary_forward_tensor,
    load_effective_ternary,
    pack_ptq1_0,
    write_canonical_ternary_checkpoint,
)
from orn_g4.ternary_core import GroupedTernary
from orn_g4.ternary_gptq import GptqConfig, sequential_gptq


def _make_grouped(name: str = "probe"):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 130)).astype(np.float64)
    x = rng.standard_normal((16, 130)).astype(np.float64)
    g = sequential_gptq(w, x, config=GptqConfig(group_size=128))
    g.source_name = name
    return g


def test_write_and_reload_canonical_checkpoint(tmp_path):
    grouped = _make_grouped("layer0.q_proj")
    write_canonical_ternary_checkpoint(
        tmp_path / "ckpt",
        {"layer0.q_proj": grouped},
        rotation={},
        precision_exceptions=["norm.weight"],
        model_config={"hidden_size": 4096},
        template_fingerprint={"source": "artifact", "hash_sha256": "sha256:abc"},
        provenance={"kaggle_handle": "mohamedshawky4340/ornith-1.5-9b/transformers/bf16/1"},
    )
    # Sharded safetensors format.
    assert (tmp_path / "ckpt" / "model.safetensors.index.json").is_file()
    assert (tmp_path / "ckpt" / "checkpoint.json").is_file()
    assert (tmp_path / "ckpt" / "tensor_manifest.json").is_file()
    assert any(
        p.name.startswith("model-") and p.name.endswith(".safetensors")
        for p in (tmp_path / "ckpt").iterdir()
    )

    cp, tensors = load_effective_ternary(tmp_path / "ckpt")
    assert "layer0.q_proj" in tensors
    t = tensors["layer0.q_proj"]
    assert set(np.unique(t["codes"]).tolist()) <= {-1, 0, 1}
    assert np.all(t["scales"] > 0)
    assert cp["precision_exceptions"] == ["norm.weight"]


def test_hard_ternary_forward_reproduces_after_reload(tmp_path):
    grouped = _make_grouped("t")
    write_canonical_ternary_checkpoint(
        tmp_path / "ckpt",
        {"t": grouped},
        rotation={},
        precision_exceptions=[],
        model_config={},
        template_fingerprint={},
        provenance={},
    )
    cp, tensors = load_effective_ternary(tmp_path / "ckpt")
    x = np.random.default_rng(1).standard_normal((8, 130)).astype(np.float32)
    rep = hard_ternary_forward_tensor(x, tensors["t"])
    ref = x @ grouped.effective_weight().T
    assert rep.shape == ref.shape
    # After reload, reproduction is exact (codes/scales round-trip losslessly).
    assert np.allclose(rep, ref, atol=1e-5)


def test_pack_ptq1_0_round_trip_verification(tmp_path):
    grouped = _make_grouped("t")
    manifest = pack_ptq1_0({"t": grouped}, tmp_path / "ptq", verify_on_real_matrices=True)
    assert manifest["packed"] is True
    assert manifest["verification"]["t"]["ok"] is True
    assert manifest["verification"]["t"]["codes_match"] is True
    assert manifest["verification"]["t"]["scales_match"] is True


def test_r26_refuses_pure_fp_ternary_export(tmp_path):
    grouped = _make_grouped("t")
    # Build a checkpoint dict where every tensor is floating point (no int8 codes).
    bad = {
        "t": GroupedTernary(
            codes=np.zeros((2, 2), dtype=np.float32),  # NOT int8 -> R-26 trigger
            scales=np.ones((2, 1), dtype=np.float32),
            group_size=128,
            in_features=2,
            out_features=2,
            n_groups=1,
        )
    }
    with pytest.raises(Exception):
        write_canonical_ternary_checkpoint(
            tmp_path / "bad",
            bad,
            rotation={},
            precision_exceptions=[],
            model_config={},
            template_fingerprint={},
            provenance={},
        )


def test_export_manifest_checksums():
    m = ExportManifest()
    import os
    import tempfile

    fd, path = tempfile.mkstemp()
    os.write(fd, b"hello world")
    os.close(fd)
    m.add("probe", path)
    m.write(path + ".manifest")
    assert "probe" in m.artifacts
    assert m.artifacts["probe"].startswith("sha256:")
