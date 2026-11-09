"""Unit tests for orn_g4/loader.py canonical persistence and buffer restoration."""

import torch
import torch.nn as nn
from pathlib import Path
from orn_g4.loader import (
    inspect_ternary,
    load_canonical,
    restore_qwen3_5_nonpersistent_buffers,
    save_canonical,
    save_trainable_only,
    load_trainable_only,
)
from orn_g4.ternary_core import TernaryLinear


class DummyRotary(nn.Module):
    def __init__(self, dim=64, base=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = base
        self._non_persistent_buffers_set = {"inv_freq"}
        self.register_buffer("inv_freq", torch.zeros(dim // 2), persistent=False)


class DummyModelWithRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary = DummyRotary(dim=64, base=10000.0)
        self.proj = TernaryLinear(128, 64, group_size=128)

    def forward(self, x):
        return self.proj(x)


def test_restore_qwen3_5_nonpersistent_buffers():
    model = DummyModelWithRotary()
    assert (model.rotary.inv_freq == 0).all()

    restored = restore_qwen3_5_nonpersistent_buffers(model)
    assert len(restored) == 1
    assert "rotary.inv_freq" in restored
    assert not (model.rotary.inv_freq == 0).all()
    assert model.rotary.inv_freq[0] == 1.0


def test_inspect_ternary():
    model = DummyModelWithRotary()
    info = inspect_ternary(model)
    assert "proj" in info
    assert info["proj"]["in_features"] == 128
    assert info["proj"]["out_features"] == 64


def test_save_and_verify_canonical_checkpoint(tmp_path):
    from orn_g4.loader import verify_canonical

    model = DummyModelWithRotary()
    dest = tmp_path / "canonical"

    manifest = save_canonical(
        model=model,
        path=dest,
        source_id="test-model",
        extra={"stage": "test"},
    )
    assert manifest["schema"] == "ternary-canonical-v1"
    assert (dest / "manifest.json").is_file()
    assert (dest / "model.safetensors.index.json").is_file()

    verified = verify_canonical(dest)
    assert verified["schema"] == "ternary-canonical-v1"
    assert "proj.codes" in verified["tensor_index"]
    assert "proj.scales" in verified["tensor_index"]

