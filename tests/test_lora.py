"""Unit tests for the behavioral SFT LoRA (work-order §5).

Covers: real PyTorch LoRA adapters attached to a model's computation graph,
projection coverage measurement, trainable-param counting from the model,
W_adapted = W_source + LoRA_delta parity, adapter save/reload, and a single
training step with finite loss and nonzero gradients.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from orn_g4.errors import LoRAError
from orn_g4.lora import (
    LoRAConfig,
    _LoRAWrapper,
    apply_lora_to_model,
    count_trainable_params,
    freeze_targets_on_model,
    load_adapter,
    make_training_step,
    measure_coverage,
    optimizer_step,
    save_adapter,
    virtual_merge_parity,
)


class _TinyModel(nn.Module):
    """Small model with nn.Linear submodules for testing LoRA attachment."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        # Tiny causal-LM-like forward returning an object with .loss
        x = self.embed(input_ids)
        x = self.q_proj(x)
        x = self.k_proj(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss})()


def test_apply_lora_to_model_matches_targets():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj", "k_proj"))
    adapted = apply_lora_to_model(model, config=config, seed=0)
    assert "q_proj" in adapted
    assert "k_proj" in adapted
    assert "norm" not in adapted
    assert "lm_head" not in adapted
    # Base layers are still registered in the model.
    assert isinstance(model.q_proj, _LoRAWrapper)
    assert isinstance(model.k_proj, _LoRAWrapper)


def test_apply_lora_raises_when_no_targets():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("nonexistent",))
    with pytest.raises(LoRAError):
        apply_lora_to_model(model, config=config, seed=0)


def test_lora_layer_forward_delta_and_adapted():
    base = nn.Linear(4, 4)
    wrapper = _LoRAWrapper(base, rank=2, alpha=2, seed=0)
    delta = wrapper.delta_weight()
    assert delta.shape == (4, 4)
    adapted = wrapper.adapted_weight()
    assert torch.allclose(adapted, base.weight + delta, atol=1e-6)
    # At init B=0, so adapted == base weight exactly.
    assert torch.allclose(adapted, base.weight, atol=1e-6)


def test_lora_forward_changes_output_after_training():
    base = nn.Linear(4, 4)
    wrapper = _LoRAWrapper(base, rank=2, alpha=2, seed=0)
    x = torch.randn(3, 4)
    out_before = wrapper.forward(x)
    # Give B a nonzero value.
    with torch.no_grad():
        wrapper.lora_B[:] = 0.1
    out_after = wrapper.forward(x)
    assert not torch.allclose(out_before, out_after, atol=1e-5)


def test_virtual_merge_parity_passes():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj",))
    adapted = apply_lora_to_model(model, config=config, seed=0)
    x = [torch.randn(2, 8)]
    max_diff, passed = virtual_merge_parity(adapted, x, tol=1e-6)
    assert passed, f"parity failed: max_diff={max_diff}"
    assert max_diff < 1e-6


def test_count_trainable_params_from_model():
    model = _TinyModel()
    # Initially nothing requires grad after freezing everything.
    for p in model.parameters():
        p.requires_grad = False
    trainable, total = count_trainable_params(model)
    assert trainable == 0
    assert total > 0

    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj", "k_proj"))
    adapted = apply_lora_to_model(model, config=config, seed=0)
    trainable, total = count_trainable_params(model)
    # Each adapted layer: A (2,8) + B (8,2) = 32 trainable. Two layers = 64.
    assert trainable == 2 * (2 * 8 + 8 * 2)
    assert total == sum(p.numel() for p in model.parameters())


def test_measure_coverage_counts_adapted():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj", "k_proj"))
    adapted = apply_lora_to_model(model, config=config, seed=0)
    coverage = measure_coverage(adapted, config=config)
    assert coverage.get("q_proj", 0) >= 1
    assert coverage.get("k_proj", 0) >= 1
    assert "norm" not in coverage
    assert "lm_head" not in coverage


def test_freeze_targets_on_model():
    model = _TinyModel()
    # Make everything trainable first.
    for p in model.parameters():
        p.requires_grad = True
    results = freeze_targets_on_model(model, ["lm_head", "nonexistent"])
    assert results["lm_head"] is True
    assert results["nonexistent"] is False
    for p in model.lm_head.parameters():
        assert p.requires_grad is False


def test_save_and_load_adapter_preserves_outputs():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj",))
    adapted = apply_lora_to_model(model, config=config, seed=0)

    # Change B so output changes.
    with torch.no_grad():
        adapted["q_proj"].lora_B[:] = 0.5
    x = torch.randn(2, 8)
    out_before = model.q_proj.forward(x)

    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        save_adapter(adapted, path)
        # Zero out the adapter, reload, verify output restored.
        with torch.no_grad():
            adapted["q_proj"].lora_B.zero_()
            adapted["q_proj"].lora_A.zero_()
        out_zeroed = model.q_proj.forward(x)
        assert not torch.allclose(out_before, out_zeroed, atol=1e-5)

        load_adapter(adapted, path)
        out_reloaded = model.q_proj.forward(x)
        assert torch.allclose(out_before, out_reloaded, atol=1e-5)
    finally:
        Path(path).unlink(missing_ok=True)


def test_make_training_step_finite_loss_and_gradients():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj", "k_proj"))
    apply_lora_to_model(model, config=config, seed=0)

    input_ids = torch.randint(0, 16, (2, 4))
    labels = torch.randint(0, 16, (2, 4))
    labels[0, 0] = -100  # ignored label
    attention_mask = torch.ones_like(input_ids)

    metrics = make_training_step(model, {}, input_ids, labels, attention_mask, config)
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] == metrics["loss"]  # not NaN
    assert metrics["supervised_tokens"] > 0
    # Gradients should exist on trainable LoRA params.
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "no gradients on trainable LoRA parameters"


def test_optimizer_step_updates_parameters():
    model = _TinyModel()
    config = LoRAConfig(rank=2, alpha=2, target_module_patterns=("q_proj",))
    adapted = apply_lora_to_model(model, config=config, seed=0)

    # Capture params before.
    a_before = adapted["q_proj"].lora_A.detach().clone()
    b_before = adapted["q_proj"].lora_B.detach().clone()

    # Create a gradient.
    input_ids = torch.randint(0, 16, (2, 4))
    labels = torch.randint(0, 16, (2, 4))
    metrics = make_training_step(model, adapted, input_ids, labels, None, config)

    optimizer_step(model, lr=config.lr)

    a_after = adapted["q_proj"].lora_A.detach().clone()
    b_after = adapted["q_proj"].lora_B.detach().clone()
    assert not torch.allclose(a_before, a_after, atol=1e-6) or not torch.allclose(
        b_before, b_after, atol=1e-6
    ), "optimizer step did not change LoRA parameters"
    # Gradients zeroed after step.
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or p.grad.abs().sum().item() == 0.0


__all__ = []
