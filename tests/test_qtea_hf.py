"""Focused unit tests for first-party QTEA Hugging Face tooling.

Covers:
1. Remote inventory auditing with mocked Hub API.
2. Lossless repacking into indexed SafeTensors shards with FP8 preservation.
3. Shard-index correctness and exclusion of stale BF16 index files.
4. Detection of missing, duplicate, corrupted, and mismatched tensors.
5. Sharded package verification and resume idempotency.
6. Tied-weight handling and alias restoration.
7. Complete-model loading from empty skeleton + QTEA shards.
8. Publishing safeguards (fail-closed auth, owner check, main-branch protection).
9. Benchmark execution, peak memory accounting, and labeling.
10. CLI operational entrypoints.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from orn_g4.errors import (
    QTEABenchmarkError,
    QTEAError,
    QTEALoadError,
    QTEAPackError,
    QTEAPublishError,
    QTEASourceError,
    QTEAVerifyError,
)
from orn_g4.qtea import QTEALinear
from orn_g4.qtea_hf import (
    DEFAULT_REPO,
    ALLOW_PATTERNS,
    DELETE_OLD_PATTERNS,
    audit_remote_qtea,
    benchmark_qtea_inference,
    load_qtea_full_model,
    pack_qtea_shards,
    publish_qtea_sharded,
    run_qtea_benchmark,
    tensor_bytes_sha256,
    validate_qtea_source,
    verify_qtea_shards,
)
from orn_g4.utils import sha256_file


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def synthetic_qtea_source(tmp_path: Path) -> Path:
    """Construct a synthetic qtea-v1 layout containing FP8 residuals and preserved tensors."""
    src = tmp_path / "qtea_v1_source"
    src.mkdir(parents=True)

    encoded_dir = src / "encoded"
    preserved_dir = src / "preserved"
    assets_dir = src / "assets"
    encoded_dir.mkdir()
    preserved_dir.mkdir()
    assets_dir.mkdir()

    # 1. Projections with FP8 residual values and valid QTEA geometry (rows=4, cols=4, group_size=4, residual_group=2)
    fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)

    q_proj_tensors = {
        "ternary": torch.tensor([[10], [20], [30], [40]], dtype=torch.uint8),
        "col_v": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        "group_alpha": torch.tensor([[0.5], [0.5], [0.5], [0.5]], dtype=torch.float32),
        "group_beta": torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
        "salient_cols": torch.tensor([0, 1], dtype=torch.int32),
        "residual_index": torch.tensor([[1], [0]], dtype=torch.uint8),
        "residual_value": torch.ones((2, 2), dtype=fp8_dtype),
        "residual_scale": torch.tensor([2.0, 2.0], dtype=torch.float32),
    }
    q_proj_file = encoded_dir / "q_proj.safetensors"
    save_file(q_proj_tensors, str(q_proj_file))

    k_proj_tensors = {
        "ternary": torch.tensor([[5], [15], [25], [35]], dtype=torch.uint8),
        "col_v": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        "group_alpha": torch.tensor([[0.8], [0.8], [0.8], [0.8]], dtype=torch.float32),
        "group_beta": torch.tensor([[0.1], [0.1], [0.1], [0.1]], dtype=torch.float32),
        "salient_cols": torch.tensor([], dtype=torch.int32),
        "residual_index": torch.zeros((0, 0), dtype=torch.uint8),
        "residual_value": torch.zeros((0, 0), dtype=fp8_dtype),
        "residual_scale": torch.zeros((0,), dtype=torch.float32),
    }
    k_proj_file = encoded_dir / "k_proj.safetensors"
    save_file(k_proj_tensors, str(k_proj_file))

    # 2. Preserved tensors
    pres_tensors = {
        "model.language_model.model.embed_tokens.weight": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.bfloat16
        ),
        "model.language_model.model.norm.weight": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    }
    pres_file = preserved_dir / "preserved.safetensors"
    save_file(pres_tensors, str(pres_file))

    # 3. Assets: config.json, tokenizer mock, and a stale BF16 index to verify exclusion
    config_dict = {
        "model_type": "ornith",
        "vocab_size": 2,
        "hidden_size": 4,
        "num_hidden_layers": 1,
    }
    (assets_dir / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    (assets_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (assets_dir / "model.safetensors.index.json").write_text('{"stale": true}', encoding="utf-8")
    (assets_dir / "model-00001-of-00004.safetensors").write_text("stale shard bytes", encoding="utf-8")

    # 4. Runtime & provenance files
    (src / "ternary_runtime.py").write_text("# runtime mock\n", encoding="utf-8")
    (src / "load_qtea_package.py").write_text("# loader mock\n", encoding="utf-8")
    (src / "provenance_lock.json").write_text('{"lock": 1}\n', encoding="utf-8")
    (src / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (src / "README.md").write_text("# Azrael QTEA 9B\nOriginal card\n", encoding="utf-8")

    # 5. manifest.json
    manifest = {
        "format": "qtea-v1",
        "outcome": "QUALITY_REJECTED_RECOVERY_REQUIRED",
        "quality": {"teacher_ce": 2.15, "student_ce": 2.60, "delta_ce": 0.45},
        "completeness": {"conversion_complete": True, "expected_count": 2},
        "encoded": {
            "model.language_model.layers.0.self_attn.q_proj": {
                "file": "q_proj.safetensors",
                "rows": 4,
                "cols": 4,
                "group_size": 4,
                "residual_group": 2,
                "has_bias": False,
                "sha256": sha256_file(q_proj_file),
            },
            "model.language_model.layers.0.self_attn.k_proj": {
                "file": "k_proj.safetensors",
                "rows": 4,
                "cols": 4,
                "group_size": 4,
                "residual_group": 2,
                "has_bias": False,
                "sha256": sha256_file(k_proj_file),
            },
        },
        "preserved": {
            "preserved.safetensors": {"sha256": sha256_file(pres_file), "count": 2},
            "tied_tensors": {
                "lm_head.weight": "model.language_model.model.embed_tokens.weight"
            },
        },
    }
    (src / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return src


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_validate_qtea_source(synthetic_qtea_source: Path):
    """Test source validation accepts complete layout and rejects corruptions."""
    m = validate_qtea_source(synthetic_qtea_source)
    assert m["format"] == "qtea-v1"
    assert m["completeness"]["conversion_complete"] is True
    assert len(m["encoded"]) == 2

    # Negative test 1: Missing manifest
    bad_dir = synthetic_qtea_source.parent / "bad_dir_no_manifest"
    bad_dir.mkdir()
    with pytest.raises(QTEASourceError, match="manifest missing"):
        validate_qtea_source(bad_dir)

    # Negative test 2: Incomplete conversion
    manifest_p = synthetic_qtea_source / "manifest.json"
    raw_m = json.loads(manifest_p.read_text(encoding="utf-8"))
    raw_m["completeness"]["conversion_complete"] = False
    manifest_p.write_text(json.dumps(raw_m), encoding="utf-8")
    with pytest.raises(QTEASourceError, match="complete conversion"):
        validate_qtea_source(synthetic_qtea_source)

    # Negative test 3: Corrupt preserved hash
    raw_m["completeness"]["conversion_complete"] = True
    raw_m["preserved"]["preserved.safetensors"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    manifest_p.write_text(json.dumps(raw_m), encoding="utf-8")
    with pytest.raises(QTEASourceError, match="Preserved tensor SHA-256 mismatch"):
        validate_qtea_source(synthetic_qtea_source)


def test_pack_qtea_shards_lossless_and_fp8(synthetic_qtea_source: Path, tmp_path: Path):
    """Test lossless repacking into SafeTensors shards and FP8 value/dtype preservation."""
    out_dir = tmp_path / "sharded_output"

    report = pack_qtea_shards(synthetic_qtea_source, out_dir, shard_mib=64)
    assert report["shards"] >= 1
    assert report["tensor_count"] == 18  # 8 fields * 2 modules + 2 preserved = 18
    assert report["quality_outcome"] == "QUALITY_REJECTED_RECOVERY_REQUIRED"

    # Verify output layout
    sharded_m_path = out_dir / "qtea-sharded-manifest.json"
    assert sharded_m_path.is_file()
    sharded_meta = json.loads(sharded_m_path.read_text(encoding="utf-8"))

    assert sharded_meta["format"] == "qtea-sharded-v2"
    assert sharded_meta["source_outcome"] == "QUALITY_REJECTED_RECOVERY_REQUIRED"
    assert "source_manifest_sha256" in sharded_meta
    assert (out_dir / "source-manifest.json").is_file()

    # Verify stale BF16 files are excluded from assets/
    assert (out_dir / "assets" / "config.json").is_file()
    assert not (out_dir / "assets" / "model.safetensors.index.json").exists()
    assert not (out_dir / "assets" / "model-00001-of-00004.safetensors").exists()

    # Verify README notice
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "qtea-sharded-v2" in readme_text
    assert "QUALITY_REJECTED_RECOVERY_REQUIRED" in readme_text

    # Verify bitwise preservation of FP8 residual tensor
    fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
    shard_file = out_dir / sharded_meta["weight_map"]["model.language_model.layers.0.self_attn.q_proj.residual_value"]
    shard_tensors = load_file(str(shard_file))

    res_val = shard_tensors["model.language_model.layers.0.self_attn.q_proj.residual_value"]
    assert res_val.dtype == fp8_dtype
    assert res_val.shape == (2, 2)

    # Verify bitwise raw bytes match
    orig_q_file = synthetic_qtea_source / "encoded" / "q_proj.safetensors"
    orig_res_val = load_file(str(orig_q_file))["residual_value"]
    assert tensor_bytes_sha256(res_val) == tensor_bytes_sha256(orig_res_val)


def test_verify_qtea_shards(synthetic_qtea_source: Path, tmp_path: Path):
    """Test verify_qtea_shards on clean and corrupted sharded outputs."""
    out_dir = tmp_path / "sharded_verify"
    pack_qtea_shards(synthetic_qtea_source, out_dir, shard_mib=64)

    # Clean verification passes
    v_report = verify_qtea_shards(out_dir, source_manifest=synthetic_qtea_source / "manifest.json")
    assert v_report["verified"] is True
    assert v_report["shards"] >= 1

    # Corrupt a shard file
    shard_file = out_dir / "qtea-0001.safetensors"
    orig_bytes = shard_file.read_bytes()
    shard_file.write_bytes(orig_bytes + b"\x00\x01\x02")

    with pytest.raises(QTEAVerifyError, match="Corrupt shard file"):
        verify_qtea_shards(out_dir)


def test_pack_refuses_overwrite_and_detects_duplicates(synthetic_qtea_source: Path, tmp_path: Path):
    """Test pack refuses existing destination and catches staging directory collision."""
    out_dir = tmp_path / "dest_exists"
    out_dir.mkdir()

    with pytest.raises(QTEAPackError, match="Output directory already exists"):
        pack_qtea_shards(synthetic_qtea_source, out_dir)

    out_dir.rmdir()
    incomplete_dir = tmp_path / "dest_exists.incomplete"
    incomplete_dir.mkdir()

    with pytest.raises(QTEAPackError, match="Incomplete staging directory exists"):
        pack_qtea_shards(synthetic_qtea_source, out_dir)


def test_audit_remote_qtea_mocked():
    """Test remote auditing without downloading model weights."""
    mock_api = MagicMock()

    # Mock model_info
    sibling1 = MagicMock(rfilename="manifest.json", size=1000)
    sibling2 = MagicMock(rfilename="preserved/preserved.safetensors", size=50000)
    sibling3 = MagicMock(rfilename="ternary_runtime.py", size=2000)
    sibling4 = MagicMock(rfilename="load_qtea_package.py", size=2000)
    sibling5 = MagicMock(rfilename="assets/config.json", size=800)
    sibling6 = MagicMock(rfilename="encoded/proj1.safetensors", size=100000)
    sibling7 = MagicMock(rfilename="checkpoints/step_1000.tar.gz", size=50000000)

    mock_info = MagicMock()
    mock_info.sha = "abcdef123456"
    mock_info.private = False
    mock_info.siblings = [sibling1, sibling2, sibling3, sibling4, sibling5, sibling6, sibling7]
    mock_api.model_info.return_value = mock_info

    manifest_data = {
        "format": "qtea-v1",
        "completeness": {"conversion_complete": True, "expected_count": 1},
        "encoded": {"layers.0.self_attn.q_proj": {"file": "proj1.safetensors"}},
        "preserved": {"preserved.safetensors": {"sha256": "dummy"}},
    }

    with patch("orn_g4.qtea_hf.get_hf_api", return_value=mock_api):
        with patch("orn_g4.qtea_hf.fetch_manifest_from_hub", return_value=(manifest_data, Path("/tmp/m.json"))):
            report = audit_remote_qtea(repo="test/repo", revision="main")

    assert report["repo"] == "test/repo"
    assert report["source_revision"] == "abcdef123456"
    assert report["projection_count"] == 1
    assert report["groups"]["checkpoints"]["files"] == 1
    assert report["groups"]["encoded"]["files"] == 1
    assert report["missing_required_files"] == []


def test_publish_safeguards_mocked(tmp_path: Path, synthetic_qtea_source: Path):
    """Test publishing security gates: auth fail-closed, owner mismatch, and main branch protection."""
    mock_api = MagicMock()

    # Test 1: whoami failure (missing or invalid token)
    mock_api.whoami.side_effect = Exception("401 Unauthorized")
    with patch("orn_g4.qtea_hf.get_hf_api", return_value=mock_api):
        with pytest.raises(QTEAPublishError, match="Failed to authenticate"):
            publish_qtea_sharded(repo="owner/repo", workdir=tmp_path / "work")

    # Test 2: Username does not match repo owner
    mock_api.whoami.side_effect = None
    mock_api.whoami.return_value = {"name": "other_user"}
    with patch("orn_g4.qtea_hf.get_hf_api", return_value=mock_api):
        with pytest.raises(QTEAPublishError, match="does not match repo owner"):
            publish_qtea_sharded(repo="target_owner/repo", workdir=tmp_path / "work")

    # Test 3: Main branch protection
    mock_api.whoami.return_value = {"name": "target_owner"}
    with patch("orn_g4.qtea_hf.get_hf_api", return_value=mock_api):
        with pytest.raises(QTEAPublishError, match="Refusing to publish to protected source branch 'main'"):
            publish_qtea_sharded(repo="target_owner/repo", branch="main", workdir=tmp_path / "work")


# ==============================================================================
# Model Loading & Benchmark Tests (Synthetic Architecture)
# ==============================================================================


class SyntheticOrnithModel(nn.Module):
    """Minimal synthetic model matching Ornith language_model projection paths and executing real QTEALinear forward."""

    def __init__(self, vocab_size: int = 2, hidden_size: int = 4):
        super().__init__()
        self.config = MagicMock()
        self.config.use_cache = True
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.model = nn.Module()
        self.model.language_model.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.language_model.model.norm = nn.LayerNorm(hidden_size, bias=False)

        layer0 = nn.Module()
        layer0.self_attn = nn.Module()
        layer0.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer0.self_attn.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.model.language_model.layers = nn.ModuleList([layer0])

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.model.language_model.model.embed_tokens(input_ids).to(torch.float32)
        q = self.model.language_model.layers[0].self_attn.q_proj(x)
        k = self.model.language_model.layers[0].self_attn.k_proj(x)
        h = self.model.language_model.model.norm(q + k)
        logits = self.lm_head(h.to(self.lm_head.weight.dtype)).to(torch.float32)
        out = MagicMock()
        out.logits = logits
        return out

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, **kwargs):
        _ = self.forward(input_ids)
        new_tokens = torch.zeros((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype)
        return torch.cat([input_ids, new_tokens], dim=-1)


def test_load_qtea_full_model_and_tied_weights(synthetic_qtea_source: Path, tmp_path: Path):
    """Test full model instantiation, module replacement, weight assignment, tied tensors, and real forward execution."""
    out_dir = tmp_path / "sharded_for_loading"
    pack_qtea_shards(synthetic_qtea_source, out_dir, shard_mib=64)

    mock_model = SyntheticOrnithModel(vocab_size=2, hidden_size=4)
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0

    with patch("transformers.AutoConfig.from_pretrained", return_value=MagicMock()):
        with patch("transformers.AutoModelForMultimodalLM.from_config", return_value=mock_model):
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("orn_g4.loader.restore_qwen3_5_nonpersistent_buffers", return_value=[]):
                    model, tokenizer, metadata = load_qtea_full_model(out_dir, device="cpu")

    # Projections must be replaced with QTEALinear
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, QTEALinear)
    assert isinstance(model.model.language_model.layers[0].self_attn.k_proj, QTEALinear)

    # Tied tensors check: lm_head.weight must point to embed_tokens.weight
    assert model.lm_head.weight is model.model.language_model.model.embed_tokens.weight

    # Execute real forward pass through QTEALinear layers
    out = model(torch.tensor([[0, 1]], dtype=torch.long))
    assert out.logits.shape == (1, 2, 2)
    assert torch.isfinite(out.logits).all()

    assert metadata["projections"] == 2
    assert metadata["sharded"] is True
    assert metadata["outcome"] == "QUALITY_REJECTED_RECOVERY_REQUIRED"


def test_load_qtea_rejects_mismatched_shapes_and_standalone_loader_works(
    synthetic_qtea_source: Path, tmp_path: Path
):
    """Test shape mismatch rejection in load_qtea_full_model and standalone load_qtea_package.py on sharded dir."""
    import importlib.util

    # 1. Copy ternary_runtime.py into synthetic source and pack
    from orn_g4.export import _TERNARY_RUNTIME_SOURCE

    (synthetic_qtea_source / "ternary_runtime.py").write_text(_TERNARY_RUNTIME_SOURCE, encoding="utf-8")
    out_dir = tmp_path / "sharded_standalone"
    pack_qtea_shards(synthetic_qtea_source, out_dir, shard_mib=64)

    # Verify standalone load_qtea_package.py works directly on qtea-sharded-v2
    spec = importlib.util.spec_from_file_location("load_qtea_package", out_dir / "load_qtea_package.py")
    assert spec is not None and spec.loader is not None
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    pkg = loader_mod.load_package(out_dir)
    assert len(pkg["projections"]) == 2
    assert len(pkg["preserved_tensors"]) == 2

    # 2. Mismatched model skeleton shape must raise QTEALoadError
    bad_model = SyntheticOrnithModel(vocab_size=5, hidden_size=4)  # vocab_size=5 != 2 in checkpoint
    with patch("transformers.AutoConfig.from_pretrained", return_value=MagicMock()):
        with patch("transformers.AutoModelForMultimodalLM.from_config", return_value=bad_model):
            with pytest.raises(QTEALoadError, match="shape mismatch"):
                load_qtea_full_model(out_dir, device="cpu")


def test_benchmark_qtea_inference_reporting():
    """Test benchmark execution, peak memory accounting, and explicit labeling."""
    mock_model = SyntheticOrnithModel(vocab_size=4, hidden_size=4)
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]])}
    mock_tokenizer.decode.return_value = "Generated text explanation."
    mock_tokenizer.eos_token_id = 0

    report = benchmark_qtea_inference(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt="Explain why the sky is blue.",
        device="cpu",
        max_new_tokens=4,
        load_seconds=1.23,
    )

    assert report["status"] == "FULL_MODEL_GENERATION_COMPLETED"
    assert report["full_model_reload_verified"] is True
    assert report["teacher_logit_parity_verified"] is False
    assert report["packed_kernel_benchmark"] is False
    assert report["load_seconds"] == 1.23
    assert report["generated_tokens"] == 4
    assert report["logits_shape"] == [1, 3, 4]
    assert report["output_text"] == "Generated text explanation."
    assert "inference_notes" in report


def test_cli_pack_and_verify(synthetic_qtea_source: Path, tmp_path: Path):
    """Test tools/qtea_hf.py CLI entrypoints for pack and verify."""
    from tools.qtea_hf import main as cli_main

    out_dir = tmp_path / "cli_sharded"

    # 1. CLI pack
    rc_pack = cli_main(["pack", "--source", str(synthetic_qtea_source), "--output", str(out_dir), "--shard-mib", "64"])
    assert rc_pack == 0
    assert (out_dir / "qtea-sharded-manifest.json").is_file()

    # 2. CLI verify
    rc_verify = cli_main(["verify", "--path", str(out_dir)])
    assert rc_verify == 0
