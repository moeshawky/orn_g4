"""Pure preflight tests: fail-closed immutable source, release and template checks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from orn_g4.source_contract import (
    SourceContractError, resolve_hub_sha, enforce_source_lock,
    validate_release, assert_template_alignment, dataset_template_hash,
)


def test_resolve_sha_model_and_dataset():
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value.sha = "a" * 40
        api.return_value.dataset_info.return_value.sha = "b" * 40
        assert resolve_hub_sha("owner/model", "model") == "a" * 40
        assert resolve_hub_sha("owner/data", "dataset", "main") == "b" * 40
        api.return_value.dataset_info.assert_called_once_with("owner/data", revision="main")


def test_source_lock_prevents_silent_revision_switch(tmp_path):
    p = tmp_path / "source_manifest.json"
    enforce_source_lock(p, "model", "owner/model", "a" * 40)
    enforce_source_lock(p, "model", "owner/model", "a" * 40)
    enforce_source_lock(p, "dataset", "owner/data", "b" * 40)
    state = json.loads(p.read_text())
    assert state["model_source"]["commit_sha"] == "a" * 40
    assert state["dataset_source"]["commit_sha"] == "b" * 40
    with pytest.raises(SourceContractError, match="source changed"):
        enforce_source_lock(p, "dataset", "owner/data", "c" * 40)


def _layout(tmp_path, *, checksum_ok=True, with_checksum=True):
    root = tmp_path / "snapshot"
    shard = root / "train_ready" / "x.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"synthetic")
    return SimpleNamespace(
        dataset_root=root,
        shards=[SimpleNamespace(
            path=shard, name="x.parquet",
            arrays=["id", "input_ids", "labels", "attention_mask"],
            num_examples=3, checksum="sha256:" + "a" * 64 if with_checksum else None,
            checksum_ok=checksum_ok,
        )],
        manifest={"dataset": {"total_records": 3, "max_seq_len": 8192}}
    )


def test_release_valid_and_legacy_counts_not_required(tmp_path):
    release = _layout(tmp_path)
    stats = validate_release(release)
    assert stats["shard_count"] == 1
    assert stats["declared_records"] == 3


def test_release_rejects_bad_checksum(tmp_path):
    layout = _layout(tmp_path, checksum_ok=False)
    with pytest.raises(SourceContractError, match="checksum mismatch"):
        validate_release(layout)


def test_release_rejects_missing_checksum(tmp_path):
    layout = _layout(tmp_path, with_checksum=False, checksum_ok=None)
    with pytest.raises(SourceContractError, match="missing or unverified"):
        validate_release(layout)


def test_release_rejects_incorrect_counts(tmp_path):
    layout = _layout(tmp_path)
    layout.manifest["dataset"]["total_records"] = 4
    with pytest.raises(SourceContractError, match="shard rows"):
        validate_release(layout)


def test_template_hash_requires_match():
    h = "a" * 64
    m = {"template": {"sha256": h}}
    assert dataset_template_hash(m) == h
    assert assert_template_alignment(m, h, require_declared_hash=True) == h
    with pytest.raises(SourceContractError, match="does not match"):
        assert_template_alignment(m, "b" * 64, require_declared_hash=True)
    with pytest.raises(SourceContractError, match="does not declare"):
        assert_template_alignment({}, h, require_declared_hash=True)


def test_release_parquet_supervised_tokens_and_v2_sha_lock(tmp_path):
    """validate_release emits summed_supervised_tokens and enforces optional SHA-256 locks."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from orn_g4.data import build_layout
    from orn_g4.source_contract import AZRAEL_SFT_V2_COMBINED_SHA256

    snap = tmp_path / "hf_v2"
    v2_root = snap / "release" / "azrael-sft-v2"
    comb_dir = v2_root / "combined"
    comb_dir.mkdir(parents=True)
    pq_path = comb_dir / "combined_train.parquet"
    table = pa.Table.from_pydict({
        "id": ["r0", "r1", "r2"],
        "source": ["sft", "sft", "sft"],
        "token_count": [6, 6, 6],
        "supervised_count": [2, 3, 4],
        "input_ids": [[1, 2, 3, 4, 5, 6]] * 3,
        "attention_mask": [[1, 1, 1, 1, 1, 1]] * 3,
        "labels": [
            [-100, -100, -100, -100, 5, 6],
            [-100, -100, -100, 4, 5, 6],
            [-100, -100, 3, 4, 5, 6],
        ],
    })
    pq.write_table(table, str(pq_path))
    actual_sha = hashlib.sha256(pq_path.read_bytes()).hexdigest()

    (v2_root / "DATASET_RELEASE_MANIFEST.json").write_text(
        json.dumps({
            "dataset": {
                "total_records": 3,
                "supervised_tokens": 9,
                "max_seq_len": 8192,
            },
            "shards": [{
                "path": "combined/combined_train.parquet",
                "rows": 3,
                "sha256": f"sha256:{actual_sha}",
            }],
        }),
        encoding="utf-8",
    )

    layout = build_layout(source=snap)
    stats = validate_release(
        layout,
        expected_shard_sha256s={"combined/combined_train.parquet": actual_sha},
    )
    assert stats["summed_supervised_tokens"] == 9
    assert stats["declared_supervised_tokens"] == 9
    assert stats["summed_shard_rows"] == 3
    assert stats["pinned_sha256_verified"] == ["combined/combined_train.parquet"]

    # Negative control: v2 combined lock rejects mismatched combined_train.parquet SHA
    assert actual_sha != AZRAEL_SFT_V2_COMBINED_SHA256
    with pytest.raises(SourceContractError, match="SHA-256 dataset lock mismatch"):
        validate_release(layout, verify_v2_combined_lock=True)

    # Negative control: mismatched supervised_tokens in manifest raises SourceContractError
    layout.manifest["dataset"]["supervised_tokens"] = 10
    with pytest.raises(SourceContractError, match="supervised_tokens=10"):
        validate_release(layout)
