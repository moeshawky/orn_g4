"""The frozen release uses sidecar SHA files and excludes the persona shard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orn_g4.errors import DatasetError
from orn_g4.frozen_release import (
    build_frozen_mvp_layout,
    inspect_frozen_mvp,
    verify_frozen_template,
)


def _fixture(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "frozen"
    directory = root / "train_ready" / "mvp"
    directory.mkdir(parents=True)
    hashes = root / "manifests" / "shard_hashes.sha256"
    hashes.parent.mkdir()
    (root / "DATASET_RELEASE_MANIFEST.json").write_text(
        json.dumps({"dataset": {"total_records": 17999}, "shards": []})
    )
    # 17,999 is the archive-wide total; exactly 17,848 of those records
    # belong to the immutable seven-shard behavioral MVP.
    counts = [17842, 1, 1, 1, 1, 1, 1]
    lines = []
    for index, count in enumerate(counts):
        target = directory / f"part{index}_frozen.parquet"
        col = pa.array([[1, 2]] * count, type=pa.list_(pa.int64()))
        table = pa.table({
            "id": pa.array([f"{index}:{k}" for k in range(count)]),
            "source": pa.array(["synthetic"] * count),
            "input_ids": col,
            "labels": col,
            "attention_mask": col,
            "token_count": pa.array([2] * count, type=pa.int64()),
            "supervised_count": pa.array([2] * count, type=pa.int64()),
        })
        pq.write_table(table, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        lines.append(f"{digest}  train_ready/mvp/{target.name}")
    hashes.write_text("\n".join(lines) + "\n")
    template_dir = root / "template"
    template_dir.mkdir()
    template = b"{% for message in messages %}{{ message.content }}{% endfor %}\n"
    (template_dir / "chat_template.jinja").write_bytes(template)
    (template_dir / "template.sha256").write_text(
        hashlib.sha256(template).hexdigest() + "  chat_template.jinja\n"
    )
    persona = root / "train_ready" / "persona"
    persona.mkdir()
    (persona / "persona_charter_sft.parquet").write_bytes(b"unrelated persona shard")
    return root, hashlib.sha256(template).hexdigest()


def test_frozen_mvp_uses_sidecar_hashes_and_not_persona(tmp_path):
    root, digest = _fixture(tmp_path)
    layout = build_frozen_mvp_layout(root)
    result = inspect_frozen_mvp(layout)
    assert result["shard_count"] == 7
    assert result["train_records"] == 17848
    assert result["verified_shards"] == 7
    assert all("persona" not in s.path.parts for s in layout.shards)
    assert verify_frozen_template(root, digest) == digest


def test_frozen_mvp_detects_tampered_shard(tmp_path):
    root, _ = _fixture(tmp_path)
    target = root / "train_ready" / "mvp" / "part0_frozen.parquet"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(DatasetError, match="SHA-256 mismatch"):
        build_frozen_mvp_layout(root)


def test_frozen_mvp_detects_template_mismatch(tmp_path):
    root, digest = _fixture(tmp_path)
    with pytest.raises(DatasetError, match="model/dataset chat template mismatch"):
        verify_frozen_template(root, "0" * 64)
    assert verify_frozen_template(root, digest) == digest


def test_frozen_mvp_rejects_missing_checksum(tmp_path):
    root, _ = _fixture(tmp_path)
    sidecar = root / "manifests" / "shard_hashes.sha256"
    text = sidecar.read_text().splitlines()
    sidecar.write_text("\n".join(text[1:]) + "\n")
    with pytest.raises(DatasetError, match="missing frozen shard checksum"):
        build_frozen_mvp_layout(root)


def test_frozen_mvp_rejects_missing_shard(tmp_path):
    root, _ = _fixture(tmp_path)
    (root / "train_ready" / "mvp" / "part6_frozen.parquet").unlink()
    layout = build_frozen_mvp_layout(root)
    with pytest.raises(DatasetError, match="exactly 7 shards"):
        inspect_frozen_mvp(layout)
