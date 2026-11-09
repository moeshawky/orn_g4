"""Unit tests for the Drive nested-ZIP dataset loader (work-order §3).

Uses synthetic fixtures: a ZIP containing a nested azrael-dataset/ root with
train_ready shards (.npz), a DATASET_RELEASE_MANIFEST.json, and an
eval_firewall/ directory that must be excluded.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from orn_g4.data import (
    DatasetLayout,
    build_layout,
    load_shard,
    resolve_dataset_root,
    validate_alignment,
)


def _make_shard_npz(path: Path, n_examples: int = 10, in_f: int = 8):
    np.savez(
        path,
        input_ids=np.arange(n_examples * in_f).reshape(n_examples, in_f),
        attention_mask=np.ones((n_examples, in_f), dtype=np.int64),
        labels=np.arange(n_examples * in_f).reshape(n_examples, in_f),
    )


def _make_release_zip(zip_path: Path):
    """Build a synthetic release ZIP mirroring the real layout."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        # nested root
        zf.writestr(
            "azrael-dataset/DATASET_RELEASE_MANIFEST.json",
            json.dumps(
                {
                    "train_shards": ["shard_a.npz", "shard_b.npz"],
                    "checksums": {},
                }
            ),
        )
        zf.writestr("azrael-dataset/train_ready/shard_a.npz", _npz_bytes(10))
        zf.writestr("azrael-dataset/train_ready/shard_b.npz", _npz_bytes(10))
        zf.writestr("azrael-dataset/eval_firewall/firewall.npz", _npz_bytes(5))
        zf.writestr("azrael-dataset/DATACARD.md", "synthetic datacard")


def _npz_bytes(n_examples: int = 10, in_f: int = 8) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez(
        buf,
        input_ids=np.arange(n_examples * in_f).reshape(n_examples, in_f),
        attention_mask=np.ones((n_examples, in_f), dtype=np.int64),
        labels=np.arange(n_examples * in_f).reshape(n_examples, in_f),
    )
    return buf.getvalue()


def test_resolve_dataset_root_finds_nested_dir(tmp_path):
    extract = tmp_path / "extract"
    (extract / "azrael-dataset" / "train_ready").mkdir(parents=True)
    (extract / "azrael-dataset" / "train_ready" / "x.npz").write_bytes(_npz_bytes())
    root = resolve_dataset_root(extract)
    assert root.name == "azrael-dataset"


def test_build_layout_excludes_eval_firewall(tmp_path):
    zip_path = tmp_path / "release.zip"
    _make_release_zip(zip_path)
    layout = build_layout(
        zip_path=zip_path,
        local_dir=tmp_path / "local",
        extract_dir=tmp_path / "extract",
        exclude_dirs=["eval_firewall"],
    )
    assert isinstance(layout, DatasetLayout)
    assert layout.eval_firewall_dir is not None
    assert "eval_firewall" in layout.excluded_dirs
    shard_names = [s.name for s in layout.shards]
    assert "shard_a.npz" in shard_names
    assert "shard_b.npz" in shard_names
    # firewall shard must not appear as a training shard.
    assert not any("firewall" in s.name for s in layout.shards)


def test_load_shard_and_validate_alignment(tmp_path):
    zip_path = tmp_path / "release.zip"
    _make_release_zip(zip_path)
    layout = build_layout(
        zip_path=zip_path,
        local_dir=tmp_path / "local",
        extract_dir=tmp_path / "extract",
        exclude_dirs=["eval_firewall"],
    )
    shard = layout.shards[0]
    bundle = load_shard(shard)
    assert bundle.input_ids.shape[0] == bundle.labels.shape[0]
    validate_alignment(bundle.input_ids, bundle.attention_mask, bundle.labels)


def test_validate_alignment_rejects_mismatch():
    ii = np.zeros((4, 8), dtype=np.int64)
    am = np.zeros((4, 8), dtype=np.int64)
    lb = np.zeros((5, 8), dtype=np.int64)  # different first dim
    with pytest.raises(Exception):
        validate_alignment(ii, am, lb)


def test_frozen_dataset_parquet_and_split_hygiene(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from orn_g4.data import FrozenDataset
    from orn_g4.errors import DataAlignmentError

    p = tmp_path / "shard_0_frozen.parquet"
    n = 200
    table = pa.Table.from_pydict({
        "id": [f"rec_{i}" for i in range(n)],
        "source": ["synthetic"] * n,
        "token_count": [8] * n,
        "supervised_count": [4] * n,
        "unmasked_len": [8] * n,
        "supervised_len": [4] * n,
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, i] for i in range(n)],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1] for _ in range(n)],
        "labels": [[-100, -100, -100, -100, 5, 6, 7, i] for i in range(n)],
    })
    pq.write_table(table, str(p))

    ds = FrozenDataset([p])
    assert len(ds) == n
    splits = ds.build_splits()
    for s in ("train", "calib", "probe", "heldout"):
        assert splits[s] > 0

    it = ds.stream("train")
    cursor, (ids, mask, labels) = next(it)
    assert cursor >= 1
    assert len(ids) == 8
    assert len(mask) == 8
    assert len(labels) == 8

    samples = ds.samples("train", 2)
    assert len(samples) == 2

    # Verify unshifted label rejection
    bad_p = tmp_path / "bad_labels.parquet"
    bad_table = pa.Table.from_pydict({
        "id": ["bad_1"],
        "source": ["synthetic"],
        "token_count": [8],
        "supervised_count": [4],
        "unmasked_len": [8],
        "supervised_len": [4],
        "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        "labels": [[-100, -100, -100, -100, 999, 999, 999, 999]],  # Mismatch with input_ids
    })
    pq.write_table(bad_table, str(bad_p))
    bad_ds = FrozenDataset([bad_p])
    with pytest.raises(DataAlignmentError, match="labels are not unshifted"):
        bad_ds.record(0)


def test_locate_dataset_local_dir_and_file(tmp_path):
    from orn_g4.data import locate_dataset, import_dataset

    d = tmp_path / "my_dataset"
    d.mkdir()
    f = tmp_path / "archive.zip"
    f.write_text("dummy")

    assert locate_dataset(d) == d.resolve()
    assert locate_dataset(str(f)) == f.resolve()
    assert import_dataset(d) == d.resolve()


def test_build_layout_from_directory_with_dict_manifest(tmp_path, monkeypatch):
    from orn_g4.data import build_layout

    dataset_root = tmp_path / "hf_snapshot"
    train_ready = dataset_root / "train_ready" / "mvp"
    train_ready.mkdir(parents=True)
    shard_file = train_ready / "part_frozen.npz"
    _make_shard_npz(shard_file)

    manifest_data = {
        "dataset": {
            "total_records": 10,
            "total_tokens": 80,
            "supervised_tokens": 40,
            "max_seq_len": 8,
        },
        "shards": [
            {
                "path": "train_ready/mvp/part_frozen.npz",
                "rows": 10,
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            }
        ],
    }
    (dataset_root / "DATASET_RELEASE_MANIFEST.json").write_text(json.dumps(manifest_data))

    layout = build_layout(source=dataset_root)
    assert layout.dataset_root == dataset_root
    assert len(layout.shards) == 1
    assert layout.shards[0].name == "part_frozen.npz"
    assert layout.shards[0].num_examples == 10


def test_locate_dataset_hf_snapshot(tmp_path, monkeypatch):
    from orn_g4.data import locate_dataset

    fake_hf_dir = tmp_path / "fake_hf_download"
    fake_hf_dir.mkdir()

    def fake_snapshot_download(repo_id, repo_type, local_dir):
        assert repo_id == "moeshawky/azrael-dataset"
        assert repo_type == "dataset"
        return str(fake_hf_dir)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    res = locate_dataset("hf://moeshawky/azrael-dataset")
    assert res == fake_hf_dir


