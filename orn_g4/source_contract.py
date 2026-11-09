"""Fail-closed source provenance and frozen-release checks for G4 runs.

No model or dataset is silently updated during a resumed run.  Revision locks
are resolved by the Hub and compared before downloading large checkpoints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SHA_RE = re.compile(r"^[a-fA-F0-9]{40}$")


class SourceContractError(RuntimeError):
    """A source, checkpoint, or dataset contract differs from the run lock."""


def resolve_hub_sha(repo_id: str, repo_type: str, revision: str | None = None) -> str:
    """Resolve a model/dataset ref to its immutable 40-digit HF commit SHA."""
    if repo_type not in ("model", "dataset"):
        raise ValueError(f"unsupported Hub repository type: {repo_type!r}")
    from huggingface_hub import HfApi

    api = HfApi()
    info = (
        api.model_info(repo_id, revision=revision)
        if repo_type == "model"
        else api.dataset_info(repo_id, revision=revision)
    )
    sha = str(getattr(info, "sha", "") or "").lower()
    if not _SHA_RE.fullmatch(sha):
        raise SourceContractError(f"Hub returned no immutable commit SHA for {repo_type} {repo_id}")
    return sha


def enforce_source_lock(path: str | Path, name: str, repo_id: str, sha: str) -> dict[str, Any]:
    """Atomically record an input SHA or reject a changed input for this run."""
    if name not in ("model", "dataset"):
        raise ValueError("source lock name must be model or dataset")
    if not _SHA_RE.fullmatch(str(sha)):
        raise SourceContractError(f"invalid immutable SHA for {name}: {sha!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{name}_source"
    current = record.get(key)
    desired = {"repo_id": repo_id, "commit_sha": sha.lower()}
    if current is not None and current != desired:
        raise SourceContractError(
            f"{name} source changed within the same run: locked={current!r}, proposed={desired!r}; "
            "create a new run or restore the originally pinned source"
        )
    record[key] = desired
    from .utils import write_json_atomic
    write_json_atomic(path, record)
    return record


def validate_release(layout: Any, *, require_checksums: bool = True, max_sequence_length: int = 8192) -> dict[str, Any]:
    """Inspect manifest-defined training shards without legacy hard-coded counts.

    Counts and hashes are checked against *this release's* manifest.  The old
    7-shard/17,848-record MVP is not treated as the current source of truth.
    """
    from .data import _rank_array_names
    if not layout.shards:
        raise SourceContractError("release has no resolved training shards")
    root = Path(layout.dataset_root).resolve()
    seen: set[str] = set()
    rows = 0
    all_rows_declared = True
    for shard in layout.shards:
        p = Path(shard.path).resolve()
        if not p.is_relative_to(root) or "eval_firewall" in p.relative_to(root).parts:
            raise SourceContractError(f"shard escapes training root or enters eval_firewall: {p}")
        key = str(p)
        if key in seen:
            raise SourceContractError(f"duplicate training shard path: {p}")
        seen.add(key)
        if shard.checksum_ok is False:
            raise SourceContractError(f"SHA-256 checksum mismatch for shard: {p}")
        if require_checksums and (not shard.checksum or shard.checksum_ok is not True):
            raise SourceContractError(f"missing or unverified shard checksum: {p}")
        mapped = _rank_array_names(shard.arrays)
        if any(mapped[k] is None for k in ("input_ids", "attention_mask", "labels")):
            raise SourceContractError(f"shard {p} lacks required columns: {mapped!r}")
        if shard.num_examples is None:
            all_rows_declared = False
        else:
            rows += int(shard.num_examples)
    m = layout.manifest
    data = m.get("dataset") if isinstance(m.get("dataset"), dict) else {}
    declared_rows = next((v for v in (
        m.get("records"), m.get("num_records"), data.get("total_records"), data.get("records")
    ) if v is not None), None)
    if declared_rows is not None and all_rows_declared and rows != int(declared_rows):
        raise SourceContractError(
            f"manifest records={declared_rows} but shard rows sum={rows}"
        )
    declared_max = next((v for v in (
        m.get("max_seq_len"), m.get("max_sequence_length"), data.get("max_seq_len")
    ) if v is not None), None)
    if declared_max is not None and int(declared_max) > max_sequence_length:
        raise SourceContractError(
            f"manifest max_seq_len={declared_max} exceeds limit {max_sequence_length}"
        )
    return {
        "shard_count": len(layout.shards),
        "declared_records": int(declared_rows) if declared_rows is not None else None,
        "summed_shard_rows": rows if all_rows_declared else None,
        "declared_max_seq_len": int(declared_max) if declared_max is not None else None,
        "sha256_verified_shards": sum(s.checksum_ok is True for s in layout.shards),
    }


def dataset_template_hash(manifest: dict[str, Any]) -> str | None:
    """Find an explicitly declared template SHA, never an unrelated tokenizer SHA."""
    candidates: list[Any] = []
    for field in ("chat_template_sha256", "template_sha256", "chat_template_hash", "template_hash"):
        candidates.append(manifest.get(field))
    for outer in ("template", "chat_template", "tokenizer", "dataset"):
        nested = manifest.get(outer)
        if isinstance(nested, dict):
            for field in ("chat_template_sha256", "template_sha256", "chat_template_hash", "template_hash"):
                candidates.append(nested.get(field))
            if outer in ("template", "chat_template"):
                candidates.extend((nested.get("sha256"), nested.get("hash_sha256")))
    for value in candidates:
        if isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[a-fA-F0-9]{64}", value):
            return value.removeprefix("sha256:").lower()
    return None


def assert_template_alignment(manifest: dict[str, Any], actual_hash: str, *, require_declared_hash: bool) -> str | None:
    expected = dataset_template_hash(manifest)
    if expected is None:
        if require_declared_hash:
            raise SourceContractError("release manifest does not declare a recognizable chat-template SHA-256")
        return None
    if actual_hash.lower() != expected:
        raise SourceContractError(
            f"pre-tokenized dataset template hash {expected} does not match model's {actual_hash}"
        )
    return expected
