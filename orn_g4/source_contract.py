"""Fail-closed source provenance and frozen-release checks for G4 runs.

No model or dataset is silently updated during a resumed run.  Revision locks
are resolved by the Hub and compared before downloading large checkpoints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import SourceContractError

_SHA_RE = re.compile(r"^[a-fA-F0-9]{40}$")
AZRAEL_SFT_V2_COMBINED_SHA256 = "a593a18a453bda2788d1e14e83cf623170d14599368e453e658cfc2f2f427e96"
AZRAEL_SFT_V2_MANIFEST_RELPATH = "release/azrael-sft-v2/DATASET_RELEASE_MANIFEST.json"
AZRAEL_SFT_V2_COMBINED_RELPATH = "combined/combined_train.parquet"


def resolve_hub_sha(repo_id: str, repo_type: str, revision: str | None = None) -> str:
    """Resolve a model/dataset ref to its immutable 40-digit HF commit SHA.

    # Errors: raises :class:`ValueError` for a repository type other than
    "model" or "dataset", and :class:`orn_g4.errors.SourceContractError` when
    the Hub returns no immutable commit SHA for the requested ref.
    """
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
    """Atomically record an input SHA or reject a changed input for this run.

    # Errors: raises :class:`ValueError` for a lock name other than "model" or
    "dataset", and :class:`orn_g4.errors.SourceContractError` when the SHA is
    not a 40-digit commit or when the pinned source changed inside the same
    run. Operator remediation: start a new run directory for a new source
    instead of re-locking an existing run.
    """
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


def _count_shard_supervised_tokens(path: Path, shard: Any) -> int | None:
    explicit = getattr(shard, "supervised_tokens", None)
    if explicit is not None:
        return int(explicit)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            with path.open("rb") as fh:
                magic = fh.read(4)
            if magic != b"PAR1":
                return None
            import pyarrow.compute as pc
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(path))
            names = set(pf.schema_arrow.names)
            if "supervised_count" in names:
                tbl = pq.read_table(str(path), columns=["supervised_count"])
                total = pc.sum(tbl["supervised_count"]).as_py()
                return int(total or 0)
            if "labels" in names:
                tbl = pq.read_table(str(path), columns=["labels"])
                sup_total = 0
                for row_labels in tbl["labels"].to_pylist():
                    sup_total += sum(1 for v in row_labels if int(v) != -100)
                return int(sup_total)
            return None
        except SourceContractError:
            raise
        except Exception as exc:
            raise SourceContractError(
                f"failed to inspect supervised tokens in parquet shard {path}: {exc}"
            ) from exc
    if suffix == ".npz":
        try:
            import numpy as np

            with np.load(path) as zf:
                if "supervised_count" in zf:
                    return int(np.sum(zf["supervised_count"]))
                if "labels" in zf:
                    return int(np.sum(np.asarray(zf["labels"]) != -100))
        except Exception:
            return None
    return None


def validate_release(
    layout: Any,
    *,
    require_checksums: bool = True,
    max_sequence_length: int = 8192,
    expected_shard_sha256s: dict[str, str] | None = None,
    verify_v2_combined_lock: bool = False,
) -> dict[str, Any]:
    """Inspect manifest-defined training shards without legacy hard-coded counts.

    Counts and hashes are checked against *this release's* manifest.  The old
    7-shard/17,848-record MVP is not treated as the current source of truth.

    # Errors: raises :class:`orn_g4.errors.SourceContractError` when the
    release resolves no shards, a shard escapes the training root or enters
    ``eval_firewall``, a shard path repeats, a checksum is missing or
    mismatched, the required array columns are absent, the manifest record
    count or supervised token sum disagrees with resolved shards, or the
    declared sequence length exceeds the limit.
    """
    from .data import _rank_array_names
    from .utils import sha256_file

    if not layout.shards:
        raise SourceContractError("release has no resolved training shards")
    root = Path(layout.dataset_root).resolve()
    seen: set[str] = set()
    rows = 0
    all_rows_declared = True
    sup_tokens = 0
    all_sup_computed = True

    pinned_map: dict[str, str] = {}
    if verify_v2_combined_lock:
        pinned_map["combined_train.parquet"] = AZRAEL_SFT_V2_COMBINED_SHA256
    if expected_shard_sha256s:
        for k, v in expected_shard_sha256s.items():
            pinned_map[str(k)] = str(v).removeprefix("sha256:").lower()

    matched_pins: set[str] = set()
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
        rel_key = str(p.relative_to(root))
        for pin_key, expected_hex in pinned_map.items():
            if rel_key == pin_key or p.name == Path(pin_key).name:
                actual_hex = sha256_file(p).removeprefix("sha256:").lower()
                expected_clean = expected_hex.removeprefix("sha256:").lower()
                if actual_hex != expected_clean:
                    raise SourceContractError(
                        f"SHA-256 dataset lock mismatch for {rel_key}: "
                        f"got {actual_hex}, expected {expected_clean}"
                    )
                matched_pins.add(pin_key)

        mapped = _rank_array_names(shard.arrays)
        if any(mapped[k] is None for k in ("input_ids", "attention_mask", "labels")):
            raise SourceContractError(f"shard {p} lacks required columns: {mapped!r}")
        if shard.num_examples is None:
            all_rows_declared = False
        else:
            rows += int(shard.num_examples)

        shard_sup = _count_shard_supervised_tokens(p, shard)
        if shard_sup is None:
            all_sup_computed = False
        else:
            sup_tokens += shard_sup

    missing_pins = sorted(set(pinned_map.keys()) - matched_pins)
    if missing_pins:
        raise SourceContractError(f"pinned dataset shard(s) not present in release: {missing_pins}")

    m = layout.manifest
    data = m.get("dataset") if isinstance(m.get("dataset"), dict) else {}
    declared_rows = next((v for v in (
        m.get("records"), m.get("num_records"), data.get("total_records"), data.get("records")
    ) if v is not None), None)
    if declared_rows is not None and all_rows_declared and rows != int(declared_rows):
        raise SourceContractError(
            f"manifest records={declared_rows} but shard rows sum={rows}"
        )
    declared_sup = next((v for v in (
        m.get("supervised_tokens"), m.get("num_supervised_tokens"), data.get("supervised_tokens")
    ) if v is not None), None)
    if declared_sup is not None and all_sup_computed and sup_tokens != int(declared_sup):
        raise SourceContractError(
            f"manifest supervised_tokens={declared_sup} but shard supervised_tokens sum={sup_tokens}"
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
        "declared_supervised_tokens": int(declared_sup) if declared_sup is not None else None,
        "summed_supervised_tokens": sup_tokens if all_sup_computed else None,
        "declared_max_seq_len": int(declared_max) if declared_max is not None else None,
        "sha256_verified_shards": sum(s.checksum_ok is True for s in layout.shards),
        "pinned_sha256_verified": sorted(matched_pins),
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
    """Compare the dataset's declared chat-template hash with the model's.

    # Errors: raises :class:`orn_g4.errors.SourceContractError` when the
    manifest declares no recognizable chat-template SHA-256 and
    ``require_declared_hash`` is set, or when the declared hash differs from
    the model's template hash. Operator remediation: regenerate the dataset
    template from the pinned model revision rather than editing the hash.
    """
    expected = dataset_template_hash(manifest)
    if expected is None:
        if require_declared_hash:
            raise SourceContractError("release manifest does not declare a recognizable chat-template SHA-256")
        return None
    actual = actual_hash.removeprefix("sha256:").lower()
    if actual != expected:
        raise SourceContractError(
            f"pre-tokenized dataset template hash {expected} does not match model's {actual_hash}"
        )
    return expected
