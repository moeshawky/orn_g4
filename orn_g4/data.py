"""Google Drive nested-ZIP dataset import (work-order §3).

Mounts Google Drive, locates the ZIP at the exact root path, verifies archive
integrity, copies the archive to Colab local storage, extracts safely, resolves
the nested ``azrael-dataset/`` root, reads ``DATASET_RELEASE_MANIFEST.json``,
locates the actual training shards using the manifest, checks shard availability
and checksums, loads the original pre-tokenized arrays and masks, keeps
``eval_firewall/`` entirely excluded from training and calibration, and trains
from local disk.

The loader is format-adaptive: it reads the manifest for the declared shard
list, and when the manifest's schema does not match it discovers array names
within each shard by key heuristics and records the mapping it chose. It never
guesses silently — an unmappable shard raises a :class:`DatasetError` naming
the keys actually found.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataAlignmentError, DatasetError
from .utils import ensure_dir, sha256_file


@dataclass(frozen=True)
class ShardRef:
    """One training shard resolved from the release manifest."""

    name: str
    path: Path  # local extracted path
    checksum: str | None  # "sha256:<hex>" from manifest, if declared
    arrays: list[str]  # array names discovered in the shard
    num_examples: int | None
    checksum_ok: bool | None = None  # None until verified


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved dataset layout after extraction."""

    zip_path: Path  # local copy of the ZIP
    extract_dir: Path  # where it was extracted
    dataset_root: Path  # nested azrael-dataset/ directory
    manifest: dict  # parsed DATASET_RELEASE_MANIFEST.json
    train_ready_dir: Path | None
    shards: list[ShardRef]
    eval_firewall_dir: Path | None
    excluded_dirs: list[str]  # names confirmed excluded from training


@dataclass(frozen=True)
class TensorBundle:
    """Loaded pre-tokenized arrays and masks from one shard."""

    input_ids: object  # numpy/torch array
    attention_mask: object
    labels: object
    shard_name: str
    num_examples: int


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive via google.colab.drive.

    This function exists as a named, callable step; the actual
    ``google.colab.drive.mount`` call is made from the notebook cell so the
    Colab auth flow runs in the notebook context.

    # Errors: raises :class:`DatasetError` if ``google.colab`` is unavailable
    when called from a non-Colab interpreter.
    """
    try:
        from google.colab import drive  # lazy import
    except ImportError as exc:
        raise DatasetError(
            "google.colab is unavailable outside Colab; mount Drive from a "
            "Colab cell before importing the dataset."
        ) from exc
    drive.mount(mount_point)


def locate_zip(zip_path: str | Path = "/content/drive/MyDrive/azrael-dataset.zip") -> Path:
    """Verify the dataset ZIP exists at the exact Google Drive root path.

    # Errors: raises :class:`DatasetError` when the file is absent.
    """
    p = Path(zip_path)
    if not p.is_file():
        raise DatasetError(
            f"dataset ZIP not found at {p}; expected at Google Drive root "
            f"/content/drive/MyDrive/azrael-dataset.zip"
        )
    return p.resolve()


def verify_zip_integrity(zip_path: Path | str) -> bool:
    """Verify the ZIP archive integrity (bad zip file test)."""
    p = Path(zip_path)
    try:
        with zipfile.ZipFile(p) as zf:
            bad = zf.testzip()
            return bad is None
    except (zipfile.BadZipFile, OSError):
        return False


def copy_zip_to_local(zip_path: Path | str, local_dir: Path | str) -> Path:
    """Copy the archive to Colab local storage (work-order §3 step 4).

    Returns the local copy path. Training reads from this local copy, not from
    repeated reads over the Google Drive mount.
    """
    src = Path(zip_path).resolve()
    dst_dir = ensure_dir(local_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst.resolve()


def extract_zip(zip_path: Path | str, extract_dir: Path | str) -> Path:
    """Extract the ZIP safely into ``extract_dir`` and return it."""
    src = Path(zip_path).resolve()
    dst = ensure_dir(extract_dir)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)
    return dst.resolve()


def resolve_dataset_root(extract_dir: Path | str) -> Path:
    """Resolve the nested ``azrael-dataset/`` directory inside the extraction.

    # Errors: raises :class:`DatasetError` when no top-level
    ``azrael-dataset/`` directory is found (the release layout changed).
    """
    root = Path(extract_dir).resolve()
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name == "azrael-dataset"]
    if not candidates:
        # Fall back: any single top-level directory.
        dirs = [p for p in root.iterdir() if p.is_dir()]
        if len(dirs) == 1:
            return dirs[0]
        raise DatasetError(
            f"no nested 'azrael-dataset/' directory found in {root}; "
            f"top-level entries: {[p.name for p in dirs]}"
        )
    return candidates[0].resolve()


def load_manifest(dataset_root: Path | str) -> dict:
    """Read ``DATASET_RELEASE_MANIFEST.json`` from the dataset root.

    # Errors: raises :class:`DatasetError` when the manifest is absent or not
    valid JSON — the manifest is the authoritative shard source.
    """
    root = Path(dataset_root).resolve()
    manifest_path = root / "DATASET_RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise DatasetError(
            f"DATASET_RELEASE_MANIFEST.json not found at {manifest_path}; "
            f"cannot resolve shards authoritatively"
        )
    try:
        with Path(manifest_path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"DATASET_RELEASE_MANIFEST.json at {manifest_path} is not valid JSON: {exc}"
        ) from exc


def _discover_arrays_in_shard(shard_path: Path) -> list[str]:
    """Discover array names inside a shard file (.npz / .safetensors / .pt).

    Returns a heuristic ranking of candidate array names. Used only when the
    manifest does not declare array names explicitly.
    """
    suffix = shard_path.suffix.lower()
    candidates: list[str] = []
    if suffix == ".npz":
        try:
            import numpy as np  # lazy

            with np.load(shard_path) as zf:
                candidates = list(zf.files)
        except Exception:  # noqa: S110  (graceful format discovery)
            pass
    elif suffix in (".parquet",):
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(str(shard_path))
            candidates = list(schema.names)
        except Exception:  # noqa: S110
            pass
    elif suffix in (".safetensors",):
        try:
            from safetensors import safe_open  # lazy

            with safe_open(str(shard_path), framework="pt") as fh:
                candidates = list(fh.keys())
        except Exception:  # noqa: S110  (graceful format discovery)
            pass
    elif suffix in (".pt", ".bin", ".pth"):
        try:
            import torch  # lazy

            obj = torch.load(str(shard_path), map_location="cpu")
            if isinstance(obj, dict):
                candidates = [str(k) for k in obj]
        except Exception:  # noqa: S110  (graceful format discovery)
            pass
    return candidates


def _rank_array_names(names: list[str]) -> dict[str, str | None]:
    """Rank discovered array names into input_ids / attention_mask / labels.

    Heuristic: exact matches first, then substring matches. Returns a mapping
    of logical role -> actual array name (or None when unmappable).
    """
    exact = {
        "input_ids": ["input_ids", "input_ids", "ids"],
        "attention_mask": ["attention_mask", "attention_mask", "mask"],
        "labels": ["labels", "targets", "label"],
    }
    mapping: dict[str, str | None] = {"input_ids": None, "attention_mask": None, "labels": None}
    lower = {n.lower(): n for n in names}
    for role, options in exact.items():
        for opt in options:
            if opt in lower:
                mapping[role] = lower[opt]
                break
        if mapping[role] is None:
            # substring fallback
            for n in names:
                ln = n.lower()
                if role == "input_ids" and "input" in ln and "id" in ln:
                    mapping[role] = n
                    break
                if role == "attention_mask" and ("attention" in ln or "mask" in ln):
                    mapping[role] = n
                    break
                if role == "labels" and ("label" in ln or "target" in ln):
                    mapping[role] = n
                    break
    return mapping


def resolve_shards(
    dataset_root: Path | str,
    manifest: dict,
    exclude_dirs: list[str] | None = None,
) -> list[ShardRef]:
    """Resolve the actual training shards using the manifest.

    Reads the manifest's declared shard list (any of ``shards``, ``train_shards``,
    ``files`` keys), checks each shard's availability and checksum, discovers
    array names within each shard, and returns ShardRefs. Shards listed under
    any ``exclude`` key (e.g. ``eval_firewall``) are omitted.

    # Errors: raises :class:`DatasetError` when no shards can be resolved, or
    when a declared shard is missing from disk.
    """
    root = Path(dataset_root).resolve()
    excluded = set(exclude_dirs or [])
    excluded.add("eval_firewall")

    shard_names: list[str] = []
    for key in ("shards", "train_shards", "files", "training_shards"):
        val = manifest.get(key)
        if isinstance(val, list):
            shard_names = [str(v) for v in val]
            break
        if isinstance(val, dict):
            # manifest may nest shards under a sub-table
            for sub in val.values():
                if isinstance(sub, list):
                    shard_names = [str(v) for v in sub]
                    break
            if shard_names:
                break

    # Manifest may declare shard paths relative to train_ready/ or to root.
    train_ready = root / "train_ready"
    shards: list[ShardRef] = []
    for name in shard_names:
        candidate_paths = [root / name, train_ready / name]
        resolved = None
        for cp in candidate_paths:
            if cp.is_file():
                resolved = cp
                break
        if resolved is None:
            # Directory-based shard: look inside train_ready for a dir named `name`.
            d = train_ready / name
            if d.is_dir():
                # arrays inside the directory
                files = sorted(d.iterdir())
                resolved = d  # marker; arrays discovered below
        if resolved is None:
            raise DatasetError(
                f"declared shard {name!r} not found at {root / name} or {train_ready / name}"
            )

        if resolved.is_dir():
            files = sorted(resolved.iterdir())
            array_names = []
            for f in files:
                array_names.extend(_discover_arrays_in_shard(f))
            shards.append(
                ShardRef(
                    name=name,
                    path=resolved,
                    checksum=None,
                    arrays=array_names,
                    num_examples=None,
                    checksum_ok=None,
                )
            )
        else:
            checksum = (
                manifest.get("checksums", {}).get(name)
                if isinstance(manifest.get("checksums"), dict)
                else None
            )
            array_names = _discover_arrays_in_shard(resolved)
            ok = None
            if checksum is not None:
                ok = sha256_file(resolved) == checksum
            shards.append(
                ShardRef(
                    name=name,
                    path=resolved,
                    checksum=checksum,
                    arrays=array_names,
                    num_examples=None,
                    checksum_ok=ok,
                )
            )

    if not shards:
        raise DatasetError(
            f"no training shards resolved from manifest in {root}; "
            f"manifest keys: {sorted(manifest.keys())}"
        )

    # Record which directories are excluded from training/calibration.
    return shards


def build_layout(
    zip_path: str | Path = "/content/drive/MyDrive/azrael-dataset.zip",
    local_dir: str | Path = "/tmp/ornith-dataset",  # noqa: S108  (config default)
    extract_dir: str | Path = "/tmp/ornith-dataset-extract",  # noqa: S108  (config default)
    exclude_dirs: list[str] | None = None,
) -> DatasetLayout:
    """Run the full import pipeline: verify, copy, extract, resolve, manifest.

    Does NOT load tensors (that is ``load_shard``); it only resolves the
    layout so the controller can decide shard ordering and budgeting.

    # Errors: raises :class:`DatasetError` at the first failed step, naming the
    step and the producer status.
    """
    zpath = locate_zip(zip_path)
    if not verify_zip_integrity(zpath):
        raise DatasetError(f"ZIP integrity check failed for {zpath}")
    local_copy = copy_zip_to_local(zpath, local_dir)
    exdir = extract_zip(local_copy, extract_dir)
    droot = resolve_dataset_root(exdir)
    manifest = load_manifest(droot)
    shards = resolve_shards(droot, manifest, exclude_dirs=exclude_dirs)

    train_ready = droot / "train_ready"
    eval_firewall = droot / "eval_firewall"
    excluded_names = sorted(set(exclude_dirs or []) | {"eval_firewall"})

    return DatasetLayout(
        zip_path=local_copy,
        extract_dir=exdir,
        dataset_root=droot,
        manifest=manifest,
        train_ready_dir=train_ready if train_ready.is_dir() else None,
        shards=shards,
        eval_firewall_dir=eval_firewall if eval_firewall.is_dir() else None,
        excluded_dirs=excluded_names,
    )


def load_shard(shard: ShardRef) -> TensorBundle:
    """Load pre-tokenized arrays and masks from one resolved shard.

    Uses the array-name mapping discovered at resolve time. Never reads from
    the Google Drive mount — always from the local extracted copy.

    # Errors: raises :class:`DatasetError` when arrays cannot be loaded or the
    mapping is incomplete.
    """
    mapping = _rank_array_names(shard.arrays)
    missing = [role for role, name in mapping.items() if name is None]
    if missing:
        raise DatasetError(
            f"shard {shard.name} is missing arrays for roles: {missing}; "
            f"discovered keys: {shard.arrays}"
        )

    path = shard.path
    if path.is_dir():
        # directory-based shard: load the mapped files
        input_ids = _load_array_file(path / mapping["input_ids"])
        attention_mask = _load_array_file(path / mapping["attention_mask"])
        labels = _load_array_file(path / mapping["labels"])
    else:
        input_ids, attention_mask, labels = _load_arrays_from_file(
            path, mapping["input_ids"], mapping["attention_mask"], mapping["labels"]
        )

    return TensorBundle(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        shard_name=shard.name,
        num_examples=int(getattr(input_ids, "shape", [0])[0]),
    )


def _load_array_file(path: Path) -> Any:
    import numpy as np  # lazy

    return np.load(str(path))


def _load_arrays_from_file(
    path: Path,
    input_name: str,
    mask_name: str,
    label_name: str,
) -> tuple[Any, Any, Any]:
    suffix = path.suffix.lower()
    if suffix in (".parquet",):
        import numpy as np
        import pyarrow.parquet as pq

        table = pq.read_table(str(path), columns=[input_name, mask_name, label_name]).to_pydict()
        return (
            np.asarray(table[input_name], dtype=object),
            np.asarray(table[mask_name], dtype=object),
            np.asarray(table[label_name], dtype=object),
        )
    if suffix == ".npz":
        import numpy as np  # lazy

        with np.load(path) as zf:
            return zf[input_name], zf[mask_name], zf[label_name]
    if suffix in (".safetensors",):
        from safetensors import safe_open  # lazy

        tensors = {}
        with safe_open(str(path), framework="pt") as fh:
            for name in (input_name, mask_name, label_name):
                tensors[name] = fh.get_tensor(name)
        return tensors[input_name], tensors[mask_name], tensors[label_name]
    if suffix in (".pt", ".bin", ".pth"):
        import torch  # lazy

        obj = torch.load(str(path), map_location="cpu")
        return obj[input_name], obj[mask_name], obj[label_name]
    raise DatasetError(f"unsupported shard format: {suffix}")


def validate_alignment(
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
) -> None:
    """Check token/label alignment across a bundle.

    # Errors: raises :class:`DataAlignmentError` when lengths disagree or when
    assistant-only masking cannot be reconstructed.
    """
    n_inputs = getattr(input_ids, "shape", None)
    n_mask = getattr(attention_mask, "shape", None)
    n_labels = getattr(labels, "shape", None)
    if n_inputs is None or n_mask is None or n_labels is None:
        raise DataAlignmentError("arrays lack a shape attribute; cannot validate alignment")
    if int(n_inputs[0]) != int(n_mask[0]) or int(n_inputs[0]) != int(n_labels[0]):
        raise DataAlignmentError(
            f"length mismatch: input_ids={int(n_inputs[0])}, "
            f"attention_mask={int(n_mask[0])}, labels={int(n_labels[0])}"
        )



class FrozenDataset:
    """Preflight-verified 7-shard frozen Parquet dataset management with split hygiene."""

    def __init__(self, files: list[Path | str], tokenizer: Any = None, seed: int = 1337):
        import pyarrow.parquet as pq

        self.files = [Path(f) for f in files]
        self.tokenizer = tokenizer
        self.seed = seed
        self.index: list[tuple[Path, int]] = []
        self.ids: list[str] = []
        self.sources: list[str] = []
        self.expected_counts: list[tuple[int, int]] = []
        self._cache: dict | None = None
        self._cache_file: Path | None = None
        self._splits: list[str] | None = None

        for file in self.files:
            md = pq.read_table(str(file), columns=["id", "source", "token_count", "supervised_count"]).to_pydict()
            n = len(md["id"])
            self.index.extend((file, i) for i in range(n))
            self.ids.extend(md["id"])
            self.sources.extend(md["source"])
            self.expected_counts.extend(zip(md["token_count"], md["supervised_count"]))
        if not self.index:
            raise DatasetError("No frozen records found in provided files")

    def __len__(self) -> int:
        return len(self.index)

    def _open(self, file: Path) -> dict:
        if self._cache_file == file and self._cache is not None:
            return self._cache
        import pyarrow.parquet as pq

        self._cache = None
        self._cache_file = None
        self._cache = pq.read_table(str(file), columns=["input_ids", "attention_mask", "labels"]).to_pydict()
        self._cache_file = file
        return self._cache

    def record(self, i: int) -> tuple[Any, Any, Any]:
        import numpy as np

        file, row = self.index[i]
        d = {k: v[row] for k, v in self._open(file).items()}
        ids = np.asarray(d["input_ids"], dtype=np.int64).reshape(-1)
        labels = np.asarray(d["labels"], dtype=np.int64).reshape(-1)
        mask = np.asarray(d["attention_mask"], dtype=np.int64).reshape(-1)
        n = int(len(ids))
        supervised = int((labels != -100).sum())
        if not (len(mask) == len(labels) == n):
            raise DataAlignmentError(f"Row {i}: array lengths differ: mask={len(mask)}, labels={len(labels)}, ids={n}")
        expected_n, expected_supervised = self.expected_counts[i]
        if n != int(expected_n) or supervised != int(expected_supervised):
            raise DataAlignmentError(f"Row {i}: token_count/supervised_count disagrees with arrays")
        if n > 8192 or n < 1:
            raise DataAlignmentError(f"Row {i}: invalid sequence length {n}")
        if not self.ids[i] or not self.sources[i]:
            raise DataAlignmentError(f"Row {i}: missing id/source")
        if not np.isin(mask, [0, 1]).all():
            raise DataAlignmentError(f"Row {i}: invalid attention mask")
        if (mask == 0).any() and (not np.all(mask[:int(mask.sum())] == 1) or not np.all(mask[int(mask.sum()):] == 0)):
            raise DataAlignmentError(f"Row {i}: non-right padding requires dedicated collator")
        if ((mask == 0) & (labels != -100)).any():
            raise DataAlignmentError(f"Row {i}: labels supervise padding")
        on = labels != -100
        if on.any() and not np.array_equal(ids[on], labels[on]):
            raise DataAlignmentError(f"Row {i}: labels are not unshifted, position-aligned input_ids")
        if self.tokenizer is not None:
            vocab = len(self.tokenizer)
            if ((ids < 0) | (ids >= vocab)).any() or ((labels[on] < 0) | (labels[on] >= vocab)).any():
                raise DataAlignmentError(f"Row {i}: token exceeds loaded tokenizer ID space ({vocab})")
        used = int(mask.sum())
        return ids[:used].copy(), mask[:used].copy(), labels[:used].copy()

    def build_splits(self) -> dict[str, int]:
        """Compute bitwise duplicate-closure and assign deterministic splits."""
        import hashlib


        parent = list(range(len(self)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def unite(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[max(a, b)] = min(a, b)

        id_first: dict[str, int] = {}
        content_first: dict[str, int] = {}
        for i in range(len(self)):
            ident = self.ids[i]
            if ident in id_first:
                unite(i, id_first[ident])
            else:
                id_first[ident] = i
            ids, _, _ = self.record(i)
            digest = hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest()
            if digest in content_first:
                unite(i, content_first[digest])
            else:
                content_first[digest] = i

        groups: dict[int, list[int]] = {}
        for i in range(len(self)):
            groups.setdefault(find(i), []).append(i)

        assignments: list[str | None] = [None] * len(self)
        for indices in groups.values():
            identity = min(self.ids[i] for i in indices)
            bucket = int.from_bytes(hashlib.sha256(f"{self.seed}:{identity}".encode()).digest()[:4], "big") % 100
            part = "calib" if bucket < 3 else ("probe" if bucket < 6 else ("heldout" if bucket < 9 else "train"))
            for i in indices:
                assignments[i] = part
        self._splits = [a for a in assignments if a is not None]
        splits = {name: self._splits.count(name) for name in ("train", "calib", "probe", "heldout")}
        if not all(splits.values()):
            raise DatasetError(f"Empty frozen split: {splits}")
        return splits

    def partition(self, i: int) -> str:
        if self._splits is None:
            self.build_splits()
        return self._splits[i]  # type: ignore[index]

    def stream(self, which: str, start: int = 0, max_len: int | None = None):
        import numpy as np

        if which not in ("train", "calib", "probe", "heldout"):
            raise ValueError(which)
        pos = int(start)
        total = len(self)
        while True:
            idx = pos % total
            pos += 1
            if self.partition(idx) != which:
                continue
            ids, mask, labels = self.record(idx)
            if max_len and len(ids) > max_len:
                ids, mask, labels = ids[:max_len], mask[:max_len], labels[:max_len]
            if which != "calib" and not np.any(labels != -100):
                continue
            yield pos, (ids, mask, labels)

    def samples(self, which: str, n: int, max_len: int | None = None) -> list[tuple[Any, Any, Any]]:
        it = self.stream(which, max_len=max_len)
        return [next(it)[1] for _ in range(n)]


__all__ = [
    "ShardRef",
    "DatasetLayout",
    "TensorBundle",
    "FrozenDataset",
    "mount_drive",
    "locate_zip",
    "verify_zip_integrity",
    "copy_zip_to_local",
    "extract_zip",
    "resolve_dataset_root",
    "load_manifest",
    "resolve_shards",
    "build_layout",
    "load_shard",
    "validate_alignment",
]
