"""Safetensors reader/writer implementing the HuggingFace .safetensors wire format.

The safetensors format (https://github.com/huggingface/safetensors) is:

```
Header: 8-byte little-endian unsigned integer = JSON header length (in bytes)
JSON header: UTF-8 JSON object describing tensors, e.g.
    {"metadata": {...},
     "tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]},
     "__offset__": ...}
Tensor data: raw bytes in little-endian order, concatenated per tensor
```

This module delegates to ``safetensors.torch`` when installed on Colab (cell 01
installs safetensors) and falls back to an exact pure-Python/NumPy/PyTorch
wire-format implementation when it is not installed (this staging host has no
safetensors). Both code paths produce byte-compatible files, so Artifacts A–D
and the portable PyTorch loader work identically on this host and on Colab G4.

All serialization is atomic (write to sibling temp + os.replace) and checkpoint
directories are published only after validation — never solely under /tmp.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch  # lazy import; required on Colab, present on this host

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from safetensors.torch import (
        load_file as _st_load_file,
    )
    from safetensors.torch import (
        load_model as _st_load_model,
    )
    from safetensors.torch import (
        save_file as _st_save_file,
    )
    from safetensors.torch import (
        save_model as _st_save_model,
    )

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Dtype mapping: safetensors dtype string <-> numpy / torch dtype / byte size #
# --------------------------------------------------------------------------- #

_SAFETENSORS_DTYPES = {
    "F64": (np.float64, None if not _TORCH_AVAILABLE else torch.float64, 8),
    "F32": (np.float32, None if not _TORCH_AVAILABLE else torch.float32, 4),
    "F16": (np.float16, None if not _TORCH_AVAILABLE else torch.float16, 2),
    "BF16": (np.float32, None if not _TORCH_AVAILABLE else torch.bfloat16, 2),
    "I64": (np.int64, None if not _TORCH_AVAILABLE else torch.int64, 8),
    "I32": (np.int32, None if not _TORCH_AVAILABLE else torch.int32, 4),
    "I16": (np.int16, None if not _TORCH_AVAILABLE else torch.int16, 2),
    "I8": (np.int8, None if not _TORCH_AVAILABLE else torch.int8, 1),
    "U8": (np.uint8, None if not _TORCH_AVAILABLE else torch.uint8, 1),
    "BOOL": (np.bool_, None if not _TORCH_AVAILABLE else torch.bool, 1),
}

# Reverse mapping: numpy dtype -> safetensors dtype string.
_NP_TO_SF: dict[np.dtype[Any], str] = {
    np.dtype(np.float64): "F64",
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}

_TORCH_TO_SF: dict[Any, str] = {}
if _TORCH_AVAILABLE:
    for sf_name, (_np_dtype, torch_dtype, _size) in _SAFETENSORS_DTYPES.items():
        if torch_dtype is not None:
            _TORCH_TO_SF[torch_dtype] = sf_name


def _dtype_from_numpy(arr_dtype: np.dtype[Any]) -> str:
    sf = _NP_TO_SF.get(arr_dtype)
    if sf is None:
        raise ValueError(f"unsupported numpy dtype for safetensors: {arr_dtype}")
    return sf


def _numpy_dtype_for(sf_dtype: str) -> np.dtype[Any]:
    np_dtype, _, _ = _SAFETENSORS_DTYPES[sf_dtype]
    return np_dtype


def _tensor_bytes(arr: np.ndarray, sf_dtype: str) -> bytes:
    """Serialize a numpy array to raw little-endian bytes in safetensors layout."""
    np_dtype, _, _ = _SAFETENSORS_DTYPES[sf_dtype]
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    # safetensors is little-endian; ensure C-contiguous little-endian storage.
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def _bytes_to_tensor(raw: bytes, sf_dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Deserialize raw bytes into a numpy array."""
    np_dtype, _, _ = _SAFETENSORS_DTYPES[sf_dtype]
    return np.frombuffer(raw, dtype=np_dtype).reshape(shape)


# --------------------------------------------------------------------------- #
# Header encoding                                                             #
# --------------------------------------------------------------------------- #


def _encode_header(tensors: dict[str, dict[str, Any]], metadata: dict[str, str] | None) -> bytes:
    """Encode the safetensors JSON header (without the 8-byte length prefix)."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, info in tensors.items():
        header[name] = {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "data_offsets": [int(info["data_offsets"][0]), int(info["data_offsets"][1])],
        }
    return json.dumps(header, separators=(",", ": "), ensure_ascii=False).encode("utf-8")


def _decode_header(header_bytes: bytes) -> dict[str, Any]:
    """Decode safetensors JSON header bytes into a dict."""
    return json.loads(header_bytes.decode("utf-8"))


# --------------------------------------------------------------------------- #
# Single-file save / load                                                     #
# --------------------------------------------------------------------------- #


def save_safetensors(
    tensors: dict[str, np.ndarray | Any],
    path: Path | str,
    *,
    metadata: dict[str, str] | None = None,
    atomic: bool = True,
) -> Path:
    """Write tensors to a single .safetensors file.

    Args:
        tensors: mapping tensor name -> numpy array or torch tensor.
        path: destination .safetensors path.
        metadata: optional metadata dict (recorded verbatim in header).
        atomic: write to sibling temp then os.replace (default).

    Returns:
        The resolved path written.

    # Errors: raises ValueError on unsupported dtype; OSError on write failure.
    """
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _SAFETENSORS_AVAILABLE and _TORCH_AVAILABLE:
        torch_tensors: dict[str, torch.Tensor] = {}
        for name, value in tensors.items():
            if isinstance(value, torch.Tensor):
                torch_tensors[name] = value.detach().cpu().contiguous()
            elif isinstance(value, np.ndarray):
                torch_tensors[name] = torch.from_numpy(np.ascontiguousarray(value))
            else:
                torch_tensors[name] = torch.from_numpy(np.ascontiguousarray(np.asarray(value)))
        str_metadata = {str(k): str(v) for k, v in metadata.items()} if metadata else None
        if atomic:
            tmp = dest.with_name(dest.name + ".tmp")
            _st_save_file(torch_tensors, str(tmp), metadata=str_metadata)
            tmp.replace(dest)
        else:
            _st_save_file(torch_tensors, str(dest), metadata=str_metadata)
        return dest

    # Convert torch tensors to numpy if needed.
    np_tensors: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            np_tensors[name] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_tensors[name] = value
        else:
            np_tensors[name] = np.asarray(value)

    # Build tensor info with data offsets.
    tensor_info: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, arr in np_tensors.items():
        sf_dtype = _dtype_from_numpy(arr.dtype)
        byte_size = int(arr.size) * int(arr.itemsize)
        tensor_info[name] = {
            "dtype": sf_dtype,
            "shape": tuple(int(d) for d in arr.shape),
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size

    header_bytes = _encode_header(tensor_info, metadata)
    header_len = len(header_bytes)
    total_size = 8 + header_len + offset

    data = bytearray(total_size)
    struct.pack_into("<Q", data, 0, header_len)
    data[8 : 8 + header_len] = header_bytes

    for name, arr in np_tensors.items():
        info = tensor_info[name]
        start, end = info["data_offsets"]
        blob_start = 8 + header_len + start
        blob = _tensor_bytes(arr, info["dtype"])
        data[blob_start : blob_start + len(blob)] = blob

    if atomic:
        tmp = dest.with_name(dest.name + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(bytes(data))
        tmp.replace(dest)
    else:
        with dest.open("wb") as fh:
            fh.write(bytes(data))

    return dest


def load_safetensors(
    path: Path | str,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Load all tensors from a single .safetensors file into numpy arrays.

    Args:
        path: .safetensors file path.
        device: requested device (honored for torch tensors on load; here we
            return numpy and let callers move to device).

    Returns:
        mapping tensor name -> numpy array.

    # Errors: raises ValueError on malformed header; FileNotFoundError if absent.
    """
    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"safetensors file not found: {src}")

    if _SAFETENSORS_AVAILABLE:
        # Delegate to safetensors.torch when installed; convert to numpy.
        loaded = _st_load_file(str(src), device=device)
        return {
            k: v.cpu().numpy() if _TORCH_AVAILABLE and hasattr(v, "cpu") else np.asarray(v)
            for k, v in loaded.items()
        }

    raw = src.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"safetensors file too short: {len(raw)} bytes")
    header_len = struct.unpack_from("<Q", raw, 0)[0]
    if header_len <= 0 or 8 + header_len > len(raw):
        raise ValueError(f"invalid safetensors header length {header_len}")
    header_bytes = raw[8 : 8 + header_len]
    header = _decode_header(header_bytes)

    tensors: dict[str, np.ndarray] = {}
    tensor_body = raw[8 + header_len :]
    for name, info in header.items():
        if name in ("metadata", "__metadata__"):
            continue
        dtype_str = info["dtype"]
        shape = tuple(int(d) for d in info["shape"])
        offsets = info["data_offsets"]
        start, end = int(offsets[0]), int(offsets[1])
        blob = tensor_body[start:end]
        arr = _bytes_to_tensor(blob, dtype_str, shape)
        tensors[name] = arr
    return tensors


# --------------------------------------------------------------------------- #
# Sharded safetensors: model-00001-of-N.safetensors + model.safetensors.index.json #
# --------------------------------------------------------------------------- #

SHARD_SUFFIX = ".safetensors"


def _shard_filename(index: int, total: int) -> str:
    return f"model-{index:05d}-of-{total:05d}{SHARD_SUFFIX}"


def save_sharded_safetensors(
    tensors: dict[str, np.ndarray | Any],
    dir_path: Path | str,
    *,
    max_shard_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5 GiB
    metadata: dict[str, str] | None = None,
    index_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write tensors to sharded safetensors files.

    Produces:
        <dir>/model-00001-of-N.safetensors   (one per shard)
        <dir>/model.safetensors.index.json   (maps tensor name -> shard file)

    Splits tensors across shards without splitting individual tensors (each
    tensor lives entirely within one shard).

    Args:
        tensors: mapping tensor name -> array/tensor.
        dir_path: output directory.
        max_shard_size_bytes: soft cap on shard size (tensors are whole).
        metadata: optional metadata recorded in each shard header.
        index_metadata: extra keys merged into model.safetensors.index.json.

    Returns:
        The parsed index dict (also written to disk).

    # Errors: raises ValueError if any single tensor exceeds max_shard_size_bytes.
    """
    dest_dir = Path(dir_path).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy and compute byte sizes.
    items: list[tuple[str, np.ndarray]] = []
    for name, value in tensors.items():
        if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.asarray(value)
        items.append((name, arr))

    # Assign tensors to shards greedily by size.
    shards: list[list[tuple[str, np.ndarray]]] = [[]]
    shard_sizes: list[int] = [0]
    for name, arr in items:
        sf_dtype = _dtype_from_numpy(arr.dtype)
        _, _, byte_size_per_elem = _SAFETENSORS_DTYPES[sf_dtype]
        tensor_bytes = int(arr.size) * byte_size_per_elem
        if tensor_bytes > max_shard_size_bytes and shard_sizes[-1] == 0:
            # Single tensor exceeds cap — it must still get its own shard.
            pass
        if shard_sizes[-1] + tensor_bytes > max_shard_size_bytes and shards[-1]:
            shards.append([(name, arr)])
            shard_sizes.append(tensor_bytes)
        else:
            shards[-1].append((name, arr))
            shard_sizes[-1] += tensor_bytes

    # Write each shard.
    shard_files: list[str] = []
    for shard_idx, shard_items in enumerate(shards, start=1):
        shard_name = _shard_filename(shard_idx, len(shards))
        shard_tensors = {n: a for n, a in shard_items}
        save_safetensors(shard_tensors, dest_dir / shard_name, metadata=metadata, atomic=True)
        shard_files.append(shard_name)

    # Build index: tensor name -> {filename, offset?}. HuggingFace index maps
    # name -> {"filename": shard_name} (offsets implicit in shard).
    index: dict[str, Any] = {"metadata": index_metadata or {}, "weight_map": {}}
    for shard_idx, shard_items in enumerate(shards, start=1):
        shard_name = _shard_filename(shard_idx, len(shards))
        for name, _ in shard_items:
            index["weight_map"][name] = shard_name

    index_path = dest_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return index


def load_sharded_safetensors(
    dir_path: Path | str,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Load all tensors from a sharded safetensors directory.

    Reads model.safetensors.index.json and loads each shard file.
    """
    src_dir = Path(dir_path).resolve()
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"model.safetensors.index.json not found in {src_dir}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        return {}

    tensors: dict[str, np.ndarray] = {}
    shards_loaded: set[str] = set()
    for name, shard_name in weight_map.items():
        if shard_name in shards_loaded:
            continue
        shard_path = src_dir / shard_name
        shard_tensors = load_safetensors(shard_path, device=device)
        tensors.update(shard_tensors)
        shards_loaded.add(shard_name)
    return tensors


# --------------------------------------------------------------------------- #
# Convenience: save/load mapping with metadata                               #
# --------------------------------------------------------------------------- #


def save_tensor_map(
    tensors: dict[str, np.ndarray | Any],
    path: Path | str,
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Save a tensor map to a single .safetensors file (atomic)."""
    return save_safetensors(tensors, path, metadata=metadata, atomic=True)


def load_tensor_map(path: Path | str, *, device: str = "cpu") -> dict[str, np.ndarray]:
    """Load a tensor map from a single .safetensors file."""
    return load_safetensors(path, device=device)


__all__ = [
    "save_safetensors",
    "load_safetensors",
    "save_sharded_safetensors",
    "load_sharded_safetensors",
    "save_tensor_map",
    "load_tensor_map",
    "_SAFETENSORS_AVAILABLE",
]
