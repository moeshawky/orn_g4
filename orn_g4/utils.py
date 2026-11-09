"""Small, side-effect-light utilities used across the campaign.

Nothing here imports torch or transformers at module import time, so this file
can be imported and unit-tested on any host (including this GPU-less one).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 with a ``Z`` suffix."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    """Return ``sha256:<hex>`` over the file's bytes.

    # Errors: raises OSError if the file cannot be read (propagated, not
    swallowed, so a missing artifact is never reported as "matches").
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def atomic_write_bytes(dest: Path | str, data: bytes) -> int:
    """Write ``data`` to ``dest`` atomically via sibling temp + os.replace.

    The temp lives in the same directory as ``dest`` so the rename is atomic on
    the target filesystem. On any failure before replace, ``dest`` is left
    byte-unchanged and no stray temp remains (unless another file occupies the
    temp path, which is left untouched rather than deleted).

    # Errors: raises OSError on write failure; ``dest`` is unchanged.
    """
    dest = Path(dest).resolve()
    parent = dest.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"destination directory does not exist: {parent}")
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
    except OSError:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return len(data)


def write_json_atomic(dest: Path | str, obj: Any) -> int:
    """Serialize ``obj`` to JSON and write atomically.

    Deterministic key order (``sort_keys``) so identical objects produce
    identical bytes — required for idempotent run manifests.
    """
    payload = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return atomic_write_bytes(dest, payload)


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON file.

    # Errors: raises FileNotFoundError / json.JSONDecodeError; not caught here
    so the caller decides whether a missing manifest means "fresh run" or
    "corrupt resume".
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (parents included) and return it as a resolved Path.

    Idempotent: existing directories are returned unchanged.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: Path | str, dst: Path | str) -> Path:
    """Copy a file to ``dst``, returning the destination path.

    # Errors: raises shutil.Error / OSError; not caught.
    """
    return Path(shutil.copy2(str(src), str(dst)))


def which(cmd: str) -> str | None:
    """Return the absolute path of an executable, or None.

    Used to detect `nvidia-smi`, `ffmpeg`, etc. without asserting hardware.
    """
    return shutil.which(cmd)


def run_command(
    argv: list[str],
    *,
    capture: bool = True,
    check: bool = False,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with stdout/stderr kept separate (no ``2>&1``).

    Preserves the producer exit code in ``returncode`` so callers can cite the
    actual producer status rather than a downstream filter's.

    # Errors: when ``check`` is True and the producer exits non-zero, raises
    ``subprocess.CalledProcessError`` whose ``returncode`` is the producer's.
    """
    return subprocess.run(  # noqa: S603  (controlled command wrapper; argv is caller-supplied)
        argv,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=capture,
        text=True,
        check=check,
    )


def free_vram_bytes() -> int | None:
    """Best-effort free CUDA VRAM in bytes, or None when CUDA is unavailable.

    Reads ``nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits``
    when the binary exists; returns None otherwise. Never raises for missing
    hardware — absence of hardware is information, not an error.
    """
    smi = which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = run_command(
            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    # First non-empty numeric line = free MiB of the first GPU.
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line) * (1 << 20)
    return None


def class_name(module_obj: object) -> str:
    """Return ``ClassName`` for an object or module, for arch-discovery logging."""
    return getattr(module_obj, "__class__", type(module_obj)).__name__


def typename(obj: object) -> str:
    """Return the fully-qualified-ish type name of an object."""
    t = type(obj)
    mod = getattr(t, "__module__", "") or ""
    name = getattr(t, "__qualname__", getattr(t, "__name__", "")) or ""
    return f"{mod}.{name}" if mod else name


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` on zero denominator (never raises)."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_bytes(num: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PiB"


__all__ = [
    "utc_now_iso",
    "sha256_file",
    "atomic_write_bytes",
    "write_json_atomic",
    "read_json",
    "ensure_dir",
    "copy_file",
    "which",
    "run_command",
    "free_vram_bytes",
    "class_name",
    "typename",
    "safe_divide",
    "format_bytes",
]
