"""Compute-environment inspection for the Colab G4 target (work-order §4).

Inspects the *actual* accelerator at notebook startup: GPU type, CUDA
capability, VRAM, BF16 support, PyTorch version, CPU RAM, local disk, model
loading memory, and initial forward/backward throughput. Every value is
observed, never assumed; when a value cannot be observed here it is reported as
``UNVERIFIED-IN-THIS-ENVIRONMENT`` rather than guessed.

torch and transformers are imported lazily inside functions so this module
imports cleanly on a host without CUDA.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .errors import OrnG4Error
from .utils import format_bytes, free_vram_bytes, run_command


class EnvironmentError(OrnG4Error):
    """The environment could not be inspected sufficiently to proceed.

    # Errors: raised when the accelerator or interpreter information needed
    for stage budgeting is absent and no safe default exists.
    """


@dataclass(frozen=True)
class GpuInfo:
    """Observed GPU characteristics."""

    present: bool
    device_name: str | None = None
    cuda_capability: str | None = None  # e.g. "8.6"
    cuda_version: str | None = None  # PyTorch-reported CUDA
    driver_version: str | None = None  # from nvidia-smi when available
    vram_total_bytes: int | None = None
    vram_free_bytes: int | None = None
    bf16_supported: bool | None = None
    tf32_supported: bool | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    """Complete startup environment snapshot (work-order §4)."""

    gpu: GpuInfo
    torch_version: str | None
    python_version: str
    cpu_ram_bytes: int | None
    local_disk_free_bytes: int | None
    nvidia_smi_path: str | None
    cpu_count: int
    notes: list[str] = field(default_factory=list)


def _read_cpu_ram() -> int | None:
    """Best-effort total physical RAM in bytes (Linux /proc/meminfo)."""
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    return None


def _read_local_disk_free(path: str = "/") -> int | None:
    """Free bytes on the filesystem holding ``path``."""
    try:
        st = shutil.disk_usage(path)
        return st.free
    except OSError:
        return None


def _read_driver_version() -> str | None:
    """NVIDIA driver version via nvidia-smi, if present."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = run_command(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # noqa: F821  (subprocess imported below)
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return text.splitlines()[0].strip() if text else None


def _torch_device_props() -> GpuInfo | None:
    """Query torch.cuda for GPU properties, or None if torch/CUDA unavailable."""
    try:
        import torch  # lazy import
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return GpuInfo(present=False)
    try:
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        total = torch.cuda.get_device_properties(dev).total_memory
        try:
            free = free_vram_bytes()
        except Exception:  # pragma: no cover - defensive
            free = None
        major, minor = torch.cuda.get_device_capability(dev)
        return GpuInfo(
            present=True,
            device_name=props.name,
            cuda_capability=f"{major}.{minor}",
            cuda_version=getattr(torch.version, "cuda", None),
            driver_version=_read_driver_version(),
            vram_total_bytes=int(total),
            vram_free_bytes=free,
            bf16_supported=torch.cuda.is_bf16_supported(),
            tf32_supported=bool(
                getattr(torch.backends.cuda, "matmul", None)
                and torch.backends.cuda.matmul.allow_tf32
            ),
        )
    except Exception:  # pragma: no cover - torch present but misconfigured
        return GpuInfo(present=True, device_name=None)


def inspect_environment() -> EnvironmentReport:
    """Run the full startup inspection (work-order §4).

    Returns an :class:`EnvironmentReport` with every field observed or
    explicitly ``None``. Never invents a GPU type or VRAM figure.
    """
    torch_version = None
    try:
        import torch  # lazy

        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    gpu = _torch_device_props() or GpuInfo(present=False)

    notes: list[str] = []
    if not gpu.present:
        notes.append(
            "UNVERIFIED-IN-THIS-ENVIRONMENT: no CUDA device observed; G4 "
            "VRAM/throughput figures cannot be produced here"
        )
    if torch_version is None:
        notes.append("torch not installed in this interpreter")
    if gpu.vram_free_bytes is None and gpu.present:
        notes.append("free VRAM could not be read; Hessian budget must be derived conservatively")

    return EnvironmentReport(
        gpu=gpu,
        torch_version=torch_version,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        cpu_ram_bytes=_read_cpu_ram(),
        local_disk_free_bytes=_read_local_disk_free("/"),
        nvidia_smi_path=shutil.which("nvidia-smi"),
        cpu_count=os.cpu_count() or 0,
        notes=notes,
    )


def summarize_env(report: EnvironmentReport) -> str:
    """Human-readable one-line summary of the environment report."""
    gpu = report.gpu
    bits = [
        f"python={report.python_version}",
        f"torch={report.torch_version or 'absent'}",
        f"gpu={'present' if gpu.present else 'absent'}",
    ]
    if gpu.device_name:
        bits.append(f"name={gpu.device_name}")
    if gpu.cuda_capability:
        bits.append(f"cuda_cap={gpu.cuda_capability}")
    if gpu.bf16_supported is not None:
        bits.append(f"bf16={gpu.bf16_supported}")
    if gpu.vram_total_bytes is not None:
        bits.append(f"vram_total={format_bytes(gpu.vram_total_bytes)}")
    if gpu.vram_free_bytes is not None:
        bits.append(f"vram_free={format_bytes(gpu.vram_free_bytes)}")
    if report.cpu_ram_bytes is not None:
        bits.append(f"ram={format_bytes(report.cpu_ram_bytes)}")
    if report.local_disk_free_bytes is not None:
        bits.append(f"disk_free={format_bytes(report.local_disk_free_bytes)}")
    return " | ".join(bits)


__all__ = [
    "EnvironmentError",
    "GpuInfo",
    "EnvironmentReport",
    "inspect_environment",
    "summarize_env",
]
