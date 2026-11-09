"""Core ternary representation and primitives (work-order §6).

Maintains four representations distinctly:
  * source floating-point weights      (W_source)
  * trainable master weights           (W_master, where necessary)
  * effective ternary codes            (ternary_codes ∈ {-1, 0, +1})
  * group scales                       (scale, positive, per group)

with grouping of ``group_size`` input features per output row and positive
group scales. ``W_effective = scale × ternary_codes``.

Pure numpy where possible so the mathematics are unit-testable on any host;
thin torch wrappers live in :mod:`orn_g4.ternary_ste` and
:mod:`orn_g4.ternary_gptq` and import torch lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GroupedTernary:
    """One projection's effective ternary representation.

    Attributes:
        codes: int8 array of shape (out_features, in_features) with values in
            {-1, 0, +1}.
        scales: float32 array of shape (out_features, n_groups) with strictly
            positive entries.
        group_size: number of input features per group (128 by mandate).
        in_features: number of input features.
        out_features: number of output features.
        n_groups: ceil(in_features / group_size).
        source_name: originating tensor name (for provenance).
    """

    codes: Any  # np.ndarray[int8] at runtime
    scales: Any  # np.ndarray[float32]
    group_size: int
    in_features: int
    out_features: int
    n_groups: int
    source_name: str = ""

    def group_for_column(self, col: int) -> int:
        """Return the group index for input column ``col``."""
        return col // self.group_size

    def effective_weight(self) -> Any:
        """Reconstruct W_effective = scale × codes (broadcast scales over groups)."""
        import numpy as np  # local import keeps this module importable

        codes = np.asarray(self.codes)
        scales = np.asarray(self.scales)
        # Broadcast (out, n_groups) scales to (out, in_features) by repeating
        # each group scale across its group_size columns.
        scale_full = np.repeat(scales, self.group_size, axis=1)[:, : self.in_features]
        return scale_full * codes

    def nonzero_fraction(self) -> float:
        """Fraction of codes that are non-zero (sparsity of the ternary code)."""
        import numpy as np

        codes = np.asarray(self.codes)
        return float(np.count_nonzero(codes) / codes.size)

    def effective_bits_per_weight(self) -> float:
        """Effective bits per weight: 2 bits/weight for codes plus 32-bit scale per group."""
        import numpy as np

        codes = np.asarray(self.codes)
        total_weights = codes.size
        code_bits = 2.0 * total_weights  # sign + zero
        scale_bits = 32.0 * self.n_groups * self.out_features
        return (code_bits + scale_bits) / total_weights


def group_scales(
    weight: Any,
    group_size: int = 128,
) -> Any:
    """Compute positive group scales for a weight matrix.

    Partitions each output row into groups of ``group_size`` input features and
    sets scale[o, g] = mean(|W[o, group_cols]|) (strictly positive; a zero
    group is nudged to a tiny positive value so division is safe).

    Returns a float32 array of shape (out_features, n_groups).

    # Errors: raises ValueError when ``weight`` is not 2-D or ``group_size`` < 1.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {w.ndim}-D")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    out_features, in_features = w.shape
    n_groups = (in_features + group_size - 1) // group_size
    scales = np.zeros((out_features, n_groups), dtype=np.float32)
    for g in range(n_groups):
        lo = g * group_size
        hi = min((g + 1) * group_size, in_features)
        group = w[:, lo:hi]
        s = np.mean(np.abs(group), axis=1)
        # Nudge zero groups to a tiny positive scale (positive group scales mandated).
        s = np.where(s <= 0, 1e-6, s)
        scales[:, g] = s
    return scales


def quantize_group_row(
    row: Any,
    scales_row: Any,
    group_size: int = 128,
) -> Any:
    """Quantize one output row to ternary codes given per-group scales.

    codes[o, j] = sign(w[o, j]) when |w[o, j]| >= 0.5 * scale[o, group(j)],
    else 0. This is the nearest-ternary assignment for a fixed scale under
    squared error with a dead-zone quantizer.

    Returns int8 array of shape (in_features,).
    """
    import numpy as np

    w = np.asarray(row, dtype=np.float32)
    scales = np.asarray(scales_row, dtype=np.float32)
    in_features = w.shape[0]
    n_groups = (in_features + group_size - 1) // group_size
    codes = np.zeros(in_features, dtype=np.int8)
    for g in range(n_groups):
        lo = g * group_size
        hi = min((g + 1) * group_size, in_features)
        seg = w[lo:hi]
        s = scales[g]
        # dead-zone: zero out small magnitudes relative to the group scale
        codes[lo:hi] = np.where(np.abs(seg) >= 0.5 * s, np.sign(seg).astype(np.int8), 0)
    return codes


def hard_ternary_forward(
    x: Any,
    grouped: GroupedTernary,
) -> Any:
    """Compute x @ W_effective where W_effective = scale × codes.

    This is the inference path: hard ternary weights, no gradient. Used both for
    the STE control forward and for the fresh-process reload parity test.

    # Errors: raises ValueError on shape mismatch.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    w_eff = grouped.effective_weight()
    if x.shape[-1] != grouped.in_features:
        raise ValueError(f"x last dim {x.shape[-1]} != grouped.in_features {grouped.in_features}")
    return x @ w_eff.T


def differentiable_effective_weight(
    codes: Any,
    scales_param: Any,
    group_size: int,
    in_features: int,
) -> Any:
    """Compute W_effective = scale × codes with scales as a torch parameter.

    Codes are fixed (not differentiable); scales_param is an nn.Parameter so the
    result is differentiable w.r.t. scales. Used by fixed-code scale distillation
    where only scales are optimized.

    # Errors: raises ValueError on shape mismatch.
    """
    import numpy as np
    import torch

    codes_t = torch.as_tensor(np.asarray(codes), dtype=scales_param.dtype)
    scale_full = torch.repeat_interleave(scales_param, group_size, dim=1)[:, :in_features]
    if scale_full.shape[1] != codes_t.shape[1]:
        raise ValueError(f"broadcast scales {scale_full.shape} do not match codes {codes_t.shape}")
    return scale_full * codes_t


def pack_ternary(grouped: GroupedTernary) -> bytes:
    """Pack codes + scales into a deterministic byte blob (round-trip testable).

    Layout (little-endian):
      magic (4 bytes) "TERN", version u32, group_size u32,
      in_features u32, out_features u32, n_groups u32,
      codes: int8 (out * in), scales: float32 (out * n_groups).
    """
    import struct

    import numpy as np

    codes = np.asarray(grouped.codes, dtype=np.int8).tobytes()
    scales = np.asarray(grouped.scales, dtype=np.float32).tobytes()
    header = struct.pack(
        "<4s5I",
        b"TERN",
        1,
        grouped.group_size,
        grouped.in_features,
        grouped.out_features,
        grouped.n_groups,
    )
    return header + codes + scales


def unpack_ternary(blob: bytes, source_name: str = "") -> GroupedTernary:
    """Inverse of :func:`pack_ternary`; validates magic/version/size consistency."""
    import struct

    import numpy as np

    if len(blob) < 4 + 5 * 4:
        raise ValueError("blob too short for ternary header")
    magic, version, group_size, in_features, out_features, n_groups = struct.unpack(
        "<4s5I", blob[: 4 + 5 * 4]
    )
    if magic != b"TERN":
        raise ValueError(f"bad ternary magic {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported ternary version {version}")
    codes_size = out_features * in_features  # int8  -> 1 byte/element
    scales_size = out_features * n_groups  # float32 -> 4 bytes/element
    expected = (
        4
        + 5 * 4
        + codes_size * np.dtype(np.int8).itemsize
        + scales_size * np.dtype(np.float32).itemsize
    )
    if len(blob) != expected:
        raise ValueError(
            f"blob length {len(blob)} != expected {expected} "
            f"(codes={codes_size}, scales={scales_size})"
        )
    header_len = 4 + 5 * 4
    codes = np.frombuffer(blob[header_len : header_len + codes_size], dtype=np.int8).reshape(
        out_features, in_features
    )
    scales = np.frombuffer(blob[header_len + codes_size :], dtype=np.float32).reshape(
        out_features, n_groups
    )
    return GroupedTernary(
        codes=codes,
        scales=scales,
        group_size=int(group_size),
        in_features=int(in_features),
        out_features=int(out_features),
        n_groups=int(n_groups),
        source_name=source_name,
    )


def group_shapes(in_features: int, group_size: int) -> tuple[int, int]:
    """Calculate number of groups and padding for in_features and group_size."""
    import math

    groups = math.ceil(in_features / group_size)
    return groups, groups * group_size - in_features


def dequantize(codes: Any, scales: Any, in_features: int, validate: bool = False) -> Any:
    """Dequantize 3D ternary codes and 2D scales to 2D effective weight matrix."""
    import torch

    codes_t = codes if torch.is_tensor(codes) else torch.as_tensor(codes)
    scales_t = scales if torch.is_tensor(scales) else torch.as_tensor(scales)
    if validate:
        if codes_t.ndim != 3 or scales_t.shape != codes_t.shape[:2]:
            raise ValueError("invalid code/scale shapes")
        if not bool(torch.all((codes_t == -1) | (codes_t == 0) | (codes_t == 1))):
            raise ValueError("ternary codes outside {-1,0,+1}")
        if not bool(torch.all(scales_t > 0)):
            raise ValueError("nonpositive scales")
    return (codes_t.float() * scales_t.float().unsqueeze(-1)).flatten(1)[:, :in_features]


def ternarize_diag(
    w: Any,
    x: Any,
    group_size: int = 128,
    iterations: int = 4,
) -> tuple[Any, Any]:
    """Activation-diagonal weighted group ternarization; X is REAL sampled input."""
    import torch
    import torch.nn.functional as F

    w_t = w.detach().float() if hasattr(w, "detach") else torch.as_tensor(w, dtype=torch.float32)
    x_t = x.detach().float() if hasattr(x, "detach") else torch.as_tensor(x, dtype=torch.float32)
    if w_t.ndim != 2 or x_t.ndim != 2 or x_t.shape[1] != w_t.shape[1]:
        raise ValueError("weight / activation shapes disagree")
    out, inn = w_t.shape
    groups, pad = group_shapes(inn, group_size)
    wp = F.pad(w_t, (0, pad)).reshape(out, groups, group_size)
    h = F.pad(x_t.square().mean(0), (0, pad)).reshape(1, groups, group_size).clamp_min(1e-8)
    s = (wp.abs() * h).sum(-1) / h.expand_as(wp).sum(-1).clamp_min(1e-8)
    s = s.clamp_min(1e-8)
    for _ in range(iterations):
        c = torch.where(wp.abs() > s.unsqueeze(-1) / 2, wp.sign(), 0.0)
        s = ((wp * c * h).sum(-1) / ((c * c * h).sum(-1)).clamp_min(1e-8)).clamp_min(1e-8)
    c = torch.where(wp.abs() > s.unsqueeze(-1) / 2, wp.sign(), 0.0).to(torch.int8)
    return c, s.float()


def lowrank_error_init(
    w: Any,
    codes: Any,
    scales: Any,
    x: Any,
    rank: int,
    iters: int = 2,
) -> tuple[Any, Any]:
    """Activation-diagonal weighted low-rank factor initialization."""
    import torch

    w_t = w.float() if torch.is_tensor(w) else torch.as_tensor(w, dtype=torch.float32)
    x_t = x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
    R = (w_t - dequantize(codes, scales, w_t.shape[1])).float()
    h = x_t.square().mean(0).clamp_min(1e-6).sqrt()
    rw = R * h[None, :]
    r = max(1, min(int(rank), min(rw.shape) - 1))
    U, S, V = torch.svd_lowrank(rw, q=min(min(rw.shape), r + 8), niter=iters)
    B = U[:, :r] * S[:r][None, :]
    A = V[:, :r].T / h[None, :]
    return A.contiguous(), B.contiguous()


def relative_output_error(w: Any, codes: Any, scales: Any, x: Any, eps: float = 1e-10) -> float:
    """Relative output error ||X@W - X@W_eff||^2 / ||X@W||^2."""
    import torch
    import torch.nn.functional as F

    w_t = w if torch.is_tensor(w) else torch.as_tensor(w)
    x_t = x.to(w_t.device).float() if torch.is_tensor(x) else torch.as_tensor(x, device=w_t.device, dtype=torch.float32)
    Wq = dequantize(codes, scales, w_t.shape[1]).to(w_t.device)
    a = F.linear(x_t, w_t.float())
    b = F.linear(x_t, Wq)
    return float(((a - b).square().sum() / a.square().sum().clamp_min(eps)).item())


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TernaryLinear(nn.Module):
        """Reference trainable ternary linear; materializes dense W per forward.

        scales / low-rank factors are FP32 parameters; shadow is optional QAT state.
        This prioritizes correctness and portability, not edge inference speed.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            group_size: int = 128,
            bias: bool = False,
            rank: int = 0,
            qat: bool = False,
            device: str = "cpu",
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.group_size = int(group_size)
            self.groups, _ = group_shapes(self.in_features, self.group_size)
            self.register_buffer(
                "codes",
                torch.zeros(
                    self.out_features,
                    self.groups,
                    self.group_size,
                    device=device,
                    dtype=torch.int8,
                ),
            )
            self.scales = nn.Parameter(
                torch.ones(self.out_features, self.groups, device=device, dtype=torch.float32)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, device=device, dtype=torch.bfloat16))
            else:
                self.register_parameter("bias", None)
            self.rank = int(rank)
            if self.rank:
                self.lora_A = nn.Parameter(
                    torch.zeros(self.rank, self.in_features, device=device, dtype=torch.float32)
                )
                self.lora_B = nn.Parameter(
                    torch.zeros(self.out_features, self.rank, device=device, dtype=torch.float32)
                )
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)
            self.qat = bool(qat)
            if self.qat:
                self.shadow = nn.Parameter(
                    torch.zeros(self.out_features, self.in_features, device=device, dtype=torch.float32)
                )
            else:
                self.register_parameter("shadow", None)

        def effective_weight(self) -> torch.Tensor:
            if self.shadow is not None:
                padded = F.pad(self.shadow.float(), (0, self.groups * self.group_size - self.in_features))
                expanded = padded.reshape(self.out_features, self.groups, self.group_size)
                c = torch.where(
                    expanded.abs() > self.scales.float().unsqueeze(-1) / 2,
                    expanded.sign(),
                    0.0,
                )
                hard = (c * self.scales.float().unsqueeze(-1)).flatten(1)[:, : self.in_features]
                w = hard + (self.shadow - self.shadow.detach())
            else:
                w = dequantize(self.codes, self.scales.clamp_min(1e-8), self.in_features)
            if self.lora_A is not None:
                w = w + self.lora_B @ self.lora_A
            return w

        @property
        def weight(self) -> torch.Tensor:
            return self.effective_weight()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weight = self.effective_weight().to(dtype=x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
            return F.linear(x, weight, bias)

        @torch.no_grad()
        def freeze_shadow_to_codes(self) -> None:
            if self.shadow is None:
                return
            padded = F.pad(self.shadow.float(), (0, self.groups * self.group_size - self.in_features))
            arr = padded.reshape(self.out_features, self.groups, self.group_size)
            c = torch.where(
                arr.abs() > self.scales.float().unsqueeze(-1) / 2,
                arr.sign(),
                0.0,
            ).to(torch.int8)
            self.codes.copy_(c)

        @classmethod
        def from_linear(cls, linear: Any, codes: Any, scales: Any, rank: int = 0, qat: bool = False) -> TernaryLinear:
            if linear.weight.shape != (linear.out_features, linear.in_features):
                raise RuntimeError("unexpected linear dimensions")
            ret = cls(
                linear.in_features,
                linear.out_features,
                codes.shape[-1],
                bias=linear.bias is not None,
                rank=rank,
                qat=qat,
                device=linear.weight.device,
            )
            ret.codes.copy_(codes.to(ret.codes.device))
            ret.scales.data.copy_(scales.to(ret.scales.device))
            if linear.bias is not None:
                ret.bias.data.copy_(linear.bias.data.to(ret.bias.device))
            if qat:
                ret.shadow.data.copy_(
                    dequantize(codes.to(ret.codes.device), scales.to(ret.scales.device), linear.in_features)
                )
            return ret

except ImportError:  # pragma: no cover
    TernaryLinear = None  # type: ignore[assignment, misc]


__all__ = [
    "GroupedTernary",
    "group_scales",
    "group_shapes",
    "dequantize",
    "ternarize_diag",
    "lowrank_error_init",
    "relative_output_error",
    "TernaryLinear",
    "quantize_group_row",
    "hard_ternary_forward",
    "pack_ternary",
    "unpack_ternary",
]
