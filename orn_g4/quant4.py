"""Protected 4-bit group-wise affine linear codec for mixed-precision Ornith-1.5.

Provides a deterministic, CPU/CUDA-consistent, bit-packed 4-bit representation
(``affine_uint4_g128``: two 4-bit unsigned integers per ``uint8`` byte with
per-group FP32 ``scale`` and ``zero_point``) for projections excluded from
1.58-bit ternary quantization (``mlp.down_proj``, ``linear_attn.in_proj_a``,
``linear_attn.in_proj_b``, ``linear_attn.in_proj_z``).

This codec is explicitly named ``affine_uint4`` (not NF4) because it uses
uniform affine quantization ``w_hat = q * scale + zero_point`` over ``[0, 15]``.
torch is imported lazily / guarded so ``orn_g4`` remains importable without it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .errors import QTEAError

CODEC_NAME = "affine_uint4"


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise QTEAError("PyTorch is required for Protected4BitLinear")


def pack_uint4(q: Any) -> Any:
    """Pack 4-bit unsigned integers ``[0, 15]`` into ``uint8`` (2 values/byte).

    Args:
        q: integer tensor of shape ``(rows, cols)`` with values in ``[0, 15]``.

    Returns:
        ``uint8`` tensor of shape ``(rows, ceil(cols / 2))`` on the same device.

    # Errors: raises QTEAError if PyTorch is unavailable or ``q`` is not 2D.
    """
    _require_torch()
    if q.ndim != 2:
        raise QTEAError(f"pack_uint4 expects 2D tensor, got shape {tuple(q.shape)}")
    rows, cols = q.shape
    q_u8 = q.to(torch.uint8)
    if cols % 2 != 0:
        pad = torch.zeros(rows, 1, dtype=torch.uint8, device=q_u8.device)
        q_u8 = torch.cat([q_u8, pad], dim=1)
    low = q_u8[:, 0::2] & 0x0F
    high = (q_u8[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous()


def unpack_uint4(packed: Any, cols: int) -> Any:
    """Unpack ``uint8`` 4-bit nibbles back to ``uint8`` values in ``[0, 15]``.

    Args:
        packed: ``uint8`` tensor of shape ``(rows, ceil(cols / 2))``.
        cols: original number of columns before packing.

    Returns:
        ``uint8`` tensor of shape ``(rows, cols)`` on ``packed.device``.

    # Errors: raises QTEAError if PyTorch is unavailable or ``packed`` is invalid.
    """
    _require_torch()
    if packed.ndim != 2:
        raise QTEAError(f"unpack_uint4 expects 2D tensor, got shape {tuple(packed.shape)}")
    expected_packed_cols = math.ceil(cols / 2)
    if packed.shape[1] != expected_packed_cols:
        raise QTEAError(
            f"unpack_uint4 packed_cols mismatch: got {packed.shape[1]}, expected {expected_packed_cols}"
        )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.stack([low, high], dim=2).reshape(packed.shape[0], -1)
    return out[:, :cols].contiguous()


def quantize_affine_uint4(
    weight: Any,
    group_size: int = 128,
) -> Dict[str, Any]:
    """Quantize a 2D weight tensor using group-wise affine 4-bit quantization.

    Formula: ``w_hat = q * scale + zero_point`` where ``q in {0..15}``.

    # Errors: raises QTEAError if PyTorch is unavailable, ``weight`` is not 2D,
    or ``group_size <= 0``.
    """
    _require_torch()
    if weight.ndim != 2:
        raise QTEAError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise QTEAError(f"group_size must be positive, got {group_size}")

    rows, cols = weight.shape
    w = weight.detach().float()
    n_groups = math.ceil(cols / group_size)
    padded_cols = n_groups * group_size
    if padded_cols > cols:
        pad = torch.zeros(rows, padded_cols - cols, dtype=w.dtype, device=w.device)
        w_pad = torch.cat([w, pad], dim=1)
    else:
        w_pad = w

    w_groups = w_pad.reshape(rows, n_groups, group_size)
    w_min = w_groups.amin(dim=2)
    w_max = w_groups.amax(dim=2)
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zeros = w_min

    q = torch.round((w_groups - zeros.unsqueeze(2)) / scales.unsqueeze(2)).clamp(0, 15).to(torch.uint8)
    q_flat = q.reshape(rows, padded_cols)[:, :cols]
    qweight = pack_uint4(q_flat)

    return {
        "codec": CODEC_NAME,
        "rows": int(rows),
        "cols": int(cols),
        "group_size": int(group_size),
        "qweight": qweight,
        "scales": scales.to(torch.float32).contiguous(),
        "zeros": zeros.to(torch.float32).contiguous(),
    }


def dequantize_affine_uint4(
    qweight: Any,
    scales: Any,
    zeros: Any,
    cols: int,
    group_size: int,
) -> Any:
    """Dequantize packed 4-bit weights to FP32 on ``qweight.device`` (out-of-place).

    Differentiable with respect to ``scales`` and ``zeros`` when they are
    parameters with ``requires_grad=True``.

    # Errors: raises QTEAError on invalid tensor dimensions.
    """
    _require_torch()
    device = qweight.device
    q = unpack_uint4(qweight, cols).to(device=device, dtype=torch.float32)
    s = scales.to(device=device, dtype=torch.float32).repeat_interleave(group_size, dim=1)[:, :cols]
    z = zeros.to(device=device, dtype=torch.float32).repeat_interleave(group_size, dim=1)[:, :cols]
    return q * s + z


if _TORCH_AVAILABLE:

    class Protected4BitLinear(nn.Module):
        """Drop-in replacement for protected 4-bit linear projections.

        Stores packed ``uint8`` 4-bit nibbles (``qweight``) plus FP32 ``scales``
        and ``zeros``. Base codes remain frozen during recovery; optional
        rank-16 low-rank adapters (``lora_A``, ``lora_B``) can be attached for
        teacher-guided recovery.
        """

        codec: str = CODEC_NAME

        def __init__(
            self,
            in_features: int,
            out_features: int,
            group_size: int = 128,
            bias: bool = False,
            device: Optional[Any] = None,
        ) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.group_size = int(group_size)
            self.lora_rank: int = 0
            self.lora_alpha: float = 0.0

            packed_cols = math.ceil(self.in_features / 2)
            n_groups = math.ceil(self.in_features / self.group_size)

            self.register_buffer(
                "qweight",
                torch.zeros(self.out_features, packed_cols, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "scales",
                torch.ones(self.out_features, n_groups, dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "zeros",
                torch.zeros(self.out_features, n_groups, dtype=torch.float32, device=device),
            )
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, dtype=torch.float32, device=device)
                )
            else:
                self.register_parameter("bias", None)

        def enable_recovery_adapters(
            self,
            *,
            rank: int = 16,
            alpha: float = 32.0,
            train_scales: bool = False,
        ) -> None:
            """Attach trainable rank-r LoRA adapters while keeping 4-bit codes frozen.

            # Errors: raises QTEAError if ``rank <= 0``.
            """
            if rank <= 0:
                raise QTEAError(f"rank must be positive, got {rank}")
            device = self.qweight.device
            if train_scales:
                for bname in ("scales", "zeros"):
                    if bname in self._buffers:
                        buf = self._buffers.pop(bname)
                        self.register_parameter(
                            bname,
                            nn.Parameter(buf.detach().clone().to(device=device, dtype=torch.float32), requires_grad=True),
                        )

            self.lora_rank = int(rank)
            self.lora_alpha = float(alpha)
            lora_a = torch.empty(self.lora_rank, self.in_features, dtype=torch.float32, device=device)
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            lora_b = torch.zeros(self.out_features, self.lora_rank, dtype=torch.float32, device=device)
            self.lora_A = nn.Parameter(lora_a, requires_grad=True)
            self.lora_B = nn.Parameter(lora_b, requires_grad=True)

        def _load_from_state_dict(
            self,
            state_dict: Any,
            prefix: str,
            local_metadata: Any,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str],
        ) -> None:
            has_lora_a = (prefix + "lora_A") in state_dict
            has_lora_b = (prefix + "lora_B") in state_dict
            if has_lora_a and has_lora_b:
                src_a = state_dict[prefix + "lora_A"]
                src_b = state_dict[prefix + "lora_B"]
                dev = self.qweight.device
                if self.lora_A is None or tuple(self.lora_A.shape) != tuple(src_a.shape):
                    self.lora_A = nn.Parameter(torch.empty_like(src_a, device=dev), requires_grad=False)
                if self.lora_B is None or tuple(self.lora_B.shape) != tuple(src_b.shape):
                    self.lora_B = nn.Parameter(torch.empty_like(src_b, device=dev), requires_grad=False)
                self.lora_rank = int(src_a.shape[0])
                if self.lora_alpha == 0.0:
                    self.lora_alpha = float(self.lora_rank * 2)
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        @classmethod
        def from_linear(
            cls,
            module: nn.Linear,
            group_size: int = 128,
            device: Optional[Any] = None,
        ) -> "Protected4BitLinear":
            """Quantize an ``nn.Linear`` into ``Protected4BitLinear`` without mutating ``module``."""
            target_dev = torch.device(device) if device is not None else module.weight.device
            payload = quantize_affine_uint4(module.weight.data.to(target_dev), group_size=group_size)
            layer = cls(
                in_features=payload["cols"],
                out_features=payload["rows"],
                group_size=payload["group_size"],
                bias=module.bias is not None,
                device=target_dev,
            )
            layer.qweight.copy_(payload["qweight"].to(target_dev))
            layer.scales.copy_(payload["scales"].to(target_dev))
            layer.zeros.copy_(payload["zeros"].to(target_dev))
            if module.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(module.bias.data.to(device=target_dev, dtype=torch.float32))
            return layer

        @classmethod
        def from_payload(
            cls,
            payload: Dict[str, Any],
            bias: Optional[torch.Tensor] = None,
            device: Optional[Any] = None,
        ) -> "Protected4BitLinear":
            """Instantiate ``Protected4BitLinear`` from a serialized payload dict."""
            target_dev = torch.device(device) if device is not None else payload["qweight"].device
            layer = cls(
                in_features=int(payload["cols"]),
                out_features=int(payload["rows"]),
                group_size=int(payload["group_size"]),
                bias=bias is not None,
                device=target_dev,
            )
            layer.qweight.copy_(payload["qweight"].to(target_dev))
            layer.scales.copy_(payload["scales"].to(target_dev))
            layer.zeros.copy_(payload["zeros"].to(target_dev))
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias.to(device=target_dev, dtype=torch.float32))
            return layer

        def to_layer_dict(self) -> Dict[str, Any]:
            """Convert buffers into canonical 4-bit payload dict."""
            return {
                "codec": self.codec,
                "rows": self.out_features,
                "cols": self.in_features,
                "group_size": self.group_size,
                "qweight": self.qweight,
                "scales": self.scales,
                "zeros": self.zeros,
            }

        def effective_weight(self, dtype: Optional[Any] = None) -> torch.Tensor:
            """Reconstruct effective dense weight (base 4-bit + optional LoRA delta)."""
            w = dequantize_affine_uint4(
                self.qweight,
                self.scales,
                self.zeros,
                self.in_features,
                self.group_size,
            )
            if self.lora_A is not None and self.lora_B is not None and self.lora_rank > 0:
                scaling = float(self.lora_alpha) / float(self.lora_rank)
                w = w + (self.lora_B.to(w.dtype) @ self.lora_A.to(w.dtype)) * scaling
            return w.to(dtype=dtype) if dtype is not None else w

        @property
        def weight(self) -> torch.Tensor:
            return self.effective_weight()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.effective_weight(dtype=x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
            return F.linear(x, w, bias)

else:  # pragma: no cover
    Protected4BitLinear = None  # type: ignore[assignment,misc]
