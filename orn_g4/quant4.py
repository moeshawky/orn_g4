"""Protected INT4 codec ``act_sym_int4_g128`` for mixed-precision Ornith-1.5 (ADV-12.1).

Single canonical serialization (ticket 007, decision B; no AWQ / asymmetric
UINT4 / NF4 / TorchAO / custom-codec mixing):

* Signed symmetric INT4 codes ``q`` in ``[-8, 7]``, stored as ``uint4`` ``q + 8``.
* Packing: two codes per ``uint8``; low nibble = even column, high nibble = odd column.
* Group size 128 along the input dimension.
* BF16 ``group_scale`` of shape ``[out, ceil(in / group_size)]``.
* BF16 ``inv_channel_scale`` of shape ``[in]``.
* Deterministic dequantization ``W_hat = (q * group_scale) * inv_channel_scale``;
  no zero point.
* Activation scaling ``s_j = (a_j / mean(a)) ** alpha`` with ``a_j = sqrt(H_jj)``
  and ``alpha`` chosen from the five-point grid ``{0, 0.25, 0.5, 0.75, 1.0}`` by
  evaluating ``tr((W - W_hat) H (W - W_hat)^T)`` directly against the calibration
  Hessian ``H``. ``inv_channel_scale = 1 / s``.
* Recovery adapters: rank-16 FP32 trainable ``lora_A`` / ``lora_B`` (BF16 only at
  export). The packed codes and scales stay frozen during recovery.

torch is imported lazily / guarded so ``orn_g4`` remains importable without it.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

CODEC_NAME = "act_sym_int4_g128"
INT8_CODEC_NAME = "sym_int8_g128"
GROUP_SIZE = 128
#: Five-point activation-scaling exponent grid (frozen, ADV-12.1 decision B).
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
INT4_MIN = -8
INT4_MAX = 7
INT8_MIN = -127
INT8_MAX = 127
_SCALE_FLOOR = 1e-8


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise QTEAError("PyTorch is required for Protected4BitLinear")


def pack_uint4(q: Any) -> Any:
    """Pack 4-bit unsigned integers ``[0, 15]`` into ``uint8`` (low nibble = even column).

    # Errors: raises QTEAError if PyTorch is unavailable, ``q`` is not 2D, or a
    value lies outside ``[0, 15]``.
    """
    _require_torch()
    if q.ndim != 2:
        raise QTEAError(f"pack_uint4 expects 2D tensor, got shape {tuple(q.shape)}")
    if q.numel() and (int(q.min()) < 0 or int(q.max()) > 15):
        raise QTEAError("pack_uint4 values must lie in [0, 15]")
    rows, cols = q.shape
    q_u8 = q.to(torch.uint8)
    if cols % 2 != 0:
        pad = torch.zeros(rows, 1, dtype=torch.uint8, device=q_u8.device)
        q_u8 = torch.cat([q_u8, pad], dim=1)
    low = q_u8[:, 0::2] & 0x0F
    high = (q_u8[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous()


def unpack_uint4(packed: Any, cols: int) -> Any:
    """Unpack ``uint8`` nibbles back to ``uint8`` values in ``[0, 15]``.

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


def unpack_int4_signed(packed: Any, cols: int) -> Any:
    """Unpack to signed codes in ``[-8, 7]`` (stored value minus 8), as ``int8``."""
    return unpack_uint4(packed, cols).to(torch.int8) - 8


def dequantize_act_sym_int4(
    qweight: Any,
    group_scale: Any,
    inv_channel_scale: Any,
    cols: int,
    group_size: int = GROUP_SIZE,
) -> Any:
    """Deterministic dequantization ``(q * group_scale) * inv_channel_scale`` in FP32.

    The BF16 scales are widened to FP32 *before* the multiplications, so every
    device computes the same two FP32 products in the same order.

    # Errors: raises QTEAError on invalid tensor dimensions.
    """
    _require_torch()
    device = qweight.device
    q = unpack_int4_signed(qweight, cols).to(device=device, dtype=torch.float32)
    gs = group_scale.to(device=device, dtype=torch.float32).repeat_interleave(group_size, dim=1)[:, :cols]
    ics = inv_channel_scale.to(device=device, dtype=torch.float32).reshape(1, -1)
    if ics.shape[1] != cols:
        raise QTEAError(f"inv_channel_scale has {ics.shape[1]} entries, expected {cols}")
    return (q * gs) * ics


def _quantize_with_scale(
    weight: Any, channel_scale: Any, group_size: int
) -> Dict[str, Any]:
    """Quantize ``weight * channel_scale`` with BF16-rounded symmetric group scales."""
    rows, cols = weight.shape
    n_groups = math.ceil(cols / group_size)
    padded = n_groups * group_size
    w = weight * channel_scale.reshape(1, -1)
    if padded > cols:
        w = torch.cat([w, torch.zeros(rows, padded - cols, dtype=w.dtype, device=w.device)], dim=1)
    groups = w.reshape(rows, n_groups, group_size)
    amax = groups.abs().amax(dim=2)
    # Round the scale to BF16 first; the codes are computed against the exact
    # value the decoder will see.
    gscale = (amax / float(INT4_MAX)).clamp(min=_SCALE_FLOOR).to(torch.bfloat16)
    gscale_f = gscale.to(torch.float32)
    q = torch.round(groups / gscale_f.unsqueeze(2)).clamp(INT4_MIN, INT4_MAX)
    q_flat = q.reshape(rows, padded)[:, :cols].to(torch.int16) + 8
    inv_cs = (1.0 / channel_scale).to(torch.bfloat16)
    return {
        "qweight": pack_uint4(q_flat),
        "group_scale": gscale.contiguous(),
        "inv_channel_scale": inv_cs.contiguous(),
    }


def activation_channel_scale(hessian_diag: Any, alpha: float) -> Any:
    """``s_j = (a_j / mean(a)) ** alpha`` with ``a_j = sqrt(H_jj)``; all-ones for alpha 0."""
    _require_torch()
    a = hessian_diag.detach().float().clamp(min=_SCALE_FLOOR).sqrt()
    if alpha == 0.0:
        return torch.ones_like(a)
    s = (a / a.mean().clamp(min=_SCALE_FLOOR)).pow(float(alpha))
    return s.clamp(min=1e-4, max=1e4)


def _sum_squared_diff_fp64(weight: Any, dequantized: Any) -> float:
    """Compute ``||W - W_hat||_F^2`` in FP64 with bounded workspace memory."""
    rows, cols = weight.shape
    if rows * cols <= 1_000_000:
        delta = weight.double() - dequantized.double()
        return float((delta * delta).sum().item())
    row_chunk = max(1, min(rows, 512))
    total = torch.zeros((), device=weight.device, dtype=torch.float64)
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        delta = weight[start:end].double() - dequantized[start:end].double()
        total = total + (delta * delta).sum()
    return float(total.item())


def hessian_weighted_error(weight: Any, dequantized: Any, hessian: Any) -> float:
    """``tr((W - W_hat) H (W - W_hat)^T)`` in FP64 with row-chunked workspace for large weights."""
    _require_torch()
    rows, cols = weight.shape
    h = (
        hessian
        if (hessian.device == weight.device and hessian.dtype == torch.float64)
        else hessian.to(device=weight.device, dtype=torch.float64)
    )
    if rows * cols <= 1_000_000:
        delta = weight.double() - dequantized.double()
        return float(((delta @ h) * delta).sum().item())
    row_chunk = max(1, min(rows, 512))
    total = torch.zeros((), device=weight.device, dtype=torch.float64)
    for start in range(0, rows, row_chunk):
        end = min(rows, start + row_chunk)
        delta = weight[start:end].double() - dequantized[start:end].double()
        total = total + ((delta @ h) * delta).sum()
    return float(total.item())


def quantize_act_sym_int4(
    weight: Any,
    hessian: Optional[Any] = None,
    group_size: int = GROUP_SIZE,
    alphas: Sequence[float] = ALPHA_GRID,
) -> Dict[str, Any]:
    """Quantize a 2D weight into the canonical ``act_sym_int4_g128`` payload.

    The activation-scaling exponent is chosen deterministically from ``alphas``
    by the smallest Hessian-weighted reconstruction error (first minimum on
    ties). Without a Hessian the identity is used, so every alpha is equivalent
    and ``alpha = 0`` (plain per-group symmetric quantization) is selected.

    # Errors: raises QTEAError if PyTorch is unavailable, ``weight`` is not 2D,
    ``group_size <= 0``, ``alphas`` is empty, or ``hessian`` has the wrong shape.
    """
    _require_torch()
    if weight.ndim != 2:
        raise QTEAError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise QTEAError(f"group_size must be positive, got {group_size}")
    if len(alphas) == 0:
        raise QTEAError("alphas must be non-empty")
    rows, cols = weight.shape
    w = weight.detach().float()
    h_64: Any | None = None
    if hessian is None:
        diag = torch.ones(cols, dtype=torch.float32, device=w.device)
    else:
        if tuple(hessian.shape) != (cols, cols):
            raise QTEAError(f"hessian shape {tuple(hessian.shape)} != ({cols}, {cols})")
        h = hessian.detach()
        diag = torch.diagonal(h).to(device=w.device, dtype=torch.float32)
        h_64 = h.to(device=w.device, dtype=torch.float64)

    best: Optional[Dict[str, Any]] = None
    best_err = math.inf
    best_alpha = float(alphas[0])
    for alpha in alphas:
        cs = activation_channel_scale(diag, float(alpha))
        cand = _quantize_with_scale(w, cs, group_size)
        w_hat = dequantize_act_sym_int4(
            cand["qweight"], cand["group_scale"], cand["inv_channel_scale"], cols, group_size
        )
        err = (
            _sum_squared_diff_fp64(w, w_hat)
            if h_64 is None
            else hessian_weighted_error(w, w_hat, h_64)
        )
        if err < best_err:  # strict: first minimum wins ties
            best, best_err, best_alpha = cand, err, float(alpha)
    assert best is not None
    return {
        "codec": CODEC_NAME,
        "rows": int(rows),
        "cols": int(cols),
        "group_size": int(group_size),
        "alpha": best_alpha,
        "hessian_error": float(best_err),
        **best,
    }


def initialization_activation_weighted_svd(
    residual: Any,
    activation_scale: Any,
    rank: int,
) -> "tuple[Any, Any]":
    """Rank-``rank`` factors minimizing ``|| (R - B A) diag(a) ||_F``.

    ``activation_scale`` is the per-input-channel activation RMS ``a``. Returns
    ``(A, B)`` with ``A`` of shape ``[rank, in]`` and ``B`` of shape ``[out, rank]``
    in FP32, such that ``B @ A`` approximates ``R`` under the activation metric
    (the adapter scaling ``alpha / rank`` is applied by the caller).

    # Errors: raises QTEAError on non-positive rank or mismatched shapes.
    """
    _require_torch()
    if rank <= 0:
        raise QTEAError(f"rank must be positive, got {rank}")
    r32 = residual.detach().float()
    a = activation_scale.detach().float().to(r32.device).clamp(min=_SCALE_FLOOR)
    if r32.ndim != 2 or a.shape[0] != r32.shape[1]:
        raise QTEAError("residual / activation_scale shape mismatch")
    u, s, vh = torch.linalg.svd(r32 * a.unsqueeze(0), full_matrices=False)
    k = min(rank, int(s.shape[0]))
    s_sqrt = s[:k].sqrt()
    b = u[:, :k] * s_sqrt.unsqueeze(0)
    a_mat = (s_sqrt.unsqueeze(1) * vh[:k, :]) / a.unsqueeze(0)
    if k < rank:
        b = torch.cat([b, torch.zeros(b.shape[0], rank - k, device=b.device)], dim=1)
        a_mat = torch.cat([a_mat, torch.zeros(rank - k, a_mat.shape[1], device=a_mat.device)], dim=0)
    return a_mat.contiguous(), b.contiguous()


if _TORCH_AVAILABLE:

    class Protected4BitLinear(nn.Module):
        """Frozen ``act_sym_int4_g128`` projection with optional FP32 rank-r adapters."""

        codec: str = CODEC_NAME

        def __init__(
            self,
            in_features: int,
            out_features: int,
            group_size: int = GROUP_SIZE,
            bias: bool = False,
            device: Optional[Any] = None,
        ) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.group_size = int(group_size)
            self.lora_rank: int = 0
            self.lora_alpha: float = 0.0
            self.conditioning: Dict[str, Any] = {}

            packed_cols = math.ceil(self.in_features / 2)
            n_groups = math.ceil(self.in_features / self.group_size)

            self.register_buffer(
                "qweight",
                torch.zeros(self.out_features, packed_cols, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "group_scale",
                torch.ones(self.out_features, n_groups, dtype=torch.bfloat16, device=device),
            )
            self.register_buffer(
                "inv_channel_scale",
                torch.ones(self.in_features, dtype=torch.bfloat16, device=device),
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
            """Attach FP32 trainable rank-r adapters; packed codes and scales stay frozen.

            # Errors: raises QTEAError if ``rank <= 0`` or ``train_scales`` is
            requested (the BF16 group / channel scales are part of the frozen
            canonical serialization).
            """
            if rank <= 0:
                raise QTEAError(f"rank must be positive, got {rank}")
            if train_scales:
                raise QTEAError("act_sym_int4_g128 scales are frozen; train_scales is unsupported")
            device = self.qweight.device
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
                # Trainable adapters are FP32 even when serialized as BF16.
                if self.lora_A is None or tuple(self.lora_A.shape) != tuple(src_a.shape):
                    self.lora_A = nn.Parameter(
                        torch.empty(src_a.shape, dtype=torch.float32, device=dev), requires_grad=False
                    )
                if self.lora_B is None or tuple(self.lora_B.shape) != tuple(src_b.shape):
                    self.lora_B = nn.Parameter(
                        torch.empty(src_b.shape, dtype=torch.float32, device=dev), requires_grad=False
                    )
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
            group_size: int = GROUP_SIZE,
            device: Optional[Any] = None,
            hessian: Optional[Any] = None,
            alphas: Sequence[float] = ALPHA_GRID,
        ) -> "Protected4BitLinear":
            """Quantize an ``nn.Linear`` without mutating ``module``."""
            target_dev = torch.device(device) if device is not None else module.weight.device
            payload = quantize_act_sym_int4(
                module.weight.data.to(target_dev),
                hessian=None if hessian is None else hessian.to(target_dev),
                group_size=group_size,
                alphas=alphas,
            )
            layer = cls.from_payload(
                payload,
                bias=module.bias.data if module.bias is not None else None,
                device=target_dev,
            )
            layer.alpha_selected = float(payload["alpha"])  # diagnostic only, not serialized
            return layer

        @classmethod
        def from_payload(
            cls,
            payload: Dict[str, Any],
            bias: Optional[torch.Tensor] = None,
            device: Optional[Any] = None,
        ) -> "Protected4BitLinear":
            """Instantiate from a serialized payload dict.

            # Errors: raises QTEAError if the payload codec is not ``act_sym_int4_g128``.
            """
            codec = payload.get("codec", CODEC_NAME)
            if codec != CODEC_NAME:
                raise QTEAError(f"Unsupported INT4 codec {codec!r}; only {CODEC_NAME!r} is accepted")
            target_dev = torch.device(device) if device is not None else payload["qweight"].device
            layer = cls(
                in_features=int(payload["cols"]),
                out_features=int(payload["rows"]),
                group_size=int(payload["group_size"]),
                bias=bias is not None,
                device=target_dev,
            )
            layer.qweight.copy_(payload["qweight"].to(target_dev))
            layer.group_scale.copy_(payload["group_scale"].to(device=target_dev, dtype=torch.bfloat16))
            layer.inv_channel_scale.copy_(
                payload["inv_channel_scale"].to(device=target_dev, dtype=torch.bfloat16)
            )
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias.to(device=target_dev, dtype=torch.float32))
            return layer

        def to_layer_dict(self) -> Dict[str, Any]:
            """Convert buffers into the canonical payload dict."""
            return {
                "codec": self.codec,
                "rows": self.out_features,
                "cols": self.in_features,
                "group_size": self.group_size,
                "qweight": self.qweight,
                "group_scale": self.group_scale,
                "inv_channel_scale": self.inv_channel_scale,
            }

        def base_weight(self) -> torch.Tensor:
            """Dequantized frozen weight in FP32 (no adapter)."""
            return dequantize_act_sym_int4(
                self.qweight,
                self.group_scale,
                self.inv_channel_scale,
                self.in_features,
                self.group_size,
            )

        def effective_weight(self, dtype: Optional[Any] = None) -> torch.Tensor:
            """Reconstruct effective dense weight (base INT4 + optional FP32 adapter delta)."""
            w = self.base_weight()
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

    def quantize_symmetric_int8(
        weight: Any, group_size: int = GROUP_SIZE
    ) -> Dict[str, Any]:
        """Quantize a 2D weight matrix to symmetric group-128 INT8 with BF16 scales.

        # Errors: raises QTEAError if PyTorch is unavailable, ``weight`` is not 2D,
        or ``group_size <= 0``.
        """
        _require_torch()
        if weight.ndim != 2:
            raise QTEAError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
        if group_size <= 0:
            raise QTEAError(f"group_size must be positive, got {group_size}")
        rows, cols = weight.shape
        n_groups = math.ceil(cols / group_size)
        padded = n_groups * group_size
        w = weight.detach().float()
        if padded > cols:
            w = torch.cat([w, torch.zeros(rows, padded - cols, dtype=w.dtype, device=w.device)], dim=1)
        groups = w.reshape(rows, n_groups, group_size)
        amax = groups.abs().amax(dim=2)
        scale = (amax / float(INT8_MAX)).clamp(min=_SCALE_FLOOR).to(torch.bfloat16)
        scale_f = scale.to(torch.float32)
        q = torch.round(groups / scale_f.unsqueeze(2)).clamp(INT8_MIN, INT8_MAX)
        qweight = q.reshape(rows, padded)[:, :cols].to(torch.int8).contiguous()
        return {
            "codec": INT8_CODEC_NAME,
            "rows": int(rows),
            "cols": int(cols),
            "group_size": int(group_size),
            "qweight": qweight,
            "scale": scale.contiguous(),
        }

    def dequantize_symmetric_int8(
        qweight: Any,
        scale: Any,
        cols: int,
        group_size: int = GROUP_SIZE,
    ) -> Any:
        """Deterministic symmetric group-INT8 dequantization in FP32."""
        _require_torch()
        dev = qweight.device
        q = qweight.to(device=dev, dtype=torch.float32)
        s = scale.to(device=dev, dtype=torch.float32).repeat_interleave(group_size, dim=1)[:, :cols]
        return q * s

    def _str_to_dtype(name: str) -> torch.dtype:
        clean = str(name).replace("torch.", "")
        if clean == "bfloat16":
            return torch.bfloat16
        if clean == "float16":
            return torch.float16
        return torch.float32

    class QuantizedInt8Embedding(nn.Module):
        """Frozen symmetric group-128 INT8 embedding with BF16 group scales."""

        codec: str = INT8_CODEC_NAME

        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            group_size: int = GROUP_SIZE,
            padding_idx: Optional[int] = None,
            out_dtype: Any = "float32",
            device: Optional[Any] = None,
        ) -> None:
            super().__init__()
            self.num_embeddings = int(num_embeddings)
            self.embedding_dim = int(embedding_dim)
            self.group_size = int(group_size)
            self.padding_idx = padding_idx
            self.out_dtype_name = str(out_dtype).replace("torch.", "")
            n_groups = math.ceil(self.embedding_dim / self.group_size)
            self.register_buffer(
                "qweight",
                torch.zeros(self.num_embeddings, self.embedding_dim, dtype=torch.int8, device=device),
            )
            self.register_buffer(
                "scale",
                torch.ones(self.num_embeddings, n_groups, dtype=torch.bfloat16, device=device),
            )

        @classmethod
        def from_embedding(
            cls,
            module: nn.Embedding,
            group_size: int = GROUP_SIZE,
            device: Optional[Any] = None,
        ) -> "QuantizedInt8Embedding":
            """Quantize an ``nn.Embedding`` without mutating it."""
            target_dev = torch.device(device) if device is not None else module.weight.device
            payload = quantize_symmetric_int8(module.weight.data.to(target_dev), group_size=group_size)
            layer = cls(
                num_embeddings=int(payload["rows"]),
                embedding_dim=int(payload["cols"]),
                group_size=int(payload["group_size"]),
                padding_idx=module.padding_idx,
                out_dtype=module.weight.dtype,
                device=target_dev,
            )
            layer.qweight.copy_(payload["qweight"].to(target_dev))
            layer.scale.copy_(payload["scale"].to(device=target_dev, dtype=torch.bfloat16))
            return layer

        def effective_weight(self, dtype: Optional[Any] = None) -> torch.Tensor:
            w = dequantize_symmetric_int8(
                self.qweight, self.scale, self.embedding_dim, self.group_size
            )
            return w.to(dtype=dtype) if dtype is not None else w

        @property
        def weight(self) -> torch.Tensor:
            return self.effective_weight(dtype=_str_to_dtype(self.out_dtype_name))

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            q_rows = self.qweight[input_ids].to(torch.float32)
            s_rows = self.scale[input_ids].to(torch.float32)
            s_exp = s_rows.repeat_interleave(self.group_size, dim=-1)[..., : self.embedding_dim]
            out = q_rows * s_exp
            if self.padding_idx is not None:
                out = out.masked_fill((input_ids == self.padding_idx).unsqueeze(-1), 0.0)
            return out.to(dtype=_str_to_dtype(self.out_dtype_name))

    class QuantizedInt8Linear(nn.Module):
        """Frozen symmetric group-128 INT8 linear layer with BF16 group scales."""

        codec: str = INT8_CODEC_NAME

        def __init__(
            self,
            in_features: int,
            out_features: int,
            group_size: int = GROUP_SIZE,
            bias: bool = False,
            device: Optional[Any] = None,
        ) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.group_size = int(group_size)
            n_groups = math.ceil(self.in_features / self.group_size)
            self.register_buffer(
                "qweight",
                torch.zeros(self.out_features, self.in_features, dtype=torch.int8, device=device),
            )
            self.register_buffer(
                "scale",
                torch.ones(self.out_features, n_groups, dtype=torch.bfloat16, device=device),
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features, dtype=torch.float32, device=device),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

        @classmethod
        def from_linear(
            cls,
            module: nn.Linear,
            group_size: int = GROUP_SIZE,
            device: Optional[Any] = None,
        ) -> "QuantizedInt8Linear":
            """Quantize an ``nn.Linear`` (such as ``lm_head``) without mutating it."""
            target_dev = torch.device(device) if device is not None else module.weight.device
            payload = quantize_symmetric_int8(module.weight.data.to(target_dev), group_size=group_size)
            layer = cls(
                in_features=int(payload["cols"]),
                out_features=int(payload["rows"]),
                group_size=int(payload["group_size"]),
                bias=module.bias is not None,
                device=target_dev,
            )
            layer.qweight.copy_(payload["qweight"].to(target_dev))
            layer.scale.copy_(payload["scale"].to(device=target_dev, dtype=torch.bfloat16))
            if module.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(module.bias.data.to(device=target_dev, dtype=torch.float32))
            return layer

        def effective_weight(self, dtype: Optional[Any] = None) -> torch.Tensor:
            w = dequantize_symmetric_int8(
                self.qweight, self.scale, self.in_features, self.group_size
            )
            return w.to(dtype=dtype) if dtype is not None else w

        @property
        def weight(self) -> torch.Tensor:
            return self.effective_weight()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            w = self.effective_weight(dtype=x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
            return F.linear(x, w, bias)

    def _set_by_dotted_name(root: nn.Module, name: str, replacement: nn.Module) -> None:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)

    def convert_vocab_int8(
        model: Any,
        *,
        group_size: int = GROUP_SIZE,
        quantize_embed: bool = True,
        quantize_lm_head: bool = True,
        inplace: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Quantize ``embed_tokens`` and/or ``lm_head`` to symmetric group-128 INT8.

        Preserves weight tying between ``embed_tokens`` and ``lm_head`` by
        aliasing the converted ``qweight`` and ``scale`` buffers when the
        source tensors shared storage.

        # Errors: raises QTEAError when PyTorch is unavailable or target
        vocabulary modules cannot be located / converted.
        """
        _require_torch()
        work = model if inplace else copy.deepcopy(model)

        embed_entry: Optional[Tuple[str, nn.Embedding]] = None
        lm_entry: Optional[Tuple[str, nn.Linear]] = None
        for name, mod in work.named_modules():
            leaf = name.split(".")[-1]
            if leaf == "embed_tokens" and isinstance(mod, nn.Embedding):
                embed_entry = (name, mod)
            elif leaf == "lm_head" and isinstance(mod, nn.Linear):
                lm_entry = (name, mod)

        tied_src = bool(
            embed_entry is not None
            and lm_entry is not None
            and embed_entry[1].weight.data_ptr() != 0
            and embed_entry[1].weight.data_ptr() == lm_entry[1].weight.data_ptr()
        )

        vocab_meta: Dict[str, Dict[str, Any]] = {}
        emb_q: Optional[QuantizedInt8Embedding] = None

        if quantize_embed and embed_entry is not None:
            ename, emod = embed_entry
            emb_q = QuantizedInt8Embedding.from_embedding(emod, group_size=group_size)
            _set_by_dotted_name(work, ename, emb_q)
            vocab_meta[ename] = {
                "kind": "embedding",
                "codec": INT8_CODEC_NAME,
                "num_embeddings": emb_q.num_embeddings,
                "embedding_dim": emb_q.embedding_dim,
                "group_size": emb_q.group_size,
                "padding_idx": emb_q.padding_idx,
                "out_dtype": emb_q.out_dtype_name,
            }

        if quantize_lm_head and lm_entry is not None:
            lname, lmod = lm_entry
            if tied_src and emb_q is not None and lmod.weight.shape == emb_q.qweight.shape:
                lm_q = QuantizedInt8Linear(
                    in_features=int(lmod.in_features),
                    out_features=int(lmod.out_features),
                    group_size=group_size,
                    bias=lmod.bias is not None,
                    device=emb_q.qweight.device,
                )
                lm_q.qweight = emb_q.qweight
                lm_q.scale = emb_q.scale
                if lmod.bias is not None and lm_q.bias is not None:
                    lm_q.bias.data.copy_(lmod.bias.data.to(device=emb_q.qweight.device, dtype=torch.float32))
            else:
                lm_q = QuantizedInt8Linear.from_linear(lmod, group_size=group_size)
            _set_by_dotted_name(work, lname, lm_q)
            vocab_meta[lname] = {
                "kind": "linear",
                "codec": INT8_CODEC_NAME,
                "in_features": lm_q.in_features,
                "out_features": lm_q.out_features,
                "group_size": lm_q.group_size,
                "has_bias": lm_q.bias is not None,
            }

        if not vocab_meta:
            for name, mod in work.named_modules():
                if isinstance(mod, QuantizedInt8Embedding):
                    vocab_meta[name] = {
                        "kind": "embedding",
                        "codec": INT8_CODEC_NAME,
                        "num_embeddings": mod.num_embeddings,
                        "embedding_dim": mod.embedding_dim,
                        "group_size": mod.group_size,
                        "padding_idx": mod.padding_idx,
                        "out_dtype": mod.out_dtype_name,
                    }
                elif isinstance(mod, QuantizedInt8Linear):
                    vocab_meta[name] = {
                        "kind": "linear",
                        "codec": INT8_CODEC_NAME,
                        "in_features": mod.in_features,
                        "out_features": mod.out_features,
                        "group_size": mod.group_size,
                        "has_bias": mod.bias is not None,
                    }

        return work, {
            "codec": INT8_CODEC_NAME,
            "group_size": int(group_size),
            "tied_vocab": tied_src,
            "modules": vocab_meta,
        }

    def measure_vocab_int8_delta_ce(
        ref_model: Any,
        int8_model: Any,
        batches: Sequence[Dict[str, Any]],
        *,
        max_regression: float = 0.05,
    ) -> Dict[str, Any]:
        """Measure token-weighted causal CE before/after INT8 vocab quantization and gate it.

        # Errors: raises QTEAError if no supervised tokens are evaluated, or
        ``QualityGateError`` if ``post_int8_ce - ref_ce > max_regression``.
        """
        from .gates import gate_ce_quality

        _require_torch()
        from .lora import _extract_backbone_and_head, chunked_causal_ce_loss

        def _call_with_no_cache(target: Any, kwargs: dict[str, Any]) -> Any:
            try:
                return target(**kwargs, use_cache=False)
            except TypeError:
                return target(**kwargs)

        def _eval_ce(model: Any) -> float:
            was_training = bool(getattr(model, "training", False))
            model.eval()
            total_nll = 0.0
            total_tok = 0
            backbone, lm_head = _extract_backbone_and_head(model)
            try:
                with torch.no_grad():
                    for batch in batches:
                        fwd: dict[str, Any] = {"input_ids": batch["input_ids"]}
                        if "attention_mask" in batch:
                            fwd["attention_mask"] = batch["attention_mask"]
                        n_valid = int(batch["labels"][:, 1:].ne(-100).sum().item())
                        if n_valid == 0:
                            continue
                        if backbone is not None and lm_head is not None:
                            b_out = _call_with_no_cache(backbone, fwd)
                            last_hidden = getattr(
                                b_out,
                                "last_hidden_state",
                                b_out[0] if isinstance(b_out, tuple) else b_out,
                            )
                            nll, n_tok = chunked_causal_ce_loss(
                                last_hidden,
                                lm_head,
                                batch["labels"].to(last_hidden.device),
                                chunk_size=256,
                                shift_labels=True,
                                grad_checkpoint=False,
                            )
                            total_nll += float(nll.item())
                            total_tok += int(n_tok)
                            continue
                        out = _call_with_no_cache(model, fwd)
                        logits = out.logits if hasattr(out, "logits") else out
                        labels = batch["labels"].to(logits.device)
                        shift_logits = logits[:, :-1, :]
                        shift_labels = labels[:, 1:]
                        seq_len = shift_logits.shape[1]
                        for start in range(0, seq_len, 256):
                            end = min(seq_len, start + 256)
                            chunk_lbl = shift_labels[:, start:end].reshape(-1)
                            if not bool(chunk_lbl.ne(-100).any().item()):
                                continue
                            chunk_log = shift_logits[:, start:end, :].float().reshape(-1, shift_logits.shape[-1])
                            nll = F.cross_entropy(chunk_log, chunk_lbl, ignore_index=-100, reduction="sum")
                            total_nll += float(nll.item())
                        total_tok += n_valid
            finally:
                if was_training:
                    model.train()
            if total_tok == 0:
                raise QTEAError("No supervised tokens available to evaluate INT8 vocab CE gate")
            return total_nll / float(total_tok)

        ref_ce = _eval_ce(ref_model)
        int8_ce = _eval_ce(int8_model)
        delta_ce = gate_ce_quality(ref_ce, int8_ce, max_regression=max_regression)
        return {
            "ref_ce": ref_ce,
            "int8_ce": int8_ce,
            "delta_ce": delta_ce,
            "max_regression": float(max_regression),
            "passed": True,
        }

else:  # pragma: no cover
    Protected4BitLinear = None  # type: ignore[assignment,misc]
    QuantizedInt8Embedding = None  # type: ignore[assignment,misc]
    QuantizedInt8Linear = None  # type: ignore[assignment,misc]

    def convert_vocab_int8(*args: Any, **kwargs: Any) -> Any:
        _require_torch()

    def measure_vocab_int8_delta_ce(*args: Any, **kwargs: Any) -> Any:
        _require_torch()
