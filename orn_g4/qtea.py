"""QTEA: Quantized Ternary Error Adaptation for Ornith-1.5-9B.

Upstream reference: Intelligent-Microsystems-Lab/QTEA @ e39bd8c0297e3427daeb404427c74cb6e202b683
Adapted for Ornith-1.5's hybrid architecture (Gated DeltaNet linear_attn + self_attn + mlp).

Core mathematical components:
  1. TernaryGroup: Iterative ternary fitting (ITF) of group alpha, beta, and dead-zone delta.
  2. Salient column selection: Top columns ranked by max_i (W_ij^2 * H_jj^2) carry 1:4 semi-sparse FP8 residuals.
  3. Pivoted LDL order: Non-salient columns ordered by symmetric LDL^T factorization of H.
  4. Rescale refinement: Alternating column rescale v_j with ternary reassignment.
  5. GPTQ sweep with error decay: Damped error propagation along each group.
  6. Base-3 packing: 5 ternary codes per byte (3^5 = 243 < 256) costing 1.6 bits/weight.
  7. Semi-sparse FP8 packing: 1:4 residual with 2-bit position indices and scaled FP8 values.
  8. Hybrid architecture targeting: Converts only linear projections in language_model.layers
     (linear_attn, self_attn, mlp), preserving multimodal vision encoders, embeddings, and norms.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

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


def _get_fp8_max() -> float:
    if _TORCH_AVAILABLE and hasattr(torch, "float8_e4m3fn"):
        return float(torch.finfo(torch.float8_e4m3fn).max)
    return 448.0  # IEEE 754 float8_e4m3fn max finite value


FP8_MAX = _get_fp8_max()

_POWERS_LIST = [1, 3, 9, 27, 81]


@dataclass
class QTEAConfig:
    """Configuration for QTEA quantization."""

    group_size: int = 128  # Columns per quantization group (== GPTQ block size)
    salient_ratio: float = 0.05  # Fraction of columns that carry an FP8 residual
    residual_group: int = 4  # 1:N semi-sparse pattern inside a salient column (N=4)
    delta_coef: float = 0.96  # Initial dead zone multiplier for mean |W|
    itf_iters: int = 1  # Iterative ternary fitting rounds per group
    rescale_iters: int = 2  # Column-wise rescale refinement rounds (0 disables)
    decay: float = 1.0  # Lambda parameter for GPTQ error decay (0 disables)
    damp: float = 0.007  # Hessian damping fraction of mean diagonal


def ternary_assign(centered: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Nearest ternary code for centered weights under dead-zone threshold delta."""
    codes = torch.zeros_like(centered)
    codes[centered > delta] = 1.0
    codes[centered < -delta] = -1.0
    return codes


class TernaryGroup:
    """Per-row ternary parameters (alpha, beta, delta) for one column group."""

    def __init__(self, delta_coef: float = 0.96, itf_iters: int = 1):
        self.delta_coef = delta_coef
        self.itf_iters = itf_iters
        self.alpha: Optional[torch.Tensor] = None  # (rows, 1) magnitude of +-1 levels
        self.beta: Optional[torch.Tensor] = None  # (rows, 1) quantization center
        self.delta: Optional[torch.Tensor] = None  # (rows, 1) dead-zone threshold

    def fit(self, W: torch.Tensor) -> "TernaryGroup":
        """Estimate group parameters from W of shape (rows, group_size)."""
        beta = W.mean(dim=1, keepdim=True)
        centered = W - beta
        magnitude = centered.abs()

        delta = self.delta_coef * magnitude.mean(dim=1, keepdim=True)
        alpha = torch.where(magnitude > delta, magnitude, torch.nan).nanmean(
            dim=1, keepdim=True
        )
        # Fallback for rows where all magnitudes <= delta
        fallback = magnitude.mean(dim=1, keepdim=True).clamp_min(1e-8)
        alpha = torch.where(torch.isnan(alpha), fallback, alpha)

        codes = ternary_assign(centered, delta)
        for _ in range(self.itf_iters):
            alpha, shift = self._least_squares(centered, codes, fallback=alpha)
            delta = 0.5 * alpha
            new_codes = ternary_assign(centered - shift, delta)
            if torch.equal(new_codes, codes):
                break
            codes = new_codes
        if self.itf_iters > 0:
            beta = beta + shift

        self.alpha, self.beta, self.delta = alpha, beta, delta
        return self

    @staticmethod
    def _least_squares(
        centered: torch.Tensor, codes: torch.Tensor, fallback: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row-wise solution of min_{a, s} || centered - s - a * codes ||^2."""
        n = centered.shape[1]
        cc = (codes * codes).sum(dim=1, keepdim=True)
        c1 = codes.sum(dim=1, keepdim=True)
        rc = (centered * codes).sum(dim=1, keepdim=True)
        r1 = centered.sum(dim=1, keepdim=True)

        det = cc * n - c1 * c1
        ok = det.abs() > 1e-8
        det_safe = torch.where(ok, det, torch.ones_like(det))

        alpha = torch.where(ok, (n * rc - c1 * r1) / det_safe, fallback)
        shift = torch.where(ok, (cc * r1 - c1 * rc) / det_safe, torch.zeros_like(r1))
        return alpha.abs().clamp_min(1e-8), shift


class QTEA:
    """Accumulates calibration Hessian for an nn.Linear layer and runs QTEA quantization."""

    def __init__(self, layer: Any, config: Optional[QTEAConfig] = None):
        if not _TORCH_AVAILABLE:
            raise QTEAError("PyTorch is required for QTEA quantization")
        self.layer = layer
        self.config = config or QTEAConfig()
        self.device = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.hessian = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

        if self.columns % self.config.group_size != 0:
            raise ValueError(
                f"{self.columns} columns is not a multiple of group size {self.config.group_size}"
            )
        if self.rows % self.config.residual_group != 0:
            raise ValueError(
                f"{self.rows} rows is not a multiple of residual group {self.config.residual_group}"
            )

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Fold one calibration batch into the running Hessian H = 2/N * sum X X^T."""
        sequences = inp.shape[0] if inp.dim() == 3 else 1
        inp_flat = inp.reshape(-1, inp.shape[-1]).t()
        self.hessian *= self.nsamples / (self.nsamples + sequences)
        self.nsamples += sequences
        scaled = math.sqrt(2.0 / self.nsamples) * inp_flat.float()
        self.hessian += scaled.matmul(scaled.t())

    def free(self) -> None:
        """Release Hessian memory."""
        self.hessian = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _select_salient(
        self, W: torch.Tensor, hessian_diag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank columns by max_i (W_ij^2 * H_jj^2) and keep top salient columns."""
        if self.config.salient_ratio <= 0:
            n_salient = 0
        else:
            blocks = self.columns // self.config.group_size
            n_salient = (
                max(1, round(blocks * self.config.salient_ratio))
                * self.config.group_size
            )
            n_salient = min(n_salient, self.columns)

        score = torch.max(W**2 * hessian_diag.reshape(1, -1) ** 2, dim=0).values
        order = torch.argsort(score, descending=True)
        return order[:n_salient], order[n_salient:]

    def _ldl_order(self, H: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        """Order non-salient columns by pivots of symmetric LDL^T of H."""
        if len(columns) == 0:
            return columns
        try:
            from scipy.linalg._decomp_ldl import _ldl_sanitize_ipiv
            from scipy.linalg.lapack import get_lapack_funcs

            sub = H[columns][:, columns].clone()
            diag = torch.arange(len(columns), device=sub.device)
            sub[diag, diag] += self.config.damp * torch.diag(sub).mean()
            matrix = sub.t().cpu().float().numpy()
            del sub

            sytrf, sytrf_lwork = get_lapack_funcs(("sytrf", "sytrf_lwork"), (matrix,))
            lwork = int(sytrf_lwork(matrix.shape[0], lower=True)[0])
            _, ipiv, info = sytrf(matrix, lwork=lwork, lower=True, overwrite_a=True)
            if info >= 0:
                swaps, _ = _ldl_sanitize_ipiv(ipiv, lower=True)
                state = np.arange(len(columns))
                for i in range(len(columns) - 1, -1, -1):
                    if swaps[i] != i:
                        state[[swaps[i], i]] = state[[i, swaps[i]]]
                permutation = torch.from_numpy(np.argsort(state)).to(columns.device)
                return columns[permutation]
        except Exception:
            pass
        return columns

    def _inverse_cholesky(self, H: torch.Tensor) -> torch.Tensor:
        """Upper-triangular Cholesky factor of H^-1, retrying with increased damping."""
        diag = torch.arange(self.columns, device=H.device)
        H[diag, diag] += self.config.damp * torch.diag(H).mean()
        for _ in range(5):
            try:
                inverse = torch.cholesky_inverse(torch.linalg.cholesky(H))
                return torch.linalg.cholesky(inverse, upper=True)
            except torch.linalg.LinAlgError:
                H[diag, diag] += 0.01 * torch.diag(H).mean()
        # Fallback to inverse square-root of diagonal
        return torch.diag(1.0 / torch.diag(H).clamp_min(1e-8).sqrt())

    def _rescale_refine(
        self, centered: torch.Tensor, group: TernaryGroup
    ) -> Tuple[torch.Tensor, float]:
        """Refine column rescale v_j and ternary codes."""
        codes = ternary_assign(centered, group.delta)
        if self.config.rescale_iters == 0:
            return codes, 1.0

        alpha_mean = group.alpha.mean()
        v = 1.0
        for _ in range(self.config.rescale_iters):
            nonzero = codes != 0
            if nonzero.any():
                v = (
                    (centered[nonzero].abs().mean() / alpha_mean)
                    .clamp(min=0.01)
                    .item()
                )
            else:
                v = 1.0
            level = group.alpha * v
            to_pos = (centered - level).abs()
            to_neg = (centered + level).abs()
            to_zero = centered.abs()
            codes = torch.zeros_like(centered)
            codes[(to_pos <= to_neg) & (to_pos <= to_zero)] = 1.0
            codes[(to_neg < to_pos) & (to_neg <= to_zero)] = -1.0
        return codes, v

    def _sparse_residual(
        self, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """1:N semi-sparse FP8 approximation of a salient column's residual (N=4)."""
        grouped = residual.view(-1, self.config.residual_group)
        index = grouped.abs().argmax(dim=1, keepdim=True)
        value = grouped.gather(1, index)

        scale = FP8_MAX / value.abs().max().clamp(min=1e-10)
        if hasattr(torch, "float8_e4m3fn"):
            val_fp8 = (value * scale).to(torch.float8_e4m3fn)
            val_dequant = val_fp8.to(residual.dtype) / scale
        else:
            val_dequant = value

        dense = torch.zeros_like(grouped).scatter(1, index, val_dequant)
        return (
            dense.view_as(residual),
            index.flatten().to(torch.uint8),
            value.flatten(),
            scale,
        )

    @torch.no_grad()
    def quantize(self) -> Dict[str, Any]:
        """Run the QTEA sweep, writes reconstructed weight to layer, and returns packed payload."""
        cfg = self.config
        W = self.layer.weight.data.clone().float()
        H = (
            self.hessian
            if self.hessian is not None
            else torch.eye(self.columns, device=self.device)
        )
        self.hessian = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        groups = [
            TernaryGroup(cfg.delta_coef, cfg.itf_iters).fit(
                W[:, i : i + cfg.group_size]
            )
            for i in range(0, self.columns, cfg.group_size)
        ]

        salient, rest = self._select_salient(W, torch.diag(H))
        n_salient = len(salient)
        order = torch.cat([salient, self._ldl_order(H, rest)])
        inverse_order = torch.argsort(order)

        W = W[:, order]
        Hinv = self._inverse_cholesky(H[order][:, order])
        del H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Q = torch.zeros_like(W)
        codes_all = torch.zeros(
            self.rows, self.columns, dtype=torch.int8, device=self.device
        )
        v_all = torch.ones(self.columns, device=self.device)
        residuals: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        residual_scales: List[torch.Tensor] = []
        group_of_column = (order // cfg.group_size).tolist()
        error = torch.zeros((), device=self.device)

        for start in range(0, self.columns, cfg.group_size):
            stop = start + cfg.group_size
            block = W[:, start:stop].clone()
            block_Q = torch.zeros(self.rows, cfg.group_size, device=self.device)
            block_err = torch.zeros_like(block_Q)
            block_Hinv = Hinv[start:stop, start:stop]
            hinv_mean = torch.diag(block_Hinv).mean()

            for i in range(cfg.group_size):
                col_idx = start + i
                column = block[:, i : i + 1]
                group = groups[group_of_column[col_idx]]

                centered = column - group.beta
                codes, v = self._rescale_refine(centered, group)
                q = group.alpha * v * codes + group.beta

                if col_idx < n_salient:
                    dense, idx_sparse, val_sparse, sc = self._sparse_residual(
                        column - q
                    )
                    q = q + dense
                    residuals.append((col_idx, idx_sparse, val_sparse))
                    residual_scales.append(sc)

                block_Q[:, i] = q.flatten()
                codes_all[:, col_idx] = codes.flatten().to(torch.int8)
                v_all[col_idx] = v

                d = block_Hinv[i, i]
                residual_error = (column.flatten() - q.flatten()) / d
                error += (residual_error**2).sum() / 2.0
                curvature = (d / hinv_mean).clamp(0.3, 3.0).item()
                gamma = math.exp(-cfg.decay * curvature * i / cfg.group_size)
                block[:, i:] -= gamma * residual_error.unsqueeze(1).matmul(
                    block_Hinv[i, i:].unsqueeze(0)
                )
                block_err[:, i] = residual_error

            Q[:, start:stop] = block_Q
            W[:, stop:] -= block_err.matmul(Hinv[start:stop, stop:])

        dequantized = Q[:, inverse_order].reshape(self.layer.weight.shape)
        self.layer.weight.data = dequantized.to(self.layer.weight.dtype)

        return self._payload(
            groups,
            codes_all,
            v_all,
            order,
            inverse_order,
            residuals,
            residual_scales,
        )

    def _payload(
        self,
        groups: List[TernaryGroup],
        codes: torch.Tensor,
        v: torch.Tensor,
        order: torch.Tensor,
        inverse_order: torch.Tensor,
        residuals: List[Tuple[int, torch.Tensor, torch.Tensor]],
        residual_scales: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Collect quantized layer payload in original column order."""
        payload: Dict[str, Any] = {
            "rows": self.rows,
            "cols": self.columns,
            "group_size": self.config.group_size,
            "residual_group": self.config.residual_group,
            "ternary": codes[:, inverse_order].cpu(),
            "col_v": v[inverse_order].cpu(),
            "group_alpha": torch.cat([g.alpha for g in groups], dim=1).cpu(),
            "group_beta": torch.cat([g.beta for g in groups], dim=1).cpu(),
        }

        if not residuals:
            payload.update(
                salient_cols=torch.zeros(0, dtype=torch.int32),
                residual_index=torch.zeros(0, 0, dtype=torch.uint8),
                residual_value=torch.zeros(0, 0),
                residual_scale=torch.zeros(0),
            )
            return payload

        positions = torch.tensor(
            [pos for pos, _, _ in residuals], device=self.device
        )
        columns = order[positions]
        by_column = torch.argsort(columns).tolist()

        payload.update(
            salient_cols=columns[by_column].to(torch.int32).cpu(),
            residual_index=torch.stack([residuals[i][1] for i in by_column]).cpu(),
            residual_value=torch.stack([residuals[i][2] for i in by_column]).cpu(),
            residual_scale=torch.stack([residual_scales[i] for i in by_column]).cpu(),
        )
        return payload


def pack_ternary(codes: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pack int8 ternary codes {-1, 0, +1} into base-3 bytes, 5 codes per byte."""
    rows, cols = codes.shape
    bytes_per_group = (group_size + 4) // 5
    digits = codes.int() + 1  # {-1, 0, +1} -> {0, 1, 2}

    groups = digits.reshape(rows, cols // group_size, group_size)
    padding = bytes_per_group * 5 - group_size
    if padding:
        pad_tensor = torch.ones(
            rows, groups.shape[1], padding, dtype=torch.int32, device=codes.device
        )
        groups = torch.cat([groups, pad_tensor], dim=2)
    groups = groups.reshape(rows, -1, 5)
    powers = torch.tensor(_POWERS_LIST, dtype=torch.int32, device=codes.device)
    return (groups * powers).sum(dim=2).to(torch.uint8).reshape(rows, -1)


def unpack_ternary(
    packed: torch.Tensor, cols: int, group_size: int = 128
) -> torch.Tensor:
    """Inverse of pack_ternary: decode base-3 bytes into int8 codes {-1, 0, +1}."""
    rows = packed.shape[0]
    bytes_per_group = (group_size + 4) // 5
    powers = torch.tensor(_POWERS_LIST, dtype=torch.int32, device=packed.device)

    values = packed.reshape(rows, -1, 1).int()
    digits = (values // powers) % 3 - 1
    digits = digits.reshape(rows, -1, bytes_per_group * 5)[:, :, :group_size]
    return digits.reshape(rows, cols).to(torch.int8)


def pack_layer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the output of QTEA.quantize into on-disk packed format."""
    rows = payload["rows"]
    group_size = payload["group_size"]
    index = payload["residual_index"]
    scale = payload["residual_scale"]

    if index.numel() > 0:
        pad = -index.shape[1] % 4
        if pad:
            index = torch.cat(
                [index, torch.zeros(index.shape[0], pad, dtype=index.dtype)], dim=1
            )
        pairs = index.reshape(index.shape[0], -1, 4).to(torch.uint8)
        index_packed = (
            pairs[:, :, 0]
            | (pairs[:, :, 1] << 2)
            | (pairs[:, :, 2] << 4)
            | (pairs[:, :, 3] << 6)
        )
        if hasattr(torch, "float8_e4m3fn"):
            val_fp8 = (payload["residual_value"] * scale[:, None]).to(
                torch.float8_e4m3fn
            )
        else:
            val_fp8 = payload["residual_value"] * scale[:, None]
    else:
        index_packed = torch.zeros(0, 0, dtype=torch.uint8)
        val_fp8 = (
            torch.zeros(0, 0, dtype=torch.float8_e4m3fn)
            if hasattr(torch, "float8_e4m3fn")
            else torch.zeros(0, 0)
        )

    ternary_packed = (
        pack_ternary(payload["ternary"], group_size)
        if payload["ternary"].dtype == torch.int8
        else payload["ternary"]
    )

    return {
        "rows": rows,
        "cols": payload["cols"],
        "group_size": group_size,
        "residual_group": payload["residual_group"],
        "ternary": ternary_packed,
        "col_v": payload["col_v"].float(),
        "group_alpha": payload["group_alpha"].float(),
        "group_beta": payload["group_beta"].float(),
        "salient_cols": payload["salient_cols"].to(torch.int32),
        "residual_index": index_packed,
        "residual_value": val_fp8,
        "residual_scale": scale.float(),
    }


def residual_columns(layer: Dict[str, Any]) -> torch.Tensor:
    """Dense (n_salient, rows) residual of a packed layer in FP32."""
    if layer["residual_value"].numel() == 0:
        return torch.zeros(0, layer["rows"])

    packed = layer["residual_index"].int()
    shifts = [0, 2, 4, 6]
    index = torch.stack([(packed >> s) & 0x3 for s in shifts], dim=2).reshape(
        packed.shape[0], -1
    )
    scale = layer["residual_scale"][:, None]
    value = layer["residual_value"].float() / scale

    dense = torch.zeros(value.shape[0], value.shape[1], layer["residual_group"])
    valid_cols = min(value.shape[1], index.shape[1])
    dense.scatter_(
        2, index[:, :valid_cols, None].long(), value[:, :valid_cols, None]
    )
    return dense.reshape(value.shape[0], -1)


def unpack_layer(layer: Dict[str, Any]) -> torch.Tensor:
    """Reconstruct dense FP32 weight matrix from a packed QTEA layer."""
    if layer["ternary"].dtype == torch.uint8:
        codes = unpack_ternary(
            layer["ternary"], layer["cols"], layer["group_size"]
        ).float()
    else:
        codes = layer["ternary"].float()

    alpha = layer["group_alpha"].repeat_interleave(layer["group_size"], dim=1)
    beta = layer["group_beta"].repeat_interleave(layer["group_size"], dim=1)

    weight = alpha * layer["col_v"][None, :] * codes + beta
    salient = layer["salient_cols"].long()
    if salient.numel() > 0:
        weight[:, salient] += residual_columns(layer).t()
    return weight


def bit_width(layer: Dict[str, Any]) -> Dict[str, float]:
    """Calculate effective bits per weight for a packed QTEA layer."""
    rows, cols = layer["rows"], layer["cols"]
    n_salient = layer["salient_cols"].numel()
    n_groups = layer["group_alpha"].shape[1]

    ternary_bits = layer["ternary"].numel() * 8
    residual_bits = (
        n_salient * (rows // layer["residual_group"]) * 10
    )  # 8-bit value + 2-bit index
    rescale_bits = cols * 32  # col_v (FP32)
    metadata_bits = 2 * rows * n_groups * 32  # alpha and beta (FP32)
    total_weights = rows * cols

    return {
        "payload": (ternary_bits + residual_bits + rescale_bits) / total_weights,
        "metadata": metadata_bits / total_weights,
        "total": (ternary_bits + residual_bits + rescale_bits + metadata_bits)
        / total_weights,
    }


class QTEALinear(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """PyTorch linear module backed by packed QTEA parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        residual_group: int = 4,
        bias: bool = False,
        device: Optional[Any] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise QTEAError("PyTorch is required for QTEALinear")
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.residual_group = int(residual_group)

        bytes_per_group = (self.group_size + 4) // 5
        n_groups = math.ceil(self.in_features / self.group_size)
        packed_cols = n_groups * bytes_per_group

        self.register_buffer(
            "ternary",
            torch.zeros(self.out_features, packed_cols, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "col_v", torch.ones(self.in_features, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "group_alpha",
            torch.ones(self.out_features, n_groups, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "group_beta",
            torch.zeros(self.out_features, n_groups, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "salient_cols", torch.zeros(0, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "residual_index", torch.zeros(0, 0, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "residual_value",
            torch.zeros(0, 0, dtype=torch.float8_e4m3fn, device=device)
            if hasattr(torch, "float8_e4m3fn")
            else torch.zeros(0, 0, device=device),
        )
        self.register_buffer(
            "residual_scale", torch.zeros(0, dtype=torch.float32, device=device)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_payload(
        cls, payload: Dict[str, Any], bias: Optional[torch.Tensor] = None
    ) -> "QTEALinear":
        """Instantiate QTEALinear from a packed layer payload."""
        layer = cls(
            in_features=payload["cols"],
            out_features=payload["rows"],
            group_size=payload["group_size"],
            residual_group=payload["residual_group"],
            bias=bias is not None,
        )
        layer.ternary.copy_(payload["ternary"])
        layer.col_v.copy_(payload["col_v"])
        layer.group_alpha.copy_(payload["group_alpha"])
        layer.group_beta.copy_(payload["group_beta"])
        layer.salient_cols = payload["salient_cols"].clone()
        layer.residual_index = payload["residual_index"].clone()
        layer.residual_value = payload["residual_value"].clone()
        layer.residual_scale = payload["residual_scale"].clone()
        if bias is not None and layer.bias is not None:
            layer.bias.data.copy_(bias)
        return layer

    def to_layer_dict(self) -> Dict[str, Any]:
        """Convert buffers back to the canonical layer dict format for unpacking."""
        return {
            "rows": self.out_features,
            "cols": self.in_features,
            "group_size": self.group_size,
            "residual_group": self.residual_group,
            "ternary": self.ternary,
            "col_v": self.col_v,
            "group_alpha": self.group_alpha,
            "group_beta": self.group_beta,
            "salient_cols": self.salient_cols,
            "residual_index": self.residual_index,
            "residual_value": self.residual_value,
            "residual_scale": self.residual_scale,
        }

    def effective_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct dense weight tensor W = alpha * col_v * T + beta + residual."""
        w = unpack_layer(self.to_layer_dict())
        return w.to(dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self.effective_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


# ==============================================================================
# Ornith-1.5 Hybrid Architecture Targeting & Adaptation
# ==============================================================================

ORNITH_TARGET_FAMILIES = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"),
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}

PILOT_TARGETS = [
    "layers.0.linear_attn.in_proj_qkv",
    "layers.0.mlp.down_proj",
    "layers.3.self_attn.q_proj",
]


def find_ornith_projections(
    model: Any, mode: str = "FULL"
) -> Dict[str, nn.Linear]:
    """Discover linear projections in Ornith-1.5 for quantization.

    Preserves:
      * model.visual (multimodal vision encoder)
      * embed_tokens and lm_head
      * norm layers (input_layernorm, post_attention_layernorm, etc.)

    Selects linear projections across language_model.layers:
      * PILOT mode: exactly 3 validated anchor projections.
      * FULL mode: all linear projections in linear_attn, self_attn, and mlp.
    """
    found: Dict[str, nn.Linear] = {}
    lm_root = getattr(model, "language_model", getattr(model, "model", model))
    layers = getattr(lm_root, "layers", None)
    if layers is None:
        return found

    for layer_idx, layer in enumerate(layers):
        for sub_name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = sub_name.split(".")[-1]
            family_match = any(
                leaf in leaves for leaves in ORNITH_TARGET_FAMILIES.values()
            )
            if not family_match:
                continue

            full_name = f"layers.{layer_idx}.{sub_name}"
            if mode == "PILOT" and full_name not in PILOT_TARGETS:
                continue
            found[full_name] = module

    return found


def calibrate_and_quantize_projection(
    name: str,
    module: nn.Linear,
    calibration_batches: List[torch.Tensor],
    config: Optional[QTEAConfig] = None,
) -> Tuple[Dict[str, Any], QTEALinear]:
    """Quantize one linear projection using upstream QTEA and return (packed_layer, QTEALinear)."""
    cfg = config or QTEAConfig()
    quantizer = QTEA(module, cfg)
    for batch in calibration_batches:
        quantizer.add_batch(batch)
    payload = quantizer.quantize()
    packed = pack_layer(payload)
    qtea_mod = QTEALinear.from_payload(
        packed, bias=module.bias.data if module.bias is not None else None
    )
    return packed, qtea_mod
