"""Triton RMSNorm backend.

One program per trailing row of ``x`` (shape [..., D]); a single masked
reduction over ``D`` matches torch's ``mean(dim=-1)`` semantics, with padded
lanes reading ``other=0.0`` and the divisor held at ``D``.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch
from torch import Tensor


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_row,
    stride_w,
    stride_out_row,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols * stride_w, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    variance = tl.sum(xf * xf, axis=0) / n_cols
    x_hat = xf * tl.rsqrt(variance + eps)
    # cast the normalized value back to the input dtype before the weight
    # scale, matching the torch reference exactly
    out = x_hat.to(x.dtype) * w
    tl.store(out_ptr + row * stride_out_row + cols, out, mask=mask)


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    grid = (x.numel() // n_cols,)
    _rms_norm_kernel[grid](
        x,
        weight,
        out,
        x.stride(-2),
        weight.stride(0),
        out.stride(-2),
        n_cols,
        eps,
        BLOCK_N=triton.next_power_of_2(n_cols),
    )
    return out
