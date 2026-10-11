"""Rowwise NVFP4 packing for experiments; hardware layout is backend-specific."""

import torch
from torch import Tensor

CODEBOOK = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def round_e2m1(x: Tensor) -> Tensor:
    magnitude = x.abs().clamp(max=6)
    rounded = torch.where(
        magnitude <= 2,
        torch.round(magnitude * 2) / 2,
        torch.where(magnitude <= 4, torch.round(magnitude), 2 * torch.round(magnitude / 2)),
    )
    return rounded * x.sign()


def pack_nvfp4(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if x.ndim < 2 or not x.numel() or x.shape[-1] % 16:
        raise ValueError("expected nonempty rows with width divisible by 16")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not torch.isfinite(x).all():
        raise ValueError("expected finite floating-point values")

    values = x.float().reshape(*x.shape[:-1], -1, 16)
    block_max = values.abs().amax(dim=-1, keepdim=True)
    global_scale = (values.abs().amax() / (448 * 6)).clamp_min(torch.finfo(torch.float32).tiny)
    local = (block_max / (6 * global_scale)).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    rounded = round_e2m1(values / (local.float() * global_scale))
    levels = torch.tensor(CODEBOOK, device=x.device)
    codes = torch.searchsorted(levels, rounded.abs().contiguous()).to(torch.uint8)
    codes |= (rounded < 0).to(torch.uint8) << 3
    codes = codes.reshape(*x.shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed, local.squeeze(-1), global_scale


def unpack_nvfp4(packed: Tensor, scales: Tensor, global_scale: Tensor, dtype: torch.dtype) -> Tensor:
    codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    levels = torch.tensor(CODEBOOK, device=packed.device)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
    return (values * scales.float().repeat_interleave(16, dim=-1) * global_scale).to(dtype)
