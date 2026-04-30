import torch
from torch import Tensor

from augur.weights import RMSNorm


def rms_norm(x: Tensor, w: RMSNorm, eps: float) -> Tensor:
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return w.weight * hidden.to(input_dtype)
