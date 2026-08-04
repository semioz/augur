import torch
from torch import Tensor

from augur.kernels import kernels_available
from augur.weights import RMSNorm


def rms_norm(x: Tensor, w: RMSNorm, eps: float) -> Tensor:
    if x.is_cuda and kernels_available(x.device):
        from augur.kernels import rms_norm as _k
        return _k.rms_norm(x, w.weight, eps)
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return w.weight * hidden.to(input_dtype)
