import torch.nn.functional as F
from torch import Tensor

from augur.weights import MLP, NVFP4Linear


def mlp(x: Tensor, w: MLP) -> Tensor:
    if isinstance(w.gate, NVFP4Linear):
        from augur.kernels.nvfp4_linear import linear

        gate = linear(x, w.gate)
    else:
        gate = F.linear(x, w.gate.weight, w.gate.bias)
    gate = F.silu(gate)
    up = F.linear(x, w.up.weight, w.up.bias)
    return F.linear(gate * up, w.down.weight, w.down.bias)
