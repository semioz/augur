import torch.nn.functional as F
from torch import Tensor

from augur.weights import MLP


def mlp(x: Tensor, w: MLP) -> Tensor:
    gate = F.silu(F.linear(x, w.gate.weight))
    up = F.linear(x, w.up.weight)
    return F.linear(gate * up, w.down.weight)
