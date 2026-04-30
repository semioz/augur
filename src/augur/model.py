from augur.block import block
from augur.rms_norm import rms_norm
import torch.nn.functional as F
from torch import Tensor
import torch

from augur.config import QwenConfig
from augur.weights import Weights

def model(input_ids: Tensor, w: Weights, cfg: QwenConfig) -> Tensor:
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq, device=input_ids.device).expand(batch, -1)

    x = F.embedding(input_ids, w.embed_tokens)

    for layer in w.layers:
        x = block(x, layer, cfg, position_ids)

    x = rms_norm(x, w.norm, cfg.rms_norm_eps)
    return F.linear(x, w.lm_head)