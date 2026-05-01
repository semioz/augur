import torch
import torch.nn.functional as F
from torch import Tensor

from augur.block import block
from augur.config import QwenConfig
from augur.kv_cache import KVCache
from augur.rms_norm import rms_norm
from augur.weights import Weights


def model(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    cache: KVCache | None = None,
    position_ids: Tensor | None = None,
) -> Tensor:
    batch, seq = input_ids.shape
    if position_ids is None:
        past_len = 0
        if cache is not None and cache.keys and cache.keys[0] is not None:
            past_len = cache.keys[0].shape[2]
        position_ids = torch.arange(
            past_len,
            past_len + seq,
            device=input_ids.device,
        ).expand(batch, -1)

    x = F.embedding(input_ids, w.embed_tokens)

    for layer_idx, layer in enumerate(w.layers):
        x = block(x, layer, cfg, position_ids, cache=cache, layer_idx=layer_idx)

    x = rms_norm(x, w.norm, cfg.rms_norm_eps)
    return F.linear(x, w.lm_head)
