import torch
import torch.nn.functional as F
from torch import Tensor

from augur.block import block
from augur.config import QwenConfig
from augur.kv_cache import KVCache, cache_attention_mask
from augur.paged_kv_cache import PagedKVCacheState
from augur.rms_norm import rms_norm
from augur.rope import rope_embeddings
from augur.weights import Weights


def model(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    cache: KVCache | None = None,
    paged_cache: PagedKVCacheState | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    cache_slots: Tensor | None = None,
) -> Tensor:
    if cache is not None and paged_cache is not None:
        raise ValueError("cache and paged_cache cannot both be provided")
    batch, seq = input_ids.shape
    if position_ids is None:
        past_len = 0
        if cache is not None and cache_slots is not None:
            position_ids = cache.seq_lens[cache_slots].unsqueeze(1) + torch.arange(
                seq,
                device=input_ids.device,
            )
        if cache is not None:
            past_len = cache.seq_len
        if paged_cache is not None:
            past_len = paged_cache.block_table.seq_len
        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + seq,
                device=input_ids.device,
            ).expand(batch, -1)

    if cache is not None:
        cache_mask = cache_attention_mask(cache, position_ids, cache_slots)
        attention_mask = cache_mask if attention_mask is None else attention_mask.to(torch.bool) & cache_mask

    x = F.embedding(input_ids, w.embed_tokens)
    rope = rope_embeddings(position_ids, cfg.head_dim, cfg.rope_theta, x.dtype)

    for layer_idx, layer in enumerate(w.layers):
        kwargs = {}
        if paged_cache is not None:
            kwargs["paged_cache"] = paged_cache
        x = block(
            x,
            layer,
            cfg,
            position_ids,
            cache=cache,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            rope=rope,
            cache_slots=cache_slots,
            **kwargs,
        )

    x = rms_norm(x, w.norm, cfg.rms_norm_eps)
    return F.linear(x, w.lm_head)
