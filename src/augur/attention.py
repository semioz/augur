import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import KVCache, write_kv
from augur.paged_kv_cache import PagedKVCacheState, read_paged_kv, write_paged_kv
from augur.rope import apply_rope
from augur.weights import Attention


def _causal_mask(query_len: int, key_len: int, device: torch.device) -> Tensor:
    past_len = key_len - query_len
    query_positions = torch.arange(query_len, device=device).unsqueeze(-1) + past_len
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    return key_positions > query_positions


def attention(
    x: Tensor,
    w: Attention,
    cfg: QwenConfig,
    position_ids: Tensor,
    cache: KVCache | None = None,
    paged_cache: PagedKVCacheState | None = None,
    layer_idx: int | None = None,
    attention_mask: Tensor | None = None,
    rope: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    batch, seq, _ = x.shape
    if w.qkv is None:
        q_proj = F.linear(x, w.q.weight, w.q.bias)
        k_proj = F.linear(x, w.k.weight, w.k.bias)
        v_proj = F.linear(x, w.v.weight, w.v.bias)
    else:
        q_proj, k_proj, v_proj = F.linear(x, w.qkv.weight, w.qkv.bias).split(
            (w.q.weight.shape[0], w.k.weight.shape[0], w.v.weight.shape[0]), dim=-1
        )
    q = rearrange(
        q_proj,
        "batch seq (heads head_dim) -> batch heads seq head_dim",
        heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
    )
    k = rearrange(
        k_proj,
        "batch seq (heads head_dim) -> batch heads seq head_dim",
        heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    v = rearrange(
        v_proj,
        "batch seq (heads head_dim) -> batch heads seq head_dim",
        heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )

    q, k = apply_rope(q, k, position_ids, cfg.rope_theta, rope)

    if cache is not None and paged_cache is not None:
        raise ValueError("cache and paged_cache cannot both be provided")
    if cache is not None:
        if layer_idx is None:
            raise ValueError("layer_idx is required when cache is provided")
        k, v = write_kv(cache, layer_idx, position_ids, k, v)
    if paged_cache is not None:
        if layer_idx is None:
            raise ValueError("layer_idx is required when paged_cache is provided")
        write_paged_kv(
            paged_cache.cache,
            layer_idx,
            paged_cache.block_table,
            position_ids,
            k,
            v,
        )
        k, v = read_paged_kv(paged_cache.cache, layer_idx, paged_cache.block_table)

    # qwen uses GQA so we repeat each shared K/V head to match the number of query heads
    k = k.repeat_interleave(cfg.num_key_value_groups, dim=1)
    v = v.repeat_interleave(cfg.num_key_value_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    # leaving the upper triangular part of matrix for causal mask, putting -inf for zeroed ones to do softmax later
    try:
        if seq > 1:
            mask = _causal_mask(seq, k.shape[2], x.device)
            scores = scores.masked_fill(mask, float("-inf"))
        if attention_mask is not None:
            if attention_mask.shape != (batch, k.shape[2]):
                raise ValueError("attention_mask must have shape [batch, key_len]")
            padding_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(~padding_mask[:, None, None, :], float("-inf"))
    except RuntimeError as exc:
        raise ValueError("attention mask is incompatible with attention scores") from exc

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v)

    out = rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)")

    return F.linear(out, w.o.weight, w.o.bias)
