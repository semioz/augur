from augur.mlp import mlp
from augur.rms_norm import rms_norm
from torch import Tensor

from augur.attention import attention
from augur.config import QwenConfig
from augur.kv_cache import KVCache
from augur.paged_kv_cache import PagedKVCacheState
from augur.weights import DecoderLayer


def block(
    x: Tensor,
    w: DecoderLayer,
    cfg: QwenConfig,
    position_ids: Tensor,
    cache: KVCache | None = None,
    paged_cache: PagedKVCacheState | None = None,
    layer_idx: int | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    x = x + attention(
        rms_norm(x, w.input_layernorm, cfg.rms_norm_eps),
        w.self_attn,
        cfg,
        position_ids,
        cache=cache,
        paged_cache=paged_cache,
        layer_idx=layer_idx,
        attention_mask=attention_mask,
    )
    x = x + mlp(
        rms_norm(x, w.post_attention_layernorm, cfg.rms_norm_eps),
        w.mlp
    )

    return x
