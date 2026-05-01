from augur.mlp import mlp
from augur.rms_norm import rms_norm
from torch import Tensor

from augur.attention import attention
from augur.config import QwenConfig
from augur.kv_cache import KVCache
from augur.weights import DecoderLayer


def block(
    x: Tensor,
    w: DecoderLayer,
    cfg: QwenConfig,
    position_ids: Tensor,
    cache: KVCache | None = None,
    layer_idx: int | None = None,
) -> Tensor:
    x = x + attention(
        rms_norm(x, w.input_layernorm, cfg.rms_norm_eps),
        w.self_attn,
        cfg,
        position_ids,
        cache=cache,
        layer_idx=layer_idx,
    )
    x = x + mlp(
        rms_norm(x, w.post_attention_layernorm, cfg.rms_norm_eps),
        w.mlp
    )

    return x
