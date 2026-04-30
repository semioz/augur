from augur.mlp import mlp
from augur.rms_norm import rms_norm
from torch import Tensor

from augur.attention import attention
from augur.config import QwenConfig
from augur.weights import DecoderLayer


def block(x: Tensor, w: DecoderLayer, cfg: QwenConfig, position_ids: Tensor) -> Tensor:
    x = x + attention(
        rms_norm(x, w.input_layernorm, cfg.rms_norm_eps),
        w.self_attn,
        cfg,
        position_ids,
    )
    x = x + mlp(
        rms_norm(x, w.post_attention_layernorm, cfg.rms_norm_eps),
        w.mlp
    )

    return x