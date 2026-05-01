"""
Run with:
  uv run pytest tests/test_attention.py -v
"""

import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

from augur.attention import attention
from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.weights import Attention, Linear


def _tiny_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )


def _build_hf_attention(w: Attention, cfg: QwenConfig) -> tuple[Qwen2Attention, Qwen2RotaryEmbedding]:
    hf_cfg = HFQwen2Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_parameters={"rope_theta": cfg.rope_theta, "rope_type": "default"},
        tie_word_embeddings=cfg.tie_word_embeddings,
        attention_bias=cfg.attention_bias,
        _attn_implementation="eager",
    )
    m = Qwen2Attention(hf_cfg, layer_idx=0).eval()
    rope = Qwen2RotaryEmbedding(hf_cfg)

    m.q_proj.weight.data = w.q.weight
    m.k_proj.weight.data = w.k.weight
    m.v_proj.weight.data = w.v.weight
    m.o_proj.weight.data = w.o.weight
    m.q_proj.bias.data.zero_()
    m.k_proj.bias.data.zero_()
    m.v_proj.bias.data.zero_()
    return m, rope


def test_attention_matches_hf_qwen() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    w = Attention(
        q=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
        k=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        v=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        o=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
    )
    hf, rope = _build_hf_attention(w, cfg)

    x = torch.randn(2, 5, cfg.hidden_size)
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [3, 4, 5, 6, 7],
        ]
    )
    cos, sin = rope(x, position_ids)
    mask = torch.triu(
        torch.full((x.shape[1], x.shape[1]), float("-inf")),
        diagonal=1,
    ).view(1, 1, x.shape[1], x.shape[1])
    expected, _ = hf(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=mask,
    )

    torch.testing.assert_close(
        attention(x, w, cfg, position_ids),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_preserves_shape() -> None:
    cfg = _tiny_config()
    w = Attention(
        q=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
        k=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        v=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        o=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
    )
    x = torch.randn(3, 7, cfg.hidden_size)
    position_ids = torch.arange(7).expand(3, -1)

    assert attention(x, w, cfg, position_ids).shape == x.shape


def test_attention_with_cache_matches_full_attention_last_token() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    w = Attention(
        q=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
        k=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        v=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
        o=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
    )
    x = torch.randn(2, 5, cfg.hidden_size)
    position_ids = torch.arange(5).expand(2, -1)

    full = attention(x, w, cfg, position_ids)

    cache = new_kv_cache(num_layers=1)
    attention(x[:, :4, :], w, cfg, position_ids[:, :4], cache=cache, layer_idx=0)
    cached_last = attention(x[:, 4:, :], w, cfg, position_ids[:, 4:], cache=cache, layer_idx=0)

    torch.testing.assert_close(cached_last, full[:, 4:, :], rtol=1e-5, atol=1e-5)
