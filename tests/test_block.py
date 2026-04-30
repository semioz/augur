"""
Run with:
  uv run pytest tests/test_block.py -v
"""

import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from augur.block import block
from augur.config import QwenConfig
from augur.weights import Attention, DecoderLayer, Linear, MLP, RMSNorm


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


def _build_hf_block(
    w: DecoderLayer,
    cfg: QwenConfig,
) -> tuple[Qwen2DecoderLayer, Qwen2RotaryEmbedding]:
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
    m = Qwen2DecoderLayer(hf_cfg, layer_idx=0).eval()
    rope = Qwen2RotaryEmbedding(hf_cfg)

    m.input_layernorm.weight.data = w.input_layernorm.weight
    m.post_attention_layernorm.weight.data = w.post_attention_layernorm.weight

    m.self_attn.q_proj.weight.data = w.self_attn.q.weight
    m.self_attn.k_proj.weight.data = w.self_attn.k.weight
    m.self_attn.v_proj.weight.data = w.self_attn.v.weight
    m.self_attn.o_proj.weight.data = w.self_attn.o.weight
    m.self_attn.q_proj.bias.data.zero_()
    m.self_attn.k_proj.bias.data.zero_()
    m.self_attn.v_proj.bias.data.zero_()

    m.mlp.gate_proj.weight.data = w.mlp.gate.weight
    m.mlp.up_proj.weight.data = w.mlp.up.weight
    m.mlp.down_proj.weight.data = w.mlp.down.weight
    return m, rope


def test_block_matches_hf_qwen() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    w = DecoderLayer(
        input_layernorm=RMSNorm(torch.randn(cfg.hidden_size)),
        self_attn=Attention(
            q=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
            k=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
            v=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
            o=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
        ),
        post_attention_layernorm=RMSNorm(torch.randn(cfg.hidden_size)),
        mlp=MLP(
            gate=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
            up=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
            down=Linear(torch.randn(cfg.hidden_size, cfg.intermediate_size)),
        ),
    )
    hf, rope = _build_hf_block(w, cfg)

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

    expected = hf(
        hidden_states=x,
        attention_mask=mask,
        position_embeddings=(cos, sin),
    )

    torch.testing.assert_close(
        block(x, w, cfg, position_ids),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_block_preserves_shape() -> None:
    cfg = _tiny_config()
    w = DecoderLayer(
        input_layernorm=RMSNorm(torch.ones(cfg.hidden_size)),
        self_attn=Attention(
            q=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
            k=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
            v=Linear(torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)),
            o=Linear(torch.randn(cfg.hidden_size, cfg.hidden_size)),
        ),
        post_attention_layernorm=RMSNorm(torch.ones(cfg.hidden_size)),
        mlp=MLP(
            gate=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
            up=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
            down=Linear(torch.randn(cfg.hidden_size, cfg.intermediate_size)),
        ),
    )
    x = torch.randn(3, 7, cfg.hidden_size)
    position_ids = torch.arange(7).expand(3, -1)

    assert block(x, w, cfg, position_ids).shape == x.shape
