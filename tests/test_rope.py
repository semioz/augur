"""
Run with:
  uv run pytest tests/test_rope.py -v
"""

import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from augur.rope import apply_rope


def test_apply_rope_matches_hf_qwen() -> None:
    cfg = HFQwen2Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )
    rope = Qwen2RotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    torch.manual_seed(0)
    q = torch.randn(2, cfg.num_attention_heads, 5, head_dim)
    k = torch.randn(2, cfg.num_key_value_heads, 5, head_dim)
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [7, 8, 9, 10, 11],
        ]
    )

    cos, sin = rope(q, position_ids)
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    actual_q, actual_k = apply_rope(q, k, position_ids, cfg.rope_parameters["rope_theta"])

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-5, atol=1e-5)


def test_apply_rope_preserves_dtype() -> None:
    position_ids = torch.arange(3).unsqueeze(0)
    q = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)

    actual_q, actual_k = apply_rope(q, k, position_ids, rope_theta=1_000_000.0)

    assert actual_q.dtype == torch.bfloat16
    assert actual_k.dtype == torch.bfloat16
