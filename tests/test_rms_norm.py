"""
Run with:
  uv run pytest tests/test_rms_norm.py -v
"""

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from augur.config import QwenConfig
from augur.rms_norm import rms_norm
from augur.weights import RMSNorm


def test_rms_norm_matches_hf() -> None:
    cfg = QwenConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    torch.manual_seed(0)
    w = RMSNorm(weight=torch.randn(cfg.hidden_size))
    hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).eval()
    hf.weight.data = w.weight

    x = torch.randn(2, 5, cfg.hidden_size)
    torch.testing.assert_close(
        rms_norm(x, w, cfg.rms_norm_eps),
        hf(x),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rms_norm_matches_hf_for_bfloat16() -> None:
    cfg = QwenConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    torch.manual_seed(0)
    w = RMSNorm(weight=torch.randn(cfg.hidden_size, dtype=torch.bfloat16))
    hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).eval()
    hf.weight.data = w.weight

    x = torch.randn(2, 5, cfg.hidden_size, dtype=torch.bfloat16)
    actual = rms_norm(x, w, cfg.rms_norm_eps)
    expected = hf(x)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
