"""
Run with:
  uv run pytest tests/test_mlp.py -v
  uv run python tests/test_mlp.py
"""

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from augur.config import QwenConfig
from augur.mlp import mlp
from augur.weights import Linear, MLP, load_weights


def _tiny_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )


def _build_hf_mlp(w: MLP, cfg: QwenConfig) -> Qwen2MLP:
    hf_cfg = HFQwen2Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        hidden_act=cfg.hidden_act,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_parameters={"rope_theta": cfg.rope_theta, "rope_type": "default"},
        tie_word_embeddings=cfg.tie_word_embeddings,
    )
    m = Qwen2MLP(hf_cfg).eval()
    m.gate_proj.weight.data = w.gate.weight
    m.up_proj.weight.data = w.up.weight
    m.down_proj.weight.data = w.down.weight
    return m


def test_qwen_config_defaults_are_qwen2_compatible() -> None:
    cfg = QwenConfig()
    assert cfg.vocab_size == 151936
    assert cfg.hidden_size == 896
    assert cfg.intermediate_size == 4864
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 14
    assert cfg.num_key_value_heads == 2
    assert cfg.hidden_act == "silu"
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.tie_word_embeddings is True


def test_qwen_mlp_matches_hf() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    w = MLP(
        gate=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
        up=Linear(torch.randn(cfg.intermediate_size, cfg.hidden_size)),
        down=Linear(torch.randn(cfg.hidden_size, cfg.intermediate_size)),
    )
    hf = _build_hf_mlp(w, cfg)

    x = torch.randn(2, 5, cfg.hidden_size)
    torch.testing.assert_close(
        mlp(x, w),
        hf(x),
        rtol=1e-5,
        atol=1e-5,
    )


def test_load_weights_reads_qwen_layout(tmp_path) -> None:
    cfg = _tiny_config()
    path = tmp_path / "model.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.randn(cfg.hidden_size),
        "lm_head.weight": torch.randn(cfg.vocab_size, cfg.hidden_size),
    }

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.randn(cfg.hidden_size)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(cfg.hidden_size)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
        )
        tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
        )
        tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
        )
        tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
        )
        tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            cfg.hidden_size,
            cfg.intermediate_size,
        )

    save_file(tensors, path)

    w = load_weights(path, cfg)

    assert w.embed_tokens.shape == (cfg.vocab_size, cfg.hidden_size)
    assert w.lm_head.shape == (cfg.vocab_size, cfg.hidden_size)
    torch.testing.assert_close(w.lm_head, w.embed_tokens)
    assert len(w.layers) == cfg.num_hidden_layers
    assert w.layers[0].self_attn.q.weight.shape == (cfg.hidden_size, cfg.hidden_size)
    assert w.layers[0].self_attn.k.weight.shape == (cfg.head_dim, cfg.hidden_size)
    assert w.layers[0].mlp.gate.weight.shape == (cfg.intermediate_size, cfg.hidden_size)
    torch.testing.assert_close(
        w.layers[1].mlp.down.weight,
        tensors["model.layers.1.mlp.down_proj.weight"],
    )


if __name__ == "__main__":
    test_qwen_config_defaults_are_qwen2_compatible()
    print("OK  test_qwen_config_defaults_are_qwen2_compatible")
    test_qwen_mlp_matches_hf()
    print("OK  test_qwen_mlp_matches_hf")
