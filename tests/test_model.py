import torch
import torch.nn.functional as F
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.rms_norm import rms_norm
from augur.weights import Attention, DecoderLayer, Linear, MLP, RMSNorm, Weights


def test_model_without_layers_matches_embedding_norm_and_lm_head() -> None:
    cfg = QwenConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    torch.manual_seed(0)
    w = Weights(
        embed_tokens=torch.randn(cfg.vocab_size, cfg.hidden_size),
        layers=(),
        norm=RMSNorm(torch.randn(cfg.hidden_size)),
        lm_head=torch.randn(cfg.vocab_size, cfg.hidden_size),
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    x = F.embedding(input_ids, w.embed_tokens)
    expected = F.linear(rms_norm(x, w.norm, cfg.rms_norm_eps), w.lm_head)

    torch.testing.assert_close(
        model(input_ids, w, cfg),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_model_returns_vocab_logits() -> None:
    cfg = QwenConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    w = Weights(
        embed_tokens=torch.randn(cfg.vocab_size, cfg.hidden_size),
        layers=(),
        norm=RMSNorm(torch.ones(cfg.hidden_size)),
        lm_head=torch.randn(cfg.vocab_size, cfg.hidden_size),
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    assert model(input_ids, w, cfg).shape == (2, 3, cfg.vocab_size)


def _tiny_config_with_layer() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )


def _build_weights(cfg: QwenConfig) -> Weights:
    return Weights(
        embed_tokens=torch.randn(cfg.vocab_size, cfg.hidden_size),
        layers=(
            DecoderLayer(
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
            ),
        ),
        norm=RMSNorm(torch.randn(cfg.hidden_size)),
        lm_head=torch.randn(cfg.vocab_size, cfg.hidden_size),
    )


def _build_hf_model(w: Weights, cfg: QwenConfig) -> Qwen2ForCausalLM:
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
        tie_word_embeddings=False,
        attention_bias=cfg.attention_bias,
        _attn_implementation="eager",
    )
    m = Qwen2ForCausalLM(hf_cfg).eval()
    layer = w.layers[0]
    hf_layer = m.model.layers[0]

    m.model.embed_tokens.weight.data = w.embed_tokens
    m.model.norm.weight.data = w.norm.weight
    m.lm_head.weight.data = w.lm_head

    hf_layer.input_layernorm.weight.data = layer.input_layernorm.weight
    hf_layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight

    hf_layer.self_attn.q_proj.weight.data = layer.self_attn.q.weight
    hf_layer.self_attn.k_proj.weight.data = layer.self_attn.k.weight
    hf_layer.self_attn.v_proj.weight.data = layer.self_attn.v.weight
    hf_layer.self_attn.o_proj.weight.data = layer.self_attn.o.weight
    hf_layer.self_attn.q_proj.bias.data.zero_()
    hf_layer.self_attn.k_proj.bias.data.zero_()
    hf_layer.self_attn.v_proj.bias.data.zero_()

    hf_layer.mlp.gate_proj.weight.data = layer.mlp.gate.weight
    hf_layer.mlp.up_proj.weight.data = layer.mlp.up.weight
    hf_layer.mlp.down_proj.weight.data = layer.mlp.down.weight
    return m


def test_model_matches_hf_qwen() -> None:
    cfg = _tiny_config_with_layer()
    torch.manual_seed(0)
    w = _build_weights(cfg)
    hf = _build_hf_model(w, cfg)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    expected = hf(input_ids=input_ids).logits

    torch.testing.assert_close(
        model(input_ids, w, cfg),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_model_with_cache_matches_full_model_last_token() -> None:
    cfg = _tiny_config_with_layer()
    torch.manual_seed(0)
    w = _build_weights(cfg)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    full = model(input_ids, w, cfg)

    cache = new_kv_cache(cfg.num_hidden_layers)
    model(input_ids[:, :3], w, cfg, cache=cache)
    cached_last = model(input_ids[:, 3:], w, cfg, cache=cache, position_ids=torch.full((2, 1), 3))

    torch.testing.assert_close(cached_last, full[:, 3:, :], rtol=1e-5, atol=1e-5)
