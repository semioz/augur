import json

from augur.config import QwenConfig


def test_qwen_config_from_pretrained_reads_model_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 151936,
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "tie_word_embeddings": True,
                "attention_bias": False,
            }
        ),
        encoding="utf-8",
    )

    config = QwenConfig.from_pretrained(tmp_path)

    assert config.hidden_size == 1536
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 12
