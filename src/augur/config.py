import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class QwenConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def __post_init__(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(f"only silu Qwen MLP is supported, got {self.hidden_act!r}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @classmethod
    def from_pretrained(cls, model_dir: Path | str) -> Self:
        config_path = Path(model_dir) / "config.json"
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return cls(
                vocab_size=int(data["vocab_size"]),
                hidden_size=int(data["hidden_size"]),
                intermediate_size=int(data["intermediate_size"]),
                num_hidden_layers=int(data["num_hidden_layers"]),
                num_attention_heads=int(data["num_attention_heads"]),
                num_key_value_heads=int(data["num_key_value_heads"]),
                hidden_act=str(data.get("hidden_act", "silu")),
                max_position_embeddings=int(data.get("max_position_embeddings", 32768)),
                rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
                rope_theta=float(data.get("rope_theta", 1_000_000.0)),
                tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
                attention_bias=bool(data.get("attention_bias", False)),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Qwen config: {config_path}") from exc

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
