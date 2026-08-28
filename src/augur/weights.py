from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from augur.config import QwenConfig


@dataclass(frozen=True)
class RMSNorm:
    weight: Tensor


@dataclass(frozen=True)
class Linear:
    weight: Tensor
    bias: Tensor | None = None


@dataclass(frozen=True)
class Attention:
    q: Linear
    k: Linear
    v: Linear
    o: Linear
    qkv: Linear | None = None


@dataclass(frozen=True)
class MLP:
    gate: Linear
    up: Linear
    down: Linear


@dataclass(frozen=True)
class DecoderLayer:
    input_layernorm: RMSNorm
    self_attn: Attention
    post_attention_layernorm: RMSNorm
    mlp: MLP


@dataclass(frozen=True)
class Weights:
    embed_tokens: Tensor
    layers: tuple[DecoderLayer, ...]
    norm: RMSNorm
    lm_head: Tensor


def load_weights(
    path: Path | str,
    config: QwenConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Weights:
    sd = load_file(str(path))

    def to(name: str) -> Tensor:
        return sd[name].to(device=device, dtype=dtype)

    def linear(prefix: str) -> Linear:
        bias_name = f"{prefix}.bias"
        return Linear(
            weight=to(f"{prefix}.weight"),
            bias=to(bias_name) if bias_name in sd else None,
        )

    def rms_norm(prefix: str) -> RMSNorm:
        return RMSNorm(weight=to(f"{prefix}.weight"))

    def attention(prefix: str) -> Attention:
        q = linear(f"{prefix}.q_proj")
        k = linear(f"{prefix}.k_proj")
        v = linear(f"{prefix}.v_proj")
        if q.bias is None:
            if k.bias is not None or v.bias is not None:
                raise ValueError("Q/K/V biases must all be present or absent")
            qkv_bias = None
        else:
            if k.bias is None or v.bias is None:
                raise ValueError("Q/K/V biases must all be present or absent")
            qkv_bias = torch.cat((q.bias, k.bias, v.bias))
        return Attention(
            q=q,
            k=k,
            v=v,
            o=linear(f"{prefix}.o_proj"),
            qkv=Linear(weight=torch.cat((q.weight, k.weight, v.weight)), bias=qkv_bias),
        )

    embed_tokens = to("model.embed_tokens.weight")
    expected_embedding_shape = (config.vocab_size, config.hidden_size)
    if embed_tokens.shape != expected_embedding_shape:
        raise ValueError(
            f"embed_tokens shape {tuple(embed_tokens.shape)} does not match config "
            f"{expected_embedding_shape}"
        )

    layers = tuple(
        DecoderLayer(
            input_layernorm=rms_norm(f"model.layers.{i}.input_layernorm"),
            self_attn=attention(f"model.layers.{i}.self_attn"),
            post_attention_layernorm=rms_norm(f"model.layers.{i}.post_attention_layernorm"),
            mlp=MLP(
                gate=linear(f"model.layers.{i}.mlp.gate_proj"),
                up=linear(f"model.layers.{i}.mlp.up_proj"),
                down=linear(f"model.layers.{i}.mlp.down_proj"),
            ),
        )
        for i in range(config.num_hidden_layers)
    )

    lm_head = embed_tokens if config.tie_word_embeddings else to("lm_head.weight")

    return Weights(
        embed_tokens=embed_tokens,
        layers=layers,
        norm=rms_norm("model.norm"),
        lm_head=lm_head,
    )
