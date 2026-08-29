import torch
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def rope_embeddings(
    position_ids: Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    )
    freqs = position_ids.to(device=position_ids.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rope(
    q: Tensor,
    k: Tensor,
    position_ids: Tensor,
    rope_theta: float,
    rope: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Qwen-style rotary position embeddings.

    q: [batch, num_heads, seq, head_dim]
    k: [batch, num_key_value_heads, seq, head_dim]
    position_ids: [batch, seq]
    """
    if rope is None:
        rope = rope_embeddings(position_ids, q.shape[-1], rope_theta, q.dtype)
    cos, sin = (part.unsqueeze(1) for part in rope)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
