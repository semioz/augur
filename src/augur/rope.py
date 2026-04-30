import torch
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    position_ids: Tensor,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    """
    Qwen-style rotary position embeddings.

    q: [batch, num_heads, seq, head_dim]
    k: [batch, num_key_value_heads, seq, head_dim]
    position_ids: [batch, seq]
    """
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = position_ids.to(device=q.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1).to(dtype=q.dtype)
    sin = emb.sin().unsqueeze(1).to(dtype=q.dtype)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
