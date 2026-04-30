import torch
import math
from torch import Tensor
import torch.nn.functional as F
from augur.rope import apply_rope

from augur.config import QwenConfig
from augur.weights import Attention

def attention(x: Tensor, w: Attention, cfg: QwenConfig, position_ids: Tensor) -> Tensor:
    batch, seq, _ = x.shape
    # we gotta move the heads dimension before the sequence dimension so attention can compute separate [seq, seq] scores for each head so tranpose
    q = F.linear(x, w.q.weight).view(batch, seq, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
    k = F.linear(x, w.k.weight).view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
    v = F.linear(x, w.v.weight).view(batch, seq, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, position_ids, cfg.rope_theta)

    # qwen uses GQA so we repeat each shared K/V head to match the number of query heads
    k = k.repeat_interleave(cfg.num_key_value_groups, dim=1)
    v = v.repeat_interleave(cfg.num_key_value_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    # leaving the upper triangular part of matrix for causal mask, putting -inf for zeroed ones to do softmax later
    mask = torch.triu(
        torch.ones(seq, seq, device=x.device, dtype=torch.bool),
        diagonal=1
    )
    scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v)

    # make memory layout contiguous after transpose so view can safely merge heads back into hidden_size
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch, seq, cfg.hidden_size)

    return F.linear(out, w.o.weight)