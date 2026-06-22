import torch
from torch import Tensor


def sample_next_token(
    logits: Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.multinomial(probs, num_samples=1)


def _apply_top_k(logits: Tensor, top_k: int | None) -> Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits

    threshold = torch.topk(logits, k=top_k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: Tensor, top_p: float | None) -> Tensor:
    if top_p is None or top_p == 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    remove = cumulative_probs > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
