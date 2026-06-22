import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.sampling import sample_next_token
from augur.weights import Weights


def generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    use_cache: bool = False,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if not use_cache:
        for _ in range(max_new_tokens):
            logits = model(input_ids, w, cfg)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if _is_eos(next_token, eos_token_id):
                break
        return input_ids

    if max_new_tokens == 0:
        return input_ids

    cache = new_kv_cache(
        cfg,
        batch_size=input_ids.shape[0],
        max_seq_len=input_ids.shape[1] + max_new_tokens,
        device=input_ids.device,
        dtype=w.embed_tokens.dtype,
    )
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
        input_ids.shape[0],
        -1,
    )
    logits = model(input_ids, w, cfg, cache=cache, position_ids=position_ids)

    for step in range(max_new_tokens):
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if _is_eos(next_token, eos_token_id):
            break
        if step == max_new_tokens - 1:
            break
        position_ids = torch.full(
            (input_ids.shape[0], 1),
            input_ids.shape[1] - 1,
            device=input_ids.device,
            dtype=torch.long,
        )
        logits = model(next_token, w, cfg, cache=cache, position_ids=position_ids)
    return input_ids


def _is_eos(next_token: Tensor, eos_token_id: int | None) -> bool:
    if eos_token_id is None:
        return False
    return bool(torch.all(next_token == eos_token_id).item())
