import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.weights import Weights


def generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    use_cache: bool = False,
) -> Tensor:
    if not use_cache:
        for _ in range(max_new_tokens):
            logits = model(input_ids, w, cfg)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    if max_new_tokens == 0:
        return input_ids

    cache = new_kv_cache(cfg.num_hidden_layers)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
        input_ids.shape[0],
        -1,
    )
    logits = model(input_ids, w, cfg, cache=cache, position_ids=position_ids)

    for step in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
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
