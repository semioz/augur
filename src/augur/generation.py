import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.model import model
from augur.weights import Weights


def generate(input_ids: Tensor, w: Weights, cfg: QwenConfig, max_new_tokens: int) -> Tensor:
    for _ in range(max_new_tokens):
        logits = model(input_ids, w, cfg)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids
