import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.prefix_cache import PrefixCache, copy_prefix_into_cache
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
    attention_mask: Tensor | None = None,
    prefix_cache: PrefixCache | None = None,
) -> Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if attention_mask is not None:
        _validate_attention_mask(input_ids, attention_mask)
    if prefix_cache is not None and not use_cache:
        raise ValueError("prefix_cache requires use_cache=True")
    if prefix_cache is not None and attention_mask is not None:
        raise ValueError("prefix_cache does not support attention_mask yet")
    finished = _new_finished(input_ids, eos_token_id)

    if not use_cache:
        for _ in range(max_new_tokens):
            position_ids = _position_ids_from_attention_mask(attention_mask)
            logits = _model(
                input_ids,
                w,
                cfg,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            next_token = sample_next_token(
                _next_token_logits(logits, attention_mask),
                temperature,
                top_k,
                top_p,
            )
            next_token = _force_finished_tokens(next_token, finished, eos_token_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            attention_mask = _append_attention_mask(attention_mask, next_token)
            finished = _update_finished(finished, next_token, eos_token_id)
            if _all_finished(finished):
                break
        return input_ids

    if max_new_tokens == 0:
        return input_ids

    prefix_entry = prefix_cache.longest_prefix(input_ids) if prefix_cache is not None else None

    cache = new_kv_cache(
        cfg,
        batch_size=input_ids.shape[0],
        max_seq_len=input_ids.shape[1] + max_new_tokens,
        device=input_ids.device,
        dtype=w.embed_tokens.dtype,
    )
    if prefix_entry is None:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
            input_ids.shape[0],
            -1,
        )
        logits = _model(
            input_ids,
            w,
            cfg,
            cache=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    else:
        copy_prefix_into_cache(prefix_entry, cache)
        suffix_ids = input_ids[:, prefix_entry.seq_len :]
        if suffix_ids.shape[1] == 0:
            logits = prefix_entry.logits.to(device=input_ids.device)
        else:
            position_ids = torch.arange(
                prefix_entry.seq_len,
                input_ids.shape[1],
                device=input_ids.device,
            ).expand(input_ids.shape[0], -1)
            logits = _model(
                suffix_ids,
                w,
                cfg,
                cache=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

    for step in range(max_new_tokens):
        next_token = sample_next_token(
            _next_token_logits(logits, attention_mask),
            temperature,
            top_k,
            top_p,
        )
        next_token = _force_finished_tokens(next_token, finished, eos_token_id)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if attention_mask is not None:
            position_ids = attention_mask.to(torch.long).sum(dim=1, keepdim=True)
            attention_mask = _mark_attention_positions(attention_mask, position_ids)
        else:
            position_ids = None
        finished = _update_finished(finished, next_token, eos_token_id)
        if _all_finished(finished):
            break
        if step == max_new_tokens - 1:
            break
        if position_ids is None:
            position_ids = _decode_position_ids(input_ids, attention_mask)
        logits = _model(
            next_token,
            w,
            cfg,
            cache=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    return input_ids


def _model(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    cache: object | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    kwargs = {}
    if cache is not None:
        kwargs["cache"] = cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(input_ids, w, cfg, **kwargs)


def _validate_attention_mask(input_ids: Tensor, attention_mask: Tensor) -> None:
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have shape [batch, seq]")


def _position_ids_from_attention_mask(attention_mask: Tensor | None) -> Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask.to(torch.long).cumsum(dim=1).sub(1).clamp_min(0)


def _next_token_logits(logits: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None or logits.shape[1] == 1:
        return logits[:, -1, :]
    last_indices = attention_mask.to(torch.long).mul(
        torch.arange(attention_mask.shape[1], device=attention_mask.device)
    ).max(dim=1).values
    return logits[torch.arange(logits.shape[0], device=logits.device), last_indices, :]


def _append_attention_mask(attention_mask: Tensor | None, next_token: Tensor) -> Tensor | None:
    if attention_mask is None:
        return None
    return torch.cat((attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1)


def _mark_attention_positions(attention_mask: Tensor, position_ids: Tensor) -> Tensor:
    max_position = int(position_ids.max().item())
    if max_position >= attention_mask.shape[1]:
        padding = torch.zeros(
            attention_mask.shape[0],
            max_position + 1 - attention_mask.shape[1],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, padding), dim=1)

    rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    attention_mask = attention_mask.clone()
    attention_mask[rows, position_ids.squeeze(1)] = 1
    return attention_mask


def _decode_position_ids(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is not None:
        return attention_mask.to(torch.long).sum(dim=1, keepdim=True).sub(1)
    return torch.full(
        (input_ids.shape[0], 1),
        input_ids.shape[1] - 1,
        device=input_ids.device,
        dtype=torch.long,
    )


def _new_finished(input_ids: Tensor, eos_token_id: int | None) -> Tensor | None:
    if eos_token_id is None:
        return None
    return torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.bool)


def _force_finished_tokens(
    next_token: Tensor,
    finished: Tensor | None,
    eos_token_id: int | None,
) -> Tensor:
    if finished is None or eos_token_id is None:
        return next_token
    return torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)


def _update_finished(
    finished: Tensor | None,
    next_token: Tensor,
    eos_token_id: int | None,
) -> Tensor | None:
    if finished is None or eos_token_id is None:
        return finished
    return finished | (next_token == eos_token_id)


def _all_finished(finished: Tensor | None) -> bool:
    if finished is None:
        return False
    return bool(torch.all(finished).item())
