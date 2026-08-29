import torch
from collections.abc import Iterator
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.paged_kv_cache import PagedKVCacheState, SequenceBlockTable, new_paged_kv_cache
from augur.prefix_cache import PrefixCache, copy_prefix_into_cache
from augur.sampling import sample_next_token
from augur.weights import Weights


class FixedSlotDecoder:
    def __init__(
        self,
        w: Weights,
        cfg: QwenConfig,
        *,
        max_slots: int,
        max_seq_len: int,
    ) -> None:
        if max_slots <= 0 or max_seq_len <= 0:
            raise ValueError("max_slots and max_seq_len must be positive")
        self.w = w
        self.cfg = cfg
        self.cache = new_kv_cache(
            cfg,
            batch_size=max_slots,
            max_seq_len=max_seq_len,
            device=w.embed_tokens.device,
            dtype=w.embed_tokens.dtype,
        )

    def prefill(
        self,
        input_ids: Tensor,
        cache_slots: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        position_ids = _position_ids_from_attention_mask(attention_mask)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(input_ids.shape[0], -1)
        return _model(
            input_ids,
            self.w,
            self.cfg,
            cache=self.cache,
            cache_slots=cache_slots,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    def decode(self, input_ids: Tensor, cache_slots: Tensor) -> Tensor:
        position_ids = self.cache.seq_lens[cache_slots].unsqueeze(1)
        return _model(
            input_ids,
            self.w,
            self.cfg,
            cache=self.cache,
            cache_slots=cache_slots,
            position_ids=position_ids,
        )


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
    cache_backend: str = "contiguous",
    paged_block_size: int = 16,
) -> Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cache_backend not in ("contiguous", "paged"):
        raise ValueError("cache_backend must be 'contiguous' or 'paged'")
    if cache_backend == "paged" and not use_cache:
        raise ValueError("paged cache backend requires use_cache=True")
    if cache_backend == "paged" and input_ids.shape[0] != 1:
        raise ValueError("paged cache backend currently supports batch size 1")
    if cache_backend == "paged" and attention_mask is not None:
        raise ValueError("paged cache backend does not support attention_mask yet")
    if cache_backend == "paged" and prefix_cache is not None:
        raise ValueError("paged cache backend does not support prefix_cache yet")
    if paged_block_size <= 0:
        raise ValueError("paged_block_size must be positive")
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

    cache = None
    paged_cache = None
    max_seq_len = input_ids.shape[1] + max_new_tokens
    if cache_backend == "paged":
        num_blocks = (max_seq_len + paged_block_size - 1) // paged_block_size
        paged_cache = PagedKVCacheState(
            cache=new_paged_kv_cache(
                cfg,
                num_blocks=num_blocks,
                block_size=paged_block_size,
                device=input_ids.device,
                dtype=w.embed_tokens.dtype,
            ),
            block_table=SequenceBlockTable.empty(block_size=paged_block_size),
        )
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
            input_ids.shape[0],
            -1,
        )
        logits = _model(
            input_ids,
            w,
            cfg,
            paged_cache=paged_cache,
            position_ids=position_ids,
        )
    else:
        prefix_entry = prefix_cache.longest_prefix(input_ids) if prefix_cache is not None else None
        cache = new_kv_cache(
            cfg,
            batch_size=input_ids.shape[0],
            max_seq_len=max_seq_len,
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
            paged_cache=paged_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    return input_ids


def generate_speculative(
    input_ids: Tensor,
    draft_weights: Weights,
    draft_cfg: QwenConfig,
    target_weights: Weights,
    target_cfg: QwenConfig,
    max_new_tokens: int,
    num_draft_tokens: int = 4,
    eos_token_id: int | None = None,
) -> Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("speculative decoding currently requires batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if num_draft_tokens <= 0:
        raise ValueError("num_draft_tokens must be positive")
    if draft_cfg.vocab_size != target_cfg.vocab_size:
        raise ValueError("draft and target models must use the same vocabulary")
    if max_new_tokens == 0:
        return input_ids

    max_seq_len = input_ids.shape[1] + max_new_tokens
    draft_cache = new_kv_cache(
        draft_cfg,
        batch_size=1,
        max_seq_len=max_seq_len,
        device=input_ids.device,
        dtype=draft_weights.embed_tokens.dtype,
    )
    target_cache = new_kv_cache(
        target_cfg,
        batch_size=1,
        max_seq_len=max_seq_len,
        device=input_ids.device,
        dtype=target_weights.embed_tokens.dtype,
    )
    prefill_positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    draft_logits = _model(
        input_ids,
        draft_weights,
        draft_cfg,
        cache=draft_cache,
        position_ids=prefill_positions,
    )
    target_logits = _model(
        input_ids,
        target_weights,
        target_cfg,
        cache=target_cache,
        position_ids=prefill_positions,
    )
    output_ids = input_ids
    pending_target_token: Tensor | None = None

    while output_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
        remaining = max_new_tokens - (output_ids.shape[1] - input_ids.shape[1])
        draft_base_len = draft_cache.seq_len
        proposals: list[Tensor] = []
        for _ in range(min(num_draft_tokens, remaining)):
            proposal = torch.argmax(_next_token_logits(draft_logits, None), dim=-1, keepdim=True)
            proposals.append(proposal)
            draft_logits = _model(
                proposal,
                draft_weights,
                draft_cfg,
                cache=draft_cache,
                position_ids=torch.full(
                    (1, 1),
                    draft_cache.seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ),
            )
            if eos_token_id is not None and bool((proposal == eos_token_id).any()):
                break

        proposal_ids = torch.cat(proposals, dim=1)
        target_base_len = target_cache.seq_len
        target_inputs = (
            proposal_ids
            if pending_target_token is None
            else torch.cat((pending_target_token, proposal_ids), dim=1)
        )
        verification_logits = _model(
            target_inputs,
            target_weights,
            target_cfg,
            cache=target_cache,
            position_ids=torch.arange(
                target_base_len,
                target_base_len + target_inputs.shape[1],
                device=input_ids.device,
            ).unsqueeze(0),
        )
        expected_logits = (
            torch.cat((target_logits[:, -1:, :], verification_logits[:, :-1, :]), dim=1)
            if pending_target_token is None
            else verification_logits[:, :-1, :]
        )
        expected_tokens = torch.argmax(expected_logits, dim=-1)
        accepted = 0
        while accepted < proposal_ids.shape[1] and bool(
            proposal_ids[0, accepted] == expected_tokens[0, accepted]
        ):
            accepted += 1

        if accepted == proposal_ids.shape[1]:
            emitted = proposal_ids
            pending_target_token = None
            if remaining > accepted:
                pending_target_token = torch.argmax(
                    verification_logits[:, -1, :], dim=-1, keepdim=True
                )
                emitted = torch.cat((emitted, pending_target_token), dim=1)
            output_ids = torch.cat((output_ids, emitted), dim=1)
            if output_ids.shape[1] - input_ids.shape[1] == max_new_tokens or (
                eos_token_id is not None and bool((emitted == eos_token_id).any())
            ):
                return output_ids
            if pending_target_token is None:
                continue
            draft_logits = _model(
                pending_target_token,
                draft_weights,
                draft_cfg,
                cache=draft_cache,
                position_ids=torch.full(
                    (1, 1),
                    draft_cache.seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ),
            )
            continue

        correction = expected_tokens[:, accepted : accepted + 1]
        emitted = torch.cat((proposal_ids[:, :accepted], correction), dim=1)
        output_ids = torch.cat((output_ids, emitted), dim=1)
        if output_ids.shape[1] - input_ids.shape[1] == max_new_tokens or (
            eos_token_id is not None and bool((emitted == eos_token_id).any())
        ):
            return output_ids

        target_cache.seq_len = target_base_len + (pending_target_token is not None) + accepted
        pending_target_token = correction
        draft_cache.seq_len = draft_base_len + accepted
        draft_logits = _model(
            correction,
            draft_weights,
            draft_cfg,
            cache=draft_cache,
            position_ids=torch.full(
                (1, 1),
                draft_cache.seq_len,
                device=input_ids.device,
                dtype=torch.long,
            ),
        )

    return output_ids


def generate_stream(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    use_cache: bool = False,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Iterator[Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("streaming generation currently supports batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return

    if not use_cache:
        for _ in range(max_new_tokens):
            logits = _model(input_ids, w, cfg)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            yield next_token
            if eos_token_id is not None and bool(torch.eq(next_token, eos_token_id).all().item()):
                break
        return

    cache = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=input_ids.shape[1] + max_new_tokens,
        device=input_ids.device,
        dtype=w.embed_tokens.dtype,
    )
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    logits = _model(input_ids, w, cfg, cache=cache, position_ids=position_ids)

    for step in range(max_new_tokens):
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        yield next_token
        if eos_token_id is not None and bool(torch.eq(next_token, eos_token_id).all().item()):
            break
        if step == max_new_tokens - 1:
            break
        position_ids = torch.full(
            (1, 1),
            input_ids.shape[1] - 1,
            device=input_ids.device,
            dtype=torch.long,
        )
        logits = _model(next_token, w, cfg, cache=cache, position_ids=position_ids)


def _model(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    cache: object | None = None,
    paged_cache: object | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    cache_slots: Tensor | None = None,
) -> Tensor:
    kwargs = {}
    if cache is not None:
        kwargs["cache"] = cache
    if paged_cache is not None:
        kwargs["paged_cache"] = paged_cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if cache_slots is not None:
        kwargs["cache_slots"] = cache_slots
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
    last_indices = (
        attention_mask.to(torch.long)
        .mul(torch.arange(attention_mask.shape[1], device=attention_mask.device))
        .max(dim=1)
        .values
    )
    return logits[torch.arange(logits.shape[0], device=logits.device), last_indices, :]


def _append_attention_mask(attention_mask: Tensor | None, next_token: Tensor) -> Tensor | None:
    if attention_mask is None:
        return None
    return torch.cat(
        (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=1
    )


def _mark_attention_positions(attention_mask: Tensor, position_ids: Tensor) -> Tensor:
    try:
        max_position = int(position_ids.max())
    except RuntimeError as exc:
        raise ValueError("position_ids must contain at least one position") from exc
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
