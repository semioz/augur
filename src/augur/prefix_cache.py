from dataclasses import dataclass

import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import KVCache
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.weights import Weights


@dataclass
class PrefixCacheEntry:
    token_ids: Tensor
    keys: Tensor
    values: Tensor
    logits: Tensor

    @property
    def seq_len(self) -> int:
        return self.token_ids.shape[0]


class PrefixCache:
    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: list[PrefixCacheEntry] = []

    def add(self, entry: PrefixCacheEntry) -> None:
        self._entries.append(entry)
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            del self._entries[0]

    def longest_prefix(self, input_ids: Tensor) -> PrefixCacheEntry | None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("prefix cache lookup currently supports batch size 1")

        best: PrefixCacheEntry | None = None
        tokens = input_ids[0].to(device="cpu")
        for entry in self._entries:
            if entry.seq_len > tokens.shape[0]:
                continue
            entry_tokens = entry.token_ids.to(device="cpu")
            if torch.equal(tokens[: entry.seq_len], entry_tokens):
                if best is None or entry.seq_len > best.seq_len:
                    best = entry
        return best


def copy_prefix_into_cache(entry: PrefixCacheEntry, cache: KVCache, batch_idx: int = 0) -> None:
    if not 0 <= batch_idx < cache.keys.shape[1]:
        raise ValueError("batch_idx is outside the cache batch range")
    if entry.seq_len > cache.max_seq_len:
        raise ValueError("prefix length exceeds cache capacity")
    expected_shape = (
        cache.keys.shape[0],
        cache.keys.shape[2],
        entry.seq_len,
        cache.keys.shape[4],
    )
    if entry.keys.shape != expected_shape or entry.values.shape != expected_shape:
        raise ValueError("prefix key/value tensors do not match cache shape")

    cache.keys[:, batch_idx, :, : entry.seq_len, :] = entry.keys.to(
        device=cache.keys.device,
        dtype=cache.keys.dtype,
    )
    cache.values[:, batch_idx, :, : entry.seq_len, :] = entry.values.to(
        device=cache.values.device,
        dtype=cache.values.dtype,
    )
    cache.seq_len = max(cache.seq_len, entry.seq_len)


def cache_prefix(input_ids: Tensor, w: Weights, cfg: QwenConfig) -> PrefixCacheEntry:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("prefix caching currently supports batch size 1")
    if input_ids.shape[1] == 0:
        raise ValueError("prefix must contain at least one token")

    cache = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=input_ids.shape[1],
        device=input_ids.device,
        dtype=w.embed_tokens.dtype,
    )
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    logits = model(input_ids, w, cfg, cache=cache, position_ids=position_ids)
    return PrefixCacheEntry(
        token_ids=input_ids[0].detach().clone(),
        keys=cache.keys[:, 0, :, : cache.seq_len, :].detach().clone(),
        values=cache.values[:, 0, :, : cache.seq_len, :].detach().clone(),
        logits=logits[:, -1:, :].detach().clone(),
    )
