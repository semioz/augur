import torch
from types import SimpleNamespace

import augur.prefix_cache as prefix_cache
from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.prefix_cache import PrefixCache, PrefixCacheEntry


def test_prefix_cache_returns_longest_matching_prefix() -> None:
    cache = PrefixCache()
    short = PrefixCacheEntry(
        token_ids=torch.tensor([1, 2]),
        keys=torch.empty(0),
        values=torch.empty(0),
        logits=torch.empty(0),
    )
    long = PrefixCacheEntry(
        token_ids=torch.tensor([1, 2, 3]),
        keys=torch.empty(0),
        values=torch.empty(0),
        logits=torch.empty(0),
    )
    miss = PrefixCacheEntry(
        token_ids=torch.tensor([2, 3]),
        keys=torch.empty(0),
        values=torch.empty(0),
        logits=torch.empty(0),
    )

    cache.add(short)
    cache.add(long)
    cache.add(miss)

    assert cache.longest_prefix(torch.tensor([[1, 2, 3, 4]])) is long


def test_prefix_cache_returns_none_when_no_prefix_matches() -> None:
    cache = PrefixCache()
    cache.add(
        PrefixCacheEntry(
            token_ids=torch.tensor([1, 2]),
            keys=torch.empty(0),
            values=torch.empty(0),
            logits=torch.empty(0),
        )
    )

    assert cache.longest_prefix(torch.tensor([[9, 2, 3]])) is None


def test_copy_prefix_into_cache_copies_entry_tensors_and_updates_length() -> None:
    from augur.prefix_cache import copy_prefix_into_cache

    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    request_cache = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    entry = PrefixCacheEntry(
        token_ids=torch.tensor([1, 2, 3]),
        keys=torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).view(2, 1, 3, 2),
        values=torch.arange(100, 100 + 2 * 1 * 3 * 2, dtype=torch.float32).view(2, 1, 3, 2),
        logits=torch.empty(0),
    )

    copy_prefix_into_cache(entry, request_cache)

    assert request_cache.seq_len == 3
    assert torch.equal(request_cache.keys[:, 0, :, :3, :], entry.keys)
    assert torch.equal(request_cache.values[:, 0, :, :3, :], entry.values)


def test_cache_prefix_prefills_and_stores_prefix_entry(monkeypatch) -> None:
    from augur.prefix_cache import cache_prefix

    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )

    def fake_model(input_ids, w, cfg, cache=None, position_ids=None):
        assert cache is not None
        assert position_ids.tolist() == [[0, 1, 2]]
        cache.keys[:, 0, :, :3, :] = 11
        cache.values[:, 0, :, :3, :] = 22
        cache.seq_len = 3
        logits = torch.zeros(1, 3, cfg.vocab_size)
        logits[:, -1, 5] = 1
        return logits

    monkeypatch.setattr(prefix_cache, "model", fake_model)

    entry = cache_prefix(
        torch.tensor([[1, 2, 3]]),
        w=SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32)),
        cfg=cfg,
    )

    assert entry.token_ids.tolist() == [1, 2, 3]
    assert entry.keys.shape == (2, 1, 3, 2)
    assert entry.values.shape == (2, 1, 3, 2)
    assert torch.all(entry.keys == 11)
    assert torch.all(entry.values == 22)
    assert entry.logits.shape == (1, 1, 8)
    assert entry.logits[0, 0, 5] == 1
