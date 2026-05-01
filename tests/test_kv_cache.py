"""
Run with:
  uv run pytest tests/test_kv_cache.py -v
"""

import torch

from augur.kv_cache import append_kv, new_kv_cache


def test_new_kv_cache_allocates_empty_layer_slots() -> None:
    cache = new_kv_cache(num_layers=2)

    assert cache.keys == [None, None]
    assert cache.values == [None, None]


def test_append_kv_stores_first_entry() -> None:
    cache = new_kv_cache(num_layers=2)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)

    cached_key, cached_value = append_kv(cache, layer_idx=1, key=key, value=value)

    torch.testing.assert_close(cached_key, key)
    torch.testing.assert_close(cached_value, value)
    assert cache.keys[0] is None
    assert cache.values[0] is None


def test_append_kv_concatenates_on_sequence_dimension() -> None:
    cache = new_kv_cache(num_layers=1)
    key_1 = torch.randn(1, 2, 3, 4)
    value_1 = torch.randn(1, 2, 3, 4)
    key_2 = torch.randn(1, 2, 1, 4)
    value_2 = torch.randn(1, 2, 1, 4)

    append_kv(cache, layer_idx=0, key=key_1, value=value_1)
    cached_key, cached_value = append_kv(cache, layer_idx=0, key=key_2, value=value_2)

    assert cached_key.shape == (1, 2, 4, 4)
    assert cached_value.shape == (1, 2, 4, 4)
    torch.testing.assert_close(cached_key[:, :, :3, :], key_1)
    torch.testing.assert_close(cached_key[:, :, 3:, :], key_2)
    torch.testing.assert_close(cached_value[:, :, :3, :], value_1)
    torch.testing.assert_close(cached_value[:, :, 3:, :], value_2)
