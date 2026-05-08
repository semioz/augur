"""
Run with:
  uv run pytest tests/test_kv_cache.py -v
"""

import torch

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache, write_kv


def tiny_cfg() -> QwenConfig:
    return QwenConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )


def test_new_kv_cache_preallocates_all_layers() -> None:
    cfg = tiny_cfg()

    cache = new_kv_cache(
        cfg,
        batch_size=3,
        max_seq_len=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cache.keys.shape == (2, 3, 1, 5, 4)
    assert cache.values.shape == (2, 3, 1, 5, 4)
    assert cache.seq_len == 0
    assert cache.keys.dtype == torch.float32


def test_write_kv_writes_positions_without_reallocating() -> None:
    cfg = tiny_cfg()
    cache = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    key_ptr = cache.keys.data_ptr()
    value_ptr = cache.values.data_ptr()
    key_1 = torch.randn(1, 1, 3, 4)
    value_1 = torch.randn(1, 1, 3, 4)
    key_2 = torch.randn(1, 1, 1, 4)
    value_2 = torch.randn(1, 1, 1, 4)

    write_kv(cache, layer_idx=1, position_ids=torch.tensor([[0, 1, 2]]), key=key_1, value=value_1)
    cached_key, cached_value = write_kv(
        cache,
        layer_idx=1,
        position_ids=torch.tensor([[3]]),
        key=key_2,
        value=value_2,
    )

    assert cache.keys.data_ptr() == key_ptr
    assert cache.values.data_ptr() == value_ptr
    assert cache.seq_len == 4
    assert cached_key.shape == (1, 1, 4, 4)
    assert cached_value.shape == (1, 1, 4, 4)
    torch.testing.assert_close(cached_key[:, :, :3, :], key_1)
    torch.testing.assert_close(cached_key[:, :, 3:, :], key_2)
    torch.testing.assert_close(cached_value[:, :, :3, :], value_1)
    torch.testing.assert_close(cached_value[:, :, 3:, :], value_2)


def test_write_kv_rejects_positions_outside_capacity() -> None:
    cfg = tiny_cfg()
    cache = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    try:
        write_kv(
            cache,
            layer_idx=0,
            position_ids=torch.tensor([[3]]),
            key=torch.randn(1, 1, 1, 4),
            value=torch.randn(1, 1, 1, 4),
        )
    except ValueError as exc:
        assert "exceeds cache capacity" in str(exc)
    else:
        raise AssertionError("write_kv should reject positions outside the cache capacity")
