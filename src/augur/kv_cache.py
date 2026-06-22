from dataclasses import dataclass

import torch
from torch import Tensor

from augur.config import QwenConfig


@dataclass
class KVCache:
    keys: Tensor
    values: Tensor
    seq_len: int = 0

    @property
    def max_seq_len(self) -> int:
        return self.keys.shape[3]


def new_kv_cache(
    cfg: QwenConfig,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCache:
    shape = (
        cfg.num_hidden_layers,
        batch_size,
        cfg.num_key_value_heads,
        max_seq_len,
        cfg.head_dim,
    )
    return KVCache(
        keys=torch.empty(shape, device=device, dtype=dtype),
        values=torch.empty(shape, device=device, dtype=dtype),
    )


def kv_cache_nbytes(
    cfg: QwenConfig,
    batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    elements_per_cache = (
        cfg.num_hidden_layers
        * batch_size
        * cfg.num_key_value_heads
        * max_seq_len
        * cfg.head_dim
    )
    return elements_per_cache * torch.empty((), dtype=dtype).element_size() * 2


def format_bytes(nbytes: int) -> str:
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if nbytes < 1024:
        return f"{nbytes} B"

    value = float(nbytes)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


def write_kv(
    cache: KVCache,
    layer_idx: int,
    position_ids: Tensor,
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")

    batch, num_heads, seq, head_dim = key.shape
    expected_shape = cache.keys.shape
    if not 0 <= layer_idx < expected_shape[0]:
        raise ValueError(f"layer_idx {layer_idx} is outside the cache layer range")
    if position_ids.shape != (batch, seq):
        raise ValueError("position_ids must have shape [batch, seq]")
    if (batch, num_heads, head_dim) != (expected_shape[1], expected_shape[2], expected_shape[4]):
        raise ValueError("key/value shape does not match cache shape")
    if key.device != cache.keys.device or value.device != cache.values.device:
        raise ValueError("key/value tensors must be on the same device as the cache")

    max_position = int(position_ids.max().item())
    min_position = int(position_ids.min().item())
    if min_position < 0:
        raise ValueError("cache positions must be non-negative")
    if max_position >= cache.max_seq_len:
        raise ValueError("position_ids exceeds cache capacity")

    for batch_idx in range(batch):
        positions = position_ids[batch_idx].to(device=cache.keys.device)
        cache.keys[layer_idx, batch_idx, :, positions, :] = key[batch_idx]
        cache.values[layer_idx, batch_idx, :, positions, :] = value[batch_idx]

    cache.seq_len = max(cache.seq_len, max_position + 1)
    return (
        cache.keys[layer_idx, :, :, : cache.seq_len, :],
        cache.values[layer_idx, :, :, : cache.seq_len, :],
    )
