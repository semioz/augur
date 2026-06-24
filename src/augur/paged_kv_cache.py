import math
from dataclasses import dataclass

import torch
from torch import Tensor

from augur.config import QwenConfig


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("no free KV cache blocks")
        return self._free_blocks.pop(0)

    def free(self, block_id: int) -> None:
        self._free_blocks.insert(0, block_id)


@dataclass
class PagedKVCache:
    keys: Tensor
    values: Tensor
    block_size: int
    allocator: BlockAllocator


@dataclass
class SequenceBlockTable:
    block_ids: list[int]
    block_size: int
    seq_len: int = 0

    @classmethod
    def empty(cls, block_size: int) -> "SequenceBlockTable":
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        return cls(block_ids=[], block_size=block_size)

    @classmethod
    def allocate(
        cls,
        allocator: BlockAllocator,
        max_seq_len: int,
        block_size: int,
    ) -> "SequenceBlockTable":
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        num_required_blocks = math.ceil(max_seq_len / block_size)
        return cls(
            block_ids=[allocator.allocate() for _ in range(num_required_blocks)],
            block_size=block_size,
            seq_len=max_seq_len,
        )

    def ensure_position(self, position: int, allocator: BlockAllocator) -> None:
        if position < 0:
            raise ValueError("position must be non-negative")
        required_blocks = position // self.block_size + 1
        while len(self.block_ids) < required_blocks:
            self.block_ids.append(allocator.allocate())
        self.seq_len = max(self.seq_len, position + 1)

    def free(self, allocator: BlockAllocator) -> None:
        for block_id in reversed(self.block_ids):
            allocator.free(block_id)
        self.block_ids.clear()
        self.seq_len = 0

    def position_to_block_offset(self, position: int) -> tuple[int, int]:
        if position < 0:
            raise ValueError("position must be non-negative")
        logical_block = position // self.block_size
        if logical_block >= len(self.block_ids):
            raise ValueError("position exceeds block table capacity")
        return self.block_ids[logical_block], position % self.block_size


def new_paged_kv_cache(
    cfg: QwenConfig,
    num_blocks: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PagedKVCache:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    shape = (
        cfg.num_hidden_layers,
        num_blocks,
        cfg.num_key_value_heads,
        block_size,
        cfg.head_dim,
    )
    return PagedKVCache(
        keys=torch.empty(shape, device=device, dtype=dtype),
        values=torch.empty(shape, device=device, dtype=dtype),
        block_size=block_size,
        allocator=BlockAllocator(num_blocks),
    )


def write_paged_kv(
    cache: PagedKVCache,
    layer_idx: int,
    block_table: SequenceBlockTable,
    position_ids: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")

    batch, num_heads, seq, head_dim = key.shape
    if batch != 1:
        raise ValueError("paged KV cache currently supports batch size 1")
    if position_ids.shape != (batch, seq):
        raise ValueError("position_ids must have shape [batch, seq]")
    if not 0 <= layer_idx < cache.keys.shape[0]:
        raise ValueError("layer_idx is outside the cache layer range")
    if (num_heads, head_dim) != (cache.keys.shape[2], cache.keys.shape[4]):
        raise ValueError("key/value shape does not match cache shape")

    for token_idx, position in enumerate(position_ids[0].tolist()):
        block_table.ensure_position(position, cache.allocator)
        physical_block, offset = block_table.position_to_block_offset(position)
        cache.keys[layer_idx, physical_block, :, offset, :] = key[0, :, token_idx, :]
        cache.values[layer_idx, physical_block, :, offset, :] = value[0, :, token_idx, :]


def read_paged_kv(
    cache: PagedKVCache,
    layer_idx: int,
    block_table: SequenceBlockTable,
    seq_len: int | None = None,
) -> tuple[Tensor, Tensor]:
    if seq_len is None:
        seq_len = block_table.seq_len
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not 0 <= layer_idx < cache.keys.shape[0]:
        raise ValueError("layer_idx is outside the cache layer range")

    key = torch.empty(
        1,
        cache.keys.shape[2],
        seq_len,
        cache.keys.shape[4],
        device=cache.keys.device,
        dtype=cache.keys.dtype,
    )
    value = torch.empty_like(key)
    for position in range(seq_len):
        physical_block, offset = block_table.position_to_block_offset(position)
        key[0, :, position, :] = cache.keys[layer_idx, physical_block, :, offset, :]
        value[0, :, position, :] = cache.values[layer_idx, physical_block, :, offset, :]
    return key, value
