import torch

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache, write_kv
from augur.paged_kv_cache import (
    BlockAllocator,
    SequenceBlockTable,
    new_paged_kv_cache,
    read_paged_kv,
    write_paged_kv,
)


def test_block_allocator_allocates_and_reuses_freed_blocks() -> None:
    allocator = BlockAllocator(num_blocks=3)

    first = allocator.allocate()
    second = allocator.allocate()

    assert first == 0
    assert second == 1

    allocator.free(first)

    reused = allocator.allocate()

    assert reused == 0


def test_sequence_block_table_allocates_blocks_for_max_sequence_length() -> None:
    allocator = BlockAllocator(num_blocks=5)

    table = SequenceBlockTable.allocate(
        allocator=allocator,
        max_seq_len=10,
        block_size=4,
    )

    assert table.block_ids == [0, 1, 2]
    assert table.block_size == 4


def test_sequence_block_table_lazily_allocates_blocks_for_positions() -> None:
    allocator = BlockAllocator(num_blocks=5)
    table = SequenceBlockTable.empty(block_size=4)

    assert table.block_ids == []

    table.ensure_position(0, allocator)
    table.ensure_position(3, allocator)

    assert table.block_ids == [0]

    table.ensure_position(4, allocator)
    table.ensure_position(9, allocator)

    assert table.block_ids == [0, 1, 2]
    assert table.position_to_block_offset(9) == (2, 1)


def test_sequence_block_table_tracks_logical_sequence_length() -> None:
    allocator = BlockAllocator(num_blocks=5)
    table = SequenceBlockTable.empty(block_size=4)

    assert table.seq_len == 0

    table.ensure_position(0, allocator)

    assert table.seq_len == 1

    table.ensure_position(4, allocator)

    assert table.seq_len == 5

    table.ensure_position(3, allocator)

    assert table.seq_len == 5


def test_sequence_block_table_maps_position_to_physical_block_and_offset() -> None:
    table = SequenceBlockTable(block_ids=[7, 1, 4], block_size=4)

    assert table.position_to_block_offset(0) == (7, 0)
    assert table.position_to_block_offset(6) == (1, 2)
    assert table.position_to_block_offset(9) == (4, 1)


def test_new_paged_kv_cache_allocates_block_storage() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )

    cache = new_paged_kv_cache(
        cfg,
        num_blocks=3,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cache.keys.shape == (2, 3, 1, 4, 2)
    assert cache.values.shape == (2, 3, 1, 4, 2)
    assert cache.block_size == 4
    assert cache.allocator.allocate() == 0


def test_write_paged_kv_writes_positions_across_blocks_and_reads_logical_order() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cache = new_paged_kv_cache(
        cfg,
        num_blocks=3,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table = SequenceBlockTable(block_ids=[2, 0, 1], block_size=4)
    key = torch.tensor([[[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]])
    value = torch.tensor([[[[110.0, 111.0], [120.0, 121.0], [130.0, 131.0]]]])

    write_paged_kv(
        cache,
        layer_idx=1,
        block_table=table,
        position_ids=torch.tensor([[3, 4, 9]]),
        key=key,
        value=value,
    )
    read_key, read_value = read_paged_kv(cache, layer_idx=1, block_table=table, seq_len=10)

    assert torch.equal(cache.keys[1, 2, :, 3, :], key[0, :, 0, :])
    assert torch.equal(cache.keys[1, 0, :, 0, :], key[0, :, 1, :])
    assert torch.equal(cache.keys[1, 1, :, 1, :], key[0, :, 2, :])
    assert torch.equal(read_key[:, :, [3, 4, 9], :], key)
    assert torch.equal(read_value[:, :, [3, 4, 9], :], value)


def test_write_paged_kv_lazily_allocates_blocks_for_written_positions() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cache = new_paged_kv_cache(
        cfg,
        num_blocks=3,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table = SequenceBlockTable.empty(block_size=4)
    key = torch.tensor([[[[10.0, 11.0], [20.0, 21.0]]]])
    value = torch.tensor([[[[110.0, 111.0], [120.0, 121.0]]]])

    write_paged_kv(
        cache,
        layer_idx=1,
        block_table=table,
        position_ids=torch.tensor([[0, 4]]),
        key=key,
        value=value,
    )

    assert table.block_ids == [0, 1]
    assert torch.equal(cache.keys[1, 0, :, 0, :], key[0, :, 0, :])
    assert torch.equal(cache.keys[1, 1, :, 0, :], key[0, :, 1, :])


def test_read_paged_kv_defaults_to_block_table_sequence_length() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    cache = new_paged_kv_cache(
        cfg,
        num_blocks=2,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table = SequenceBlockTable.empty(block_size=4)
    key = torch.arange(1 * 1 * 5 * 2, dtype=torch.float32).view(1, 1, 5, 2)
    value = torch.arange(100, 100 + 1 * 1 * 5 * 2, dtype=torch.float32).view(1, 1, 5, 2)

    write_paged_kv(
        cache,
        layer_idx=1,
        block_table=table,
        position_ids=torch.tensor([[0, 1, 2, 3, 4]]),
        key=key,
        value=value,
    )
    read_key, read_value = read_paged_kv(cache, layer_idx=1, block_table=table)

    assert table.seq_len == 5
    assert torch.equal(read_key, key)
    assert torch.equal(read_value, value)


def test_paged_kv_read_matches_contiguous_kv_cache() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    contiguous = new_kv_cache(
        cfg,
        batch_size=1,
        max_seq_len=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    paged = new_paged_kv_cache(
        cfg,
        num_blocks=2,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    table = SequenceBlockTable(block_ids=[1, 0], block_size=4)
    key = torch.arange(1 * 1 * 6 * 2, dtype=torch.float32).view(1, 1, 6, 2)
    value = torch.arange(100, 100 + 1 * 1 * 6 * 2, dtype=torch.float32).view(1, 1, 6, 2)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])

    contiguous_key, contiguous_value = write_kv(
        contiguous,
        layer_idx=1,
        position_ids=position_ids,
        key=key,
        value=value,
    )
    write_paged_kv(
        paged,
        layer_idx=1,
        block_table=table,
        position_ids=position_ids,
        key=key,
        value=value,
    )
    paged_key, paged_value = read_paged_kv(paged, layer_idx=1, block_table=table, seq_len=6)

    assert torch.equal(paged_key, contiguous_key)
    assert torch.equal(paged_value, contiguous_value)
