"""
Run with:
  uv run pytest tests/test_generation.py -v
"""

import torch
import pytest
from types import SimpleNamespace

import augur.generation as generation
from augur.config import QwenConfig


def test_generate_appends_greedy_tokens(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    calls: list[torch.Tensor] = []
    greedy_tokens = [5, 6, 7]

    def fake_model(input_ids: torch.Tensor, w: object, cfg: QwenConfig) -> torch.Tensor:
        calls.append(input_ids.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(calls) - 1]] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    input_ids = torch.tensor([[1, 2]])
    output = generation.generate(input_ids, w=object(), cfg=cfg, max_new_tokens=3)

    assert output.tolist() == [[1, 2, 5, 6, 7]]
    assert [call.tolist() for call in calls] == [
        [[1, 2]],
        [[1, 2, 5]],
        [[1, 2, 5, 6]],
    ]


def test_generate_zero_new_tokens_returns_input() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    input_ids = torch.tensor([[1, 2]])

    output = generation.generate(input_ids, w=object(), cfg=cfg, max_new_tokens=0)

    assert output is input_ids


def test_generate_rejects_negative_max_new_tokens() -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )

    with pytest.raises(ValueError, match="max_new_tokens"):
        generation.generate(torch.tensor([[1, 2]]), w=object(), cfg=cfg, max_new_tokens=-1)


def test_generate_stops_when_eos_token_is_selected(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    calls: list[torch.Tensor] = []
    greedy_tokens = [5, 3, 7]

    def fake_model(input_ids: torch.Tensor, w: object, cfg: QwenConfig) -> torch.Tensor:
        calls.append(input_ids.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(calls) - 1]] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    input_ids = torch.tensor([[1, 2]])
    output = generation.generate(input_ids, w=object(), cfg=cfg, max_new_tokens=5, eos_token_id=3)

    assert output.tolist() == [[1, 2, 5, 3]]
    assert [call.tolist() for call in calls] == [
        [[1, 2]],
        [[1, 2, 5]],
    ]


def test_generate_stops_each_batch_row_after_eos(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    calls: list[torch.Tensor] = []
    greedy_tokens = [
        [3, 6],
        [5, 7],
        [5, 3],
        [5, 5],
        [5, 5],
    ]

    def fake_model(input_ids: torch.Tensor, w: object, cfg: QwenConfig) -> torch.Tensor:
        calls.append(input_ids.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        for row_idx, token_id in enumerate(greedy_tokens[len(calls) - 1]):
            logits[row_idx, -1, token_id] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    output = generation.generate(
        torch.tensor([[1, 2], [4, 5]]),
        w=object(),
        cfg=cfg,
        max_new_tokens=5,
        eos_token_id=3,
    )

    assert output.tolist() == [[1, 2, 3, 3, 3], [4, 5, 6, 7, 3]]
    assert [call.tolist() for call in calls] == [
        [[1, 2], [4, 5]],
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3, 3], [4, 5, 6, 7]],
    ]


def test_generate_uses_sampling_parameters(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_sampling_args: list[tuple[float, int | None, float | None]] = []

    def fake_model(input_ids: torch.Tensor, w: object, cfg: QwenConfig) -> torch.Tensor:
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)

    def fake_sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        seen_sampling_args.append((temperature, top_k, top_p))
        return torch.tensor([[4]])

    monkeypatch.setattr(generation, "model", fake_model)
    monkeypatch.setattr(generation, "sample_next_token", fake_sample_next_token)

    output = generation.generate(
        torch.tensor([[1, 2]]),
        w=object(),
        cfg=cfg,
        max_new_tokens=1,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )

    assert output.tolist() == [[1, 2, 4]]
    assert seen_sampling_args == [(0.8, 20, 0.9)]


def test_generate_with_attention_mask_samples_from_last_real_token(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_attention_masks: list[torch.Tensor] = []

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        seen_attention_masks.append(attention_mask.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[0, 1, 5] = 1.0
        logits[0, 2, 6] = 2.0
        logits[1, 2, 7] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    output = generation.generate(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        w=object(),
        cfg=cfg,
        max_new_tokens=1,
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )

    assert output.tolist() == [[1, 2, 0, 5], [3, 4, 5, 7]]
    assert seen_attention_masks[0].tolist() == [[1, 1, 0], [1, 1, 1]]


def test_generate_with_cache_uses_sampling_parameters(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_sampling_args: list[tuple[float, int | None, float | None]] = []

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        cache: object | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache is not None
        assert position_ids is not None
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)

    def fake_sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> torch.Tensor:
        seen_sampling_args.append((temperature, top_k, top_p))
        return torch.tensor([[4]])

    monkeypatch.setattr(generation, "model", fake_model)
    monkeypatch.setattr(generation, "sample_next_token", fake_sample_next_token)

    output = generation.generate(
        torch.tensor([[1, 2]]),
        w=SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32)),
        cfg=cfg,
        max_new_tokens=1,
        use_cache=True,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )

    assert output.tolist() == [[1, 2, 4]]
    assert seen_sampling_args == [(0.8, 20, 0.9)]


def test_generate_with_cache_updates_batched_attention_mask_and_positions(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_attention_masks: list[list[list[int]]] = []
    seen_position_ids: list[list[list[int]]] = []

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        cache: object | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache is not None
        assert position_ids is not None
        assert attention_mask is not None
        seen_position_ids.append(position_ids.tolist())
        seen_attention_masks.append(attention_mask.tolist())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        if len(seen_position_ids) == 1:
            logits[0, 1, 5] = 1.0
            logits[0, 2, 6] = 2.0
            logits[1, 2, 7] = 1.0
        else:
            logits[0, -1, 1] = 1.0
            logits[1, -1, 2] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    output = generation.generate(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        w=SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32)),
        cfg=cfg,
        max_new_tokens=2,
        use_cache=True,
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )

    assert output.tolist() == [[1, 2, 0, 5, 1], [3, 4, 5, 7, 2]]
    assert seen_position_ids == [[[0, 1, 2], [0, 1, 2]], [[2], [3]]]
    assert seen_attention_masks == [
        [[1, 1, 0], [1, 1, 1]],
        [[1, 1, 1, 0], [1, 1, 1, 1]],
    ]


def test_generate_can_use_cache(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_shapes: list[tuple[int, int]] = []
    greedy_tokens = [5, 6, 7]
    seen_cache_shapes: list[tuple[int, ...]] = []

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        cache: object | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache is not None
        assert position_ids is not None
        seen_cache_shapes.append(tuple(cache.keys.shape))
        seen_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(seen_shapes) - 1]] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    input_ids = torch.tensor([[1, 2]])
    weights = SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32))
    output = generation.generate(input_ids, w=weights, cfg=cfg, max_new_tokens=3, use_cache=True)

    assert output.tolist() == [[1, 2, 5, 6, 7]]
    assert seen_shapes == [(1, 2), (1, 1), (1, 1)]
    assert seen_cache_shapes == [(2, 1, 1, 5, 2)] * 3


def test_generate_with_cache_stops_when_eos_token_is_selected(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    seen_shapes: list[tuple[int, int]] = []
    greedy_tokens = [5, 3, 7]

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        cache: object | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache is not None
        assert position_ids is not None
        seen_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(seen_shapes) - 1]] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    input_ids = torch.tensor([[1, 2]])
    weights = SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32))
    output = generation.generate(
        input_ids,
        w=weights,
        cfg=cfg,
        max_new_tokens=5,
        use_cache=True,
        eos_token_id=3,
    )

    assert output.tolist() == [[1, 2, 5, 3]]
    assert seen_shapes == [(1, 2), (1, 1)]


def test_generate_with_cache_stops_each_batch_row_after_eos(monkeypatch) -> None:
    cfg = QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    calls: list[torch.Tensor] = []
    greedy_tokens = [
        [3, 6],
        [5, 7],
        [5, 3],
        [5, 5],
        [5, 5],
    ]

    def fake_model(
        input_ids: torch.Tensor,
        w: object,
        cfg: QwenConfig,
        cache: object | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert cache is not None
        assert position_ids is not None
        calls.append(input_ids.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        for row_idx, token_id in enumerate(greedy_tokens[len(calls) - 1]):
            logits[row_idx, -1, token_id] = 1.0
        return logits

    monkeypatch.setattr(generation, "model", fake_model)

    output = generation.generate(
        torch.tensor([[1, 2], [4, 5]]),
        w=SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32)),
        cfg=cfg,
        max_new_tokens=5,
        use_cache=True,
        eos_token_id=3,
    )

    assert output.tolist() == [[1, 2, 3, 3, 3], [4, 5, 6, 7, 3]]
    assert [call.tolist() for call in calls] == [
        [[1, 2], [4, 5]],
        [[3], [6]],
        [[3], [7]],
    ]
