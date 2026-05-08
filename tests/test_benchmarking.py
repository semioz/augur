import pytest
import torch
from types import SimpleNamespace

import augur.benchmarking as benchmarking
from augur.config import QwenConfig


class ManualClock:
    def __init__(self, times: list[float]) -> None:
        self._times = iter(times)

    def __call__(self) -> float:
        return next(self._times)


def tiny_cfg() -> QwenConfig:
    return QwenConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
    )


def test_benchmark_generate_measures_cached_prefill_and_decode(monkeypatch) -> None:
    cfg = tiny_cfg()
    seen_shapes: list[tuple[int, int]] = []
    seen_positions: list[list[list[int]]] = []
    greedy_tokens = [5, 6, 7]

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
        seen_positions.append(position_ids.tolist())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(seen_shapes) - 1]] = 1.0
        return logits

    monkeypatch.setattr(benchmarking, "model", fake_model)

    result = benchmarking.benchmark_generate(
        torch.tensor([[1, 2]]),
        w=SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float32)),
        cfg=cfg,
        max_new_tokens=3,
        use_cache=True,
        clock=ManualClock([0.0, 0.25, 0.25, 0.35, 0.35, 0.50]),
    )

    assert result.output_ids.tolist() == [[1, 2, 5, 6, 7]]
    assert seen_shapes == [(1, 2), (1, 1), (1, 1)]
    assert seen_positions == [[[0, 1]], [[2]], [[3]]]
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 3
    assert result.decode_model_tokens == 2
    assert result.prefill_seconds == pytest.approx(0.25)
    assert result.decode_seconds == pytest.approx(0.25)
    assert result.total_seconds == pytest.approx(0.50)
    assert result.decode_tokens_per_second == pytest.approx(8.0)
    assert result.total_tokens_per_second == pytest.approx(6.0)


def test_benchmark_generate_measures_uncached_recompute(monkeypatch) -> None:
    cfg = tiny_cfg()
    seen_shapes: list[tuple[int, int]] = []
    greedy_tokens = [5, 6]

    def fake_model(input_ids: torch.Tensor, w: object, cfg: QwenConfig) -> torch.Tensor:
        seen_shapes.append(tuple(input_ids.shape))
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
        logits[:, -1, greedy_tokens[len(seen_shapes) - 1]] = 1.0
        return logits

    monkeypatch.setattr(benchmarking, "model", fake_model)

    result = benchmarking.benchmark_generate(
        torch.tensor([[1, 2]]),
        w=object(),
        cfg=cfg,
        max_new_tokens=2,
        use_cache=False,
        clock=ManualClock([0.0, 0.1, 0.1, 0.3]),
    )

    assert result.output_ids.tolist() == [[1, 2, 5, 6]]
    assert seen_shapes == [(1, 2), (1, 3)]
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 2
    assert result.decode_model_tokens == 2
    assert result.prefill_seconds == 0.0
    assert result.decode_seconds == pytest.approx(0.3)
    assert result.total_seconds == pytest.approx(0.3)
    assert result.decode_tokens_per_second == pytest.approx(2 / 0.3)
    assert result.total_tokens_per_second == pytest.approx(2 / 0.3)
