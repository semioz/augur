"""
Run with:
  uv run pytest tests/test_generation.py -v
"""

import torch

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
