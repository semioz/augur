"""
Run with:
  uv run pytest tests/test_sampling.py -v
"""

import pytest
import torch

from augur.sampling import sample_next_token


def test_sample_next_token_uses_greedy_when_temperature_is_zero() -> None:
    logits = torch.tensor(
        [
            [1.0, 4.0, 2.0],
            [3.0, 2.0, 5.0],
        ]
    )

    assert sample_next_token(logits, temperature=0.0).tolist() == [[1], [2]]


def test_sample_next_token_samples_when_temperature_is_positive() -> None:
    logits = torch.tensor([[100.0, -100.0, -100.0]])

    assert sample_next_token(logits, temperature=1.0).tolist() == [[0]]


def test_sample_next_token_top_k_keeps_only_highest_k_logits() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0]])

    assert sample_next_token(logits, temperature=1.0, top_k=1).tolist() == [[0]]


def test_sample_next_token_top_p_keeps_smallest_high_probability_prefix() -> None:
    logits = torch.tensor([[5.0, 4.0, 0.0]])

    assert sample_next_token(logits, temperature=1.0, top_p=0.7).tolist() == [[0]]


def test_sample_next_token_rejects_invalid_sampling_parameters() -> None:
    logits = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=-1.0)

    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(logits, temperature=1.0, top_k=0)

    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, temperature=1.0, top_p=0.0)
