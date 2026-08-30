import asyncio
from types import SimpleNamespace

import pytest
import torch

from augur.scheduler import ActiveRequest, GenerationRequest
from augur.server import ContinuousEngine


def make_state(request_id: str, *, temperature: float = 0.0, slot: int = 0) -> ActiveRequest:
    return ActiveRequest(
        request=GenerationRequest(
            request_id=request_id,
            prompt="hello",
            max_new_tokens=8,
            temperature=temperature,
            top_k=None,
            top_p=None,
            stop=[],
        ),
        slot=slot,
        token_queue=asyncio.Queue(),
    )


def test_continuous_engine_samples_one_token_for_each_active_request() -> None:
    engine = ContinuousEngine.__new__(ContinuousEngine)
    states = [make_state("one"), make_state("two", slot=1)]
    logits = torch.tensor([[[0.0, 2.0]], [[3.0, 0.0]]])

    assert engine._sample(logits, None, states) == [1, 0]


def test_continuous_engine_rejects_mixed_sampling_params() -> None:
    engine = ContinuousEngine.__new__(ContinuousEngine)
    logits = torch.tensor([[[0.0, 2.0]], [[3.0, 0.0]]])

    with pytest.raises(ValueError, match="matching generation params"):
        engine._sample(logits, None, [make_state("one"), make_state("two", temperature=0.5)])


def test_continuous_engine_releases_cache_slots() -> None:
    engine = ContinuousEngine.__new__(ContinuousEngine)
    cache = SimpleNamespace(seq_lens=torch.tensor([4, 7]))
    engine._engine = SimpleNamespace(device=torch.device("cpu"))
    engine._decoder = SimpleNamespace(cache=cache)

    engine.release([make_state("one", slot=1)])

    assert cache.seq_lens.tolist() == [4, 0]
