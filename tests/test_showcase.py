from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from augur.server import GenerateRequest
from augur.showcase import showcase_html


def test_showcase_html_includes_the_streaming_demo() -> None:
    page = showcase_html()

    assert "Augur" in page
    assert "/generate_stream" in page
    assert "Qwen2.5-0.5B" in page
    assert "PD · 2 GPUs" in page
    assert "augur-pd-pdworker-web.modal.run" in page


def test_public_generation_request_limits_output_tokens() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", max_new_tokens=129)


class FakeEngine:
    cfg = SimpleNamespace(max_position_embeddings=16)
    device = torch.device("cpu")
    weights = object()
    tokenizer = SimpleNamespace(
        eos_token_id=2,
        encode=lambda _text: [1],
        decode=lambda token_ids: "Hello" if len(token_ids) == 1 else "Hello world",
    )

    def generate_batch(self, _requests):
        return []


class FakeContinuousEngine:
    def __init__(self, *_args, **_kwargs):
        pass

    def prefill(self, states):
        return [1] * len(states)

    def decode(self, states):
        return [2] * len(states)

    def release(self, _states):
        pass


def test_stream_reports_measured_token_rate(monkeypatch) -> None:
    import augur.server as server
    from augur.showcase import create_showcase_app

    monkeypatch.setattr(server, "ContinuousEngine", FakeContinuousEngine)
    with TestClient(create_showcase_app(FakeEngine())) as client:
        response = client.post("/generate_stream", json={"prompt": "hello"})

    assert '"generated_tokens": 2' in response.text
    assert '"tokens_per_second":' in response.text


def test_generate_uses_continuous_scheduler(monkeypatch) -> None:
    import augur.server as server
    from augur.showcase import create_showcase_app

    monkeypatch.setattr(server, "ContinuousEngine", FakeContinuousEngine)
    with TestClient(create_showcase_app(FakeEngine())) as client:
        response = client.post("/generate", json={"prompt": "hello"})

    assert response.json()["text"] == "Hello world"
    assert response.json()["generated_tokens"] == 2
