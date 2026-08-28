import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from augur.server import GenerateRequest
from augur.showcase import showcase_html


def test_showcase_html_includes_the_streaming_demo() -> None:
    page = showcase_html()

    assert "Augur" in page
    assert "/generate_stream" in page
    assert "Qwen2.5-0.5B" in page


def test_public_generation_request_limits_output_tokens() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", max_new_tokens=129)


class FakeEngine:
    def generate_batch(self, _requests):
        return []

    def generate_stream(self, **kwargs):
        for text in ("Hello", " world"):
            kwargs["on_token"]()
            yield text


def test_stream_reports_measured_token_rate() -> None:
    from augur.showcase import create_showcase_app

    with TestClient(create_showcase_app(FakeEngine())) as client:
        response = client.post("/generate_stream", json={"prompt": "hello"})

    assert '"generated_tokens": 2' in response.text
    assert '"tokens_per_second":' in response.text
