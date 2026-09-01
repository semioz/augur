import json
import time
from collections.abc import Iterator
from pathlib import Path

import modal
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from augur.config import QwenConfig
from augur.generation import FixedSlotDecoder
from augur.pd import DecodeService, PrefillService
from augur.sampling import sample_next_token
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

app = modal.App("augur-pd")
models = modal.Volume.from_name("augur-models", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("einops>=0.8", "fastapi>=0.138.1", "packaging>=24.0", "pydantic>=2.0", "regex>=2024.0", "safetensors>=0.4")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=1, le=128)


def load_decoder(device: torch.device) -> tuple[FixedSlotDecoder, Tokenizer]:
    model_dir = Path("/models/qwen2.5-0.5b")
    cfg = QwenConfig.from_pretrained(model_dir)
    tokenizer = Tokenizer.from_pretrained(model_dir)
    weights = load_weights(model_dir / "model.safetensors", cfg, device=device, dtype=torch.float16)
    return FixedSlotDecoder(weights, cfg, max_slots=1, max_seq_len=8192), tokenizer


def sse_event(data: dict[str, object]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.cls(image=image, gpu="A10G:2", timeout=300, volumes={"/models": models})
@modal.concurrent(max_inputs=1)
class PDWorker:
    @modal.enter()
    def load(self) -> None:
        prefill_decoder, self.tokenizer = load_decoder(torch.device("cuda:0"))
        decode_decoder, _ = load_decoder(torch.device("cuda:1"))
        self.peer_access = torch.cuda.can_device_access_peer(0, 1)
        self.prefill = PrefillService(prefill_decoder, export_device=torch.device("cuda:1"))
        self.decode = DecodeService(decode_decoder)

    def _stream_tokens(self, prompt: str, max_new_tokens: int) -> Iterator[tuple[int, float, float]]:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=torch.device("cuda:0"))
        with torch.no_grad():
            result, prefill_metrics = self.prefill.prefill(input_ids)
            import_metrics = self.decode.start(result)
            yield result.first_token, prefill_metrics.export_seconds, import_metrics.import_seconds
            token_id = result.first_token
            while token_id != self.tokenizer.eos_token_id:
                logits, _ = self.decode.decode(token_id)
                token_id = int(sample_next_token(logits[:, -1]).item())
                yield token_id, prefill_metrics.export_seconds, import_metrics.import_seconds

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> dict[str, object]:
        started_at = time.perf_counter()
        token_ids = []
        handoff_seconds = import_seconds = 0.0
        for token_id, handoff_seconds, import_seconds in self._stream_tokens(prompt, max_new_tokens):
            token_ids.append(token_id)
            if len(token_ids) == max_new_tokens or token_id == self.tokenizer.eos_token_id:
                break
        elapsed = time.perf_counter() - started_at
        return {
            "text": self.tokenizer.decode(token_ids).lstrip(),
            "generated_tokens": len(token_ids),
            "peer_access": self.peer_access,
            "handoff_seconds": handoff_seconds,
            "import_seconds": import_seconds,
            "tokens_per_second": len(token_ids) / elapsed,
        }

    @modal.asgi_app()
    def web(self) -> FastAPI:
        web = FastAPI(title="augur-pd")

        @web.post("/generate_stream")
        def generate_stream(request: GenerateRequest) -> StreamingResponse:
            def events() -> Iterator[str]:
                started_at = time.perf_counter()
                token_ids: list[int] = []
                emitted_text = ""
                handoff_seconds = import_seconds = 0.0
                for token_id, handoff_seconds, import_seconds in self._stream_tokens(request.prompt, request.max_new_tokens):
                    token_ids.append(token_id)
                    text = self.tokenizer.decode(token_ids).lstrip()
                    delta = text[len(emitted_text) :]
                    emitted_text = text
                    if delta:
                        yield sse_event({"text": delta})
                    if len(token_ids) == request.max_new_tokens or token_id == self.tokenizer.eos_token_id:
                        break
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                yield sse_event(
                    {
                        "generated_tokens": len(token_ids),
                        "tokens_per_second": len(token_ids) / elapsed,
                        "handoff_milliseconds": handoff_seconds * 1_000,
                        "import_milliseconds": import_seconds * 1_000,
                    }
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return web


@app.local_entrypoint()
def main(prompt: str = "What is the capital of France?", max_new_tokens: int = 32) -> None:
    print(PDWorker().generate.remote(prompt, max_new_tokens))
