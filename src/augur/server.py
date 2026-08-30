import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from augur.config import QwenConfig
from augur.generation import FixedSlotDecoder, generate
from augur.generation import generate_stream as generate_token_stream
from augur.sampling import sample_next_token
from augur.scheduler import ActiveRequest, AsyncContinuousScheduler, GenerationRequest
from augur.text import apply_stop_strings
from augur.tokenizer import Tokenizer
from augur.weights import Weights, load_weights


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=0, le=128)
    temperature: float = Field(default=0.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    output_tokens: int
    generated_tokens: int


class TextGenerator(Protocol):
    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerateResponse]: ...


class AugurEngine:
    def __init__(
        self,
        *,
        model_dir: Path,
        cfg: QwenConfig,
        device: torch.device,
        dtype: torch.dtype,
        tokenizer: Tokenizer | None = None,
        weights: Weights | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer or Tokenizer.from_pretrained(model_dir)
        self.weights = weights or load_weights(
            model_dir / "model.safetensors",
            cfg,
            device=device,
            dtype=dtype,
        )

    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        stop: list[str],
    ) -> GenerateResponse:
        return self.generate_batch(
            [
                GenerationRequest(
                    request_id="single",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop=stop,
                )
            ]
        )[0]

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerateResponse]:
        if not requests:
            return []

        params = requests[0].params
        if any(request.params != params for request in requests):
            raise ValueError("all requests in a batch must have matching generation params")

        input_ids, attention_mask, prompt_lengths = self._encode_prompts(
            [request.prompt for request in requests]
        )
        with torch.no_grad():
            output_ids = generate(
                input_ids,
                self.weights,
                self.cfg,
                max_new_tokens=params.max_new_tokens,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                attention_mask=attention_mask,
            )

        prompt_width = max(prompt_lengths)
        generated_tokens = output_ids.shape[1] - prompt_width
        responses = []
        for row_idx, request in enumerate(requests):
            text = self.tokenizer.decode(output_ids[row_idx, prompt_width:].tolist()).lstrip()
            text = apply_stop_strings(text, request.stop)
            responses.append(
                GenerateResponse(
                    text=text,
                    prompt_tokens=prompt_lengths[row_idx],
                    output_tokens=prompt_lengths[row_idx] + generated_tokens,
                    generated_tokens=generated_tokens,
                )
            )
        return responses

    def generate_stream(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        stop: list[str],
        on_token: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        output_token_ids: list[int] = []
        emitted_text = ""

        with torch.no_grad():
            for next_token in generate_token_stream(
                input_ids,
                self.weights,
                self.cfg,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ):
                if on_token is not None:
                    on_token()
                output_token_ids.extend(next_token[0].tolist())
                text = self.tokenizer.decode(output_token_ids).lstrip()
                stop_positions = [idx for stop_text in stop if (idx := text.find(stop_text)) != -1]
                if stop_positions:
                    text = text[: min(stop_positions)]
                    delta = text[len(emitted_text) :]
                    if delta:
                        yield delta
                    break

                delta = text[len(emitted_text) :]
                emitted_text = text
                if delta:
                    yield delta

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        encoded = [self.tokenizer.encode(prompt) for prompt in prompts]
        max_len = max(len(ids) for ids in encoded)
        input_ids = []
        attention_mask = []
        prompt_lengths = []
        for ids in encoded:
            prompt_lengths.append(len(ids))
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.eos_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return (
            torch.tensor(input_ids, device=self.device),
            torch.tensor(attention_mask, device=self.device),
            prompt_lengths,
        )


class ContinuousEngine:
    def __init__(self, engine: AugurEngine, *, max_slots: int, max_seq_len: int) -> None:
        self._engine = engine
        self._decoder = FixedSlotDecoder(
            engine.weights,
            engine.cfg,
            max_slots=max_slots,
            max_seq_len=max_seq_len,
        )

    def prefill(self, states: list[ActiveRequest]) -> list[int]:
        input_ids, attention_mask, _ = self._engine._encode_prompts(
            [state.request.prompt for state in states]
        )
        slots = torch.tensor([state.slot for state in states], device=self._engine.device)
        logits = self._decoder.prefill(input_ids, slots, attention_mask)
        return self._sample(logits, attention_mask, states)

    def decode(self, states: list[ActiveRequest]) -> list[int]:
        slots = torch.tensor([state.slot for state in states], device=self._engine.device)
        input_ids = torch.tensor(
            [[state.pending_token] for state in states],
            device=self._engine.device,
        )
        logits = self._decoder.decode(input_ids, slots)
        return self._sample(logits, None, states)

    def release(self, states: list[ActiveRequest]) -> None:
        slots = torch.tensor([state.slot for state in states], device=self._engine.device)
        cache = self._decoder.cache
        cache.seq_lens[slots] = 0
        cache.seq_len = int(cache.seq_lens.max().item())

    def _sample(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None,
        states: list[ActiveRequest],
    ) -> list[int]:
        params = states[0].request.params
        if any(state.request.params != params for state in states):
            raise ValueError("continuous batches require matching generation params")
        if attention_mask is not None:
            positions = attention_mask.to(torch.long).sum(dim=1) - 1
            logits = logits[torch.arange(logits.shape[0], device=logits.device), positions]
        else:
            logits = logits[:, -1]
        return sample_next_token(logits, params.temperature, params.top_k, params.top_p).flatten().tolist()


def create_app(engine: AugurEngine) -> FastAPI:
    continuous_scheduler = AsyncContinuousScheduler(
        ContinuousEngine(
            engine,
            max_slots=8,
            max_seq_len=min(2048, engine.cfg.max_position_embeddings),
        ),
        max_slots=8,
        eos_token_id=engine.tokenizer.eos_token_id,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        continuous_scheduler.start()
        yield
        await continuous_scheduler.shutdown()

    app = FastAPI(title="augur", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        token_ids: list[int] = []
        emitted_text = ""
        generation_request = GenerationRequest(
            request_id=uuid4().hex,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop=request.stop,
        )
        async for token_id in continuous_scheduler.stream(generation_request):
            token_ids.append(token_id)
            text = engine.tokenizer.decode(token_ids).lstrip()
            if any(stop_text in text for stop_text in request.stop):
                emitted_text = apply_stop_strings(text, request.stop)
                break
            emitted_text = text
        prompt_tokens = len(engine.tokenizer.encode(request.prompt))
        return GenerateResponse(
            text=emitted_text,
            prompt_tokens=prompt_tokens,
            output_tokens=prompt_tokens + len(token_ids),
            generated_tokens=len(token_ids),
        )

    @app.post("/generate_stream")
    async def generate_stream_endpoint(request: GenerateRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            started_at = time.perf_counter()
            token_ids: list[int] = []
            emitted_text = ""
            generation_request = GenerationRequest(
                request_id=uuid4().hex,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop=request.stop,
            )
            async for token_id in continuous_scheduler.stream(generation_request):
                token_ids.append(token_id)
                text = engine.tokenizer.decode(token_ids).lstrip()
                stop_positions = [idx for stop_text in request.stop if (idx := text.find(stop_text)) != -1]
                if stop_positions:
                    text = text[: min(stop_positions)]
                delta = text[len(emitted_text) :]
                emitted_text = text
                if delta:
                    yield sse_event({"text": delta})
                if stop_positions:
                    break
            elapsed = max(time.perf_counter() - started_at, 1e-6)
            yield sse_event(
                {
                    "generated_tokens": len(token_ids),
                    "tokens_per_second": len(token_ids) / elapsed,
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def sse_event(data: dict[str, object]) -> str:
    return f"data: {json.dumps(data)}\n\n"
