from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from augur.config import QwenConfig
from augur.generation import generate
from augur.scheduler import AsyncBatchScheduler, GenerationRequest
from augur.text import apply_stop_strings
from augur.tokenizer import Tokenizer
from augur.weights import Weights, load_weights


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=0)
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


def create_app(engine: TextGenerator) -> FastAPI:
    scheduler: AsyncBatchScheduler[GenerateResponse] = AsyncBatchScheduler(engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler.start()
        yield
        await scheduler.shutdown()

    app = FastAPI(title="augur", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        return await scheduler.generate(
            GenerationRequest(
                request_id=uuid4().hex,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop=request.stop,
            )
        )

    return app
