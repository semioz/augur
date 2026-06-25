from pathlib import Path
from typing import Protocol

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from augur.config import QwenConfig
from augur.generation import generate
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
    def generate_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        stop: list[str],
    ) -> GenerateResponse: ...


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
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        with torch.no_grad():
            output_ids = generate(
                input_ids,
                self.weights,
                self.cfg,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        prompt_tokens = input_ids.shape[1]
        text = self.tokenizer.decode(output_ids[0, prompt_tokens:].tolist()).lstrip()
        text = apply_stop_strings(text, stop)
        return GenerateResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_ids.shape[1],
            generated_tokens=output_ids.shape[1] - prompt_tokens,
        )


def create_app(engine: TextGenerator) -> FastAPI:
    app = FastAPI(title="augur")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        return engine.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            stop=request.stop,
        )

    return app
