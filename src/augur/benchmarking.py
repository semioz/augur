import csv
import io
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.generation import (
    _append_attention_mask,
    _mark_attention_positions,
    _next_token_logits,
    _validate_attention_mask,
)
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.sampling import sample_next_token
from augur.weights import Weights

Clock = Callable[[], float]


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    output_ids: Tensor
    use_cache: bool
    prompt_tokens: int
    generated_tokens_per_sequence: int
    prefill_seconds: float
    decode_seconds: float
    decode_model_tokens_per_sequence: int

    @property
    def batch_size(self) -> int:
        return self.output_ids.shape[0]

    @property
    def total_generated_tokens(self) -> int:
        return self.generated_tokens_per_sequence * self.batch_size

    @property
    def total_decode_model_tokens(self) -> int:
        return self.decode_model_tokens_per_sequence * self.batch_size

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds

    @property
    def decode_tokens_per_second(self) -> float:
        return tokens_per_second(self.total_decode_model_tokens, self.decode_seconds)

    @property
    def total_tokens_per_second(self) -> float:
        return tokens_per_second(self.total_generated_tokens, self.total_seconds)


def tokens_per_second(tokens: int, seconds: float) -> float:
    if tokens <= 0 or seconds <= 0:
        return 0.0
    return tokens / seconds


def benchmark_generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    use_cache: bool,
    attention_mask: Tensor | None = None,
    clock: Clock = time.perf_counter,
) -> GenerationBenchmarkResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if attention_mask is not None:
        _validate_attention_mask(input_ids, attention_mask)

    if use_cache:
        return _benchmark_cached_generate(input_ids, w, cfg, max_new_tokens, attention_mask, clock)
    return _benchmark_uncached_generate(input_ids, w, cfg, max_new_tokens, attention_mask, clock)


def format_benchmark_result(result: GenerationBenchmarkResult) -> str:
    prefill = (
        f"{result.prefill_seconds:.4f}s"
        if result.use_cache
        else "n/a (full sequence is recomputed every token)"
    )
    cache = "on" if result.use_cache else "off"
    return "\n".join(
        [
            f"cache: {cache}",
            f"batch size: {result.batch_size}",
            f"prompt tokens/seq: {result.prompt_tokens}",
            f"generated tokens/seq: {result.generated_tokens_per_sequence}",
            f"total generated tokens: {result.total_generated_tokens}",
            f"prefill time: {prefill}",
            f"decode time: {result.decode_seconds:.4f}s",
            f"total time: {result.total_seconds:.4f}s",
            (
                "decode tokens/sec: "
                f"{result.decode_tokens_per_second:.2f} "
                f"({result.total_decode_model_tokens} measured decode tokens)"
            ),
            f"total tokens/sec: {result.total_tokens_per_second:.2f}",
        ]
    )


def format_comparison(
    uncached: GenerationBenchmarkResult,
    cached: GenerationBenchmarkResult,
) -> str:
    speedup = tokens_per_second(uncached.total_seconds, cached.total_seconds)
    return f"cached speedup vs uncached total time: {speedup:.2f}x"


def format_benchmark_csv(results: Iterable[GenerationBenchmarkResult]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "variant",
            "batch_size",
            "prompt_tokens",
            "generated_tokens_per_sequence",
            "total_generated_tokens",
            "prefill_seconds",
            "decode_seconds",
            "total_seconds",
            "decode_model_tokens_per_sequence",
            "total_decode_model_tokens",
            "decode_tokens_per_second",
            "total_tokens_per_second",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "variant": "cached" if result.use_cache else "uncached",
                "batch_size": result.batch_size,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens_per_sequence": result.generated_tokens_per_sequence,
                "total_generated_tokens": result.total_generated_tokens,
                "prefill_seconds": f"{result.prefill_seconds:.6f}",
                "decode_seconds": f"{result.decode_seconds:.6f}",
                "total_seconds": f"{result.total_seconds:.6f}",
                "decode_model_tokens_per_sequence": result.decode_model_tokens_per_sequence,
                "total_decode_model_tokens": result.total_decode_model_tokens,
                "decode_tokens_per_second": f"{result.decode_tokens_per_second:.6f}",
                "total_tokens_per_second": f"{result.total_tokens_per_second:.6f}",
            }
        )
    return output.getvalue()


@torch.inference_mode()
def _benchmark_uncached_generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    attention_mask: Tensor | None,
    clock: Clock,
) -> GenerationBenchmarkResult:
    output_ids = input_ids
    decode_seconds = 0.0

    for _ in range(max_new_tokens):
        logits, elapsed = _time_forward(
            output_ids.device,
            clock,
            lambda: _model(output_ids, w, cfg, attention_mask=attention_mask),
        )
        decode_seconds += elapsed
        next_token = sample_next_token(_next_token_logits(logits, attention_mask))
        output_ids = torch.cat((output_ids, next_token), dim=1)
        attention_mask = _append_attention_mask(attention_mask, next_token)

    return GenerationBenchmarkResult(
        output_ids=output_ids,
        use_cache=False,
        prompt_tokens=input_ids.shape[1],
        generated_tokens_per_sequence=output_ids.shape[1] - input_ids.shape[1],
        prefill_seconds=0.0,
        decode_seconds=decode_seconds,
        decode_model_tokens_per_sequence=max_new_tokens,
    )


@torch.inference_mode()
def _benchmark_cached_generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    attention_mask: Tensor | None,
    clock: Clock,
) -> GenerationBenchmarkResult:
    if max_new_tokens == 0:
        return GenerationBenchmarkResult(
            output_ids=input_ids,
            use_cache=True,
            prompt_tokens=input_ids.shape[1],
            generated_tokens_per_sequence=0,
            prefill_seconds=0.0,
            decode_seconds=0.0,
            decode_model_tokens_per_sequence=0,
        )

    cache = new_kv_cache(
        cfg,
        batch_size=input_ids.shape[0],
        max_seq_len=input_ids.shape[1] + max_new_tokens,
        device=input_ids.device,
        dtype=w.embed_tokens.dtype,
    )
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand(
        input_ids.shape[0],
        -1,
    )
    logits, prefill_seconds = _time_forward(
        input_ids.device,
        clock,
        lambda: _model(
            input_ids,
            w,
            cfg,
            cache=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ),
    )

    output_ids = input_ids
    decode_seconds = 0.0
    decode_model_tokens = 0

    for step in range(max_new_tokens):
        next_token = sample_next_token(_next_token_logits(logits, attention_mask))
        output_ids = torch.cat((output_ids, next_token), dim=1)
        if attention_mask is not None:
            position_ids = attention_mask.to(torch.long).sum(dim=1, keepdim=True)
            attention_mask = _mark_attention_positions(attention_mask, position_ids)
        else:
            position_ids = None
        if step == max_new_tokens - 1:
            break

        if position_ids is None:
            position_ids = torch.full(
                (output_ids.shape[0], 1),
                output_ids.shape[1] - 1,
                device=output_ids.device,
                dtype=torch.long,
            )
        logits, elapsed = _time_forward(
            output_ids.device,
            clock,
            lambda: _model(
                next_token,
                w,
                cfg,
                cache=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ),
        )
        decode_seconds += elapsed
        decode_model_tokens += 1

    return GenerationBenchmarkResult(
        output_ids=output_ids,
        use_cache=True,
        prompt_tokens=input_ids.shape[1],
        generated_tokens_per_sequence=output_ids.shape[1] - input_ids.shape[1],
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_model_tokens_per_sequence=decode_model_tokens,
    )


def _model(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    cache: object | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    kwargs = {}
    if cache is not None:
        kwargs["cache"] = cache
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(input_ids, w, cfg, **kwargs)


def _time_forward(
    device: torch.device,
    clock: Clock,
    forward: Callable[[], Tensor],
) -> tuple[Tensor, float]:
    _sync_device(device)
    start = clock()
    logits = forward()
    _sync_device(device)
    return logits, clock() - start


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
