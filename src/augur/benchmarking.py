import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from augur.config import QwenConfig
from augur.kv_cache import new_kv_cache
from augur.model import model
from augur.weights import Weights

Clock = Callable[[], float]


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    output_ids: Tensor
    use_cache: bool
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode_model_tokens: int

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds

    @property
    def decode_tokens_per_second(self) -> float:
        return tokens_per_second(self.decode_model_tokens, self.decode_seconds)

    @property
    def total_tokens_per_second(self) -> float:
        return tokens_per_second(self.generated_tokens, self.total_seconds)


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
    clock: Clock = time.perf_counter,
) -> GenerationBenchmarkResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    if use_cache:
        return _benchmark_cached_generate(input_ids, w, cfg, max_new_tokens, clock)
    return _benchmark_uncached_generate(input_ids, w, cfg, max_new_tokens, clock)


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
            f"prompt tokens: {result.prompt_tokens}",
            f"generated tokens: {result.generated_tokens}",
            f"prefill time: {prefill}",
            f"decode time: {result.decode_seconds:.4f}s",
            f"total time: {result.total_seconds:.4f}s",
            (
                "decode tokens/sec: "
                f"{result.decode_tokens_per_second:.2f} "
                f"({result.decode_model_tokens} measured decode forwards)"
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


@torch.inference_mode()
def _benchmark_uncached_generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    clock: Clock,
) -> GenerationBenchmarkResult:
    output_ids = input_ids
    decode_seconds = 0.0

    for _ in range(max_new_tokens):
        logits, elapsed = _time_forward(
            output_ids.device,
            clock,
            lambda: model(output_ids, w, cfg),
        )
        decode_seconds += elapsed
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        output_ids = torch.cat((output_ids, next_token), dim=1)

    return GenerationBenchmarkResult(
        output_ids=output_ids,
        use_cache=False,
        prompt_tokens=input_ids.shape[1],
        generated_tokens=output_ids.shape[1] - input_ids.shape[1],
        prefill_seconds=0.0,
        decode_seconds=decode_seconds,
        decode_model_tokens=max_new_tokens,
    )


@torch.inference_mode()
def _benchmark_cached_generate(
    input_ids: Tensor,
    w: Weights,
    cfg: QwenConfig,
    max_new_tokens: int,
    clock: Clock,
) -> GenerationBenchmarkResult:
    if max_new_tokens == 0:
        return GenerationBenchmarkResult(
            output_ids=input_ids,
            use_cache=True,
            prompt_tokens=input_ids.shape[1],
            generated_tokens=0,
            prefill_seconds=0.0,
            decode_seconds=0.0,
            decode_model_tokens=0,
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
        lambda: model(input_ids, w, cfg, cache=cache, position_ids=position_ids),
    )

    output_ids = input_ids
    decode_seconds = 0.0
    decode_model_tokens = 0

    for step in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        if step == max_new_tokens - 1:
            break

        position_ids = torch.full(
            (output_ids.shape[0], 1),
            output_ids.shape[1] - 1,
            device=output_ids.device,
            dtype=torch.long,
        )
        logits, elapsed = _time_forward(
            output_ids.device,
            clock,
            lambda: model(next_token, w, cfg, cache=cache, position_ids=position_ids),
        )
        decode_seconds += elapsed
        decode_model_tokens += 1

    return GenerationBenchmarkResult(
        output_ids=output_ids,
        use_cache=True,
        prompt_tokens=input_ids.shape[1],
        generated_tokens=output_ids.shape[1] - input_ids.shape[1],
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_model_tokens=decode_model_tokens,
    )


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
