import argparse
import importlib
import time
from pathlib import Path

import torch
import uvicorn

from augur.benchmarking import (
    benchmark_generate,
    format_benchmark_csv,
    format_benchmark_result,
    format_comparison,
    tokens_per_second,
)
from augur.config import QwenConfig
from augur.generation import generate, generate_speculative
from augur.kv_cache import format_bytes, kv_cache_nbytes
from augur.server import AugurEngine, create_app
from augur.text import apply_stop_strings
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b"
DEFAULT_PROMPT = (
    "In a few paragraphs, explain why the night sky looks dark even though "
    "the universe has billions of stars."
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen inference engine utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a prompt.")
    add_generate_args(generate_parser)
    generate_parser.set_defaults(func=run_generate)

    bench_parser = subparsers.add_parser("bench", help="Benchmark cached and uncached generation.")
    add_bench_args(bench_parser)
    bench_parser.set_defaults(func=run_bench)

    vllm_bench_parser = subparsers.add_parser("bench-vllm", help="Benchmark vLLM as a baseline.")
    add_vllm_bench_args(vllm_bench_parser)
    vllm_bench_parser.set_defaults(func=run_bench_vllm)

    speculate_parser = subparsers.add_parser(
        "speculate", help="Generate with a draft and target model."
    )
    add_speculate_args(speculate_parser)
    speculate_parser.set_defaults(func=run_speculate)

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP generation server.")
    add_serve_args(serve_parser)
    serve_parser.set_defaults(func=run_serve)

    return parser.parse_args(argv)


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--stop", action="append", default=[])
    add_runtime_args(parser)


def add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    add_runtime_args(parser)
    parser.add_argument(
        "--skip-uncached",
        action="store_true",
        help="Only run the cached benchmark.",
    )
    parser.add_argument("--csv", action="store_true", help="Print benchmark results as CSV.")


def add_speculate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--draft-model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    add_runtime_args(parser)


def add_vllm_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--draft-model-dir", type=Path)
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    add_runtime_args(parser)


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    add_runtime_args(parser)


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )


def run_generate(args: argparse.Namespace) -> None:
    cfg = QwenConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=dtype)

    prompts = args.prompt or [DEFAULT_PROMPT]
    input_ids, attention_mask, prompt_lengths = encode_prompts(tokenizer, prompts, device)
    output_ids = generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        attention_mask=attention_mask,
    )

    generated_texts = [
        apply_stop_strings(text, args.stop)
        for text in decode_generated_texts(tokenizer, prompt_lengths, output_ids)
    ]
    if len(generated_texts) == 1:
        print(generated_texts[0])
        return
    for idx, text in enumerate(generated_texts):
        print(f"[{idx}] {text}")


def run_speculate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    target_cfg = QwenConfig.from_pretrained(args.model_dir)
    draft_cfg = QwenConfig.from_pretrained(args.draft_model_dir)
    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    target_weights = load_weights(
        args.model_dir / "model.safetensors", target_cfg, device=device, dtype=dtype
    )
    draft_weights = load_weights(
        args.draft_model_dir / "model.safetensors", draft_cfg, device=device, dtype=dtype
    )
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    output_ids = generate_speculative(
        input_ids,
        draft_weights,
        draft_cfg,
        target_weights,
        target_cfg,
        max_new_tokens=args.max_new_tokens,
        num_draft_tokens=args.num_draft_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(decode_generated_text(tokenizer, input_ids, output_ids))


def run_bench(args: argparse.Namespace) -> None:
    cfg = QwenConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=dtype)
    prompts = args.prompt or [DEFAULT_PROMPT]
    input_ids, attention_mask, _ = encode_prompts(tokenizer, prompts, device)

    cached = benchmark_generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        attention_mask=attention_mask,
    )
    results = [cached]
    if not args.skip_uncached:
        results.append(
            benchmark_generate(
                input_ids,
                weights,
                cfg,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
                attention_mask=attention_mask,
            )
        )

    if args.csv:
        print(format_benchmark_csv(results), end="")
        return

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"batch size: {input_ids.shape[0]}")
    print(f"max new tokens: {args.max_new_tokens}")
    print(
        "kv cache memory: "
        + format_bytes(
            kv_cache_nbytes(
                cfg,
                batch_size=input_ids.shape[0],
                max_seq_len=input_ids.shape[1] + args.max_new_tokens,
                dtype=dtype,
            )
        )
    )
    print()
    print(format_benchmark_result(cached))
    if args.skip_uncached:
        return

    uncached = results[1]
    print()
    print(format_benchmark_result(uncached))
    print()
    print(format_comparison(uncached, cached))


def run_bench_vllm(args: argparse.Namespace) -> None:
    try:
        vllm = importlib.import_module("vllm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install vLLM to use this benchmark: uv run --with vllm augur bench-vllm ..."
        ) from exc

    device = resolve_device(args.device)
    dtype = {
        "auto": "half" if device.type == "cuda" else "float",
        "float32": "float",
        "float16": "half",
        "bfloat16": "bfloat16",
    }[args.dtype]
    prompts = args.prompt or [DEFAULT_PROMPT]
    llm_kwargs: dict[str, object] = {"model": str(args.model_dir), "dtype": dtype}
    if args.draft_model_dir is not None:
        llm_kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": str(args.draft_model_dir),
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    llm = vllm.LLM(**llm_kwargs)
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    generated_tokens = sum(
        len(completion.token_ids) for output in outputs for completion in output.outputs
    )

    print("variant: vllm")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    if args.draft_model_dir is not None:
        print(f"speculative draft model: {args.draft_model_dir}")
    print(f"batch size: {len(prompts)}")
    print(f"prompt tokens: {prompt_tokens}")
    print(f"generated tokens: {generated_tokens}")
    print(f"total time: {elapsed:.4f}s")
    print(f"total tokens/sec: {tokens_per_second(generated_tokens, elapsed):.2f}")
    metrics = getattr(outputs[0], "metrics", None) if len(outputs) == 1 else None
    if metrics is not None:
        arrival_time = getattr(metrics, "arrival_time", None)
        first_token_time = getattr(metrics, "first_token_time", None)
        last_token_time = getattr(metrics, "last_token_time", None)
        if (
            arrival_time is not None
            and first_token_time is not None
            and last_token_time is not None
        ):
            ttft = first_token_time - arrival_time
            decode_seconds = last_token_time - first_token_time
            print(f"ttft: {ttft:.4f}s")
            print(f"decode time: {decode_seconds:.4f}s")
            print(f"decode tokens/sec: {tokens_per_second(generated_tokens, decode_seconds):.2f}")


def run_serve(args: argparse.Namespace) -> None:
    cfg = QwenConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    engine = AugurEngine(
        model_dir=args.model_dir,
        cfg=cfg,
        device=device,
        dtype=dtype,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def decode_generated_text(
    tokenizer: Tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor
) -> str:
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids.tolist()).lstrip()


def encode_prompts(
    tokenizer: Tokenizer,
    prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = [tokenizer.encode(prompt) for prompt in prompts]
    if not encoded:
        raise ValueError("at least one prompt is required")
    max_len = max(len(ids) for ids in encoded)
    pad_token_id = tokenizer.eos_token_id

    input_ids = []
    attention_mask = []
    prompt_lengths = []
    for ids in encoded:
        prompt_lengths.append(len(ids))
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attention_mask, device=device),
        prompt_lengths,
    )


def decode_generated_texts(
    tokenizer: Tokenizer,
    prompt_lengths: list[int],
    output_ids: torch.Tensor,
) -> list[str]:
    prompt_width = max(prompt_lengths)
    return [
        tokenizer.decode(output_ids[row_idx, prompt_width:].tolist()).lstrip()
        for row_idx in range(len(prompt_lengths))
    ]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
