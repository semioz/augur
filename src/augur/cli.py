import argparse
from pathlib import Path

import torch

from augur.benchmarking import (
    benchmark_generate,
    format_benchmark_csv,
    format_benchmark_result,
    format_comparison,
)
from augur.config import QwenConfig
from augur.generation import generate
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

    return parser.parse_args(argv)


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    add_runtime_args(parser)


def add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    add_runtime_args(parser)
    parser.add_argument(
        "--skip-uncached",
        action="store_true",
        help="Only run the cached benchmark.",
    )
    parser.add_argument("--csv", action="store_true", help="Print benchmark results as CSV.")


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

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
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
    )

    print(decode_generated_text(tokenizer, input_ids, output_ids))


def run_bench(args: argparse.Namespace) -> None:
    cfg = QwenConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=dtype)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    cached = benchmark_generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
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
            )
        )

    if args.csv:
        print(format_benchmark_csv(results), end="")
        return

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"max new tokens: {args.max_new_tokens}")
    print()
    print(format_benchmark_result(cached))
    if args.skip_uncached:
        return

    uncached = results[1]
    print()
    print(format_benchmark_result(uncached))
    print()
    print(format_comparison(uncached, cached))


def decode_generated_text(tokenizer: Tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids.tolist()).lstrip()


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
