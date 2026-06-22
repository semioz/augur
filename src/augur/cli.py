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
from augur.kv_cache import format_bytes, kv_cache_nbytes
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


def decode_generated_text(tokenizer: Tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
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


def apply_stop_strings(text: str, stop: list[str]) -> str:
    stop_positions = [idx for stop_text in stop if (idx := text.find(stop_text)) != -1]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


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
