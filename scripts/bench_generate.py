import argparse
from pathlib import Path

import torch

from augur.benchmarking import (
    benchmark_generate,
    format_benchmark_result,
    format_comparison,
)
from augur.config import QwenConfig
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"
DEFAULT_PROMPT = (
    "In a few paragraphs, explain why the night sky looks dark even though "
    "the universe has billions of stars."
)


def main() -> None:
    args = parse_args()
    cfg = QwenConfig()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=dtype)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"max new tokens: {args.max_new_tokens}")
    print()

    cached = benchmark_generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    print(format_benchmark_result(cached))

    if args.skip_uncached:
        return

    print()
    uncached = benchmark_generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
    )
    print(format_benchmark_result(uncached))
    print()
    print(format_comparison(uncached, cached))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy Qwen generation.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--skip-uncached",
        action="store_true",
        help="Only run the cached benchmark.",
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
