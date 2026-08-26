import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPT = "Write one short sentence about GPUs."


@dataclass(frozen=True)
class ModalRunConfig:
    profile: bool
    engine: str
    gpu: str
    timeout: int
    model: str
    model_dir: str
    prompt: str
    max_new_tokens: int
    dtype: str


def parse_runner_args(argv: list[str] | None = None) -> ModalRunConfig:
    parser = argparse.ArgumentParser(description="Run an Augur or vLLM benchmark on Modal.")
    parser.add_argument("--engine", default="augur", choices=["augur", "vllm"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gpu", default="A10G")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-dir", default="qwen2.5-0.5b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args(argv)

    if args.profile and args.engine != "augur":
        parser.error("--profile currently supports only --engine augur")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if Path(args.model_dir).name != args.model_dir:
        parser.error("--model-dir must be a directory name under the Modal model volume")

    return ModalRunConfig(
        profile=args.profile,
        engine=args.engine,
        gpu=args.gpu,
        timeout=args.timeout,
        model=args.model,
        model_dir=args.model_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )


def build_bench_args(config: ModalRunConfig) -> list[str]:
    args = [
        "bench" if config.engine == "augur" else "bench-vllm",
        "--model-dir",
        str(Path("/models") / config.model_dir),
        "--prompt",
        config.prompt,
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--device",
        "cuda",
        "--dtype",
        config.dtype,
    ]
    if config.engine == "augur":
        args.append("--skip-uncached")
    return args
