import argparse
from pathlib import Path

import torch

from augur.cli import encode_prompts, resolve_dtype
from augur.config import QwenConfig
from augur.generation import generate
from augur.tokenizer import Tokenizer
from augur.weights import load_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile cached Augur generation.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = resolve_dtype(args.dtype, device)
    cfg = QwenConfig.from_pretrained(args.model_dir)
    tokenizer = Tokenizer.from_pretrained(args.model_dir)
    weights = load_weights(args.model_dir / "model.safetensors", cfg, device=device, dtype=dtype)
    input_ids, attention_mask, _ = encode_prompts(tokenizer, [args.prompt], device)

    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        generate(
            input_ids,
            weights,
            cfg,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
    torch.cuda.synchronize(device)
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
