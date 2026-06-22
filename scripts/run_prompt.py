import argparse
from pathlib import Path

import torch

from augur.config import QwenConfig
from augur.generation import generate
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"
DEFAULT_PROMPT = (
    "In a few paragraphs, explain why the night sky looks dark even though "
    "the universe has billions of stars."
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = QwenConfig()
    device = "cpu"
    dtype = torch.float32

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen generation on a prompt.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args(argv)


def decode_generated_text(tokenizer: Tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids.tolist()).lstrip()


if __name__ == "__main__":
    main()
