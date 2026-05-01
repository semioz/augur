import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from augur.config import QwenConfig
from augur.generation import generate
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"


def main() -> None:
    prompt = "In a few paragraphs, explain why the night sky looks dark even though the universe has billions of stars."
    cfg = QwenConfig()
    device = "cpu"  
    dtype = torch.float32
    max_new = 200

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    weights = load_weights(MODEL_DIR / "model.safetensors", cfg, device=device, dtype=dtype)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    out = generate(input_ids, weights, cfg, max_new_tokens=max_new, use_cache=True)
    dt = time.perf_counter() - t0
    gen = out.shape[1] - prompt_len
    print(f"cached: {gen} toks in {dt:.2f}s -> {gen / dt:.2f} tok/s (prompt={prompt_len})")


if __name__ == "__main__":
    main()
