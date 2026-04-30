from pathlib import Path

import torch
from transformers import AutoTokenizer

from augur.config import QwenConfig
from augur.generation import generate
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"


def main() -> None:
    prompt = "The capital of France is"
    cfg = QwenConfig()
    device = "cpu"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    weights = load_weights(MODEL_DIR / "model.safetensors", cfg, device=device, dtype=dtype)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = generate(input_ids, weights, cfg, max_new_tokens=20)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
