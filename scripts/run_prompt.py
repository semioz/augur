from pathlib import Path

import torch

from augur.config import QwenConfig
from augur.generation import generate
from augur.tokenizer import Tokenizer
from augur.weights import load_weights

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"


def main() -> None:
    prompt = "In a few paragraphs, explain why the night sky looks dark even though the universe has billions of stars."
    cfg = QwenConfig()
    device = "cpu"
    dtype = torch.float32

    tokenizer = Tokenizer.from_pretrained(MODEL_DIR)
    weights = load_weights(MODEL_DIR / "model.safetensors", cfg, device=device, dtype=dtype)

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    output_ids = generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens=200,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
