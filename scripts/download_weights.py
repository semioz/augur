"""Download Qwen2.5-0.5B tokenizer + weight files into models/qwen2.5-0.5b/.

Run:  uv run python scripts/download_weights.py
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main"
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
DEST = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DEST / name
        if path.exists():
            print(f"skip  {name}  ({path.stat().st_size / 1e6:.1f} MB present)")
            continue
        print(f"fetch {name}")
        urlretrieve(f"{BASE}/{name}", path)
        print(f"  -> {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
