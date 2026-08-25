"""Download a Qwen2.5 tokenizer and weights.

Run:  uv run python scripts/download_weights.py --model Qwen/Qwen2.5-1.5B --dest models/qwen2.5-1.5b
"""

import argparse
import re
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b"
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Qwen2.5 model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if MODEL_ID.fullmatch(args.model) is None:
        raise ValueError("model must be a Hugging Face organization/model identifier")
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = args.dest / name
        if path.exists():
            print(f"skip  {name}  ({path.stat().st_size / 1e6:.1f} MB present)")
            continue
        print(f"fetch {name}")
        urlretrieve("https://huggingface.co/" + args.model + "/resolve/main/" + name, path)
        print(f"  -> {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
