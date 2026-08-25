import importlib.util
from pathlib import Path

import pytest


def load_downloader():
    path = Path(__file__).parents[1] / "scripts" / "download_weights.py"
    spec = importlib.util.spec_from_file_location("download_weights", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_weights_accepts_model_and_destination() -> None:
    downloader = load_downloader()

    args = downloader.parse_args(["--model", "Qwen/Qwen2.5-1.5B", "--dest", "models/qwen2.5-1.5b"])

    assert args.model == "Qwen/Qwen2.5-1.5B"
    assert args.dest == Path("models/qwen2.5-1.5b")


def test_download_weights_rejects_non_hugging_face_model_ids(monkeypatch, tmp_path) -> None:
    downloader = load_downloader()
    monkeypatch.setattr(downloader, "urlretrieve", lambda *_: pytest.fail("download attempted"))

    with pytest.raises(ValueError, match="model"):
        downloader.main(["--model", "file:///etc/passwd", "--dest", str(tmp_path)])
