import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def load_run_prompt():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_prompt.py"
    spec = importlib.util.spec_from_file_location("run_prompt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTokenizer:
    def decode(self, ids: list[int]) -> str:
        return " " + " ".join(str(token_id) for token_id in ids)


def test_decode_generated_text_skips_prompt_tokens() -> None:
    run_prompt = load_run_prompt()
    input_ids = torch.tensor([[10, 11, 12]])
    output_ids = torch.tensor([[10, 11, 12, 13, 14]])

    assert run_prompt.decode_generated_text(DummyTokenizer(), input_ids, output_ids) == "13 14"


def test_parse_args_accepts_generation_controls() -> None:
    run_prompt = load_run_prompt()

    args = run_prompt.parse_args(
        [
            "--prompt",
            "hello",
            "--max-new-tokens",
            "7",
            "--temperature",
            "0.8",
            "--top-k",
            "20",
            "--top-p",
            "0.9",
        ]
    )

    assert args.prompt == "hello"
    assert args.max_new_tokens == 7
    assert args.temperature == 0.8
    assert args.top_k == 20
    assert args.top_p == 0.9


def test_main_passes_generation_controls(monkeypatch, capsys) -> None:
    run_prompt = load_run_prompt()
    seen_generate_args = {}

    class TokenizerStub:
        eos_token_id = 99

        def encode(self, text: str) -> list[int]:
            assert text == "hello"
            return [10, 11]

        def decode(self, ids: list[int]) -> str:
            return " " + " ".join(str(token_id) for token_id in ids)

    def fake_generate(input_ids, weights, cfg, **kwargs):
        seen_generate_args.update(kwargs)
        return torch.tensor([[10, 11, 12]])

    monkeypatch.setattr(
        run_prompt.Tokenizer,
        "from_pretrained",
        staticmethod(lambda model_dir: TokenizerStub()),
    )
    monkeypatch.setattr(
        run_prompt,
        "load_weights",
        lambda path, cfg, device, dtype: SimpleNamespace(embed_tokens=torch.empty(0)),
    )
    monkeypatch.setattr(run_prompt, "generate", fake_generate)

    run_prompt.main(
        [
            "--prompt",
            "hello",
            "--max-new-tokens",
            "7",
            "--temperature",
            "0.8",
            "--top-k",
            "20",
            "--top-p",
            "0.9",
        ]
    )

    assert seen_generate_args == {
        "max_new_tokens": 7,
        "use_cache": True,
        "eos_token_id": 99,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.9,
    }
    assert capsys.readouterr().out == "12\n"
