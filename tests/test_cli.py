from pathlib import Path
from types import SimpleNamespace

import torch

import augur.cli as cli


class TokenizerStub:
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        assert text == "hello"
        return [10, 11]

    def decode(self, ids: list[int]) -> str:
        return " " + " ".join(str(token_id) for token_id in ids)


def test_parse_args_accepts_generate_subcommand_controls() -> None:
    args = cli.parse_args(
        [
            "generate",
            "--model-dir",
            "models/qwen2.5-0.5b",
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
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    assert args.command == "generate"
    assert args.model_dir == Path("models/qwen2.5-0.5b")
    assert args.prompt == "hello"
    assert args.max_new_tokens == 7
    assert args.temperature == 0.8
    assert args.top_k == 20
    assert args.top_p == 0.9
    assert args.device == "cpu"
    assert args.dtype == "float32"


def test_generate_command_passes_runtime_and_sampling_controls(monkeypatch, capsys) -> None:
    seen_load_weights_args = {}
    seen_generate_args = {}

    def fake_load_weights(path, cfg, device, dtype):
        seen_load_weights_args.update({"path": path, "device": device, "dtype": dtype})
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_generate(input_ids, weights, cfg, **kwargs):
        assert input_ids.device.type == "cpu"
        assert input_ids.tolist() == [[10, 11]]
        seen_generate_args.update(kwargs)
        return torch.tensor([[10, 11, 12]])

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "generate", fake_generate)

    cli.main(
        [
            "generate",
            "--model-dir",
            "models/qwen2.5-0.5b",
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
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    assert seen_load_weights_args == {
        "path": Path("models/qwen2.5-0.5b/model.safetensors"),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    assert seen_generate_args == {
        "max_new_tokens": 7,
        "use_cache": True,
        "eos_token_id": 99,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.9,
    }
    assert capsys.readouterr().out == "12\n"


def test_decode_generated_text_skips_prompt_tokens() -> None:
    input_ids = torch.tensor([[10, 11, 12]])
    output_ids = torch.tensor([[10, 11, 12, 13, 14]])

    assert cli.decode_generated_text(TokenizerStub(), input_ids, output_ids) == "13 14"


def test_resolve_dtype_uses_float32_for_auto_cpu() -> None:
    assert cli.resolve_dtype("auto", torch.device("cpu")) == torch.float32
