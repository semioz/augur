from pathlib import Path
from types import SimpleNamespace

import torch

import augur.cli as cli
from augur.benchmarking import GenerationBenchmarkResult


class TokenizerStub:
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return {
            "hello": [10, 11],
            "short": [12],
            "longer": [13, 14, 15],
        }[text]

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
            "--stop",
            "Human:",
            "--stop",
            "\n",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    assert args.command == "generate"
    assert args.model_dir == Path("models/qwen2.5-0.5b")
    assert args.prompt == ["hello"]
    assert args.max_new_tokens == 7
    assert args.temperature == 0.8
    assert args.top_k == 20
    assert args.top_p == 0.9
    assert args.stop == ["Human:", "\n"]
    assert args.device == "cpu"
    assert args.dtype == "float32"


def test_parse_args_accepts_bench_subcommand_controls() -> None:
    args = cli.parse_args(
        [
            "bench",
            "--model-dir",
            "models/qwen2.5-0.5b",
            "--prompt",
            "hello",
            "--prompt",
            "short",
            "--max-new-tokens",
            "7",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--skip-uncached",
            "--csv",
        ]
    )

    assert args.command == "bench"
    assert args.model_dir == Path("models/qwen2.5-0.5b")
    assert args.prompt == ["hello", "short"]
    assert args.max_new_tokens == 7
    assert args.device == "cpu"
    assert args.dtype == "float32"
    assert args.skip_uncached is True
    assert args.csv is True


def test_generate_command_passes_runtime_and_sampling_controls(monkeypatch, capsys) -> None:
    seen_load_weights_args = {}
    seen_generate_args = {}

    def fake_load_weights(path, cfg, device, dtype):
        seen_load_weights_args.update({"path": path, "device": device, "dtype": dtype})
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_generate(input_ids, weights, cfg, **kwargs):
        assert input_ids.device.type == "cpu"
        assert input_ids.tolist() == [[10, 11]]
        assert kwargs["attention_mask"].tolist() == [[1, 1]]
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
    attention_mask = seen_generate_args.pop("attention_mask")
    assert attention_mask.tolist() == [[1, 1]]
    assert seen_generate_args == {
        "max_new_tokens": 7,
        "use_cache": True,
        "eos_token_id": 99,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.9,
    }
    assert capsys.readouterr().out == "12\n"


def test_generate_command_batches_repeated_prompts_with_padding(monkeypatch, capsys) -> None:
    seen_generate_args = {}

    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_generate(input_ids, weights, cfg, **kwargs):
        assert input_ids.tolist() == [[12, 99, 99], [13, 14, 15]]
        assert kwargs["attention_mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
        seen_generate_args.update(kwargs)
        return torch.tensor([[12, 99, 99, 20], [13, 14, 15, 21]])

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "generate", fake_generate)

    cli.main(
        [
            "generate",
            "--prompt",
            "short",
            "--prompt",
            "longer",
            "--max-new-tokens",
            "1",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    assert seen_generate_args["max_new_tokens"] == 1
    assert capsys.readouterr().out == "[0] 20\n[1] 21\n"


def test_generate_command_applies_stop_strings(monkeypatch, capsys) -> None:
    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_generate(input_ids, weights, cfg, **kwargs):
        return torch.tensor([[10, 11, 20, 21, 22]])

    class StopTokenizer(TokenizerStub):
        def decode(self, ids: list[int]) -> str:
            return " GPUs are useful.Human: ignore this"

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: StopTokenizer()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "generate", fake_generate)

    cli.main(
        [
            "generate",
            "--prompt",
            "hello",
            "--stop",
            "Human:",
        ]
    )

    assert capsys.readouterr().out == "GPUs are useful.\n"


def test_bench_command_prints_text_results_and_comparison(monkeypatch, capsys) -> None:
    seen_calls: list[bool] = []

    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_benchmark_generate(input_ids, weights, cfg, max_new_tokens, use_cache, **kwargs):
        assert input_ids.tolist() == [[10, 11]]
        assert kwargs["attention_mask"].tolist() == [[1, 1]]
        assert max_new_tokens == 7
        seen_calls.append(use_cache)
        return benchmark_result(use_cache=use_cache)

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "benchmark_generate", fake_benchmark_generate)

    cli.main(
        [
            "bench",
            "--model-dir",
            "models/qwen2.5-0.5b",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "7",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    output = capsys.readouterr().out
    assert seen_calls == [True, False]
    assert "device: cpu" in output
    assert "dtype: torch.float32" in output
    assert "kv cache memory: 216.00 KiB" in output
    assert "cache: on" in output
    assert "cache: off" in output
    assert "cached speedup vs uncached total time" in output


def test_bench_command_batches_repeated_prompts_with_padding(monkeypatch, capsys) -> None:
    seen_calls: list[tuple[bool, list[list[int]], list[list[int]]]] = []

    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_benchmark_generate(
        input_ids,
        weights,
        cfg,
        max_new_tokens,
        use_cache,
        attention_mask=None,
    ):
        assert attention_mask is not None
        seen_calls.append((use_cache, input_ids.tolist(), attention_mask.tolist()))
        return benchmark_result(use_cache=use_cache)

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "benchmark_generate", fake_benchmark_generate)

    cli.main(
        [
            "bench",
            "--prompt",
            "short",
            "--prompt",
            "longer",
            "--max-new-tokens",
            "7",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--skip-uncached",
        ]
    )

    assert seen_calls == [(True, [[12, 99, 99], [13, 14, 15]], [[1, 0, 0], [1, 1, 1]])]
    assert "batch size: 2" in capsys.readouterr().out


def test_bench_command_can_print_csv_results(monkeypatch, capsys) -> None:
    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_benchmark_generate(input_ids, weights, cfg, max_new_tokens, use_cache, **kwargs):
        return benchmark_result(use_cache=use_cache)

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "benchmark_generate", fake_benchmark_generate)

    cli.main(
        [
            "bench",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "7",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--csv",
        ]
    )

    assert capsys.readouterr().out == (
        "variant,batch_size,prompt_tokens,generated_tokens_per_sequence,total_generated_tokens,"
        "prefill_seconds,decode_seconds,total_seconds,decode_model_tokens_per_sequence,"
        "total_decode_model_tokens,decode_tokens_per_second,total_tokens_per_second\n"
        "cached,1,2,3,3,0.250000,0.250000,0.500000,2,2,8.000000,6.000000\n"
        "uncached,1,2,3,3,0.000000,0.750000,0.750000,3,3,4.000000,4.000000\n"
    )


def test_bench_command_can_skip_uncached(monkeypatch, capsys) -> None:
    seen_calls: list[bool] = []

    def fake_load_weights(path, cfg, device, dtype):
        return SimpleNamespace(embed_tokens=torch.empty(0, dtype=dtype))

    def fake_benchmark_generate(input_ids, weights, cfg, max_new_tokens, use_cache, **kwargs):
        seen_calls.append(use_cache)
        return benchmark_result(use_cache=use_cache)

    monkeypatch.setattr(cli.Tokenizer, "from_pretrained", staticmethod(lambda model_dir: TokenizerStub()))
    monkeypatch.setattr(cli, "load_weights", fake_load_weights)
    monkeypatch.setattr(cli, "benchmark_generate", fake_benchmark_generate)

    cli.main(
        [
            "bench",
            "--prompt",
            "hello",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--skip-uncached",
            "--csv",
        ]
    )

    assert seen_calls == [True]
    assert "uncached" not in capsys.readouterr().out


def test_decode_generated_text_skips_prompt_tokens() -> None:
    input_ids = torch.tensor([[10, 11, 12]])
    output_ids = torch.tensor([[10, 11, 12, 13, 14]])

    assert cli.decode_generated_text(TokenizerStub(), input_ids, output_ids) == "13 14"


def test_decode_generated_texts_skip_each_prompt_length() -> None:
    output_ids = torch.tensor([[12, 99, 99, 20], [13, 14, 15, 21]])

    assert cli.decode_generated_texts(TokenizerStub(), [1, 3], output_ids) == ["20", "21"]


def test_apply_stop_strings_uses_earliest_match() -> None:
    assert cli.apply_stop_strings("abc STOP def END ghi", ["END", "STOP"]) == "abc "
    assert cli.apply_stop_strings("abc def", ["STOP"]) == "abc def"


def test_resolve_dtype_uses_float32_for_auto_cpu() -> None:
    assert cli.resolve_dtype("auto", torch.device("cpu")) == torch.float32


def benchmark_result(use_cache: bool) -> GenerationBenchmarkResult:
    if use_cache:
        return GenerationBenchmarkResult(
            output_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            use_cache=True,
            prompt_tokens=2,
            generated_tokens_per_sequence=3,
            prefill_seconds=0.25,
            decode_seconds=0.25,
            decode_model_tokens_per_sequence=2,
        )
    return GenerationBenchmarkResult(
        output_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        use_cache=False,
        prompt_tokens=2,
        generated_tokens_per_sequence=3,
        prefill_seconds=0.0,
        decode_seconds=0.75,
        decode_model_tokens_per_sequence=3,
    )
