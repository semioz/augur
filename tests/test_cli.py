import sys
from types import SimpleNamespace

from augur.cli import parse_args, run_bench_vllm


def test_parse_args_registers_vllm_benchmark_command() -> None:
    args = parse_args(
        [
            "bench-vllm",
            "--model-dir",
            "weights/qwen",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "8",
            "--device",
            "cuda",
            "--dtype",
            "float16",
        ]
    )

    assert args.func is run_bench_vllm
    assert str(args.model_dir) == "weights/qwen"
    assert args.prompt == ["hello"]
    assert args.max_new_tokens == 8
    assert args.device == "cuda"
    assert args.dtype == "float16"


def test_parse_args_registers_speculative_generation_command() -> None:
    args = parse_args(
        [
            "speculate",
            "--model-dir",
            "models/qwen2.5-1.5b",
            "--draft-model-dir",
            "models/qwen2.5-0.5b",
            "--num-draft-tokens",
            "6",
        ]
    )

    assert args.func.__name__ == "run_speculate"
    assert str(args.model_dir) == "models/qwen2.5-1.5b"
    assert str(args.draft_model_dir) == "models/qwen2.5-0.5b"
    assert args.num_draft_tokens == 6


def test_run_bench_vllm_reports_end_to_end_generation_throughput(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            calls["llm"] = kwargs

        def generate(self, prompts, sampling_params):
            calls["prompts"] = prompts
            calls["sampling_params"] = sampling_params
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=[3, 4, 5])],
                )
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            calls["sampling_params_kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    args = parse_args(
        [
            "bench-vllm",
            "--model-dir",
            "weights/qwen",
            "--prompt",
            "hello",
            "--max-new-tokens",
            "3",
            "--device",
            "cuda",
            "--dtype",
            "float16",
            "--draft-model-dir",
            "weights/draft",
            "--num-speculative-tokens",
            "4",
        ]
    )

    run_bench_vllm(args)

    assert calls["llm"] == {
        "model": "weights/qwen",
        "device": "cuda",
        "dtype": "half",
        "speculative_config": {
            "method": "draft_model",
            "model": "weights/draft",
            "num_speculative_tokens": 4,
        },
    }
    assert calls["prompts"] == ["hello"]
    assert calls["sampling_params_kwargs"] == {
        "temperature": 0.0,
        "max_tokens": 3,
        "ignore_eos": True,
    }
    output = capsys.readouterr().out
    assert "variant: vllm" in output
    assert "prompt tokens: 2" in output
    assert "generated tokens: 3" in output
    assert "total tokens/sec:" in output
    assert "speculative draft model: weights/draft" in output
