from augur.modal_runner import build_bench_args, parse_runner_args


def test_modal_runner_uses_a10g_and_cached_benchmark_defaults() -> None:
    config = parse_runner_args([])

    assert config.gpu == "A10G"
    assert config.model == "Qwen/Qwen2.5-0.5B"
    assert config.model_dir == "qwen2.5-0.5b"
    assert build_bench_args(config) == [
        "bench",
        "--model-dir",
        "/models/qwen2.5-0.5b",
        "--prompt",
        config.prompt,
        "--max-new-tokens",
        "32",
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--skip-uncached",
    ]


def test_modal_runner_accepts_warmup_and_measurement_counts() -> None:
    config = parse_runner_args(["--warmup", "2", "--runs", "5"])

    assert config.warmup == 2
    assert config.runs == 5


def test_modal_runner_enables_decode_profiling() -> None:
    config = parse_runner_args(["--profile"])

    assert config.profile is True


def test_modal_runner_builds_a_vllm_baseline_command() -> None:
    config = parse_runner_args(["--engine", "vllm"])

    assert build_bench_args(config) == [
        "bench-vllm",
        "--model-dir",
        "/models/qwen2.5-0.5b",
        "--prompt",
        config.prompt,
        "--max-new-tokens",
        "32",
        "--device",
        "cuda",
        "--dtype",
        "float16",
    ]
