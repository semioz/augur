# Augur

A compact Qwen2.5 inference runtime for studying and benchmarking the mechanics of LLM serving. Augur makes the complete path—from checkpoint loading through cached token-by-token decode—explicit, testable, and easy to modify.

It provides a PyTorch reference implementation validated against Hugging Face, plus optional Triton kernels behind the same interfaces. The goal is not to replicate a production scheduler: it is to establish a correct baseline, profile the real decode workload, and retain only optimizations that improve measured performance.

**Try the hosted A10G demo:** [semioz--augur-showcase-augurshowcase-web.modal.run](https://semioz--augur-showcase-augurshowcase-web.modal.run)

## What is here

- Qwen2.5 tokenizer, checkpoint loader, RoPE, grouped-query attention, SwiGLU MLP, RMSNorm, and full LM forward pass
- Cached generation with contiguous or paged KV storage, batched prompts, temperature/top-k/top-p sampling, EOS handling, and stop strings
- Hugging Face parity tests for the core Qwen math
- Local benchmarks that separate prefill from decode, plus a Modal A10G runner and a vLLM baseline
- A small FastAPI server with JSON generation and Server-Sent Events streaming

## Install and run

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python scripts/download_weights.py
uv run augur generate \
  --prompt "Write one short sentence about GPUs." \
  --max-new-tokens 40
```

The downloader stores the default `Qwen/Qwen2.5-0.5B` checkpoint in `models/qwen2.5-0.5b/`. To use the optional Triton kernel path on a CUDA machine:

```bash
uv sync --extra kernel
uv run pytest tests/test_kernels -v
```

## Generate

By default generation is greedy. Add sampling controls when needed:

```bash
uv run augur generate \
  --prompt "Write one sentence about GPUs." \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --max-new-tokens 64
```

Pass `--prompt` more than once for a fixed batch:

```bash
uv run augur generate \
  --prompt "Write one sentence about GPUs." \
  --prompt "Write one sentence about CPUs." \
  --max-new-tokens 32 \
  --stop "Human:"
```

## Server

Start the local API:

```bash
uv run augur serve --device cpu --dtype float32
```

Generate JSON with `POST /generate`:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write one short sentence about GPUs.","max_new_tokens":32}'
```

`POST /generate_stream` emits token text as SSE; `GET /health` reports server health.

## Performance

The current benchmark workload is Modal A10G, Qwen2.5-0.5B FP16, batch 1, a 7-token prompt, and 256 generated tokens. The runner warms the GPU process once, then reports three measurements.

| Engine | Warmed total throughput |
| --- | ---: |
| Augur | 48.35 tok/s |
| vLLM | 300.38 tok/s |

Packed QKV projections and bypassing the redundant causal mask during single-token decode improved Augur throughput by 4% and 8.4%, respectively. vLLM remains substantially faster because it has a production scheduler and optimized execution path; this comparison is directional, not an apples-to-apples serving benchmark.

Run Augur on the same workload:

```bash
uv run --with modal python scripts/modal_gpu.py \
  --warmup 1 --runs 3 --max-new-tokens 256
```

Append `--engine vllm` for the vLLM baseline. See [the benchmark log](docs/benchmark-log.md) for raw runs, profiles, rejected experiments, and methodology.

## Architecture

```text
prompt → tokenizer → prefill → KV cache → one-token decode loop → sampled tokens
```

| Area | Role |
| --- | --- |
| `config.py`, `weights.py`, `tokenizer.py` | Qwen configuration, tensors, and text/token conversion |
| `model.py`, `block.py`, `attention.py`, `mlp.py` | Transformer forward pass |
| `generation.py`, `sampling.py` | Cached generation and token selection |
| `kv_cache.py`, `paged_kv_cache.py`, `prefix_cache.py` | Cache allocation and reuse |
| `benchmarking.py`, `modal_runner.py` | Reproducible measurement harness |
| `kernels/` | Optional Triton implementations behind PyTorch fallbacks |

## Benchmark and develop

```bash
# Compare cached and uncached generation locally.
uv run augur bench --max-new-tokens 32

# Machine-readable benchmark output.
uv run augur bench --max-new-tokens 32 --csv

# Run correctness and style checks.
uv run pytest -v
uv run ruff check .
```

`augur speculate` runs greedy draft-model speculative decoding. `augur --help` lists all CLI options.

## Current scope

Augur currently targets Qwen2.5-0.5B. It is a single-process reference engine, not a production serving stack: it has no continuous batching scheduler or paged-attention kernel. RMSNorm is the only Triton kernel today; attention, RoPE, and MLP use the PyTorch reference path.
