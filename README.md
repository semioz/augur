# augur

A small Qwen inference engine in PyTorch, built as a correctness reference and then accelerated with hand-written Triton-CUDA kernels.

Loads Qwen weights, tokenizes prompts, runs transformer forwards, and generates text.

## Features

- **Qwen weight loading**: loads Qwen checkpoint tensors from `safetensors`, including projection weights and optional attention biases.
- **Qwen tokenizer path**: encodes text with Qwen BPE files and decodes generated token ids back to text.
- **Qwen model config**: uses the Qwen hidden size, attention head counts, key/value head counts, MLP size, RMSNorm epsilon, RoPE theta, and vocabulary size.
- **RMSNorm**: matches Qwen's normalization layer before attention, before the MLP, and after the final decoder layer.
- **RoPE**: applies rotary position embeddings to query and key tensors so attention understands token positions.
- **Grouped-query attention**: supports Qwen's layout where many query heads share fewer key/value heads.
- **Causal masking**: prevents each token from attending to future tokens during generation.
- **Padding masks**: supports right-padded prompt batches so padded tokens do not affect attention or next-token selection.
- **Qwen MLP**: implements the SwiGLU feed-forward path used by Qwen.
- **Decoder blocks**: mirrors the transformer block structure: norm, attention, residual, norm, MLP, residual.
- **Full forward pass**: turns token ids into logits through embeddings, decoder layers, final norm, and LM head.
- **Greedy generation**: generates text by repeatedly choosing the highest-probability next token.
- **Sampling controls**: supports temperature, top-k, and top-p token selection.
- **EOS stopping**: stops generation when each sequence emits the configured end-of-sequence token.
- **Stop strings**: trims decoded CLI output at user-provided stop sequences.
- **Prefill/decode split**: processes the prompt once, then decodes one token at a time.
- **Preallocated KV-cache**: stores key/value tensors in fixed cache memory instead of recomputing the whole prompt every token.
- **Manual prefix cache API**: can prefill and reuse a single-sequence prefix cache for cached generation.
- **Static batched generation**: accepts multiple prompts in one fixed batch through the CLI.
- **KV-cache memory accounting**: reports estimated cache memory for benchmark runs.
- **Cache benchmarking**: measures cached vs uncached generation speed, prefill time, decode time, tokens/sec, and CSV output.
- **Local HTTP server**: serves a simple `/generate` JSON endpoint backed by the same generation path as the CLI.
- **Hugging Face parity tests**: checks core math against Hugging Face Qwen modules so the implementation stays aligned with real Qwen behavior.

## Kernels

GPU backends live in `src/augur/kernels/`. Each hot op keeps its public function
in the main `augur` package as the single entry point; on a CUDA device with
Triton available it dispatches to the kernel, otherwise it falls back to the
reference PyTorch implementation. The torch path always stays as the fallback
and the correctness ground truth.

- **Triton RMSNorm** (`src/augur/kernels/rms_norm.py`): first kernel landed.
  One program per row, fp32 accumulate, matches the torch reference cast order.

Kernels only run when Triton is installed *and* a CUDA device is present —
`augur.kernels.kernels_available()` gates the dispatch, and kernel tests skip
on CPU-only machines via the `gpu_kernel` fixture.

Install the kernel extra (GPU box only):

```bash
uv sync --extra kernel
```

Run the GPU-gated parity tests on a CUDA machine:

```bash
uv run pytest tests/test_kernels -v
```

Upcoming kernels, in porting order (see `docs/superpowers/plans/2026-08-02-triton-kernels.md`):

- **Triton RoPE** (`src/augur/kernels/rope.py`)
- **Triton fused SwiGLU MLP** (`src/augur/kernels/mlp.py`)
- **Triton flash attention** over the contiguous KV cache (`src/augur/kernels/flash_attention.py`) — the largest expected speedup
- **Raw CUDA kernels**: paged/block-table flash attention and custom fused decode land behind the same dispatch seam.

## Run

```bash
uv sync
uv run python scripts/download_weights.py
uv run augur generate --prompt "Write one short sentence about GPUs." --max-new-tokens 40
```

Batched generation:

```bash
uv run augur generate \
  --prompt "Write one sentence about GPUs." \
  --prompt "Write one sentence about CPUs." \
  --max-new-tokens 32 \
  --stop "Human:"
```

Benchmark:

```bash
uv run augur bench --max-new-tokens 32
```

Batched benchmark with CSV output:

```bash
uv run augur bench \
  --prompt "Write one sentence about GPUs." \
  --prompt "Write one sentence about CPUs." \
  --max-new-tokens 32 \
  --csv
```

vLLM baseline (GPU environment with vLLM installed):

```bash
uv run --with vllm augur bench-vllm \
  --model-dir models/qwen2.5-1.5b \
  --draft-model-dir models/qwen2.5-0.5b \
  --num-speculative-tokens 4 \
  --max-new-tokens 32 \
  --device cuda \
  --dtype float16
```

Omit the draft options for regular vLLM decoding.

Greedy speculative decoding with compatible Qwen2.5 draft and target models. Download the target first (the default downloader only fetches 0.5B):

```bash
uv run python scripts/download_weights.py \
  --model Qwen/Qwen2.5-1.5B \
  --dest models/qwen2.5-1.5b
```

```bash
uv run augur speculate \
  --model-dir models/qwen2.5-1.5b \
  --draft-model-dir models/qwen2.5-0.5b \
  --num-draft-tokens 4 \
  --device cuda \
  --dtype float16
```

HTTP server:

```bash
uv run augur serve \
  --model-dir models/qwen2.5-0.5b \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu \
  --dtype float32
```

Generate through HTTP:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write one short sentence about GPUs.",
    "max_new_tokens": 32,
    "temperature": 0.0
  }'
```

Stream generated text with Server-Sent Events:

```bash
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write one short sentence about GPUs.",
    "max_new_tokens": 32,
    "temperature": 0.0
  }'
```

The streaming endpoint emits chunks like:

```text
data: {"text": "GPUs"}

data: {"text": " are"}

data: [DONE]
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Test:

```bash
uv run pytest -v
uv run ruff check .
```

## Not Yet

- Only Qwen2.5-0.5B is targeted right now.
- No presence, frequency, or repetition penalties yet.
- Prefix cache is core-only for now: batch size 1, cached generation only, no attention masks, no CLI flag yet.
- No continuous batching scheduler yet.
- Kernel coverage is partial: only RMSNorm is ported to Triton; RoPE, MLP, and flash attention are still torch (see the Kernels section).
- No paged attention yet.
- No raw CUDA kernels yet.
