# augur

A small Qwen inference engine written from scratch in PyTorch as a correctness reference before Triton/CUDA work.

The goal is to mirror the real Qwen inference path closely enough that the model can load actual Qwen weights, tokenize a prompt, run transformer forward passes, and generate text.

## Features

- **Real Qwen weight loading**: loads Qwen checkpoint tensors from `safetensors`, including projection weights and optional attention biases.
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
- **Static batched generation**: accepts multiple prompts in one fixed batch through the CLI.
- **KV-cache memory accounting**: reports estimated cache memory for benchmark runs.
- **Cache benchmarking**: measures cached vs uncached generation speed, prefill time, decode time, tokens/sec, and CSV output.
- **Hugging Face parity tests**: checks core math against Hugging Face Qwen modules so the implementation stays aligned with real Qwen behavior.

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

Test:

```bash
uv run pytest -v
uv run ruff check .
```

## Not Yet

- Only Qwen2.5-0.5B is targeted right now.
- No presence, frequency, or repetition penalties yet.
- No continuous batching scheduler yet.
- No Triton/CUDA kernels yet.
- No paged attention yet.
