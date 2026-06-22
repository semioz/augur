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
- **Qwen MLP**: implements the SwiGLU feed-forward path used by Qwen.
- **Decoder blocks**: mirrors the transformer block structure: norm, attention, residual, norm, MLP, residual.
- **Full forward pass**: turns token ids into logits through embeddings, decoder layers, final norm, and LM head.
- **Greedy generation**: generates text by repeatedly choosing the highest-probability next token.
- **Sampling controls**: supports temperature, top-k, and top-p token selection.
- **EOS stopping**: stops generation early when the model emits the configured end-of-sequence token.
- **Prefill/decode split**: processes the prompt once, then decodes one token at a time.
- **Preallocated KV-cache**: stores key/value tensors in fixed cache memory instead of recomputing the whole prompt every token.
- **Cache benchmarking**: measures cached vs uncached generation speed, prefill time, decode time, and tokens/sec.
- **Hugging Face parity tests**: checks core math against Hugging Face Qwen modules so the implementation stays aligned with real Qwen behavior.

## Run

```bash
uv sync
uv run python scripts/download_weights.py
uv run python scripts/run_prompt.py
```

Benchmark:

```bash
uv run python scripts/bench_generate.py --max-new-tokens 32
```

Test:

```bash
uv run pytest -v
uv run ruff check .
```

## Not Yet

- Only Qwen2.5-0.5B is targeted right now.
- No presence, frequency, or repetition penalties yet.
- No batching with padding masks yet.
- No Triton/CUDA kernels yet.
- No paged attention yet.
