# Qwen2.5-0.5B Architecture Notes

## Exact Model

Use this model first:

```text
Qwen/Qwen2.5-0.5B
```

This is **not a 2.5B-parameter model**. The name means **Qwen version 2.5**,
model size **0.5B**. Qwen2.5 open-weight dense sizes include 0.5B, 1.5B, 3B,
7B, 14B, 32B, and 72B. There is no standard `Qwen2.5-2.5B` checkpoint in that
main family.

For this repo, `Qwen/Qwen2.5-0.5B` is the right first target because it has the
modern Qwen inference architecture while staying small enough to inspect and
debug locally.

## What Kind Of Model Is It?

Qwen2.5-0.5B is a **decoder-only causal language model**.

That means:

1. It reads a prefix of tokens.
2. Each token can attend only to previous tokens and itself.
3. The model predicts logits for the next token.
4. Generation loops by appending one sampled/generated token at a time.

High-level forward pass:

```text
input_ids
  -> token embeddings
  -> decoder layer 0
  -> decoder layer 1
  -> ...
  -> decoder layer 23
  -> final RMSNorm
  -> LM head
  -> logits over vocab
```

The LM head is tied to the token embedding matrix for this checkpoint.

## Core Config

From the model config:

```text
architecture:             Qwen2ForCausalLM
model_type:               qwen2
vocab_size:               151936
hidden_size:              896
intermediate_size:        4864
num_hidden_layers:        24
num_attention_heads:      14
num_key_value_heads:      2
head_dim:                 64
num_key_value_groups:     7
max_position_embeddings:  32768
sliding_window:           32768
use_sliding_window:       false
hidden_act:               silu
rms_norm_eps:             1e-6
rope_theta:               1000000.0
attention_dropout:        0.0
tie_word_embeddings:      true
torch_dtype:              bfloat16
```

Two derived values matter a lot:

```text
head_dim = hidden_size / num_attention_heads
         = 896 / 14
         = 64

num_key_value_groups = num_attention_heads / num_key_value_heads
                     = 14 / 2
                     = 7
```

That `7` is the GQA repeat factor: each K/V head serves seven query heads.

## Decoder Layer

Each decoder layer has this shape:

```text
residual = x
x = RMSNorm(x)
x = self_attention(x)
x = residual + x

residual = x
x = RMSNorm(x)
x = mlp(x)
x = residual + x
```

In code terms:

```python
x = x + attention(rms_norm(x, input_layernorm), self_attn)
x = x + mlp(rms_norm(x, post_attention_layernorm), mlp_weights)
```

This is a **pre-norm** transformer block: normalization happens before the
attention and MLP sublayers, not after the residual add.

## RMSNorm

Qwen uses RMSNorm instead of LayerNorm.

LayerNorm subtracts the mean and divides by standard deviation. RMSNorm skips
mean subtraction and normalizes by root mean square:

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms
out = y * weight
```

Implementation shape:

```python
variance = x.float().pow(2).mean(-1, keepdim=True)
y = x.float() * torch.rsqrt(variance + eps)
out = weight * y.to(x.dtype)
```

There is no bias term.

## Attention

Qwen2.5-0.5B uses grouped-query attention.

Projection weights:

```text
q_proj: [896, 896]
k_proj: [128, 896]
v_proj: [128, 896]
o_proj: [896, 896]
```

Why K/V are 128 wide:

```text
num_key_value_heads * head_dim = 2 * 64 = 128
```

Attention flow:

```text
x
  -> q_proj, k_proj, v_proj
  -> reshape Q to [batch, 14, seq, 64]
  -> reshape K to [batch, 2,  seq, 64]
  -> reshape V to [batch, 2,  seq, 64]
  -> apply RoPE to Q and K
  -> repeat K/V heads from 2 to 14
  -> causal scaled dot-product attention
  -> merge heads back to [batch, seq, 896]
  -> o_proj
```

The attention score scale is:

```text
1 / sqrt(head_dim) = 1 / sqrt(64) = 1 / 8
```

## RoPE

Qwen does not use learned positional embeddings like GPT-2. It uses rotary
positional embeddings on Q and K.

Important points:

- RoPE is applied after Q/K projection and head reshaping.
- RoPE is applied to Q and K, not V.
- The position index must be absolute within the sequence.
- During cached decoding, the next token position is the cache length, not 0.
- This checkpoint uses `rope_theta = 1000000.0`.

For an inference engine, this means `attention.py` must accept `position_ids` or
a cache-position value. Otherwise generation with KV cache will produce wrong
answers.

## MLP

Qwen uses SwiGLU, not GPT-2 GELU.

Weights:

```text
gate_proj: [4864, 896]
up_proj:   [4864, 896]
down_proj: [896, 4864]
```

Forward:

```python
hidden = silu(gate_proj(x)) * up_proj(x)
out = down_proj(hidden)
```

The gate branch decides what information passes through. The up branch carries
the candidate features. The elementwise multiply is the GLU part.

## Embeddings And LM Head

Token embedding:

```text
embed_tokens.weight: [151936, 896]
```

For this checkpoint, `tie_word_embeddings = true`, so the LM head reuses the
same matrix:

```text
logits = hidden @ embed_tokens.weight.T
```

The logits shape is:

```text
[batch, seq, vocab_size] = [batch, seq, 151936]
```

## Tokenizer

The tokenizer class is `Qwen2Tokenizer`.

Special token details from `tokenizer_config.json`:

```text
eos_token: <|endoftext|>
pad_token: <|endoftext|>
<|endoftext|>: 151643
<|im_start|>:   151644
<|im_end|>:     151645
```

The base model is pretrained, not instruction tuned. It can generate text, but
it is not the best checkpoint for chat behavior. For chat templates and
assistant-style behavior, use the instruct checkpoint later:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

For learning the engine, the base checkpoint is simpler.

## KV Cache

KV cache stores keys and values per layer so decoding does not recompute the
whole prefix every step.

For this model, per-layer cache shape should be:

```text
key_cache:   [batch, 2, past_seq, 64]
value_cache: [batch, 2, past_seq, 64]
```

Do not store repeated K/V heads in the cache. Store the compact 2-head GQA
version and repeat to 14 heads only inside attention.

Prefill:

```text
input seq len = prompt length
compute Q/K/V for all prompt tokens
apply RoPE with positions [0, 1, ..., prompt_len - 1]
store K/V in cache
```

Decode:

```text
input seq len = 1
position = current cache length
compute Q/K/V for the new token
apply RoPE using the absolute position
append K/V to cache
attend over cached prefix + current token
```

This is the first major inference-engineering feature to implement after plain
attention works.

## Speculative Decoding

Speculative decoding works with Qwen, but it needs two models:

```text
draft model:  smaller/faster proposer
target model: larger/verifying model
```

Good learning setup:

```text
draft:  Qwen/Qwen2.5-0.5B
target: Qwen/Qwen2.5-1.5B or Qwen/Qwen2.5-3B
```

First implement normal greedy cached generation. Then implement a simple greedy
speculative path:

1. Draft model proposes `k` tokens.
2. Target model verifies those `k` positions in one forward pass.
3. Accept the longest prefix where target greedy tokens match draft tokens.
4. If a draft token is rejected, use the target token at that position.

After that works, add the full sampling acceptance rule.

## Implementation Order For This Repo

Current foundation:

```text
config.py
weights.py
mlp.py
tokenizer.py
```

Next files:

1. `rms_norm.py`
2. `rope.py`
3. `attention.py`
4. `kv_cache.py`
5. `block.py`
6. `model.py`
7. `sampling.py`
8. `generation.py`
9. `cli.py`

The important rule: test each component against Hugging Face before composing
the full model.

## Sources

- Qwen2.5-0.5B model card:
  https://huggingface.co/Qwen/Qwen2.5-0.5B
- Qwen2.5-0.5B config:
  https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
- Qwen2.5 Technical Report:
  https://arxiv.org/abs/2412.15115
- Hugging Face Qwen2 Transformers documentation:
  https://huggingface.co/docs/transformers/model_doc/qwen2
