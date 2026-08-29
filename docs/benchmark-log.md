# Augur Performance Log

## Workload

- Hardware: Modal A10G
- Model: `Qwen/Qwen2.5-0.5B` from `/models/qwen2.5-0.5b`
- Precision: FP16
- Batch size: 1
- Prompt: 7 tokens (`Write one short sentence about GPUs.`)
- Generated tokens: 256

## Baseline

| Engine | Metric | Runs (tokens/sec) | Median (tokens/sec) |
| --- | --- | --- | ---: |
| Augur | total generation | 32.77, 26.46, 32.56 | 32.56 |
| Augur | decode only | 42.42, 33.18, 41.77 | 41.77 |
| vLLM | total generation | 281.74, 294.55, 288.48 | 288.48 |

vLLM's median generation throughput is 8.9x Augur's under this harness.

## Decode Profile

A cached 256-token Augur generation profile on the same A10G reported these leading GPU operations:

| Operation | Self CUDA time share |
| --- | ---: |
| `aten::mm` | 43.7% |
| CUTLASS tensor-core kernels | 16.3% |
| GEMV kernels | 14.9% |
| `aten::addmm` | 8.0% |
| `aten::bmm` | 2.2% |

The profile does not support attention as the first optimization: projection and MLP linear layers dominate this short-context decode workload. The profiler runs asynchronous generation rather than the benchmark's per-token synchronization, so use it to rank operation classes, not to replace the wall-clock benchmark.

## Nsight Systems Trace

A real cached 256-token Augur decode was captured on Modal A10G with `nsys profile --trace=cuda,nvtx --sample=none`. A second trace used `cudaProfilerStart/Stop` after a warm-up to exclude initialization. The warmed report is retained at `augur-nsys-traces/augur-warmed-decode.nsys-rep` on the Modal Volume and downloaded locally to `/tmp/augur-nsys/augur-warmed-decode.nsys-rep`.

CUDA/NVTX tracing works under Modal gVisor; only CPU sampling is unavailable. The warmed trace recorded 9,478 `cudaLaunchKernel` calls with 4.7 µs median API duration and 47.3 ms aggregate launch API time. That is under 1% of the roughly 5.7 s eager decode benchmark, so kernel-launch overhead alone does not justify the static-cache CUDA Graph refactor.

## Packed QKV Projection

Q/K/V weights are concatenated at load time and decoded with one `F.linear` call, then split into Q, K, and V. This preserves the existing attention path and avoids a custom kernel.

| Metric | Baseline median | Packed QKV runs | Packed QKV median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 32.56 | 33.67, 34.01, 33.11 | 33.67 | +3.4% |
| decode tokens/sec | 41.77 | 43.20, 43.64, 43.44 | 43.44 | +4.0% |

The improvement is real but small. Keep the change: it is low-risk and validates that reducing projection launches helps, but MLP/projection fusion remains the larger opportunity.

## Cached Single-Token Causal Mask Bypass

Cached decode forwards contain one query token and only past/current KV entries, so their causal mask is always false. The attention path now skips mask construction and `masked_fill` when `seq == 1`; prefill and multi-token forwards still use the causal mask.

| Metric | Prior warmed median | Mask-bypass runs | Mask-bypass median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 44.59 | 48.35, 49.20, 48.15 | 48.35 | +8.4% |
| decode tokens/sec | 44.60 | 48.39, 49.24, 48.18 | 48.39 | +8.5% |

Keep the change. It removes an unnecessary per-layer decode operation without changing attention semantics.

## Shared RoPE Tables

RoPE cos/sin tables depend only on position IDs, head dimension, dtype, and `rope_theta`; the prior path rebuilt them in every attention layer. The model now builds them once per forward and passes the same tensors through all layers.

| Metric | Prior warmed median | Shared-RoPE runs | Shared-RoPE median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 48.44 | 55.54, 55.42, 55.75 | 55.54 | +14.7% |
| decode tokens/sec | 48.48 | 55.59, 55.46, 55.80 | 55.59 | +14.7% |

Keep the change. It removes 23 redundant RoPE table constructions per forward without changing model outputs.

## Rejected: Fused SwiGLU Activation

A Triton kernel fused the `SiLU(gate) * up` elementwise step, while leaving the gate, up, and down GEMVs unchanged. It matched PyTorch FP16 output on A10G but added enough launch/dispatch overhead to regress throughput.

| Metric | Shared-RoPE median | Fused-SwiGLU runs | Fused-SwiGLU median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 55.54 | 54.71, 54.11, 54.05 | 54.11 | -2.6% |
| decode tokens/sec | 55.59 | 54.75, 54.16, 54.09 | 54.16 | -2.6% |

The implementation was reverted. A useful MLP kernel must also reduce projection/intermediate-memory work, not merely fuse the elementwise activation.

## Warmed Modal Runs

The Modal runner now performs one warm-up followed by three measurements in the same remote process.

| Engine | Metric | Measurement runs | Median |
| --- | --- | --- | ---: |
| Augur | prefill time | 24.4, 24.2, 24.6 ms | 24.4 ms |
| Augur | decode throughput | 44.10, 45.26, 44.60 tok/s | 44.60 tok/s |
| Augur | total throughput | 44.09, 45.25, 44.59 tok/s | 44.59 tok/s |
| vLLM | total throughput | 300.80, 300.38, 300.34 tok/s | 300.38 tok/s |

The warmed vLLM/Augur total-throughput ratio is 6.7x. vLLM 0.28's offline `LLM.generate` path did not expose per-request timing fields and rejects the newer per-request-metrics engine argument, so this runner cannot yet report vLLM TTFT or pure decode throughput. That requires a vLLM server or async streaming benchmark path.

## Rejected: Packed SwiGLU Gate/Up Projection

Gate and up weights were concatenated at load time, evaluated by one `F.linear`, then split before `SiLU(gate) * up`. On this batch-1 decode workload, the larger fused GEMV is slower than two separate projections.

| Metric | Warmed QKV median | Packed gate/up runs | Packed gate/up median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 44.59 | 37.55, 37.32, 37.35 | 37.35 | -16.2% |
| decode tokens/sec | 44.60 | 37.56, 37.34, 37.36 | 37.36 | -16.2% |

The implementation was reverted. A real fused SwiGLU kernel remains a separate possibility, but simple weight packing is not beneficial here.

## Rejected: `torch.compile`

`torch.compile(model, dynamic=True)` was benchmarked with the same one-warm-up/three-measurement runner. Dynamo graph-broke at `position_ids.max().item()` in KV-cache writes, then hit its recompilation limit because the Python `layer_idx` changes per layer. The compiled fragments are slower than eager execution.

| Metric | Warmed QKV median | `torch.compile` runs | `torch.compile` median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 44.59 | 34.28, 34.48, 33.90 | 34.28 | -23.1% |
| decode tokens/sec | 44.60 | 34.30, 34.50, 33.93 | 34.30 | -23.1% |

The implementation was reverted. Revisit compilation only after the KV write path is tensorized and layer-specific work is made compile-friendly.

## Comparability

Both benchmarks exclude model-weight loading. vLLM also completes engine compilation and warm-up before its timer starts, while Augur's first prefill can include cold GPU work. Treat this as a directional baseline, not a final apples-to-apples result.

## Change Log

1. **Baseline recorded** — no engine change.
2. **Decode profile captured** — linear projections dominate; attention is deferred.
3. **Packed QKV projection** — +3.4% total throughput and +4.0% decode throughput.
4. **Warmed Modal comparison** — Augur 44.59 total tok/s; vLLM 300.38 total tok/s.
5. **Packed SwiGLU gate/up rejected** — -16.2% total/decode throughput; reverted.
6. **`torch.compile` rejected** — graph breaks/recompilation caused a 23.1% throughput regression; reverted.
7. **Warmed Nsight Systems trace captured** — 47.3 ms aggregate kernel-launch API time; CUDA Graph is not the next priority.
8. **Cached single-token causal mask bypass** — +8.4% total throughput and +8.5% decode throughput.
9. **Shared RoPE tables** — +14.7% total/decode throughput by constructing cos/sin once per forward.
10. **Fused SwiGLU activation rejected** — -2.6%; elementwise fusion alone is too small.
11. **Next: profile GPU kernels rather than refactor for CUDA Graph** — simple MLP packing is also not beneficial.
