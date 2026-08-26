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

## Packed QKV Projection

Q/K/V weights are concatenated at load time and decoded with one `F.linear` call, then split into Q, K, and V. This preserves the existing attention path and avoids a custom kernel.

| Metric | Baseline median | Packed QKV runs | Packed QKV median | Change |
| --- | ---: | --- | ---: | ---: |
| total generation tokens/sec | 32.56 | 33.67, 34.01, 33.11 | 33.67 | +3.4% |
| decode tokens/sec | 41.77 | 43.20, 43.64, 43.44 | 43.44 | +4.0% |

The improvement is real but small. Keep the change: it is low-risk and validates that reducing projection launches helps, but MLP/projection fusion remains the larger opportunity.

## Comparability

Both benchmarks exclude model-weight loading. vLLM also completes engine compilation and warm-up before its timer starts, while Augur's first prefill can include cold GPU work. Treat this as a directional baseline, not a final apples-to-apples result.

## Change Log

1. **Baseline recorded** — no engine change.
2. **Decode profile captured** — linear projections dominate; attention is deferred.
3. **Packed QKV projection** — +3.4% total throughput and +4.0% decode throughput.
4. **Next: fused MLP investigation** — preserve correctness, then repeat the same three-run benchmark.
