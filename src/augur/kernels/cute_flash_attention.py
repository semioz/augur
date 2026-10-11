import math

import torch
from torch import Tensor


def flash_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        raise ValueError("q, k, v must be float16")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected q, k, v with shape [batch, heads, seq, 64]")
    batch, heads, seq, dim = q.shape
    if dim != 64 or k.shape[0] != batch or k.shape[2:] != (seq, dim):
        raise ValueError("expected matching batch, sequence and head dimension 64")
    if seq < 1 or k.shape[1] < 1 or heads % k.shape[1]:
        raise ValueError("sequence must be nonempty and query heads divisible by KV heads")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must share a CUDA device")

    import cutlass
    import cutlass.cute as cute

    # ponytail: one warp per query rereads KV; use tiled MMA/shared memory for throughput.
    @cute.kernel
    def online_attention(
        Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, output: cute.Tensor,
        group_size: cutlass.Constexpr, scale: cutlass.Constexpr,
    ):
        query, b, h = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        kv_head = h // group_size
        q0 = Q[b, h, query, lane].to(cutlass.Float32)
        q1 = Q[b, h, query, lane + 32].to(cutlass.Float32)
        max_score = cutlass.Float32(-3.4028235e38)
        denominator = cutlass.Float32(0.0)
        numerator0 = cutlass.Float32(0.0)
        numerator1 = cutlass.Float32(0.0)

        for key in range(query + 1):
            dot = q0 * K[b, kv_head, key, lane].to(cutlass.Float32)
            dot += q1 * K[b, kv_head, key, lane + 32].to(cutlass.Float32)
            for offset in [16, 8, 4, 2, 1]:
                dot += cute.arch.shuffle_sync_down(dot, offset)
            score = cute.arch.shuffle_sync(dot, 0) * scale
            new_max = cute.math.max(max_score, score)
            rescale = cute.math.exp(max_score - new_max)
            probability = cute.math.exp(score - new_max)
            denominator = denominator * rescale + probability
            numerator0 = numerator0 * rescale + probability * V[b, kv_head, key, lane].to(cutlass.Float32)
            numerator1 = numerator1 * rescale + probability * V[b, kv_head, key, lane + 32].to(cutlass.Float32)
            max_score = new_max

        output[b, h, query, lane] = (numerator0 / denominator).to(cutlass.Float16)
        output[b, h, query, lane + 32] = (numerator1 / denominator).to(cutlass.Float16)

    @cute.jit
    def launch(
        Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, output: cute.Tensor,
        group_size: cutlass.Constexpr, scale: cutlass.Constexpr,
        batch_size: cutlass.Constexpr, num_heads: cutlass.Constexpr,
        seq_len: cutlass.Constexpr,
    ):
        online_attention(Q, K, V, output, group_size, scale).launch(
            grid=(seq_len, batch_size, num_heads), block=(32, 1, 1)
        )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        launch(
            *(cute.runtime.from_dlpack(t, assumed_align=16) for t in (q, k, v, out)),
            heads // k.shape[1], 1 / math.sqrt(dim), batch, heads, seq,
        )
    return out
