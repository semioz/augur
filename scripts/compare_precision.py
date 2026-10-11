"""Compare lower-precision linear math on Qwen-sized synthetic tensors."""

import math
import statistics

import torch
import torch.nn.functional as F
from torch import Tensor

from augur.nvfp4 import round_e2m1


def error_metrics(actual: Tensor, reference: Tensor) -> tuple[float, float, float]:
    difference = (actual.float() - reference.float()).abs()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return difference.mean().item(), difference.max().item(), relative_l2.item()


def fake_fp4_weight(weight: Tensor, scheme: str) -> Tensor:
    if scheme not in ("MXFP4", "NVFP4"):
        raise ValueError(f"unsupported format: {scheme}")
    block_size = 32 if scheme == "MXFP4" else 16
    if weight.ndim != 2 or weight.shape[1] % block_size or weight.numel() == 0:
        raise ValueError(f"expected nonempty 2D weights with width divisible by {block_size}")
    if weight.dtype != torch.float32 or not torch.isfinite(weight).all():
        raise ValueError("weights must contain finite FP32 values")

    blocks = weight.reshape(weight.shape[0], -1, block_size)
    block_max = blocks.abs().amax(dim=-1, keepdim=True)
    if scheme == "MXFP4":
        exponent = torch.ceil(torch.log2(block_max / 6)).clamp(-127, 127)
        scale = torch.where(block_max == 0, 1.0, torch.exp2(exponent))
    else:
        global_scale = (weight.abs().amax() / (448 * 6)).clamp_min(torch.finfo(torch.float32).tiny)
        local_scale = (block_max / (6 * global_scale)).clamp(2**-9, 448)
        scale = local_scale.to(torch.float8_e4m3fn).float() * global_scale
    return (round_e2m1(blocks / scale) * scale).reshape_as(weight)


def latency_ms(x: Tensor, weight: Tensor) -> float:
    for _ in range(10):
        F.linear(x, weight)
    torch.cuda.synchronize()
    samples = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        F.linear(x, weight)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    weight = torch.randn(4864, 896, device=device) / math.sqrt(896)
    print(f"device: {device}; synthetic FP32 weights, Qwen2.5-0.5B MLP gate shape")
    print("timings exclude dtype conversion; FP32 GPU matmul uses no TF32")
    mx_weight = fake_fp4_weight(weight, "MXFP4")
    nv_weight = fake_fp4_weight(weight, "NVFP4")
    count = weight.numel()
    fp4_sizes = {
        "MXFP4": (count / 2 + count / 32) / 1024**2,
        "NVFP4": (count / 2 + count / 16 + 4) / 1024**2,
    }

    for tokens in (1, 128):
        x = torch.randn(tokens, 896, device=device)
        reference = F.linear(x, weight)
        print(f"\ntokens: {tokens}")
        print("dtype    MAE        max abs    relative L2   weight MiB   linear ms")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = x.to(dtype)
            weights = weight.to(dtype)
            output = F.linear(values, weights)
            mae, max_abs, relative_l2 = error_metrics(output, reference)
            memory = weights.numel() * weights.element_size() / 1024**2
            timing = f"{latency_ms(values, weights):.4f}" if device.type == "cuda" else "n/a"
            print(f"{str(dtype):<8} {mae:.6f}   {max_abs:.6f}   {relative_l2:.6f}      {memory:.2f}      {timing}")

        print("weight-only: FP32 activations; FP4 weights dequantized for FP32 matmul")
        print("FP4 MiB = ideal packed data + scales (not allocated memory); no FP4 speed claim")
        print("weights  MAE        max abs    relative L2   weight MiB")
        for name, weights, memory in (
            ("FP16", weight.half().float(), count * 2 / 1024**2),
            ("BF16", weight.bfloat16().float(), count * 2 / 1024**2),
            ("MXFP4", mx_weight, fp4_sizes["MXFP4"]),
            ("NVFP4", nv_weight, fp4_sizes["NVFP4"]),
        ):
            mae, max_abs, relative_l2 = error_metrics(F.linear(x, weights), reference)
            print(f"{name:<6}   {mae:.6f}   {max_abs:.6f}   {relative_l2:.6f}      {memory:.2f}")


if __name__ == "__main__":
    main()
