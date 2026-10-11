import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from augur.weights import NVFP4Linear

_COMPILED: dict[tuple[int, int, int, int], object] = {}

@cute.jit
def _decode_e2m1(code: cutlass.Int32) -> cutlass.Float32:
    magnitude = code & 7
    value = cutlass.Float32(0)
    if magnitude == 1:
        value = cutlass.Float32(0.5)
    elif magnitude >= 2:
        factor = cutlass.Float32(1)
        if magnitude >= 6:
            factor = cutlass.Float32(4)
        elif magnitude >= 4:
            factor = cutlass.Float32(2)
        value = factor * (1 + (magnitude & 1) * 0.5)
    if (code & 8) != 0:
        value = -value
    return value


# ponytail: one warp per output, replace with tiled tensor-core MMA if throughput matters.
@cute.kernel
def _linear_kernel(
    x: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor,
    global_scale: cute.Tensor, out: cute.Tensor, in_features: cutlass.Constexpr,
):
    out_row, token, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()
    acc = cutlass.Float32(0)
    for block in range(in_features // 64):
        byte = block * 32 + lane
        bits = packed[out_row, byte].to(cutlass.Int32)
        scale = scales[out_row, byte // 8].to(cutlass.Float32)
        weight0 = _decode_e2m1(bits & 15)
        weight1 = _decode_e2m1(bits >> 4)
        acc += (
            x[token, byte * 2].to(cutlass.Float32) * weight0
            + x[token, byte * 2 + 1].to(cutlass.Float32) * weight1
        ) * scale
    for offset in [16, 8, 4, 2, 1]:
        acc += cute.arch.shuffle_sync_down(acc, offset)
    if lane == 0:
        out[token, out_row] = (acc * global_scale[0]).to(cutlass.Float16)


@cute.jit
def _launch(
    x: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor,
    global_scale: cute.Tensor, out: cute.Tensor,
    in_features: cutlass.Constexpr, out_features: cutlass.Constexpr,
    tokens: cutlass.Constexpr,
):
    _linear_kernel(x, packed, scales, global_scale, out, in_features).launch(
        grid=(out_features, tokens, 1), block=(32, 1, 1)
    )


def linear(x: Tensor, w: NVFP4Linear) -> Tensor:
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device)[0] < 10:
        raise ValueError("NVFP4 gate requires a Blackwell CUDA GPU")
    if x.dtype != torch.float16 or x.shape[-1] != w.packed.shape[1] * 2:
        raise ValueError("expected FP16 activations matching the NVFP4 weight width")
    if x.shape[-1] % 64:
        raise ValueError("NVFP4 gate requires input width divisible by 64")
    if w.packed.device != x.device or w.scales.device != x.device or w.global_scale.device != x.device:
        raise ValueError("NVFP4 weights and activations must share a device")
    if w.scales.shape != (w.packed.shape[0], x.shape[-1] // 16):
        raise ValueError("invalid NVFP4 scale shape")

    flat = x.contiguous().reshape(-1, x.shape[-1])
    out = torch.empty((flat.shape[0], w.packed.shape[0]), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        tensors = (
            cute.runtime.from_dlpack(flat, assumed_align=16),
            cute.runtime.from_dlpack(w.packed, assumed_align=16),
            cute.runtime.from_dlpack(w.scales, assumed_align=1),
            cute.runtime.from_dlpack(w.global_scale.reshape(1), assumed_align=4),
            cute.runtime.from_dlpack(out, assumed_align=16),
        )
        key = (torch.cuda.current_device(), flat.shape[0], x.shape[-1], w.packed.shape[0])
        if key not in _COMPILED:
            # Compile once per shape: CuTe's JIT launcher otherwise adds milliseconds per token.
            _COMPILED[key] = cute.compile(_launch, *tensors, x.shape[-1], w.packed.shape[0], flat.shape[0])
        _COMPILED[key](*tensors)
    result = out.reshape(*x.shape[:-1], w.packed.shape[0])
    return result if w.bias is None else result + w.bias
