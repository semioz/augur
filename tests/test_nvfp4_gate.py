import pytest
import torch
import torch.nn.functional as F

from augur.config import QwenConfig
from augur.weights import load_weights


def test_nvfp4_gate_requires_blackwell():
    with pytest.raises(ValueError, match="Blackwell"):
        load_weights("unused", QwenConfig(), device="cpu", dtype=torch.float16, nvfp4_gate=True)


@pytest.mark.gpu_kernel
def test_nvfp4_gate_matches_dequantized_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Blackwell required")
    pytest.importorskip("cutlass")
    from augur.kernels.nvfp4_linear import linear
    from augur.nvfp4 import pack_nvfp4, unpack_nvfp4
    from augur.weights import NVFP4Linear

    torch.manual_seed(1)
    x = torch.randn(2, 1, 64, device="cuda", dtype=torch.float16)
    weight = (torch.randn(16, 64, device="cuda") / 8).half()
    data, scales, global_scale = pack_nvfp4(weight)
    gate = NVFP4Linear(data, scales, global_scale)

    actual = linear(x, gate)
    expected = F.linear(x, unpack_nvfp4(data, scales, global_scale, torch.float16))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu_kernel
def test_nvfp4_gate_reuses_compilation_for_same_shape():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Blackwell required")
    from augur.kernels.nvfp4_linear import _COMPILED, linear
    from augur.nvfp4 import pack_nvfp4
    from augur.weights import NVFP4Linear

    _COMPILED.clear()
    x = torch.ones(1, 1, 64, device="cuda", dtype=torch.float16)
    for factor in (1.0, 2.0):
        data, scales, global_scale = pack_nvfp4(torch.full((16, 64), factor, device="cuda"))
        linear(x, NVFP4Linear(data, scales, global_scale))
    assert len(_COMPILED) == 1


@pytest.mark.gpu_kernel
def test_mlp_runs_with_packed_nvfp4_gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Blackwell required")
    from augur.mlp import mlp
    from augur.nvfp4 import pack_nvfp4, unpack_nvfp4
    from augur.weights import Linear, MLP, NVFP4Linear

    torch.manual_seed(2)
    x = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    gate = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    up = Linear(torch.randn_like(gate))
    down = Linear(torch.randn(64, 16, device="cuda", dtype=torch.float16))
    data, scales, global_scale = pack_nvfp4(gate)
    quantized = mlp(x, MLP(NVFP4Linear(data, scales, global_scale), up, down))
    reference = mlp(x, MLP(Linear(unpack_nvfp4(data, scales, global_scale, x.dtype)), up, down))
    torch.testing.assert_close(quantized, reference, rtol=2e-2, atol=2e-2)
