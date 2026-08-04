from __future__ import annotations

import pytest
import torch

from augur.rms_norm import rms_norm as torch_backend
from augur.weights import RMSNorm


@pytest.mark.gpu_kernel
def test_rms_norm_parity(gpu_kernel):
    from augur.kernels.rms_norm import rms_norm as triton_backend

    torch.manual_seed(0)
    x = torch.randn(4, 8, 256, device="cuda")
    w = torch.randn(256, device="cuda")
    weight = RMSNorm(weight=w)

    torch.testing.assert_close(
        triton_backend(x, w, 1e-6),
        torch_backend(x, weight, 1e-6),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.gpu_kernel
def test_rms_norm_bf16_parity(gpu_kernel):
    from augur.kernels.rms_norm import rms_norm as triton_backend

    torch.manual_seed(0)
    x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    weight = RMSNorm(weight=w)

    torch.testing.assert_close(
        triton_backend(x, w, 1e-6),
        torch_backend(x, weight, 1e-6),
        rtol=1e-2,
        atol=1e-2,
    )
