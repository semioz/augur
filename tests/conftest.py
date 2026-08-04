from __future__ import annotations

import pytest

import augur.kernels as kernels


@pytest.fixture
def gpu_kernel():
    """Return the kernels package if a triton+CUDA runtime exists, else skip."""
    if not kernels.kernels_available():
        pytest.skip("no Triton/CUDA runtime available")
    return kernels
