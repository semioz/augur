import pytest
import torch
import torch.nn.functional as F

from augur.kernels.cute_flash_attention import flash_attention


@pytest.mark.gpu_kernel
@pytest.mark.parametrize("batch,heads,kv_heads,seq", [(1, 14, 2, 1), (1, 14, 2, 17), (2, 14, 2, 65)])
def test_cute_flash_attention_matches_pytorch(batch, heads, kv_heads, seq):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("cutlass")
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, kv_heads, seq, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(heads // kv_heads, dim=1),
        v.repeat_interleave(heads // kv_heads, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(flash_attention(q, k, v), expected, rtol=1e-2, atol=1e-2)


def test_cute_flash_attention_rejects_cpu_tensors():
    x = torch.ones(1, 2, 1, 64)
    with pytest.raises(ValueError, match="CUDA"):
        flash_attention(x, x, x)
