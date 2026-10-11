import torch

from scripts.compare_kv_precision import pack_nvfp4, unpack_nvfp4
from scripts.compare_precision import fake_fp4_weight


def test_nvfp4_cache_pack_matches_numerical_reference():
    x = torch.zeros(2, 64, dtype=torch.float16)
    x[0, :4] = torch.tensor([5.0, -5.0, 1.0, -0.5])
    x[1, 16] = 0.25

    data, scales, global_scale = pack_nvfp4(x)
    restored = unpack_nvfp4(data, scales, global_scale, torch.float32)
    expected = fake_fp4_weight(x.float(), "NVFP4")

    assert data.shape == (2, 32) and data.dtype == torch.uint8
    assert scales.shape == (2, 4) and scales.dtype == torch.float8_e4m3fn
    assert data[0, 0] == 0xF7
    assert data.numel() + scales.numel() + global_scale.element_size() == 76
    torch.testing.assert_close(restored, expected)


def test_nvfp4_cache_pack_matches_reference_for_signed_values():
    torch.manual_seed(1)
    x = (torch.randn(3, 64) * 5).half()
    data, scales, global_scale = pack_nvfp4(x)

    torch.testing.assert_close(
        unpack_nvfp4(data, scales, global_scale, torch.float32),
        fake_fp4_weight(x.float(), "NVFP4"),
    )
