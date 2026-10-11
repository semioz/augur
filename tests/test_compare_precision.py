import math

import torch

from scripts.compare_precision import error_metrics, fake_fp4_weight, round_e2m1


def test_error_metrics_against_fp32_reference():
    reference = torch.tensor([1.0, 2.0])
    actual = torch.tensor([1.0, 3.0])

    mae, max_abs, relative_l2 = error_metrics(actual, reference)

    assert mae == 0.5
    assert max_abs == 1.0
    assert math.isclose(relative_l2, 1 / math.sqrt(5), rel_tol=1e-6)


def test_e2m1_rounds_to_finite_values_with_ties_to_even():
    x = torch.tensor([0.0, 0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 9.0, -5.0])
    assert round_e2m1(x).tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 6.0, -4.0]


def test_fp4_scales_and_empty_block():
    weight = torch.zeros(1, 32)
    weight[0, 0] = 5.0

    mx = fake_fp4_weight(weight, "MXFP4")
    nv = fake_fp4_weight(weight, "NVFP4")

    assert mx[0, 0] == 4.0
    assert nv[0, 0] == 5.0
    assert torch.isfinite(mx).all() and torch.isfinite(nv).all()
    assert torch.count_nonzero(mx) == torch.count_nonzero(nv) == 1
