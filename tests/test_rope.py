"""
Run with:
  uv run pytest tests/test_rope.py -v
"""

import pytest
import torch

from augur.rope import apply_rope


def test_apply_rope_is_not_implemented_yet() -> None:
    q = torch.empty(1, 2, 3, 4)
    k = torch.empty(1, 1, 3, 4)
    position_ids = torch.arange(3).unsqueeze(0)

    with pytest.raises(NotImplementedError):
        apply_rope(q, k, position_ids, rope_theta=1_000_000.0)
