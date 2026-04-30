from torch import Tensor


def apply_rope(
    q: Tensor,
    k: Tensor,
    position_ids: Tensor,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError
