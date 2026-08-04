"""GPU kernel backends for augur compute ops, with a torch fallback.

Nothing in this package imports triton at module scope, so it stays inert on
CPU-only boxes. ``kernels_available()`` is the single gate every op dispatch
uses. A future raw-CUDA backend slots in here behind the same gate.
"""

from __future__ import annotations

import torch


def have_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def kernels_available(device: torch.device | None = None) -> bool:
    if not have_triton():
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    if device is not None and device.type != "cuda":
        return False
    return True
