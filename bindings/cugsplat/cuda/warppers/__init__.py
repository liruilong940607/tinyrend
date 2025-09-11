from typing import Callable, Optional

import torch.autograd.forward_ad as fwAD
from torch import Tensor


def has_tangent(tensor: Optional[Tensor]) -> bool:
    if tensor is None:
        return False
    return fwAD.unpack_dual(tensor).tangent is not None


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from .._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


