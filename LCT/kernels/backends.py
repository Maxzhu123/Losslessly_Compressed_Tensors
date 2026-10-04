"""Select individual codec kernels, with model and family overrides."""

from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from .generic.decode import _decode_kernel
from .generic.encode import _encode_kernel
from .. import config
from .device import device_profile


@dataclass(frozen=True)
class KernelBackend:
    # Triton and TileLang expose different compiled-kernel launch interfaces.
    encode: Any
    decode: Any
    encode_tilelang: bool = False
    decode_tilelang: bool = False


GENERIC = KernelBackend(encode=_encode_kernel, decode=_decode_kernel)

# Overrides replace individual kernels while retaining the shared codec pipeline.
FAMILY_BACKENDS: dict[str, KernelBackend] = {"generic": GENERIC}
MODEL_BACKENDS: dict[str, KernelBackend] = {}


@cache
def _tilelang_backend() -> KernelBackend:
    """Import TileLang only when its kernels are selected."""
    from .tilelang.encode import encode_kernel
    from .tilelang.decode import decode_kernel

    return KernelBackend(encode=encode_kernel, decode=decode_kernel,
                         encode_tilelang=True, decode_tilelang=True)


def backend_for(device: torch.device) -> KernelBackend:
    """Resolve exact model -> family -> configured default without caching registries."""
    profile = device_profile(device)
    if profile.model in MODEL_BACKENDS:
        return MODEL_BACKENDS[profile.model]
    if profile.family != "generic" and profile.family in FAMILY_BACKENDS:
        return FAMILY_BACKENDS[profile.family]
    default = FAMILY_BACKENDS["generic"]
    return _tilelang_backend() if config.use_tilelang and default is GENERIC else default
