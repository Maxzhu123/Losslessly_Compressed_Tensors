"""Shared codec kernels and Triton/TileLang engine selection.

Triton decorators import their GPU-selected settings from kernels/autotune.py.
This module binds the shared entry points once; it does not select or rewrap
GPU-specific kernel packages.
"""
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import torch

from .. import config
from .common.tables import (
    _estimate_and_shift_encoding_table_kernel,
    _shift_decoding_table_kernel, _shift_encoding_table_kernel,
)
from .generic.compaction import (
    _compact_bad_streams_kernel, _compact_extra_kernel,
    _count_bad_streams_kernel, _scatter_blocked_fallback_kernel,
)
from .generic.decode import _decode_kernel, _decode_overflow_kernel
from .generic.encode import _encode_kernel
from .generic.metadata import _prefix_allocate_kernel
from .generic.pointwise import (
    pointwise_compressed_dense_fallback_kernel,
    pointwise_compressed_dense_kernel,
)
from .generic.pointwise_scalar_dual import (
    _initialize_dual_maps_kernel, _prepare_dual_tables_and_maps_kernel,
    pointwise_scalar_mul_add_compressed_compressed_mapped_kernel,
)
from .device import device_profile


@dataclass
class FallbackStorage:
    """Compact overflow metadata and its private or shared tail allocation."""
    tail: torch.Tensor
    count: torch.Tensor
    lanes: torch.Tensor
    starts: torch.Tensor
    blocks: torch.Tensor
    descriptor: torch.Tensor | None = None
    allocation: torch.Tensor | None = None
    base: int = 0

    @property
    def buffered(self) -> bool:
        return self.descriptor is not None


@dataclass(frozen=True)
class KernelBackend:
    """Shared kernel entry points for one execution engine."""
    encode: Any
    decode: Any
    decode_overflow: Any = None
    encode_tilelang: bool = False
    decode_tilelang: bool = False
    shift_encoding_table: Any = None
    estimate_and_shift_encoding_table: Any = None
    shift_decoding_table: Any = None
    prefix_allocate: Any = None
    count_bad_streams: Any = None
    compact_bad_streams: Any = None
    compact_extra: Any = None
    scatter_fallback: Any = None
    pointwise_dense: Any = None
    pointwise_fallback: Any = None
    dual_maps_init: Any = None
    dual_tables_and_maps: Any = None
    dual_mapped: Any = None
    # Optional per-call auxiliary decode table (e.g. the A100 escape table).
    # When set, ``dispatch.decode_kernel`` builds it and passes it to
    # ``decode``/``decode_overflow`` immediately after the shifted decode table.
    decode_make_aux: Any = None


TRITON = KernelBackend(
    encode=_encode_kernel, decode=_decode_kernel,
    decode_overflow=_decode_overflow_kernel,
    shift_encoding_table=_shift_encoding_table_kernel,
    estimate_and_shift_encoding_table=_estimate_and_shift_encoding_table_kernel,
    shift_decoding_table=_shift_decoding_table_kernel,
    prefix_allocate=_prefix_allocate_kernel,
    count_bad_streams=_count_bad_streams_kernel,
    compact_bad_streams=_compact_bad_streams_kernel,
    compact_extra=_compact_extra_kernel,
    scatter_fallback=_scatter_blocked_fallback_kernel,
    pointwise_dense=pointwise_compressed_dense_kernel,
    pointwise_fallback=pointwise_compressed_dense_fallback_kernel,
    dual_maps_init=_initialize_dual_maps_kernel,
    dual_tables_and_maps=_prepare_dual_tables_and_maps_kernel,
    dual_mapped=pointwise_scalar_mul_add_compressed_compressed_mapped_kernel,
)


@cache
def _tilelang_backend() -> KernelBackend:
    from .tilelang.encode import encode_kernel
    from .tilelang.decode import decode_kernel
    return replace(TRITON, encode=encode_kernel, decode=decode_kernel,
                   encode_tilelang=True, decode_tilelang=True)


@cache
def _a100_backend() -> KernelBackend:
    """Ampere kernels that replace shared kernels on A100 parts."""
    if config.lane_mapping != "swizzle":
        raise ValueError("The A100 decoder requires config.lane_mapping = 'swizzle'")
    from .a100.decode import (
        _a100_decode_kernel, _a100_decode_overflow_kernel, build_escape_table,
    )
    return replace(
        TRITON,
        decode=_a100_decode_kernel,
        decode_overflow=_a100_decode_overflow_kernel,
        decode_make_aux=build_escape_table,
    )


def backend_for(device: torch.device) -> KernelBackend:
    """Select the engine; GPU tuning is already bound by autotune.py."""
    if device.type != "cuda":
        raise ValueError(f"LCT kernel dispatch requires a CUDA device, got {device}")
    if config.use_tilelang:
        return _tilelang_backend()
    if "a100" in device_profile(device).model:
        return _a100_backend()
    return TRITON
