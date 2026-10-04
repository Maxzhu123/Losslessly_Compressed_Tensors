"""Shared output allocation and overflow restoration for codec decoding."""

import torch
import triton

from ..comp_tensor import CompressedTensor
from ..compression.huffman_tables import get_distribution_tables
from ..kernels.generic.compaction import _scatter_blocked_fallback_kernel
from ..kernels import dispatch
from .autotune import SCATTER_GRID_LIMIT


def decode(data: CompressedTensor) -> torch.Tensor:
    """Decode the fixed payload and restore its overflow tails."""
    logical_numel = data.logical_numel
    output = torch.empty(logical_numel, dtype=torch.int16, device=data.data.device)
    if logical_numel:
        _, decode_table, rare_length = get_distribution_tables(data.distribution)
        block_symbols, lanes, steps, fixed_words = data.codec_geometry
        dispatch.decode_kernel(
            data.data, data.sign_mantissa, output, decode_table, data.center, data.size,
            logical_numel=logical_numel, block_symbols=block_symbols, lanes=lanes,
            steps=steps, fixed_words=fixed_words, rare_length=rare_length,
        )
        _restore_fallback(data, output)
    return output.view(torch.bfloat16).reshape(data.shape)


def _restore_fallback(data: CompressedTensor, output: torch.Tensor) -> None:
    """Restore compact overflow tails for either fixed-payload decoder."""
    logical_numel = data.logical_numel
    block_symbols, lanes, steps, _ = data.codec_geometry
    blocks = triton.cdiv(data.size, block_symbols)
    streams = blocks * lanes
    scatter_tile = 64
    scatter_meta = dict(
        LOGICAL_NUMEL=logical_numel,
        TILE=scatter_tile, BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
    )

    def scatter_grid(stream_count):
        programs = triton.cdiv(stream_count, scatter_tile)
        return (min(programs, SCATTER_GRID_LIMIT),)

    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        _scatter_blocked_fallback_kernel[scatter_grid(streams)](
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, output, data.size,
            BUFFERED=True, **scatter_meta,
        )
    elif data.offsets.numel():
        _scatter_blocked_fallback_kernel[scatter_grid(data.offsets.numel())](
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            output, data.size, BUFFERED=False, **scatter_meta,
        )
