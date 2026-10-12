"""Shared output allocation and overflow restoration for codec decoding."""

import torch
import triton

from ..comp_tensor import CompressedTensor
from ..compression.huffman_tables import get_distribution_tables
from ..kernels import dispatch
from ..kernels.backends import FallbackStorage

# Stream count below which a separate scatter pass beats folding the fallback
# into the decoder. The fused overlay wins at every size on the 5090.
OVERLAY_MIN_STREAMS = 0


def _buffered(data: CompressedTensor) -> bool:
    return data.fallback_descriptor is not None


def _storage(data: CompressedTensor) -> FallbackStorage:
    """Rebuild the fallback storage view from a compressed tensor.

    Shared arenas carry lane/start bytes followed by raw tail bytes; private
    tensors carry the same fields in separate allocations.
    """
    if _buffered(data):
        arena = data.fallback_buffer
        return FallbackStorage(
            tail=arena, count=data.fallback_count,
            lanes=arena, starts=arena,
            blocks=data.fallback_blocks, descriptor=data.fallback_descriptor,
            allocation=data.fallback_descriptor,
        )
    # The kernels ignore their descriptor when the tails are private; the
    # generator's compact list stands in for it.
    return FallbackStorage(
        tail=data.fallback_buffer, count=data.fallback_count,
        lanes=data.fallback_lanes, starts=data.fallback_starts,
        blocks=data.fallback_blocks,
        allocation=data.fallback_lanes,
        base=data.fallback_base,
    )


def has_fallback(data: CompressedTensor) -> bool:
    """True when this tensor actually carries overflow streams."""
    if data.layout != data.layout.COMPRESSED:
        return False
    if _buffered(data):
        return True
    return data.fallback_lanes is not None and data.fallback_lanes.numel() > 0


def decode(data: CompressedTensor) -> torch.Tensor:
    """Decode the fixed payload and restore its overflow tails."""
    logical_numel = data.logical_numel
    output = torch.empty(logical_numel, dtype=torch.int16, device=data.data.device)
    if logical_numel:
        backend = dispatch.backend_for(data.data.device)
        _, decode_table, rare_length = get_distribution_tables(data.distribution)
        block_symbols, lanes, steps, fixed_words = data.codec_geometry
        blocks = triton.cdiv(data.size, block_symbols)
        streams = blocks * lanes
        # Only backends with a fused decoder can absorb the fallback in place.
        overlay = None
        if (
            has_fallback(data)
            and backend.decode_overflow is not None
            and not backend.decode_tilelang
            and streams >= OVERLAY_MIN_STREAMS
        ):
            storage = _storage(data)
            overlay = (dispatch.fallback_tail_args(storage), storage.buffered)
        dispatch.decode_kernel(
            data.data, data.sign_mantissa, output, decode_table, data.center, data.size,
            logical_numel=logical_numel, block_symbols=block_symbols, lanes=lanes,
            steps=steps, fixed_words=fixed_words, rare_length=rare_length,
            overflow=overlay, shifted_decode=data.decode_table,
        )
        if overlay is None and has_fallback(data):
            _restore_fallback(data, output, lanes, steps)
    return output.view(torch.bfloat16).reshape(data.shape)


def _restore_fallback(data: CompressedTensor, output: torch.Tensor,
                      lanes: int, steps: int) -> None:
    """Restore overflow tails with a separate scatter pass."""
    block_symbols = data.codec_geometry[0]
    dispatch.scatter_fallback(
        _storage(data), data.sign_mantissa, output, data.size,
        data.logical_numel, block_symbols, lanes, steps,
    )
