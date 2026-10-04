"""Shared BF16 encoding, allocation, and overflow storage pipeline."""

import torch
import triton

from ..comp_format import Distribution, StorageLayout
from ..comp_tensor import CompressedTensor
from ..compression.huffman_tables import get_distribution_tables
from ..tensor_buffer import Allocation, TensorBuffer
from ..kernels.common.tables import (
    _shift_encoding_table_kernel,
    _estimate_and_shift_encoding_table_kernel,
)
from ..kernels.generic.compaction import (
    _count_bad_streams_kernel, _compact_bad_streams_kernel,
    _compact_extra_kernel, _compact_components_extra_kernel,
)
from ..kernels.generic.metadata import _prefix_allocate_kernel, _compact_metadata_kernel
from ..kernels import dispatch
from .autotune import SUMMARY_BLOCK_LIMIT, COMPACT_GRID_LIMIT
from .geometry import geometry

CENTER_SAMPLE_SIZE = 4096


def _compact_bad_streams(
    extra_starts,
    bad_streams_out, bad_starts_out, fallback_offsets_out,
    metadata_buffer, allocation_descriptor, final_counts,
    bad_count, fallback_total, streams, steps,
    *,
    buffered,
):
    """Compact overflow-stream metadata for a buffered or private fallback path."""
    compact_grid = lambda meta: (triton.cdiv(streams, meta["BLOCK"]),)
    _compact_bad_streams_kernel[compact_grid](
        extra_starts, bad_streams_out, bad_starts_out, fallback_offsets_out,
        metadata_buffer, allocation_descriptor, final_counts,
        bad_count, fallback_total, streams, steps,
        BUFFERED=buffered,
    )


def _compact_extra(
    source_values,
    bad_streams, bad_starts, fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    size, streams, *,
    precomputed, buffered,
    logical_numel,
    block_symbols, lanes, steps,
):
    """Compact fallback tail values for a buffered or private fallback path."""
    def compact_grid(meta):
        return (min(triton.cdiv(streams, meta["TILE"]), COMPACT_GRID_LIMIT),)

    kernel = _compact_components_extra_kernel if precomputed else _compact_extra_kernel
    kernel[compact_grid](
        source_values, bad_streams, bad_starts, fallback_offsets,
        fallback_data, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, size,
        BUFFERED=buffered, LOGICAL_NUMEL=logical_numel,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
    )


def encode_components(
    source_values: torch.Tensor,
    sign_mantissa: torch.Tensor,
    size: int,
    distribution: Distribution,
    buffer: TensorBuffer | None,
    shape: tuple[int, ...],
    *,
    precomputed: bool,
    center: torch.Tensor | None = None,
    logical_numel: int,
) -> CompressedTensor:
    """Encode BF16 bits or split components into a ``CompressedTensor``.

    Args:
        source_values: Int16 BF16 bits, or raw uint8 exponents if ``precomputed``.
        sign_mantissa: Output side-byte stream, or precomputed side bytes.
        size: Padded codec element count (storage element count).
        distribution: Codebook and stream geometry.
        buffer: Optional shared fallback arena.
        shape: Original tensor shape.
        precomputed: ``False`` extracts both fields from BF16 bits; ``True``
            consumes raw exponent bytes plus existing side bytes from a fused op.
        center: Precomputed exponent center, or ``None`` to estimate it.
        logical_numel: Flattened logical element count (1D storage mapping).
    """
    if buffer is not None and buffer.capacity_bytes % 4:
        raise ValueError("TensorBuffer capacity must be divisible by 4")
    # Geometry fixes the independent stream count and per-stream bit budget.
    stream_geometry = geometry(distribution)
    block_symbols, lanes, steps, fixed_words = stream_geometry
    blocks = triton.cdiv(size, block_symbols)
    streams = blocks * lanes
    encode_table, _, _ = get_distribution_tables(distribution)
    shifted_encode = torch.empty(256, dtype=torch.int32, device=source_values.device)
    if center is None:
        center = torch.empty(1, dtype=torch.int32, device=source_values.device)
        _estimate_and_shift_encoding_table_kernel[(1,)](
            source_values, center, logical_numel, encode_table, shifted_encode,
            SAMPLE_SIZE=min(logical_numel, CENTER_SAMPLE_SIZE),
            PRECOMPUTED=precomputed, IGNORE_ZERO=distribution.zero_prob > 0,
            BLOCK=4096, num_warps=4, num_stages=2,
        )
    else:
        _shift_encoding_table_kernel[(1,)](
            encode_table, center, shifted_encode, BLOCK=256,
        )
    # Encode and decode share this sampled center through the result metadata.
    encoded = torch.empty(
        streams * fixed_words + 4,
        dtype=torch.int32,
        device=source_values.device,
    )
    extra_starts = torch.empty(
        streams, dtype=torch.uint8, device=source_values.device
    )
    # Bound the single-program prefix scan; larger tensors keep the prior path.
    use_summaries = blocks <= SUMMARY_BLOCK_LIMIT
    summaries = (
        torch.empty(4 * blocks, dtype=torch.int32, device=source_values.device)
        if use_summaries else extra_starts
    )
    dispatch.encode_kernel(
        source_values, sign_mantissa, encoded, shifted_encode, extra_starts, summaries,
        size, streams, precomputed=precomputed, logical_numel=logical_numel,
        block_symbols=block_symbols, lanes=lanes, steps=steps,
        fixed_words=fixed_words, write_summary=use_summaries,
    )

    # Count overflow streams and bytes once.  For a shared buffer these counts
    # are used asynchronously by the allocator; for private fallback they tell
    # us how much exact-size storage to allocate.
    make_counts = torch.empty if use_summaries else torch.zeros
    counts = make_counts(4, dtype=torch.int32, device=source_values.device)
    final_count = counts[:1]
    final_total = counts[1:2]
    compact_count = counts[2:3]
    compact_total = counts[3:]
    if use_summaries:
        descriptor = (
            torch.empty(4, dtype=torch.int32, device=source_values.device)
            if buffer is not None else counts
        )
        state = (
            (buffer._free_starts, buffer._free_sizes, buffer._free_count,
             buffer._lock, buffer._generation)
            if buffer is not None else (counts,) * 5
        )
        _prefix_allocate_kernel[(1,)](
            summaries, counts, descriptor, *state, blocks,
            BUFFERED=buffer is not None, BLOCK=triton.next_power_of_2(blocks),
            MAX_FREE_REGIONS=buffer.max_free_regions if buffer is not None else 256,
            num_warps=4, num_stages=2,
        )
    else:
        count_grid = lambda meta: (triton.cdiv(streams, meta["BLOCK"]),)
        _count_bad_streams_kernel[count_grid](
            extra_starts, final_count, final_total, streams, steps,
        )

    buffered = buffer is not None
    if buffer is not None:
        if use_summaries:
            allocation = Allocation(descriptor, buffer)
        else:
            allocation = buffer.allocate_with_items(final_total, final_count, 9)
        metadata = buffer.data.view(torch.int32)
        fallback_buffer = buffer.data
        bad_streams_out = metadata
        bad_starts_out = buffer.data
        fallback_offsets_out = metadata
        metadata_buffer = metadata
        allocation_descriptor = allocation.descriptor
        descriptor = allocation.descriptor
        # counts[0] and counts[1] are the final counts used by the buffered
        # metadata layout; counts[2] and counts[3] are zeroed compaction
        # accumulators.
        final_counts = final_count
        bad_count = compact_count
        fallback_total = compact_total
    else:
        count, fallback_size = (int(value) for value in counts[:2].tolist())
        bad_streams_out = torch.empty(count, dtype=torch.int32, device=source_values.device)
        bad_starts_out = torch.empty(count, dtype=torch.uint8, device=source_values.device)
        fallback_offsets_out = torch.empty(
            count, dtype=torch.int32, device=source_values.device
        )
        fallback_buffer = torch.empty(
            fallback_size, dtype=torch.int8, device=source_values.device
        )
        metadata_buffer = bad_streams_out
        # Private compaction does not use the descriptor/final count path, so
        # the zeroed counts[2] and counts[3] act as the atomic accumulators.
        allocation_descriptor = compact_count
        descriptor = None
        final_counts = compact_count
        bad_count = compact_count
        fallback_total = compact_total

    if use_summaries:
        _compact_metadata_kernel[(blocks,)](
            extra_starts, summaries, counts, allocation_descriptor, metadata_buffer,
            bad_streams_out, bad_starts_out, fallback_offsets_out,
            blocks, streams, BUFFERED=buffered, N_LANES=lanes, N_STEPS=steps,
            num_warps=4, num_stages=2,
        )
        bad_count = final_count
        fallback_total = final_total
    else:
        _compact_bad_streams(
            extra_starts, bad_streams_out, bad_starts_out, fallback_offsets_out,
            metadata_buffer, allocation_descriptor, final_counts,
            bad_count, fallback_total, streams, steps,
            buffered=buffered,
        )
    _compact_extra(
        source_values, bad_streams_out, bad_starts_out, fallback_offsets_out,
        fallback_buffer, metadata_buffer, allocation_descriptor,
        final_counts, bad_count, size, streams,
        precomputed=precomputed, buffered=buffered,
        logical_numel=logical_numel,
        block_symbols=block_symbols, lanes=lanes, steps=steps,
    )
    return CompressedTensor(
        encoded, size, sign_mantissa,
        offsets=None if buffered else bad_streams_out,
        fallback_starts=None if buffered else bad_starts_out,
        fallback_offsets=None if buffered else fallback_offsets_out,
        fallback_buffer=fallback_buffer,
        fallback_descriptor=descriptor,
        buffer=buffer,
        fallback_count=final_count if buffered else bad_count,
        fallback_used=final_total if buffered else fallback_total,
        distribution=distribution, center=center, shape=shape,
        layout=StorageLayout.COMPRESSED,
        stream_geometry=stream_geometry,
    )


def encode(
    data: torch.Tensor, distribution: Distribution, buffer: TensorBuffer | None = None, *,
    allow_raw: bool = False,
) -> CompressedTensor:
    """Compress any tensor through the flattened 1D blocked storage mapping."""
    shape = tuple(data.shape)
    source = data.contiguous().view(-1)
    logical_numel = source.numel()
    block_symbols, lanes, _, fixed_words = geometry(distribution)
    # Always use the flattened 1D layout: the codec operates on the
    # contiguous row-major stream and the original shape is restored by
    # reshape on decode.
    blocks = triton.cdiv(logical_numel, block_symbols)
    storage_numel = blocks * block_symbols
    streams = blocks * lanes
    minimum_bytes = storage_numel + (streams * fixed_words + 4) * 4
    if allow_raw and minimum_bytes > logical_numel * data.element_size():
        return CompressedTensor(
            source, logical_numel, buffer=buffer,
            distribution=distribution, shape=shape,
            layout=StorageLayout.RAW,
        )
    sign_mantissa = torch.empty(
        storage_numel, dtype=torch.uint8, device=source.device
    )
    return encode_components(
        source.view(torch.int16), sign_mantissa, storage_numel,
        distribution, buffer, shape, precomputed=False,
        logical_numel=logical_numel,
    )
