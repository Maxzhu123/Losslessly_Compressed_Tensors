"""Shared BF16 encoding, allocation, and overflow storage pipeline.

All kernels use compact per-block overflow metadata. This module reserves
private or shared storage and launches the same pipeline for every GPU.
"""

import torch
import triton

from ..comp_format import Distribution, StorageLayout
from ..comp_tensor import CompressedTensor
from ..compression.huffman_tables import get_distribution_tables
from ..tensor_buffer import Allocation, TensorBuffer
from ..kernels import dispatch
from ..kernels.backends import FallbackStorage
from ..kernels.autotune import SUMMARY_BLOCK_LIMIT
from .geometry import geometry

CENTER_SAMPLE_SIZE = 4096


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
    encode_table, decode_table, _ = get_distribution_tables(distribution)
    shifted_encode = torch.empty(256, dtype=torch.int32, device=source_values.device)
    # Build once and reuse for decoding and fused pointwise reads.
    shifted_decode = torch.empty_like(decode_table)
    if center is None:
        center = torch.empty(1, dtype=torch.int32, device=source_values.device)
        dispatch.estimate_and_shift_encoding_table(
            source_values, center, logical_numel, encode_table, shifted_encode,
            decode_table, shifted_decode,
            min(logical_numel, CENTER_SAMPLE_SIZE), precomputed,
        )
    else:
        dispatch.shift_encoding_table(
            encode_table, center, shifted_encode, decode_table, shifted_decode,
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
    # Bound the single-program prefix scan; larger tensors keep the atomic path.
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
        dispatch.prefix_allocate(
            summaries, counts, descriptor, buffer, blocks, buffer is not None,
            buffer.max_free_regions if buffer is not None else 256,
        )
    else:
        dispatch.count_bad_streams(
            extra_starts, final_count, final_total, streams, steps,
        )

    buffered = buffer is not None
    # Each overflowing stream stores a lane byte and a start byte.
    blocks_out = torch.empty(
        3 * blocks, dtype=torch.int32, device=source_values.device,
    )
    if buffer is not None:
        allocation = (
            Allocation(descriptor, buffer) if use_summaries
            else buffer.allocate_with_items(final_total, final_count, 2)
        )
        arena = buffer.data
        allocation_descriptor = allocation.descriptor
        descriptor = allocation.descriptor
        storage = FallbackStorage(
            tail=arena, count=final_count, lanes=arena, starts=arena,
            blocks=blocks_out, descriptor=allocation_descriptor,
            allocation=allocation_descriptor,
        )
        final_counts = final_count
        bad_count = compact_count
        fallback_total = compact_total
    else:
        count, fallback_size = (int(value) for value in counts[:2].tolist())
        lanes_out = torch.empty(count, dtype=torch.uint8, device=source_values.device)
        starts_out = torch.empty(count, dtype=torch.uint8, device=source_values.device)
        tail_buffer = torch.empty(
            fallback_size, dtype=torch.uint8, device=source_values.device,
        )
        allocation_descriptor = compact_count
        descriptor = None
        final_counts = compact_count
        bad_count = compact_count
        fallback_total = compact_total
        storage = FallbackStorage(
            tail=tail_buffer, count=bad_count, lanes=lanes_out,
            starts=starts_out, blocks=blocks_out,
            allocation=allocation_descriptor, base=0,
        )

    if use_summaries:
        bad_count = final_count
        fallback_total = final_total
    else:
        dispatch.compact_bad_streams(
            extra_starts, storage, final_counts,
            bad_count, fallback_total, streams, steps, blocks, buffered, lanes,
        )
    # The readers need the final stream count, not the pre-compaction
    # accumulator the storage was built with a moment ago.
    storage.count = final_count if buffered else bad_count
    dispatch.compact_extra(
        source_values, storage, final_counts, blocks, size,
        precomputed=precomputed, buffered=buffered, logical_numel=logical_numel,
        block_symbols=block_symbols, lanes=lanes, steps=steps,
        extra_starts=extra_starts, summaries=summaries, n_streams=streams,
        build_meta=use_summaries,
    )
    return CompressedTensor(
        encoded, size, sign_mantissa,
        fallback_starts=None if buffered else storage.starts,
        fallback_lanes=None if buffered else storage.lanes,
        fallback_blocks=storage.blocks,
        fallback_buffer=storage.tail,
        fallback_descriptor=descriptor,
        buffer=buffer,
        fallback_count=storage.count,
        fallback_used=final_total if buffered else fallback_total,
        distribution=distribution, center=center, decode_table=shifted_decode,
        shape=shape,
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
