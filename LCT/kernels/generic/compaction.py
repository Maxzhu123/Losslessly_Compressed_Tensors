"""Generic overflow counting, compaction, and decode scatter kernels."""

import triton
from triton import language as tl

from ...codec.autotune import (
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    DENSE_SCATTER_AUTOTUNE_CONFIGS,
)
from ..primitives import pack_bf16


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _count_bad_streams_kernel(
    extra_starts,
    bad_count, fallback_total, n_streams, steps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    tl.atomic_add(fallback_total, total_len, mask=total_len != 0)


@triton.autotune(
    configs=COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _compact_bad_streams_kernel(
    extra_starts,
    bad_streams_out, bad_starts_out, fallback_offsets_out,
    metadata_buffer, allocation_descriptor, final_counts,
    bad_count, fallback_total, n_streams, steps,
    BUFFERED: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_streams
    start = tl.load(extra_starts + offs, mask=mask, other=255)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    block_start = tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    fallback_block_start = tl.atomic_add(
        fallback_total, total_len, mask=total_len != 0
    )
    prefix = tl.cumsum(bad.to(tl.int32), axis=0) - bad.to(tl.int32)
    offset_prefix = tl.cumsum(lengths, axis=0) - lengths
    dense_offset = fallback_block_start + offset_prefix
    pos = block_start + prefix
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        tl.store(metadata_buffer + base_words + pos, offs.to(tl.int32), mask=bad)
        tl.store(
            metadata_buffer + base_words + count + pos,
            dense_offset,
            mask=bad,
        )
        tl.store(bad_starts_out + base + 8 * count + pos, start, mask=bad)
    else:
        tl.store(bad_streams_out + pos, offs.to(tl.int32), mask=bad)
        tl.store(bad_starts_out + pos, start, mask=bad)
        tl.store(
            fallback_offsets_out + pos,
            dense_offset,
            mask=bad,
        )


@triton.jit
def _compact_extra_impl(
    source_bits,
    extra_streams, extra_starts,
    fallback_offsets, fallback_data,
    metadata_buffer, allocation_descriptor, final_counts, bad_count,
    n_elements,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr, TILE: tl.constexpr,
    ROW_TILE: tl.constexpr, tile_id,
):
    pid = tile_id
    tile = pid * TILE + tl.arange(0, TILE)
    if BUFFERED:
        count = tl.load(final_counts).to(tl.int32)
    else:
        count = tl.load(bad_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(allocation_descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata_buffer + base_words + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(
            extra_starts + base + 8 * count + tile,
            mask=valid,
            other=N_STEPS,
        ).to(tl.int32)
        fallback_offset = tl.load(
            metadata_buffer + base_words + count + tile,
            mask=valid,
            other=0,
        ).to(tl.int32)
        fallback_base = base + 9 * count
    else:
        stream = tl.load(extra_streams + tile, mask=valid, other=0).to(tl.int32)
        start = tl.load(extra_starts + tile, mask=valid, other=N_STEPS).to(tl.int32)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0).to(tl.int32)
    block = stream // N_LANES
    lane = stream - block * N_LANES
    # Hoisted swizzle (see _encode_kernel): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    # Absolute rows keep adjacent source lanes together even when streams
    # have different overflow starts. Process several rows per iteration.
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    row_offsets = tl.arange(0, ROW_TILE)
    for first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = first + row_offsets
        source_offset = block_base[None, :] + row[:, None] * N_LANES + lane[None, :]
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & (source_offset < n_elements)
        )
        logical_n = block[None, :] * N_STEPS + row[:, None]
        logical_k = (lane[None, :] + ((block_shift[None, :] + row[:, None]) & 255)) & 255
        input_offset = logical_n * N_LANES + logical_k
        value = tl.load(
            source_bits + input_offset,
            mask=active & (input_offset < LOGICAL_NUMEL), other=0,
        ).to(tl.int32)
        if PRECOMPUTED:
            values = (value - 127).to(tl.int8)
        else:
            values = (((value >> 7) & 255) - 127).to(tl.int8)
        destination = fallback_offset[None, :] + row[:, None] - start[None, :]
        if BUFFERED:
            tl.store(fallback_data + fallback_base + destination, values, mask=active)
        else:
            tl.store(fallback_data + destination, values, mask=active)


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_components_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Compact overflow exponents from flattened precomputed planes."""
    count = tl.load(final_counts if BUFFERED else bad_count).to(tl.int32)
    for tile_id in tl.range(tl.program_id(0), tl.cdiv(count, TILE), tl.num_programs(0)):
        _compact_extra_impl(
            source_bits, extra_streams, extra_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, n_elements, BUFFERED, True,
            LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS, TILE, ROW_TILE,
            tile_id,
        )


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _compact_extra_kernel(
    source_bits, extra_streams, extra_starts,
    fallback_offsets, fallback_data, metadata_buffer,
    allocation_descriptor, final_counts, bad_count, n_elements,
    BUFFERED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Compact flattened overflow values through the 1D source mapping."""
    count = tl.load(final_counts if BUFFERED else bad_count).to(tl.int32)
    for tile_id in tl.range(tl.program_id(0), tl.cdiv(count, TILE), tl.num_programs(0)):
        _compact_extra_impl(
            source_bits, extra_streams, extra_starts, fallback_offsets,
            fallback_data, metadata_buffer, allocation_descriptor,
            final_counts, bad_count, n_elements, BUFFERED, False,
            LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS, TILE, ROW_TILE,
            tile_id,
        )


@triton.jit
def _scatter_blocked_fallback_impl(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, output, n_elements,
    BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    ROW_TILE: tl.constexpr, tile_id,
):
    """Overwrite mapped decode output with compact fallback stream tails."""
    pid = tile_id
    tile = pid * TILE + tl.arange(0, TILE)
    count = tl.load(fallback_count).to(tl.int32)
    if pid * TILE >= count:
        return
    valid = tile < count
    if BUFFERED:
        base = tl.load(descriptor).to(tl.int32)
        base_words = base // 4
        stream = tl.load(metadata + base_words + tile, mask=valid, other=0)
        start = tl.load(
            bad_starts + base + 8 * count + tile,
            mask=valid, other=N_STEPS,
        )
        fallback_offset = tl.load(
            metadata + base_words + count + tile, mask=valid, other=0,
        )
        fallback_base = base + 9 * count
    else:
        stream = tl.load(bad_streams + tile, mask=valid, other=0)
        start = tl.load(bad_starts + tile, mask=valid, other=N_STEPS)
        fallback_offset = tl.load(fallback_offsets + tile, mask=valid, other=0)
    stream = stream.to(tl.int32)
    start = start.to(tl.int32)
    fallback_offset = fallback_offset.to(tl.int32)
    block = stream // N_LANES
    lane = stream % N_LANES
    # Hoisted swizzle (see _encode_kernel): block_shift is loop-invariant.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    # Skip the inactive prefix, then restore several absolute rows at once.
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    row_offsets = tl.arange(0, ROW_TILE)
    for first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = first + row_offsets
        storage_offset = block_base[None, :] + row[:, None] * N_LANES + lane[None, :]
        logical_n = block[None, :] * N_STEPS + row[:, None]
        logical_k = (lane[None, :] + ((block_shift[None, :] + row[:, None]) & 255)) & 255
        logical_offset = logical_n * N_LANES + logical_k
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & (storage_offset < n_elements)
            & (logical_offset < LOGICAL_NUMEL)
        )
        source_offset = fallback_offset[None, :] + row[:, None] - start[None, :]
        exponent = tl.load(
            fallback_buffer + fallback_base + source_offset,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + storage_offset, mask=active, other=0)
        tl.store(output + logical_offset, pack_bf16(exponent, sm).to(tl.int16), mask=active)


@triton.autotune(
    configs=DENSE_SCATTER_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _scatter_blocked_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, output, n_elements,
    BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    count = tl.load(fallback_count).to(tl.int32)
    for tile_id in tl.range(tl.program_id(0), tl.cdiv(count, TILE), tl.num_programs(0)):
        _scatter_blocked_fallback_impl(
            bad_streams, bad_starts, fallback_offsets, fallback_buffer,
            fallback_base, metadata, descriptor, fallback_count,
            sign_mantissa, output, n_elements, BUFFERED, LOGICAL_NUMEL,
            TILE, BLOCK, N_LANES, N_STEPS, ROW_TILE,
            tile_id,
        )
