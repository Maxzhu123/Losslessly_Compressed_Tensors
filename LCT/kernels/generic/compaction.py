"""Generic overflow counting, compaction, and decode scatter kernels.

The compact fallback list is stored **per storage block**:

* ``lanes``   uint8[count]   storage lane of each overflowing stream
* ``starts``  uint8[count]   first step held in fallback storage for that stream
* ``blocks``  int32[3*n_blocks]
    - ``blocks[b]``              number of overflowing streams in block ``b``
    - ``blocks[n_blocks + b]``   index of the block's first entry in the list
    - ``blocks[2*n_blocks + b]`` byte offset of the block's first tail

Entries and tail bytes of one block are contiguous and are positioned by that
block alone, so the list order never has to agree with the tail order. That is
what lets every consumer walk the list a block at a time and recover each
entry's dense tail offset from the block base plus a local prefix sum, with no
offset array and no separate offset pass.
"""
import triton
from triton import language as tl

from ..autotune import (
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS,
    COMPACT_BLOCK_AUTOTUNE_CONFIGS,
    COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    DENSE_SCATTER_AUTOTUNE_CONFIGS,
)
from ..primitives import pack_bf16
from .layout import swizzle_lane


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
    configs=COMPACT_BLOCK_AUTOTUNE_CONFIGS,
    key=["n_streams", "steps"],
    restore_value=["bad_count", "fallback_total"],
)
@triton.jit
def _compact_bad_streams_kernel(
    extra_starts,
    lanes, starts, descriptor, blocks, final_counts,
    bad_count, fallback_total, n_streams, steps, n_blocks,
    BUFFERED: tl.constexpr, N_LANES: tl.constexpr,
):
    """One program per storage block; reserves its own slots and tail bytes.

    The entry slots and the tail bytes are reserved with two independent
    atomics. That is safe here precisely because each block records its own
    ``first`` and ``byte_base``: nothing has to match a global list order.
    """
    block = tl.program_id(0)
    lane = tl.arange(0, N_LANES)
    stream = block * N_LANES + lane
    mask = stream < n_streams
    start = tl.load(extra_starts + stream, mask=mask, other=255).to(tl.int32)
    bad = mask & (start != 255)
    lengths = tl.where(bad, steps - start, 0)
    cnt = tl.sum(bad.to(tl.int32), axis=0)
    total_len = tl.sum(lengths, axis=0)
    block_start = tl.atomic_add(bad_count, cnt, mask=cnt != 0)
    byte_start = tl.atomic_add(fallback_total, total_len, mask=total_len != 0)
    tl.store(blocks + block, cnt)
    tl.store(blocks + n_blocks + block, block_start)
    tl.store(blocks + 2 * n_blocks + block, byte_start)
    if cnt > 0:
        rank = tl.cumsum(bad.to(tl.int32), axis=0) - bad.to(tl.int32)
        pos = block_start + rank
        if BUFFERED:
            total = tl.load(final_counts).to(tl.int32)
            base = tl.load(descriptor).to(tl.int32)
            tl.store(lanes + base + pos, lane.to(tl.uint8), mask=bad)
            tl.store(starts + base + total + pos, start.to(tl.uint8), mask=bad)
        else:
            tl.store(lanes + pos, lane.to(tl.uint8), mask=bad)
            tl.store(starts + pos, start.to(tl.uint8), mask=bad)


@triton.jit
def _compact_extra_rows(
    source_bits, tail, tail_off, n_elements, block_id,
    lane, start, valid, fallback_offset,
    PRECOMPUTED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Copy the tail bytes of one set of overflowing lanes, one row tile deep."""
    # The rows start close to the set's own smallest overflow step instead of
    # at the whole block's minimum.
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    block_base = block_id * BLOCK
    row_offsets = tl.arange(0, ROW_TILE)
    for row_first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = row_first + row_offsets
        source_offset = block_base + row[:, None] * N_LANES + lane[None, :]
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & (source_offset < n_elements)
        )
        logical_n = block_id * N_STEPS + row[:, None]
        logical_k = swizzle_lane(block_id * N_STEPS, row[:, None], lane[None, :])
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
        tl.store(tail + tail_off + destination, values, mask=active)


@triton.jit
def _compact_extra_impl(
    source_bits,
    lanes, starts, tail, descriptor, fallback_count, blocks,
    fallback_base, n_blocks, n_elements,
    extra_starts, summaries, n_streams,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr, BUILD_META: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    ROW_TILE: tl.constexpr, block_id,
):
    if BUFFERED:
        total = tl.load(fallback_count).to(tl.int32)
        base = tl.load(descriptor).to(tl.int32)
        lane_off = base
        start_off = base + total
        tail_off = base + 2 * total
    else:
        lane_off = 0
        start_off = 0
        tail_off = fallback_base
    if BUILD_META:
        # Publish this block's compact entries and its slice positions, so the
        # summaries path needs only one launch per block. The tail copy takes
        # the overflow steps straight from extra_starts in registers rather
        # than reading back the lanes/starts this program is about to write,
        # which would be an unsynchronised cross-thread read of its own stores.
        count = tl.load(summaries + block_id).to(tl.int32)
        first = tl.load(summaries + 2 * n_blocks + block_id).to(tl.int32)
        byte_base = tl.load(summaries + 3 * n_blocks + block_id).to(tl.int32)
        if count > 0:
            lane = tl.arange(0, N_LANES)
            stream = block_id * N_LANES + lane
            start = tl.load(extra_starts + stream).to(tl.int32)
            bad = (stream < n_streams) & (start != 255)
            length = tl.where(bad, N_STEPS - start, 0)
            local = tl.cumsum(length, axis=0) - length
            _compact_extra_rows(
                source_bits, tail, tail_off, n_elements, block_id,
                lane, start, bad, byte_base + local,
                PRECOMPUTED, LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS, ROW_TILE,
            )
            bad_i = bad.to(tl.int32)
            rank = tl.cumsum(bad_i, axis=0) - bad_i
            pos = first + rank
            if BUFFERED:
                tl.store(lanes + base + pos, lane.to(tl.uint8), mask=bad)
                tl.store(starts + base + total + pos, start.to(tl.uint8), mask=bad)
            else:
                tl.store(lanes + pos, lane.to(tl.uint8), mask=bad)
                tl.store(starts + pos, start.to(tl.uint8), mask=bad)
        # Every block publishes its own slice positions, including empty blocks.
        tl.store(blocks + block_id, count)
        tl.store(blocks + n_blocks + block_id, first)
        tl.store(blocks + 2 * n_blocks + block_id, byte_base)
        return
    count = tl.load(blocks + block_id).to(tl.int32)
    if count == 0:
        return
    first = tl.load(blocks + n_blocks + block_id).to(tl.int32)
    byte_base = tl.load(blocks + 2 * n_blocks + block_id).to(tl.int32)
    # A block's entries are contiguous and in increasing lane order, so one
    # N_LANES-wide pass visits all of them and the running offset is just the
    # prefix sum of the tail lengths in that order.
    k = tl.arange(0, N_LANES)
    valid = k < count
    lane = tl.load(lanes + lane_off + first + k, mask=valid, other=0).to(tl.int32)
    start = tl.load(
        starts + start_off + first + k, mask=valid, other=N_STEPS,
    ).to(tl.int32)
    length = tl.where(valid, N_STEPS - start, 0)
    local = tl.cumsum(length, axis=0) - length
    _compact_extra_rows(
        source_bits, tail, tail_off, n_elements, block_id,
        lane, start, valid, byte_base + local,
        PRECOMPUTED, LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS, ROW_TILE,
    )


@triton.autotune(
    configs=COMPACT_EXTRA_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK", "PRECOMPUTED", "BUILD_META"],
)
@triton.jit
def _compact_extra_kernel(
    source_bits, lanes, starts, tail, descriptor, fallback_count, blocks,
    fallback_base, n_blocks, n_elements,
    extra_starts, summaries, n_streams,
    BUFFERED: tl.constexpr, PRECOMPUTED: tl.constexpr, LOGICAL_NUMEL: tl.constexpr,
    BUILD_META: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, ROW_TILE: tl.constexpr,
):
    """Compact overflow tails, and optionally this block's metadata, per block."""
    _compact_extra_impl(
        source_bits, lanes, starts, tail, descriptor, fallback_count, blocks,
        fallback_base, n_blocks, n_elements, extra_starts, summaries, n_streams,
        BUFFERED, PRECOMPUTED,
        LOGICAL_NUMEL, BUILD_META, BLOCK, N_LANES, N_STEPS, ROW_TILE,
        tl.program_id(0),
    )


@triton.autotune(
    configs=DENSE_SCATTER_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK"],
)
@triton.jit
def _scatter_blocked_fallback_kernel(
    lanes, starts, tail, descriptor, fallback_count, blocks,
    fallback_base,
    sign_mantissa, output, n_elements,
    BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    """Overwrite mapped decode output with compact fallback stream tails."""
    block_id = tl.program_id(0)
    n_blocks = tl.num_programs(0)
    if BUFFERED:
        total = tl.load(fallback_count).to(tl.int32)
        base = tl.load(descriptor).to(tl.int32)
        lane_off = base
        start_off = base + total
        tail_off = base + 2 * total
        tail_ptr = tail
    else:
        lane_off = 0
        start_off = 0
        tail_off = fallback_base
        tail_ptr = tail

    count = tl.load(blocks + block_id).to(tl.int32)
    if count == 0:
        return
    first = tl.load(blocks + n_blocks + block_id).to(tl.int32)
    byte_base = tl.load(blocks + 2 * n_blocks + block_id).to(tl.int32)

    k = tl.arange(0, N_LANES)
    valid = k < count
    lane = tl.load(lanes + lane_off + first + k, mask=valid, other=0).to(tl.int32)
    start = tl.load(
        starts + start_off + first + k, mask=valid, other=N_STEPS,
    ).to(tl.int32)
    length = tl.where(valid, N_STEPS - start, 0)
    local = tl.cumsum(length, axis=0) - length
    fallback_offset = byte_base + local

    block_base = block_id * BLOCK
    # Skip inactive prefixes, then restore several absolute rows at once.
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    row_offsets = tl.arange(0, ROW_TILE)
    for row_first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = row_first + row_offsets
        storage_offset = block_base + row[:, None] * N_LANES + lane[None, :]
        logical_n = block_id * N_STEPS + row[:, None]
        logical_k = swizzle_lane(block_id * N_STEPS, row[:, None], lane[None, :])
        logical_offset = logical_n * N_LANES + logical_k
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & (storage_offset < n_elements)
            & (logical_offset < LOGICAL_NUMEL)
        )
        # Reassemble BF16 bits and write only active logical elements.
        source_offset = fallback_offset[None, :] + row[:, None] - start[None, :]
        exponent = tl.load(
            tail_ptr + tail_off + source_offset, mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + storage_offset, mask=active, other=0)
        tl.store(output + logical_offset, pack_bf16(exponent, sm).to(tl.int16), mask=active)
