"""Block summaries, prefix offsets, and GPU allocation for overflow storage."""

import triton
from triton import language as tl

from ...tensor_buffer import _allocate_kernel


@triton.jit
def _prefix_allocate_kernel(
    summaries, counts, descriptor, free_starts, free_sizes, free_count,
    lock, generation, n_blocks,
    BUFFERED: tl.constexpr, BLOCK: tl.constexpr, MAX_FREE_REGIONS: tl.constexpr,
):
    """Scan block summaries, initialize counts, and reserve the shared arena."""
    idx = tl.arange(0, BLOCK)
    block_count = tl.load(summaries + idx, mask=idx < n_blocks, other=0)
    block_bytes = tl.load(summaries + n_blocks + idx, mask=idx < n_blocks, other=0)
    count_prefix = tl.cumsum(block_count, axis=0) - block_count
    byte_prefix = tl.cumsum(block_bytes, axis=0) - block_bytes
    tl.store(summaries + 2 * n_blocks + idx, count_prefix, mask=idx < n_blocks)
    tl.store(summaries + 3 * n_blocks + idx, byte_prefix, mask=idx < n_blocks)
    total_count = tl.sum(block_count, axis=0)
    total_bytes = tl.sum(block_bytes, axis=0)
    out_idx = tl.arange(0, 4)
    out = tl.where(out_idx == 0, total_count, tl.where(out_idx == 1, total_bytes, 0))
    tl.store(counts + out_idx, out)
    if BUFFERED:
        tl.debug_barrier()
        _allocate_kernel(
            counts + 1, counts, descriptor, free_starts, free_sizes,
            free_count, lock, generation, 16, 9, MAX_FREE_REGIONS,
        )


@triton.jit
def _compact_metadata_kernel(
    extra_starts, summaries, counts, descriptor, metadata,
    streams_out, starts_out, offsets_out,
    n_blocks, n_streams,
    BUFFERED: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Write disjoint block metadata using scan offsets instead of atomics."""
    block = tl.program_id(0)
    block_count = tl.load(summaries + block)
    if block_count == 0:
        return
    lane = tl.arange(0, N_LANES)
    stream = block * N_LANES + lane
    start = tl.load(extra_starts + stream).to(tl.int32)
    bad = (stream < n_streams) & (start != 255)
    length = tl.where(bad, N_STEPS - start, 0)
    rank = tl.cumsum(bad.to(tl.int32), axis=0) - bad.to(tl.int32)
    local_offset = tl.cumsum(length, axis=0) - length
    pos = tl.load(summaries + 2 * n_blocks + block) + rank
    offset = tl.load(summaries + 3 * n_blocks + block) + local_offset
    if BUFFERED:
        count = tl.load(counts)
        base = tl.load(descriptor)
        tl.store(metadata + base // 4 + pos, stream, mask=bad)
        tl.store(metadata + base // 4 + count + pos, offset, mask=bad)
        tl.store(starts_out + base + 8 * count + pos, start, mask=bad)
    else:
        tl.store(streams_out + pos, stream, mask=bad)
        tl.store(starts_out + pos, start, mask=bad)
        tl.store(offsets_out + pos, offset, mask=bad)
