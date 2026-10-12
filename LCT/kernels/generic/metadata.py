"""Block summaries, prefix offsets, and GPU allocation for overflow storage.

The compact fallback list is stored per storage block (see
``LCT/kernels/generic/compaction.py``), so every consumer walks it one block at
a time and needs no global offset array.
"""
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
        # Two metadata bytes per overflowing stream: lane, then start step.
        _allocate_kernel(
            counts + 1, counts, descriptor, free_starts, free_sizes,
            free_count, lock, generation, 16, 2, MAX_FREE_REGIONS,
        )
