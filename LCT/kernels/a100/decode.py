"""A100 single-symbol decoder with a precomputed escape table.

``decode_symbol`` evaluates the escape (rare-code) path for every symbol even
though only ~0.1% of clean Gaussian symbols take it. On Ampere that dead
arithmetic -- symbol extraction, zero-delta bias, sign fixups -- is the largest
single block of work in the decode inner loop, and it sits on the loop-carried
dependency chain alongside the table lookup.

This decoder replaces those ~13 instructions with one extra load from a
256-entry table built from the center. The table is 1 KiB, stays in L1, and is
only read (never written) by the decode loop.

Everything else -- payload ABI, sign/mantissa layout, compact overflow
metadata, window bookkeeping -- matches the shared decoder, so payloads are
interchangeable.
"""

import torch
import triton
from triton import language as tl



@triton.jit
def _build_escape_table_kernel(center, escape_table, BLOCK: tl.constexpr):
    """Materialise the escape exponent for every raw 8-bit remainder."""
    raw = tl.arange(0, BLOCK)
    center_value = tl.load(center).to(tl.int32)
    zero_delta = (-127 - center_value) & 255
    is_zero = raw == 0
    delta = tl.where(is_zero, 0, raw - (raw <= zero_delta).to(tl.int32))
    delta = tl.where(delta >= 128, delta - 256, delta)
    tl.store(escape_table + raw, tl.where(is_zero, -127, delta + center_value))


@triton.jit
def _fast_symbol(
    current, decode_table, escape_table, FIRST_MASK: tl.constexpr,
    RARE_LENGTH: tl.constexpr,
):
    """Decode one symbol, reading the escape exponent from the table."""
    first = tl.load(decode_table + (current & FIRST_MASK).to(tl.int32), cache_modifier='.ca')
    first_length = first & 255
    continuation = first_length == 0
    raw = ((current >> RARE_LENGTH) & 255).to(tl.int32)
    escaped = tl.load(escape_table + raw)
    value = tl.where(continuation, escaped, first >> 8)
    length = tl.where(continuation, RARE_LENGTH + 8, first_length)
    return value, length


def build_escape_table(center: torch.Tensor,
                       decode_table: torch.Tensor) -> torch.Tensor:
    """Return the 256-entry escape exponent table for one center.

    The table depends only on the center, so it is cached against the shifted
    decode table it was built beside. The cache keeps one strong reference and
    compares by identity, so a freed tensor can never hand back a stale table,
    while a repeat decode of the same compressed tensor rebuilds nothing.
    """
    cached = _ESCAPE_CACHE
    if cached["decode_table"] is decode_table:
        return cached["table"]
    escape_table = torch.empty(256, dtype=torch.int32, device=center.device)
    _build_escape_table_kernel[(1,)](center, escape_table, BLOCK=256)
    cached["decode_table"] = decode_table
    cached["table"] = escape_table
    return escape_table


# One-entry identity cache; see ``build_escape_table``.
_ESCAPE_CACHE: dict = {"decode_table": None, "table": None}


from ..autotune import (
    DECODE_AUTOTUNE_CONFIGS, DECODE_OVERFLOW_AUTOTUNE_CONFIGS,
)
from ..primitives import pack_bf16

@triton.jit
def _a100_decode_rows_impl(
    encoded, sign_mantissa, output, decode_table, escape_table, n_elements, n_streams, center,
    tail, tail_base, stream_start, stream_offset,
    LOGICAL_NUMEL: tl.constexpr, FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    FIXED_WORDS: tl.constexpr, ON_DEMAND: tl.constexpr, OVERFLOW: tl.constexpr,
):
    """Decode one codec block per program; each lane owns one Huffman stream."""
    block = tl.program_id(0)
    lane_ids = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lane_ids
    # word/shift form the current 64-bit decoding window into the fixed payload.
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(
        encoded + word * n_streams + lane_index
    ).to(tl.uint32).to(tl.uint64)
    word1 = tl.load(
        encoded + (word + 1) * n_streams + lane_index
    ).to(tl.uint32).to(tl.uint64)
    window = word0 | (word1 << 32)
    center_value = tl.load(center).to(tl.int32)

    # Fast path: fully-contained blocks skip all per-element validity checks.
    # The swizzled logical mapping (storage -> flattened output) still applies.
    block_base = block * BLOCK
    # The compact tail offset of a lane only changes when the lane overflows,
    # so fold the constant part once instead of rebuilding it every step.
    fallback_base = stream_offset - stream_start
    # Even swizzle masks keep adjacent lane pairs adjacent in memory.
    pair_ids = tl.arange(0, N_LANES // 2)
    pair_bits = tl.arange(0, 2)
    swizzle_mask = (block * N_STEPS) & (N_LANES - 2)
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block_base + lane_ids
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            # Rows 2k and 2k+1 share an even mask, so both stores in one
            # step use the same permutation.
            swizzle0 = (swizzle_mask + step) & (N_LANES - 2)
            swizzle1 = swizzle0
            # The swizzle mask is even, so lanes 2p and 2p+1 stay adjacent in
            # memory.  Expressing the offsets as a (N_LANES/2, 2) tile with an
            # explicit contiguous minor axis lets Triton vectorise the store
            # instead of staging every row through shared memory.
            output_offset0 = (
                logical_n0 * N_LANES
                + ((pair_ids ^ (swizzle0 >> 1)) << 1)[:, None] + pair_bits[None, :]
            )
            output_offset1 = (
                logical_n1 * N_LANES
                + ((pair_ids ^ (swizzle1 >> 1)) << 1)[:, None] + pair_bits[None, :]
            )
            if not ON_DEMAND:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
                ).to(tl.uint32).to(tl.uint64)
            if OVERFLOW:
                # Issue both tail reads before the dependent symbol chain so the
                # gather latency overlaps the table lookups.
                fb0 = step >= stream_start
                fb1 = step + 1 >= stream_start
                fb_index0 = fallback_base + step
                fb_index1 = fb_index0 + 1
                fallback0 = tl.load(
                    tail + tail_base + fb_index0, mask=fb0, other=0,
                ).to(tl.int32)
                fallback1 = tl.load(
                    tail + tail_base + fb_index1, mask=fb1, other=0,
                ).to(tl.int32)
            # Decode symbol 0 from the current prefix bits.
            # A symbol consumes at most 16 bits, so the two symbols of one
            # step read at most bits [0, 32) of the window. Keeping ``current``
            # in 32 bits lets symbol 1 chain off it with a 32-bit shift instead
            # of shifting the 64-bit window a second time.
            current = (window >> shift).to(tl.uint32)
            value, length = _fast_symbol(
                current, decode_table, escape_table, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value = tl.where(fb0, fallback0, value)
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            packed = pack_bf16(value, sm)
            tl.store(
                output + output_offset0,
                tl.reshape(packed.to(tl.int16), (N_LANES // 2, 2)),
                cache_modifier='.cs',
            )
            # Decode symbol 1 at the bit offset after symbol 0.
            shift1 = shift + length
            current1 = current >> length
            value1, length1 = _fast_symbol(
                current1, decode_table, escape_table, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value1 = tl.where(fb1, fallback1, value1)
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            packed1 = pack_bf16(value1, sm1)
            tl.store(
                output + output_offset1,
                tl.reshape(packed1.to(tl.int16), (N_LANES // 2, 2)),
                cache_modifier='.cs',
            )
            # Advance the 64-bit window when both symbols cross a 32-bit word boundary.
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            if ON_DEMAND:
                next_word = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index,
                    mask=crosses_word, other=0,
                ).to(tl.uint32).to(tl.uint64)
                next_window = (window >> 32) | (next_word << 32)
            else:
                next_window = (window >> 32) | (word2_prefetch << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES
    else:
        # Tail path: map the codec storage block back to flattened coordinates
        # with bounds masks.
        storage_offset = block_base + lane_ids
        logical_n = block * N_STEPS
        for step in tl.range(0, N_STEPS, 2):
            word2_prefetch = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
            ).to(tl.uint32).to(tl.uint64)
            logical_n0 = logical_n + step
            logical_n1 = logical_n0 + 1
            # Rows 2k and 2k+1 share an even mask, so both stores in one
            # step use the same permutation.
            swizzle0 = (swizzle_mask + step) & (N_LANES - 2)
            swizzle1 = swizzle0
            # The swizzle mask is even, so lanes 2p and 2p+1 stay adjacent in
            # memory.  Expressing the offsets as a (N_LANES/2, 2) tile with an
            # explicit contiguous minor axis lets Triton vectorise the store
            # instead of staging every row through shared memory.
            output_offset0 = (
                logical_n0 * N_LANES
                + ((pair_ids ^ (swizzle0 >> 1)) << 1)[:, None] + pair_bits[None, :]
            )
            output_offset1 = (
                logical_n1 * N_LANES
                + ((pair_ids ^ (swizzle1 >> 1)) << 1)[:, None] + pair_bits[None, :]
            )
            storage_valid = storage_offset < n_elements
            valid = output_offset0 < LOGICAL_NUMEL
            if OVERFLOW:
                fb0 = step >= stream_start
                fb1 = step + 1 >= stream_start
                fb_index0 = fallback_base + step
                fb_index1 = fb_index0 + 1
                fallback0 = tl.load(
                    tail + tail_base + fb_index0, mask=fb0, other=0,
                ).to(tl.int32)
                fallback1 = tl.load(
                    tail + tail_base + fb_index1, mask=fb1, other=0,
                ).to(tl.int32)
            # Decode symbol 0 from the current prefix bits.
            # A symbol consumes at most 16 bits, so the two symbols of one
            # step read at most bits [0, 32) of the window. Keeping ``current``
            # in 32 bits lets symbol 1 chain off it with a 32-bit shift instead
            # of shifting the 64-bit window a second time.
            current = (window >> shift).to(tl.uint32)
            value, length = _fast_symbol(
                current, decode_table, escape_table, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value = tl.where(fb0, fallback0, value)
            sm = tl.load(
                sign_mantissa + storage_offset,
                mask=storage_valid, other=0, cache_modifier='.cg',
            )
            packed = pack_bf16(value, sm)
            tl.store(
                output + output_offset0,
                tl.reshape(packed.to(tl.int16), (N_LANES // 2, 2)),
                mask=valid, cache_modifier='.cs',
            )
            step_length = tl.where(storage_valid, length, 0)
            shift1 = shift + step_length
            current1 = current >> step_length
            value1, length1 = _fast_symbol(
                current1, decode_table, escape_table, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value1 = tl.where(fb1, fallback1, value1)
            storage_offset1 = storage_offset + N_LANES
            storage_valid1 = storage_offset1 < n_elements
            valid1 = output_offset1 < LOGICAL_NUMEL
            sm1 = tl.load(
                sign_mantissa + storage_offset1,
                mask=storage_valid1, other=0, cache_modifier='.cg',
            )
            packed1 = pack_bf16(value1, sm1)
            tl.store(
                output + output_offset1,
                tl.reshape(packed1.to(tl.int16), (N_LANES // 2, 2)),
                mask=valid1, cache_modifier='.cs',
            )
            next_shift = shift1 + tl.where(storage_valid1, length1, 0)
            crosses_word = next_shift >= 32
            next_window = (window >> 32) | (word2_prefetch << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES


@triton.jit
def _a100_decode_impl(
    encoded, sign_mantissa, output,
    decode_table, escape_table, n_elements, n_streams, center,
    lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    ON_DEMAND: tl.constexpr,
    OVERFLOW: tl.constexpr, BUFFERED: tl.constexpr, PARALLEL_MAP: tl.constexpr,
    EMPTY_FAST_PATH: tl.constexpr,
):
    block = tl.program_id(0)
    lane_ids = tl.arange(0, N_LANES)
    tail_base = 0
    stream_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    stream_offset = tl.zeros((N_LANES,), tl.int32)
    # Overflow streams carry their tail in separate compact storage. Resolving
    # it here replaces a second scatter pass over the whole tensor, so the
    # sign/mantissa byte and the output row are each touched only once.
    if OVERFLOW:
        total = tl.load(fallback_count).to(tl.int32)
        if BUFFERED:
            base = tl.load(descriptor).to(tl.int32)
            lane_off = base
            start_off = base + total
            tail_base = base + 2 * total
        else:
            lane_off = 0
            start_off = 0
            tail_base = fallback_base
        # This block's entries are contiguous, so a running sum over them gives
        # every one of its streams its dense tail offset. That is cheaper than
        # building and reading a full per-stream map array.
        n_blocks = tl.num_programs(0)
        stream_start = tl.full((N_LANES,), N_STEPS, tl.int32)
        stream_offset = tl.zeros((N_LANES,), tl.int32)
        block_count = tl.load(blocks + block).to(tl.int32)
        if block_count > 0:
            first = tl.load(blocks + n_blocks + block).to(tl.int32)
            byte_base = tl.load(blocks + 2 * n_blocks + block).to(tl.int32)
            if PARALLEL_MAP:
                # Entries are sorted by lane. A histogram and prefix sum map each
                # storage lane to its compact rank in parallel, avoiding a serial
                # loop over as many as 256 overflowing streams on Ampere.
                entry = tl.arange(0, N_LANES)
                valid = entry < block_count
                entry_lane = tl.load(lanes + lane_off + first + entry, mask=valid, other=0).to(tl.int32)
                entry_start = tl.load(starts + start_off + first + entry, mask=valid, other=N_STEPS).to(tl.int32)
                length = tl.where(valid, N_STEPS - entry_start, 0)
                offsets = byte_base + tl.cumsum(length, axis=0) - length
                present = tl.histogram(entry_lane, N_LANES, mask=valid)
                rank = tl.maximum(tl.cumsum(present, axis=0) - 1, 0)
                stream_start = tl.where(present != 0, tl.gather(entry_start, rank, axis=0), N_STEPS)
                stream_offset = tl.gather(offsets, rank, axis=0)
            else:
                # Preserve the measured 5090 path; Ampere uses the parallel map.
                running = tl.zeros((), tl.int32)
                for entry in tl.range(0, block_count):
                    entry_lane = tl.load(lanes + lane_off + first + entry).to(tl.int32)
                    entry_start = tl.load(starts + start_off + first + entry).to(tl.int32)
                    entry_offset = byte_base + running
                    running += N_STEPS - entry_start
                    hit = entry_lane == lane_ids
                    stream_start = tl.where(hit, entry_start, stream_start)
                    stream_offset = tl.where(hit, entry_offset, stream_offset)

    if OVERFLOW and EMPTY_FAST_PATH:
        if block_count == 0:
            _a100_decode_rows_impl(
                encoded, sign_mantissa, output, decode_table, escape_table, n_elements, n_streams, center,
                tail, tail_base, stream_start, stream_offset,
                LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS,
                FIXED_WORDS, ON_DEMAND, False,
            )
            return
    _a100_decode_rows_impl(
        encoded, sign_mantissa, output, decode_table, escape_table, n_elements, n_streams, center,
        tail, tail_base, stream_start, stream_offset,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS,
        FIXED_WORDS, ON_DEMAND, OVERFLOW,
    )


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_STEPS", "FIXED_WORDS", "ON_DEMAND"],
)
@triton.jit
def _a100_decode_kernel(
    encoded, sign_mantissa, output,
    decode_table, escape_table, n_elements, n_streams, center,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    ON_DEMAND: tl.constexpr,
):
    """Plain dense decode.

    Deliberately keeps the original argument count: Triton's launcher cost
    grows with the parameter list, and this kernel runs on every decode.
    """
    _a100_decode_impl(
        encoded, sign_mantissa, output, decode_table, escape_table, n_elements, n_streams, center,
        encoded, encoded, encoded, encoded, encoded, encoded, encoded,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
        ON_DEMAND, False, False, False, False,
    )


@triton.autotune(
    configs=DECODE_OVERFLOW_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_STEPS", "FIXED_WORDS", "ON_DEMAND", "BUFFERED"],
)
@triton.jit
def _a100_decode_overflow_kernel(
    encoded, sign_mantissa, output,
    decode_table, escape_table, n_elements, n_streams, center,
    lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    ON_DEMAND: tl.constexpr, BUFFERED: tl.constexpr,
    PARALLEL_MAP: tl.constexpr = True, EMPTY_FAST_PATH: tl.constexpr = True,
):
    """Dense decode that substitutes compact fallback tails in place."""
    _a100_decode_impl(
        encoded, sign_mantissa, output, decode_table, escape_table, n_elements, n_streams, center,
        lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
        ON_DEMAND, True, BUFFERED, PARALLEL_MAP, EMPTY_FAST_PATH,
    )
