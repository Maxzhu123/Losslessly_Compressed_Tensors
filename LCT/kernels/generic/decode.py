"""Triton fixed-payload Huffman decoding kernel.
"""
import triton
from triton import language as tl

from ..autotune import (
    DECODE_AUTOTUNE_CONFIGS, DECODE_OVERFLOW_AUTOTUNE_CONFIGS,
)
from ..primitives import decode_symbol, pack_bf16
from .layout import swizzle_lane

@triton.jit
def _decode_rows_impl(
    encoded, sign_mantissa, output, decode_table, n_elements, n_streams, center,
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
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block_base + lane_ids
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            out_k0 = swizzle_lane(block * N_STEPS, step, lane_ids)
            out_k1 = swizzle_lane(block * N_STEPS, step + 1, lane_ids)
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            if not ON_DEMAND:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
                ).to(tl.uint32).to(tl.uint64)
            if OVERFLOW:
                # Issue both tail reads before the dependent symbol chain so the
                # gather latency overlaps the table lookups.
                fb0 = step >= stream_start
                fb1 = step + 1 >= stream_start
                fb_index0 = tl.where(fb0, stream_offset + step - stream_start, 0)
                fb_index1 = tl.where(fb1, stream_offset + step + 1 - stream_start, 0)
                fallback0 = tl.load(
                    tail + tail_base + fb_index0, mask=fb0, other=0,
                ).to(tl.int32)
                fallback1 = tl.load(
                    tail + tail_base + fb_index1, mask=fb1, other=0,
                ).to(tl.int32)
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            value, length = decode_symbol(
                current, decode_table, center_value, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value = tl.where(fb0, fallback0, value)
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            packed = pack_bf16(value, sm)
            tl.store(output + output_offset0, packed.to(tl.int16), cache_modifier='.cs')
            # Decode symbol 1 at the bit offset after symbol 0.
            shift1 = shift + length
            current1 = window >> shift1
            value1, length1 = decode_symbol(
                current1, decode_table, center_value, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value1 = tl.where(fb1, fallback1, value1)
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            packed1 = pack_bf16(value1, sm1)
            tl.store(
                output + output_offset1,
                packed1.to(tl.int16),
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
            out_k0 = swizzle_lane(block * N_STEPS, step, lane_ids)
            out_k1 = swizzle_lane(block * N_STEPS, step + 1, lane_ids)
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            storage_valid = storage_offset < n_elements
            valid = output_offset0 < LOGICAL_NUMEL
            if OVERFLOW:
                fb0 = step >= stream_start
                fb1 = step + 1 >= stream_start
                fb_index0 = tl.where(fb0, stream_offset + step - stream_start, 0)
                fb_index1 = tl.where(fb1, stream_offset + step + 1 - stream_start, 0)
                fallback0 = tl.load(
                    tail + tail_base + fb_index0, mask=fb0, other=0,
                ).to(tl.int32)
                fallback1 = tl.load(
                    tail + tail_base + fb_index1, mask=fb1, other=0,
                ).to(tl.int32)
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            value, length = decode_symbol(
                current, decode_table, center_value, FIRST_MASK, RARE_LENGTH,
            )
            if OVERFLOW:
                value = tl.where(fb0, fallback0, value)
            sm = tl.load(
                sign_mantissa + storage_offset,
                mask=storage_valid, other=0, cache_modifier='.cg',
            )
            packed = pack_bf16(value, sm)
            tl.store(output + output_offset0, packed.to(tl.int16), mask=valid, cache_modifier='.cs')
            shift1 = shift + tl.where(storage_valid, length, 0)
            current1 = window >> shift1
            value1, length1 = decode_symbol(
                current1, decode_table, center_value, FIRST_MASK, RARE_LENGTH,
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
                packed1.to(tl.int16), mask=valid1, cache_modifier='.cs',
            )
            next_shift = shift1 + tl.where(storage_valid1, length1, 0)
            crosses_word = next_shift >= 32
            next_window = (window >> 32) | (word2_prefetch << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES


@triton.jit
def _decode_impl(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams, center,
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
            _decode_rows_impl(
                encoded, sign_mantissa, output, decode_table, n_elements, n_streams, center,
                tail, tail_base, stream_start, stream_offset,
                LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS,
                FIXED_WORDS, ON_DEMAND, False,
            )
            return
    _decode_rows_impl(
        encoded, sign_mantissa, output, decode_table, n_elements, n_streams, center,
        tail, tail_base, stream_start, stream_offset,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS,
        FIXED_WORDS, ON_DEMAND, OVERFLOW,
    )


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_STEPS", "FIXED_WORDS", "ON_DEMAND"],
)
@triton.jit
def _decode_kernel(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams, center,
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
    _decode_impl(
        encoded, sign_mantissa, output, decode_table, n_elements, n_streams, center,
        encoded, encoded, encoded, encoded, encoded, encoded, encoded,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
        ON_DEMAND, False, False, False, False,
    )


@triton.autotune(
    configs=DECODE_OVERFLOW_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_STEPS", "FIXED_WORDS", "ON_DEMAND", "BUFFERED"],
)
@triton.jit
def _decode_overflow_kernel(
    encoded, sign_mantissa, output,
    decode_table, n_elements, n_streams, center,
    lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
    ON_DEMAND: tl.constexpr, BUFFERED: tl.constexpr,
    PARALLEL_MAP: tl.constexpr = True, EMPTY_FAST_PATH: tl.constexpr = True,
):
    """Dense decode that substitutes compact fallback tails in place."""
    _decode_impl(
        encoded, sign_mantissa, output, decode_table, n_elements, n_streams, center,
        lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
        LOGICAL_NUMEL, FIRST_MASK, RARE_LENGTH, BLOCK, N_LANES, N_STEPS, FIXED_WORDS,
        ON_DEMAND, True, BUFFERED, PARALLEL_MAP, EMPTY_FAST_PATH,
    )
