"""Triton fixed-payload Huffman decoding kernel."""

import triton
from triton import language as tl

from ...codec.autotune import DECODE_AUTOTUNE_CONFIGS


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
    # One program decodes one codec block; each lane owns one fixed Huffman stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
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
    zero_delta = (-127 - center_value) & 255
    # Fast path: fully-contained blocks skip all per-element validity checks.
    # The swizzled logical mapping (storage -> flattened output) still applies.
    # Hoisted swizzle: shift depends only on step + loop-invariant block_shift.
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block_base + lanes
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            out_k0 = (lanes + ((block_shift + step) & 255)) & 255
            out_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            if not ON_DEMAND:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
                ).to(tl.uint32).to(tl.uint64)
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            first = tl.load(
                decode_table + (current & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length = first & 255
            continuation = first_length == 0
            length = tl.where(continuation, RARE_LENGTH + 8, first_length)
            symbol = ((current >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero = symbol == 0
            delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
            delta = tl.where(delta >= 128, delta - 256, delta)
            escaped_value = tl.where(is_zero, -127, delta + center_value)
            value = tl.where(continuation, escaped_value, first >> 8)
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            packed = (
                (((value.to(tl.int32) + 127) & 255) << 7)
                | (sm.to(tl.int32) & 0x7F)
                | ((sm.to(tl.int32) & 0x80) << 8)
            )
            tl.store(output + output_offset0, packed.to(tl.int16), cache_modifier='.cs')
            # Decode symbol 1 at the bit offset after symbol 0.
            shift1 = shift + length
            current1 = window >> shift1
            first1 = tl.load(
                decode_table + (current1 & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = tl.where(continuation1, RARE_LENGTH + 8, first_length1)
            symbol1 = ((current1 >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero1 = symbol1 == 0
            delta1 = tl.where(symbol1 == 0, 0, symbol1 - (symbol1 <= zero_delta).to(tl.int32))
            delta1 = tl.where(delta1 >= 128, delta1 - 256, delta1)
            escaped_value1 = tl.where(is_zero1, -127, delta1 + center_value)
            value1 = tl.where(continuation1, escaped_value1, first1 >> 8)
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            packed1 = (
                (((value1.to(tl.int32) + 127) & 255) << 7)
                | (sm1.to(tl.int32) & 0x7F)
                | ((sm1.to(tl.int32) & 0x80) << 8)
            )
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
        storage_offset = block_base + lanes
        logical_n = block * N_STEPS
        for step in tl.range(0, N_STEPS, 2):
            word2_prefetch = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams + lane_index
            ).to(tl.uint32).to(tl.uint64)
            logical_n0 = logical_n + step
            logical_n1 = logical_n0 + 1
            out_k0 = (lanes + ((block_shift + step) & 255)) & 255
            out_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            output_offset0 = logical_n0 * N_LANES + out_k0
            output_offset1 = logical_n1 * N_LANES + out_k1
            storage_valid = storage_offset < n_elements
            valid = output_offset0 < LOGICAL_NUMEL
            # Decode symbol 0 from the current prefix bits.
            current = window >> shift
            first = tl.load(
                decode_table + (current & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length = first & 255
            continuation = first_length == 0
            length = tl.where(continuation, RARE_LENGTH + 8, first_length)
            symbol = ((current >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero = symbol == 0
            delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
            delta = tl.where(delta >= 128, delta - 256, delta)
            escaped_value = tl.where(is_zero, -127, delta + center_value)
            value = tl.where(continuation, escaped_value, first >> 8)
            sm = tl.load(
                sign_mantissa + storage_offset,
                mask=storage_valid, other=0, cache_modifier='.cg',
            )
            packed = (
                (((value.to(tl.int32) + 127) & 255) << 7)
                | (sm.to(tl.int32) & 0x7F)
                | ((sm.to(tl.int32) & 0x80) << 8)
            )
            tl.store(output + output_offset0, packed.to(tl.int16), mask=valid, cache_modifier='.cs')
            shift1 = shift + tl.where(storage_valid, length, 0)
            current1 = window >> shift1
            first1 = tl.load(
                decode_table + (current1 & FIRST_MASK).to(tl.int32),
                cache_modifier='.ca',
            )
            first_length1 = first1 & 255
            continuation1 = first_length1 == 0
            length1 = tl.where(continuation1, RARE_LENGTH + 8, first_length1)
            symbol1 = ((current1 >> RARE_LENGTH) & 255).to(tl.int32)
            is_zero1 = symbol1 == 0
            delta1 = tl.where(symbol1 == 0, 0, symbol1 - (symbol1 <= zero_delta).to(tl.int32))
            delta1 = tl.where(delta1 >= 128, delta1 - 256, delta1)
            escaped_value1 = tl.where(is_zero1, -127, delta1 + center_value)
            value1 = tl.where(continuation1, escaped_value1, first1 >> 8)
            storage_offset1 = storage_offset + N_LANES
            storage_valid1 = storage_offset1 < n_elements
            valid1 = output_offset1 < LOGICAL_NUMEL
            sm1 = tl.load(
                sign_mantissa + storage_offset1,
                mask=storage_valid1, other=0, cache_modifier='.cg',
            )
            packed1 = (
                (((value1.to(tl.int32) + 127) & 255) << 7)
                | (sm1.to(tl.int32) & 0x7F)
                | ((sm1.to(tl.int32) & 0x80) << 8)
            )
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
