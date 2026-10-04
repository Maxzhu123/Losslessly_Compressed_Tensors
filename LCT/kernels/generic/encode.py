"""Triton fixed-payload Huffman encoding kernel."""

import triton
from triton import language as tl

from ...codec.autotune import ENCODE_AUTOTUNE_CONFIGS


@triton.autotune(
    configs=ENCODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "PRECOMPUTED", "WRITE_SUMMARY"],
)
@triton.jit
def _encode_kernel(
    source_bits, sign_mantissa, encoded, encode_table,
    extra_starts, summaries, n_elements, n_streams,
    PRECOMPUTED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIXED_WORDS: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
    WRITE_SUMMARY: tl.constexpr = False,
):
    """Encode one flattened contiguous block per program (1D storage).

    Storage offset ``block * BLOCK + step * N_LANES + lane`` maps to the
    flattened logical element ``logical_n * N_LANES + out_k`` with the
    row-dependent swizzle ``out_k = (lane + (logical_n & 255)) & 255``,
    which spreads neighbouring values across lanes.
    """
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word_value = tl.zeros((N_LANES,), tl.uint32)
    overflow = tl.zeros((N_LANES,), tl.int1)
    extra_start = tl.full((N_LANES,), N_STEPS, tl.int32)
    has_data = block * BLOCK + lanes < n_elements

    logical_n_base = block * N_STEPS
    # Hoisted swizzle: (block*N_STEPS + step) & 255 == (block_shift + step) & 255
    # with block_shift loop-invariant, so the hot loop only sees `step`.
    # (N_STEPS=256 -> block_shift=0; N_STEPS=128 -> 128*(block & 1).)
    block_shift = (block * N_STEPS) & 255
    block_base = block * BLOCK
    # Fully-valid blocks can skip all per-element validity/mask checks.
    full_block = (block + 1) * BLOCK <= LOGICAL_NUMEL
    if full_block:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            value0 = tl.load(source_bits + input_offset0).to(tl.int32)
            value1 = tl.load(source_bits + input_offset1).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(sign_mantissa + source_offset, sm0.to(tl.uint8))
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8),
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            length0 = (packed0 >> 20).to(tl.int32)
            length1 = (packed1 >> 20).to(tl.int32)
            length = length0 + length1
            code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

            new_word = word_value | (code << shift)
            crosses_word = shift + length >= 32
            pair_overflow = word * 32 + shift + length > FIXED_WORDS * 32
            first_overflow = (~overflow) & pair_overflow

            extra_start = tl.where(first_overflow, step, extra_start)

            store_value = tl.where(first_overflow, word_value, new_word)
            safe_word = tl.minimum(word, FIXED_WORDS - 1)
            tl.store(
                encoded + safe_word * n_streams + lane_index,
                store_value,
                mask=crosses_word & (word < FIXED_WORDS),
            )
            word_value = tl.where(crosses_word, code >> (32 - shift), new_word)
            word += tl.where(crosses_word, 1, 0)
            shift = tl.where(crosses_word, shift + length - 32, shift + length)
            overflow |= pair_overflow
    else:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=4):
            source_offset = block_base + step * N_LANES + lanes
            valid0 = source_offset < n_elements
            valid1 = source_offset + N_LANES < n_elements
            logical_n = logical_n_base + step
            shift0 = (block_shift + step) & 255
            shift1 = (block_shift + step + 1) & 255
            input_k0 = (lanes + shift0) & 255
            input_k1 = (lanes + shift1) & 255
            input_offset0 = logical_n * N_LANES + input_k0
            input_offset1 = (logical_n + 1) * N_LANES + input_k1
            input_valid0 = input_offset0 < LOGICAL_NUMEL
            input_valid1 = input_offset1 < LOGICAL_NUMEL
            value0 = tl.load(
                source_bits + input_offset0, mask=input_valid0, other=0,
            ).to(tl.int32)
            value1 = tl.load(
                source_bits + input_offset1, mask=input_valid1, other=0,
            ).to(tl.int32)
            if PRECOMPUTED:
                byte0 = value0 & 255
                byte1 = value1 & 255
            else:
                byte0 = (value0 >> 7) & 0xFF
                byte1 = (value1 >> 7) & 0xFF
                sm0 = (value0 & 0x7F) | ((value0 >> 8) & 0x80)
                sm1 = (value1 & 0x7F) | ((value1 >> 8) & 0x80)
                tl.store(
                    sign_mantissa + source_offset, sm0.to(tl.uint8),
                    mask=valid0 & input_valid0,
                )
                tl.store(
                    sign_mantissa + source_offset + N_LANES,
                    sm1.to(tl.uint8), mask=valid1 & input_valid1,
                )
            packed0 = tl.load(encode_table + byte0).to(tl.uint32)
            packed1 = tl.load(encode_table + byte1).to(tl.uint32)
            packed0 = tl.where(input_valid0, packed0, 0)
            packed1 = tl.where(input_valid1, packed1, 0)
            length0 = (packed0 >> 20).to(tl.int32)
            length1 = (packed1 >> 20).to(tl.int32)
            length = length0 + length1
            code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)

            new_word = word_value | (code << shift)
            crosses_word = shift + length >= 32
            pair_overflow = (word * 32 + shift + length > FIXED_WORDS * 32) & valid0
            first_overflow = (~overflow) & pair_overflow

            extra_start = tl.where(first_overflow, step, extra_start)

            store_value = tl.where(first_overflow, word_value, new_word)
            safe_word = tl.minimum(word, FIXED_WORDS - 1)
            tl.store(
                encoded + safe_word * n_streams + lane_index,
                store_value,
                mask=crosses_word & (word < FIXED_WORDS),
            )
            word_value = tl.where(crosses_word, code >> (32 - shift), new_word)
            word += tl.where(crosses_word, 1, 0)
            shift = tl.where(crosses_word, shift + length - 32, shift + length)
            overflow |= pair_overflow

    safe_word = tl.minimum(word, FIXED_WORDS - 1)
    tl.store(
        encoded + safe_word * n_streams + lane_index,
        word_value,
        mask=has_data & (word < FIXED_WORDS) & (shift != 0),
    )
    tl.store(
        extra_starts + lane_index,
        tl.where(has_data & overflow, extra_start, 255),
    )

    if WRITE_SUMMARY:
        bad = has_data & overflow
        count = tl.sum(bad.to(tl.int32), axis=0)
        total = tl.sum(tl.where(bad, N_STEPS - extra_start, 0), axis=0)
        tl.store(summaries + block, count)
        tl.store(summaries + tl.num_programs(0) + block, total)
