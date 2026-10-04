"""Dedicated fused scalar multiply-add pointwise kernels.

Kept separate from the binary pointwise kernels so the existing add/multiply
hot path does not pay any scalar/alpha overhead.
"""

import triton
from triton import language as tl

from ...codec.autotune import (
    DECODE_AUTOTUNE_CONFIGS,
    POINTWISE_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..primitives import decode_symbol, pack_bf16
from .pointwise import (
    _pointwise_location,
    _pointwise_fallback_impl,
    _store_result,
)


@triton.jit
def _scaled_sum(left, right, alpha, beta, SCALE_OTHER: tl.constexpr, ALPHA_IS_ONE: tl.constexpr):
    if ALPHA_IS_ONE:
        if SCALE_OTHER:
            return tl.math.fma(right.to(tl.float32), beta, left.to(tl.float32))
        return left.to(tl.float32) + right.to(tl.float32)
    if SCALE_OTHER:
        right = right.to(tl.float32) * beta
    return tl.math.fma(left, alpha, right)


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY", "SCALE_OTHER", "ALPHA_IS_ONE"],
)
@triton.jit
def pointwise_scalar_mul_add_dense_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center, alpha, beta,
    SCALE_OTHER: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """Apply ``alpha * decoded + other``, optionally scaling other by beta."""
    # One program handles one codec block; each lane decodes one fixed stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(encoded + word * n_streams + lane_index)
    word1 = tl.load(encoded + (word + 1) * n_streams + lane_index)
    window = word0.to(tl.uint32).to(tl.uint64)
    window |= word1.to(tl.uint32).to(tl.uint64) << 32
    center_value = tl.load(center).to(tl.int32)
    alpha_value = 1.0
    if not ALPHA_IS_ONE:
        alpha_value = tl.load(alpha).to(tl.float32)
    beta_value = 1.0
    if SCALE_OTHER:
        beta_value = tl.load(beta).to(tl.float32)

    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        # Hoisted swizzle: shift depends only on step + loop-invariant block_shift.
        block_shift = (block * N_STEPS) & 255
        for step in tl.range(0, N_STEPS, 2, flatten=True, warp_specialize=True):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            logical_k0 = (lanes + ((block_shift + step) & 255)) & 255
            logical_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            logical_offset0 = logical_n0 * N_LANES + logical_k0
            logical_offset1 = logical_n1 * N_LANES + logical_k1
            current = window >> shift
            value, length = decode_symbol(
                current, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm = tl.load(sign_mantissa + storage_offset, cache_modifier='.cg')
            left = pack_bf16(value, sm).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right = tl.load(other + logical_offset0, cache_modifier='.cg')
            _store_result(
                _scaled_sum(left, right, alpha_value, beta_value, SCALE_OTHER, ALPHA_IS_ONE),
                output, auxiliary, storage_offset,
                logical_offset0, true_mask, true_mask, OUTPUT_POLICY,
            )
            shift1 = shift + length
            current1 = window >> shift1
            value1, length1 = decode_symbol(
                current1, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm1 = tl.load(sign_mantissa + storage_offset + N_LANES, cache_modifier='.cg')
            left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right1 = tl.load(other + logical_offset1, cache_modifier='.cg')
            _store_result(
                _scaled_sum(left1, right1, alpha_value, beta_value, SCALE_OTHER, ALPHA_IS_ONE),
                output, auxiliary,
                storage_offset + N_LANES, logical_offset1,
                true_mask, true_mask, OUTPUT_POLICY,
            )
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            word2 = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=crosses_word, other=0,
            ).to(tl.uint32).to(tl.uint64)
            next_window = (window >> 32) | (word2 << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)
            storage_offset += 2 * N_LANES
    else:
        # Tail path: flattened logical coordinates with bounds masks.
        for step in tl.range(0, N_STEPS, 2):
            offset, logical_offset, storage_valid, valid = _pointwise_location(
                block, step, lanes, n_elements, LOGICAL_NUMEL,
                BLOCK, N_LANES, N_STEPS,
            )
            value, length = decode_symbol(
                window >> shift, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm = tl.load(sign_mantissa + offset, mask=storage_valid, other=0, cache_modifier='.cg')
            left = pack_bf16(value, sm).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right = tl.load(other + logical_offset, mask=valid, other=0.0, cache_modifier='.cg')
            _store_result(
                _scaled_sum(left, right, alpha_value, beta_value, SCALE_OTHER, ALPHA_IS_ONE),
                output, auxiliary, offset, logical_offset,
                valid, storage_valid, OUTPUT_POLICY,
            )
            shift1 = shift + tl.where(storage_valid, length, 0)
            offset1, logical_offset1, storage_valid1, valid1 = _pointwise_location(
                block, step + 1, lanes, n_elements, LOGICAL_NUMEL,
                BLOCK, N_LANES, N_STEPS,
            )
            value1, length1 = decode_symbol(
                window >> shift1, decode_table, center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            sm1 = tl.load(sign_mantissa + offset1, mask=storage_valid1, other=0, cache_modifier='.cg')
            left1 = pack_bf16(value1, sm1).to(tl.int16).to(
                tl.bfloat16, bitcast=True
            )
            right1 = tl.load(other + logical_offset1, mask=valid1, other=0.0, cache_modifier='.cg')
            _store_result(
                _scaled_sum(left1, right1, alpha_value, beta_value, SCALE_OTHER, ALPHA_IS_ONE),
                output, auxiliary, offset1, logical_offset1,
                valid1, storage_valid1, OUTPUT_POLICY,
            )
            next_shift = shift1 + tl.where(storage_valid1, length1, 0)
            crosses_word = next_shift >= 32
            word2 = tl.load(
                encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                + lane_index,
                mask=crosses_word, other=0,
            ).to(tl.uint32).to(tl.uint64)
            next_window = (window >> 32) | (word2 << 32)
            window = tl.where(crosses_word, next_window, window)
            word += crosses_word
            shift = tl.where(crosses_word, next_shift - 32, next_shift)


@triton.autotune(
    configs=POINTWISE_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK", "OUTPUT_POLICY", "SCALE_OTHER", "ALPHA_IS_ONE"],
)
@triton.jit
def pointwise_scalar_mul_add_dense_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements, alpha, beta,
    SCALE_OTHER: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr, ROW_TILE: tl.constexpr,
):
    count = tl.load(fallback_count).to(tl.int32)
    if tl.program_id(0) * TILE >= count:
        return
    alpha_value = 1.0
    if not ALPHA_IS_ONE:
        alpha_value = tl.load(alpha).to(tl.float32)
    beta_value = 1.0
    if SCALE_OTHER:
        beta_value = tl.load(beta).to(tl.float32)
    for tile_id in tl.range(tl.program_id(0), tl.cdiv(count, TILE), tl.num_programs(0)):
        _pointwise_fallback_impl(
            bad_streams, bad_starts, fallback_offsets, fallback_buffer,
            fallback_base, metadata, descriptor, count,
            sign_mantissa, other, output, auxiliary, n_elements, tile_id,
            alpha_value, beta_value,
            OP=_scaled_sum, SCALED=True,
            SCALE_OTHER=SCALE_OTHER, ALPHA_IS_ONE=ALPHA_IS_ONE,
            OUTPUT_POLICY=OUTPUT_POLICY, BUFFERED=BUFFERED,
            LOGICAL_NUMEL=LOGICAL_NUMEL, TILE=TILE, ROW_TILE=ROW_TILE,
            BLOCK=BLOCK, N_LANES=N_LANES, N_STEPS=N_STEPS,
        )
