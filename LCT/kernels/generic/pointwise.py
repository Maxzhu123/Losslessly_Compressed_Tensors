"""Generic Triton templates for pointwise compressed-tensor operations.
"""
import triton
from triton import language as tl

from ..autotune import (
    POINTWISE_DENSE_AUTOTUNE_CONFIGS,
    POINTWISE_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..primitives import decode_symbol, pack_bf16
from .layout import swizzle_lane


DENSE_OUTPUT = tl.constexpr(0)
COMPRESSED_OUTPUT = tl.constexpr(1)


@triton.jit
def _load_scalars(alpha, beta, ALPHA_IS_ONE: tl.constexpr, BETA_IS_ONE: tl.constexpr):
    """Load only the scalar values required by this specialization."""
    alpha_value = 1.0
    beta_value = 1.0
    if not ALPHA_IS_ONE:
        alpha_value = tl.load(alpha).to(tl.float32)
    if not BETA_IS_ONE:
        beta_value = tl.load(beta).to(tl.float32)
    return alpha_value, beta_value


@triton.jit
def _apply_pointwise(
    left, right, alpha, beta,
    OP: tl.constexpr, SCALED: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr, BETA_IS_ONE: tl.constexpr,
):
    """Apply the selected arithmetic with the same rounding in every kernel."""
    if SCALED:
        return OP(left, right, alpha, beta, ALPHA_IS_ONE, BETA_IS_ONE)
    else:
        return OP(left, right)


@triton.jit
def _store_result(
    result, output, auxiliary, storage_offset, logical_offset,
    logical_mask, storage_mask,
    OUTPUT_POLICY: tl.constexpr,
):
    """Round an operation result to BF16 and emit dense or component output."""
    result = result.to(tl.bfloat16)
    if OUTPUT_POLICY == DENSE_OUTPUT:
        tl.store(output + logical_offset, result, mask=logical_mask, cache_modifier='.cs')
    else:
        result = tl.where(logical_mask, result, 0.0).to(tl.bfloat16)
        bits = result.to(tl.int16, bitcast=True).to(tl.int32)
        sign_mantissa = (bits & 0x7F) | ((bits >> 8) & 0x80)
        exponent = (bits >> 7) & 0xFF
        tl.store(
            output + storage_offset, sign_mantissa.to(tl.uint8),
            mask=storage_mask,
        )
        tl.store(
            auxiliary + logical_offset, exponent.to(tl.uint8),
            mask=logical_mask,
        )


@triton.jit
def _pointwise_location(
    block, step, lane, n_elements,
    LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Map a storage-stream position to its logical offset and validity masks."""
    storage_offset = block * BLOCK + step * N_LANES + lane
    storage_valid = storage_offset < n_elements
    logical_n = block * N_STEPS + step
    logical_k = swizzle_lane(block * N_STEPS, step, lane)
    logical_offset = logical_n * N_LANES + logical_k
    logical_valid = logical_offset < LOGICAL_NUMEL
    return storage_offset, logical_offset, storage_valid, logical_valid


@triton.autotune(
    configs=POINTWISE_DENSE_AUTOTUNE_CONFIGS,
    key=[
        "n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY",
        "OP", "SCALED", "ALPHA_IS_ONE", "BETA_IS_ONE",
    ],
)
@triton.jit
def pointwise_compressed_dense_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center, alpha, beta,
    OP: tl.constexpr, SCALED: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr, BETA_IS_ONE: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """ compA + B, compA * B, or alpha * compA + beta * B
        Elementwise. Dense BF16 values or components for recompression.
    """
    # One program handles one codec block; each lane decodes one fixed stream.
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    # word/shift track the current position inside the 64-bit fixed-payload window.
    word = tl.zeros((N_LANES,), tl.int32)
    shift = tl.zeros((N_LANES,), tl.int32)
    word0 = tl.load(encoded + word * n_streams + lane_index)
    word1 = tl.load(encoded + (word + 1) * n_streams + lane_index)
    window = word0.to(tl.uint32).to(tl.uint64)
    window |= word1.to(tl.uint32).to(tl.uint64) << 32
    center_value = tl.load(center).to(tl.int32)

    # Unit scalars specialize away their loads and multiplications.
    alpha_value, beta_value = _load_scalars(alpha, beta, ALPHA_IS_ONE, BETA_IS_ONE)

    # Fast path: fully-contained blocks skip all per-element validity checks.
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        # Preserve the existing binary unrolling and scaled-operation scheduling.
        UNROLL_FACTOR: tl.constexpr = 1 if SCALED else 2
        for step in tl.range(
            0, N_STEPS, 2, loop_unroll_factor=UNROLL_FACTOR,
            flatten=SCALED, warp_specialize=SCALED,
        ):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            logical_k0 = swizzle_lane(block * N_STEPS, step, lanes)
            logical_k1 = swizzle_lane(block * N_STEPS, step + 1, lanes)
            logical_offset0 = logical_n0 * N_LANES + logical_k0
            logical_offset1 = logical_n1 * N_LANES + logical_k1
            # Binary operations prefetch; scaled operations load on word crossing.
            if not SCALED:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                ).to(tl.uint32).to(tl.uint64)
            # Decode symbol 0, reconstruct its BF16 value, then apply the op.
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
            result = _apply_pointwise(
                left, right, alpha_value, beta_value, OP, SCALED, ALPHA_IS_ONE, BETA_IS_ONE,
            )
            _store_result(
                result, output, auxiliary, storage_offset,
                logical_offset0, true_mask, true_mask, OUTPUT_POLICY,
            )

            # Decode symbol 1 at the next bit offset and process it the same way.
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
            result1 = _apply_pointwise(
                left1, right1, alpha_value, beta_value, OP, SCALED, ALPHA_IS_ONE, BETA_IS_ONE,
            )
            _store_result(
                result1, output, auxiliary,
                storage_offset + N_LANES, logical_offset1,
                true_mask, true_mask, OUTPUT_POLICY,
            )

            # Advance the 64-bit window and storage offset after processing two symbols.
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
            if SCALED:
                word2_prefetch = tl.load(
                    encoded + tl.minimum(word + 2, FIXED_WORDS - 1) * n_streams
                    + lane_index,
                    mask=crosses_word, other=0,
                ).to(tl.uint32).to(tl.uint64)
            next_window = (window >> 32) | (word2_prefetch << 32)
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
            result = _apply_pointwise(
                left, right, alpha_value, beta_value, OP, SCALED, ALPHA_IS_ONE, BETA_IS_ONE,
            )
            _store_result(
                result, output, auxiliary, offset, logical_offset,
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
            result1 = _apply_pointwise(
                left1, right1, alpha_value, beta_value, OP, SCALED, ALPHA_IS_ONE, BETA_IS_ONE,
            )
            _store_result(
                result1, output, auxiliary, offset1, logical_offset1,
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
    key=[
        "n_elements", "N_LANES", "N_STEPS", "BLOCK", "OUTPUT_POLICY",
        "OP", "SCALED", "ALPHA_IS_ONE", "BETA_IS_ONE",
    ],
)
@triton.jit
def pointwise_compressed_dense_fallback_kernel(
    lanes, starts, tail, descriptor, fallback_count, blocks, fallback_base,
    sign_mantissa, other, output, auxiliary, n_elements, alpha, beta,
    OP: tl.constexpr, SCALED: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr, BETA_IS_ONE: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr, ROW_TILE: tl.constexpr,
):
    """Recompute one storage block of overflow results."""
    block_id = tl.program_id(0)
    n_blocks = tl.num_programs(0)
    if BUFFERED:
        total = tl.load(fallback_count).to(tl.int32)
        base = tl.load(descriptor).to(tl.int32)
        lane_off = base
        start_off = base + total
        tail_base = base + 2 * total
        tail_ptr = tail
    else:
        lane_off = 0
        start_off = 0
        tail_base = fallback_base
        tail_ptr = tail
    # Match the main kernel's scalar specialization after the empty-work check.
    alpha_value, beta_value = _load_scalars(alpha, beta, ALPHA_IS_ONE, BETA_IS_ONE)
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

    # Process only rows at or beyond each stream's overflow start.
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    row_offsets = tl.arange(0, ROW_TILE)
    for row_first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = row_first + row_offsets
        offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
            block_id, row[:, None], lane[None, :], n_elements,
            LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS,
        )
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & storage_valid
        )
        logical_active = active & logical_valid

        # Reconstruct BF16 operands and apply the same arithmetic as the main kernel.
        tail_offset = fallback_offset[None, :] + row[:, None] - start[None, :]
        exponent = tl.load(
            tail_ptr + tail_base + tail_offset, mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + offset, mask=active, other=0, cache_modifier='.cg')
        left = pack_bf16(exponent, sm).to(tl.int16).to(tl.bfloat16, bitcast=True)
        right = tl.load(other + logical_offset, mask=logical_active, other=0.0, cache_modifier='.cg')
        result = _apply_pointwise(
            left, right, alpha_value, beta_value, OP, SCALED, ALPHA_IS_ONE, BETA_IS_ONE,
        )
        _store_result(
            result, output, auxiliary, offset, logical_offset,
            logical_active, active, OUTPUT_POLICY,
        )


@triton.jit
def add_op(left, right):
    """Add operands inside a specialized pointwise kernel."""
    return left + right


@triton.jit
def multiply_op(left, right):
    """Multiply operands inside a specialized pointwise kernel."""
    return left * right


@triton.jit
def _scaled_sum(left, right, alpha, beta, ALPHA_IS_ONE: tl.constexpr, BETA_IS_ONE: tl.constexpr):
    """ alpha * compA + beta * B
        Elementwise. Preserve FP32 scaling and fused multiply-add ordering.
    """
    if ALPHA_IS_ONE:
        if not BETA_IS_ONE:
            return tl.math.fma(right.to(tl.float32), beta, left.to(tl.float32))
        return left.to(tl.float32) + right.to(tl.float32)
    if not BETA_IS_ONE:
        right = right.to(tl.float32) * beta
    return tl.math.fma(left, alpha, right)
