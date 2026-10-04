"""Generic Triton templates for pointwise compressed-tensor operations."""

import triton
from triton import language as tl

from ...codec.autotune import (
    DECODE_AUTOTUNE_CONFIGS,
    POINTWISE_FALLBACK_AUTOTUNE_CONFIGS,
)
from ..primitives import decode_symbol, pack_bf16


DENSE_OUTPUT = tl.constexpr(0)
COMPRESSED_OUTPUT = tl.constexpr(1)


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
    """Map one storage-stream position to its flattened logical offset (1D).

    The row swizzle ``(n & 255)`` is hoisted: ``(block*N_STEPS+step) & 255``
    equals ``(block_shift + step) & 255`` with ``block_shift`` loop-invariant,
    so callers in hot loops should prefer the hoisted form below.
    """
    storage_offset = block * BLOCK + step * N_LANES + lane
    storage_valid = storage_offset < n_elements
    logical_n = block * N_STEPS + step
    logical_k = (lane + (((block * N_STEPS) + step) & 255)) & 255
    logical_offset = logical_n * N_LANES + logical_k
    logical_valid = logical_offset < LOGICAL_NUMEL
    return storage_offset, logical_offset, storage_valid, logical_valid


@triton.autotune(
    configs=DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_compressed_dense_kernel(
    encoded, sign_mantissa, other, output, auxiliary, decode_table,
    n_elements, n_streams, center,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
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

    # Fast path: fully-contained blocks skip all per-element validity checks.
    # Hoisted swizzle: shift depends only on step + loop-invariant block_shift.
    block_shift = (block * N_STEPS) & 255
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        storage_offset = block * BLOCK + lanes
        true_mask = tl.full((N_LANES,), True, tl.int1)
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=2):
            logical_n0 = block * N_STEPS + step
            logical_n1 = logical_n0 + 1
            logical_k0 = (lanes + ((block_shift + step) & 255)) & 255
            logical_k1 = (lanes + ((block_shift + step + 1) & 255)) & 255
            logical_offset0 = logical_n0 * N_LANES + logical_k0
            logical_offset1 = logical_n1 * N_LANES + logical_k1
            # Prefetch the next 32-bit Huffman word while decoding the pair.
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
            _store_result(
                OP(left, right), output, auxiliary, storage_offset,
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
            _store_result(
                OP(left1, right1), output, auxiliary,
                storage_offset + N_LANES, logical_offset1,
                true_mask, true_mask, OUTPUT_POLICY,
            )

            # Advance the 64-bit window and storage offset after processing two symbols.
            next_shift = shift1 + length1
            crosses_word = next_shift >= 32
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
            _store_result(
                OP(left, right), output, auxiliary, offset, logical_offset,
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
                OP(left1, right1), output, auxiliary, offset1, logical_offset1,
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


@triton.jit
def _pointwise_fallback_impl(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, count,
    sign_mantissa, other, output, auxiliary, n_elements, tile_id,
    alpha_value, beta_value,
    OP: tl.constexpr, SCALED: tl.constexpr,
    SCALE_OTHER: tl.constexpr, ALPHA_IS_ONE: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, ROW_TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr,
):
    """Process absolute overflow rows with coalesced side-byte and operand loads."""
    tile = tile_id * TILE + tl.arange(0, TILE)
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
    first_step = tl.min(tl.where(valid, start, N_STEPS), axis=0)
    row_offsets = tl.arange(0, ROW_TILE)
    for first in tl.range(first_step, N_STEPS, ROW_TILE):
        row = first + row_offsets
        offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
            block[None, :], row[:, None], lane[None, :], n_elements,
            LOGICAL_NUMEL, BLOCK, N_LANES, N_STEPS,
        )
        active = (
            valid[None, :] & (row[:, None] >= start[None, :])
            & (row[:, None] < N_STEPS) & storage_valid
        )
        logical_active = active & logical_valid
        tail_offset = fallback_offset[None, :] + row[:, None] - start[None, :]
        exponent = tl.load(
            fallback_buffer + fallback_base + tail_offset,
            mask=active, other=0,
        ).to(tl.int32)
        sm = tl.load(sign_mantissa + offset, mask=active, other=0, cache_modifier='.cg')
        left = pack_bf16(exponent, sm).to(tl.int16).to(tl.bfloat16, bitcast=True)
        right = tl.load(other + logical_offset, mask=logical_active, other=0.0, cache_modifier='.cg')
        if SCALED:
            result = OP(left, right, alpha_value, beta_value, SCALE_OTHER, ALPHA_IS_ONE)
        else:
            result = OP(left, right)
        _store_result(
            result, output, auxiliary, offset, logical_offset,
            logical_active, active, OUTPUT_POLICY,
        )


@triton.autotune(
    configs=POINTWISE_FALLBACK_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "BLOCK", "OUTPUT_POLICY", "OP"],
)
@triton.jit
def pointwise_compressed_dense_fallback_kernel(
    bad_streams, bad_starts, fallback_offsets,
    fallback_buffer, fallback_base, metadata, descriptor, fallback_count,
    sign_mantissa, other, output, auxiliary, n_elements,
    OP: tl.constexpr, OUTPUT_POLICY: tl.constexpr, BUFFERED: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
    N_LANES: tl.constexpr, N_STEPS: tl.constexpr, ROW_TILE: tl.constexpr,
):
    """Recompute overflow results using a bounded grid of persistent programs."""
    count = tl.load(fallback_count).to(tl.int32)
    if tl.program_id(0) * TILE >= count:
        return
    for tile_id in tl.range(tl.program_id(0), tl.cdiv(count, TILE), tl.num_programs(0)):
        _pointwise_fallback_impl(
            bad_streams, bad_starts, fallback_offsets, fallback_buffer,
            fallback_base, metadata, descriptor, count,
            sign_mantissa, other, output, auxiliary, n_elements, tile_id,
            1.0, 1.0,
            OP=OP, SCALED=False, SCALE_OTHER=False, ALPHA_IS_ONE=False,
            OUTPUT_POLICY=OUTPUT_POLICY, BUFFERED=BUFFERED,
            LOGICAL_NUMEL=LOGICAL_NUMEL, TILE=TILE, ROW_TILE=ROW_TILE,
            BLOCK=BLOCK, N_LANES=N_LANES, N_STEPS=N_STEPS,
        )


@triton.jit
def add_op(left, right):
    """Add operands inside a specialized pointwise kernel."""
    return left + right


@triton.jit
def multiply_op(left, right):
    """Multiply operands inside a specialized pointwise kernel."""
    return left * right
