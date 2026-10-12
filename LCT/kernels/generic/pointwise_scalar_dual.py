"""Fused pointwise kernels for two compressed BF16 operands.

Computes ``output = alpha * a + b`` without materialising either operand as a
dense tensor. Per-stream maps let the main kernel read either fixed payload
or overflow storage, supporting private and buffered allocations.
"""
import triton
from triton import language as tl

from ..autotune import DUAL_DECODE_AUTOTUNE_CONFIGS
from ..common.tables import _shift_decoding_table_impl
from ..primitives import decode_symbol, pack_bf16
from .layout import swizzle_lane
from .pointwise import _pointwise_location, _store_result


@triton.jit
def _initialize_dual_maps_kernel(
    a_stream_starts, a_stream_offsets, b_stream_starts, b_stream_offsets,
    a_fallback_count, b_fallback_count, n_streams,
    N_STEPS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Initialize both operands' maps in one launch when the decoder needs them."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < n_streams
    tl.store(a_stream_starts + idx, N_STEPS, mask=valid)
    tl.store(b_stream_starts + idx, N_STEPS, mask=valid)
    tl.store(a_stream_offsets + idx, 0, mask=valid)
    tl.store(b_stream_offsets + idx, 0, mask=valid)


@triton.jit
def _prepare_dual_tables_and_maps_kernel(
    a_decode_table, a_center, a_shifted_decode,
    b_decode_table, b_center, b_shifted_decode,
    a_lanes, a_starts, a_descriptor, a_blocks, a_fallback_count,
    a_stream_starts, a_stream_offsets,
    b_lanes, b_starts, b_descriptor, b_blocks, b_fallback_count,
    b_stream_starts, b_stream_offsets,
    N_SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    N_STEPS: tl.constexpr,
    N_LANES: tl.constexpr,
    BUFFERED: tl.constexpr,
):
    """Shift both decode tables and expand both compact fallback lists.

    One program handles one storage block of each operand, matching the
    per-block layout written by the encoder's compaction pass.
    """
    pid = tl.program_id(0)
    if pid < N_SHIFT:
        if pid == 0:
            base_decode = a_decode_table
            center = a_center
            shifted_decode = a_shifted_decode
        else:
            base_decode = b_decode_table
            center = b_center
            shifted_decode = b_shifted_decode
        center_value = tl.load(center).to(tl.int32)
        _shift_decoding_table_impl(base_decode, center_value, shifted_decode, BLOCK)
        return

    n_blocks = tl.num_programs(0) - N_SHIFT
    block = pid - N_SHIFT
    if BUFFERED:
        a_total = tl.load(a_fallback_count).to(tl.int32)
        a_base = tl.load(a_descriptor).to(tl.int32)
        a_lane_off = a_base
        a_start_off = a_base + a_total
        b_total = tl.load(b_fallback_count).to(tl.int32)
        b_base = tl.load(b_descriptor).to(tl.int32)
        b_lane_off = b_base
        b_start_off = b_base + b_total
    else:
        a_lane_off = 0
        a_start_off = 0
        b_lane_off = 0
        b_start_off = 0

    k = tl.arange(0, N_LANES)
    a_count = tl.load(a_blocks + block).to(tl.int32)
    if a_count > 0:
        a_first = tl.load(a_blocks + n_blocks + block).to(tl.int32)
        a_byte = tl.load(a_blocks + 2 * n_blocks + block).to(tl.int32)
        a_valid = k < a_count
        a_lane = tl.load(
            a_lanes + a_lane_off + a_first + k, mask=a_valid, other=0,
        ).to(tl.int32)
        a_st = tl.load(
            a_starts + a_start_off + a_first + k, mask=a_valid, other=N_STEPS,
        ).to(tl.int32)
        a_len = tl.where(a_valid, N_STEPS - a_st, 0)
        a_local = tl.cumsum(a_len, axis=0) - a_len
        a_stream = block * N_LANES + a_lane
        tl.store(a_stream_starts + a_stream, a_st, mask=a_valid)
        tl.store(a_stream_offsets + a_stream, a_byte + a_local, mask=a_valid)

    b_count = tl.load(b_blocks + block).to(tl.int32)
    if b_count > 0:
        b_first = tl.load(b_blocks + n_blocks + block).to(tl.int32)
        b_byte = tl.load(b_blocks + 2 * n_blocks + block).to(tl.int32)
        b_valid = k < b_count
        b_lane = tl.load(
            b_lanes + b_lane_off + b_first + k, mask=b_valid, other=0,
        ).to(tl.int32)
        b_st = tl.load(
            b_starts + b_start_off + b_first + k, mask=b_valid, other=N_STEPS,
        ).to(tl.int32)
        b_len = tl.where(b_valid, N_STEPS - b_st, 0)
        b_local = tl.cumsum(b_len, axis=0) - b_len
        b_stream = block * N_LANES + b_lane
        tl.store(b_stream_starts + b_stream, b_st, mask=b_valid)
        tl.store(b_stream_offsets + b_stream, b_byte + b_local, mask=b_valid)


@triton.autotune(
    configs=DUAL_DECODE_AUTOTUNE_CONFIGS,
    key=["n_elements", "N_LANES", "N_STEPS", "FIXED_WORDS", "OUTPUT_POLICY"],
)
@triton.jit
def pointwise_scalar_mul_add_compressed_compressed_mapped_kernel(
    a_encoded, a_sign_mantissa, a_decode_table, a_center,
    a_stream_starts, a_stream_offsets,
    a_descriptor, a_fallback_count,
    a_fallback_buffer, a_fallback_base,
    b_encoded, b_sign_mantissa, b_decode_table, b_center,
    b_stream_starts, b_stream_offsets,
    b_descriptor, b_fallback_count,
    b_fallback_buffer, b_fallback_base,
    output, auxiliary, alpha,
    n_elements, n_streams,
    BUFFERED: tl.constexpr,
    OUTPUT_POLICY: tl.constexpr,
    LOGICAL_NUMEL: tl.constexpr,
    FIRST_MASK: tl.constexpr, RARE_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr, N_LANES: tl.constexpr,
    N_STEPS: tl.constexpr, FIXED_WORDS: tl.constexpr,
):
    """One-pass fused kernel using low-count scan or per-stream maps."""
    block = tl.program_id(0)
    lanes = tl.arange(0, N_LANES)
    lane_index = block * N_LANES + lanes
    stream = lane_index

    # The compact fallback lists are always expanded into per-stream maps by
    # _prepare_dual_tables_and_maps_kernel, so the main loop only reads them.
    a_count = tl.load(a_fallback_count).to(tl.int32)
    b_count = tl.load(b_fallback_count).to(tl.int32)
    a_start = tl.load(a_stream_starts + stream).to(tl.int32)
    a_offset = tl.load(a_stream_offsets + stream).to(tl.int32)
    b_start = tl.load(b_stream_starts + stream).to(tl.int32)
    b_offset = tl.load(b_stream_offsets + stream).to(tl.int32)
    if BUFFERED:
        a_real_base = tl.load(a_descriptor).to(tl.int32) + 2 * a_count
        b_real_base = tl.load(b_descriptor).to(tl.int32) + 2 * b_count
    else:
        a_real_base = a_fallback_base
        b_real_base = b_fallback_base

    a_word = tl.zeros((N_LANES,), tl.int32)
    a_shift = tl.zeros((N_LANES,), tl.int32)
    b_word = tl.zeros((N_LANES,), tl.int32)
    b_shift = tl.zeros((N_LANES,), tl.int32)

    a_word0 = tl.load(a_encoded + a_word * n_streams + stream)
    a_word1 = tl.load(a_encoded + (a_word + 1) * n_streams + stream)
    a_window = a_word0.to(tl.uint32).to(tl.uint64)
    a_window |= a_word1.to(tl.uint32).to(tl.uint64) << 32

    b_word0 = tl.load(b_encoded + b_word * n_streams + stream)
    b_word1 = tl.load(b_encoded + (b_word + 1) * n_streams + stream)
    b_window = b_word0.to(tl.uint32).to(tl.uint64)
    b_window |= b_word1.to(tl.uint32).to(tl.uint64) << 32

    a_center_value = tl.load(a_center).to(tl.int32)
    b_center_value = tl.load(b_center).to(tl.int32)
    alpha_value = tl.load(alpha).to(tl.float32)

    # Fast path: fully-contained blocks skip all per-element validity checks.
    if (block + 1) * BLOCK <= LOGICAL_NUMEL:
        for step in tl.range(0, N_STEPS, 2, loop_unroll_factor=6):
            a_word2_prefetch = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
            ).to(tl.uint32).to(tl.uint64)
            b_word2_prefetch = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
            ).to(tl.uint32).to(tl.uint64)
            a0_is_fb = step >= a_start
            b0_is_fb = step >= b_start

            a0, a0_len = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b0, b0_len = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a0_fb = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step - a_start),
                mask=a0_is_fb, other=0,
            ).to(tl.int32)
            b0_fb = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step - b_start),
                mask=b0_is_fb, other=0,
            ).to(tl.int32)
            a0 = tl.where(a0_is_fb, a0_fb, a0)
            b0 = tl.where(b0_is_fb, b0_fb, b0)
            a0_len = tl.where(a0_is_fb, 0, a0_len)
            b0_len = tl.where(b0_is_fb, 0, b0_len)

            offset0 = block * BLOCK + step * N_LANES + lanes
            logical_n0 = block * N_STEPS + step
            # Use the same storage-to-logical mapping as the codec.
            logical_k0 = swizzle_lane(block * N_STEPS, step, lanes)
            logical_offset0 = logical_n0 * N_LANES + logical_k0
            sm_a0 = tl.load(a_sign_mantissa + offset0, cache_modifier='.cg')
            sm_b0 = tl.load(b_sign_mantissa + offset0, cache_modifier='.cg')
            a_left0 = pack_bf16(a0, sm_a0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left0 = pack_bf16(b0, sm_b0).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left0, alpha_value, b_left0), output, auxiliary,
                offset0, logical_offset0, True, True, OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a0_len
            b_shift1 = b_shift + b0_len
            a1_is_fb = step + 1 >= a_start
            b1_is_fb = step + 1 >= b_start

            a1, a1_len = decode_symbol(
                a_window >> a_shift1, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b1, b1_len = decode_symbol(
                b_window >> b_shift1, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a1_fb = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step + 1 - a_start),
                mask=a1_is_fb, other=0,
            ).to(tl.int32)
            b1_fb = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step + 1 - b_start),
                mask=b1_is_fb, other=0,
            ).to(tl.int32)
            a1 = tl.where(a1_is_fb, a1_fb, a1)
            b1 = tl.where(b1_is_fb, b1_fb, b1)
            a1_len = tl.where(a1_is_fb, 0, a1_len)
            b1_len = tl.where(b1_is_fb, 0, b1_len)

            offset1 = offset0 + N_LANES
            logical_n1 = logical_n0 + 1
            logical_k1 = swizzle_lane(block * N_STEPS, step + 1, lanes)
            logical_offset1 = logical_n1 * N_LANES + logical_k1
            sm_a1 = tl.load(a_sign_mantissa + offset1, cache_modifier='.cg')
            sm_b1 = tl.load(b_sign_mantissa + offset1, cache_modifier='.cg')
            a_left1 = pack_bf16(a1, sm_a1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left1 = pack_bf16(b1, sm_b1).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left1, alpha_value, b_left1), output, auxiliary,
                offset1, logical_offset1, True, True, OUTPUT_POLICY,
            )

            a_next_shift = a_shift1 + a1_len
            a_crosses = a_next_shift >= 32
            a_next_window = (a_window >> 32) | (a_word2_prefetch << 32)
            a_window = tl.where(a_crosses, a_next_window, a_window)
            a_word += a_crosses
            a_shift = tl.where(a_crosses, a_next_shift - 32, a_next_shift)

            b_next_shift = b_shift1 + b1_len
            b_crosses = b_next_shift >= 32
            b_next_window = (b_window >> 32) | (b_word2_prefetch << 32)
            b_window = tl.where(b_crosses, b_next_window, b_window)
            b_word += b_crosses
            b_shift = tl.where(b_crosses, b_next_shift - 32, b_next_shift)

    else:
        for step in tl.range(0, N_STEPS):
            a_is_fb = step >= a_start
            b_is_fb = step >= b_start

            a_value, a_length = decode_symbol(
                a_window >> a_shift, a_decode_table, a_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            b_value, b_length = decode_symbol(
                b_window >> b_shift, b_decode_table, b_center_value,
                FIRST_MASK, RARE_LENGTH,
            )
            a_fb_value = tl.load(
                a_fallback_buffer + a_real_base + a_offset + (step - a_start),
                mask=a_is_fb, other=0,
            ).to(tl.int32)
            b_fb_value = tl.load(
                b_fallback_buffer + b_real_base + b_offset + (step - b_start),
                mask=b_is_fb, other=0,
            ).to(tl.int32)
            a_value = tl.where(a_is_fb, a_fb_value, a_value)
            b_value = tl.where(b_is_fb, b_fb_value, b_value)
            a_length = tl.where(a_is_fb, 0, a_length)
            b_length = tl.where(b_is_fb, 0, b_length)

            offset, logical_offset, storage_valid, logical_valid = _pointwise_location(
                block, step, lanes, n_elements, LOGICAL_NUMEL,
                BLOCK, N_LANES, N_STEPS,
            )
            valid = storage_valid
            logical_active = valid & logical_valid
            sm_a = tl.load(a_sign_mantissa + offset, mask=valid, other=0, cache_modifier='.cg')
            sm_b = tl.load(b_sign_mantissa + offset, mask=valid, other=0, cache_modifier='.cg')
            a_left = pack_bf16(a_value, sm_a).to(tl.int16).to(tl.bfloat16, bitcast=True)
            b_left = pack_bf16(b_value, sm_b).to(tl.int16).to(tl.bfloat16, bitcast=True)
            _store_result(
                tl.math.fma(a_left, alpha_value, b_left), output, auxiliary,
                offset, logical_offset, logical_active, valid, OUTPUT_POLICY,
            )

            a_shift1 = a_shift + a_length
            a_crosses = a_shift1 >= 32
            a_word2 = tl.load(
                a_encoded + tl.minimum(a_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
                mask=a_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            a_next_window = (a_window >> 32) | (a_word2 << 32)
            a_window = tl.where(a_crosses, a_next_window, a_window)
            a_word += a_crosses
            a_shift = tl.where(a_crosses, a_shift1 - 32, a_shift1)

            b_shift1 = b_shift + b_length
            b_crosses = b_shift1 >= 32
            b_word2 = tl.load(
                b_encoded + tl.minimum(b_word + 2, FIXED_WORDS - 1) * n_streams
                + stream,
                mask=b_crosses, other=0,
            ).to(tl.uint32).to(tl.uint64)
            b_next_window = (b_window >> 32) | (b_word2 << 32)
            b_window = tl.where(b_crosses, b_next_window, b_window)
            b_word += b_crosses
            b_shift = tl.where(b_crosses, b_shift1 - 32, b_shift1)
