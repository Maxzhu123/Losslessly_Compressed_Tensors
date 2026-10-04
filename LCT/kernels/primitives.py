"""Small Triton helpers shared by codec and fused-operation kernels."""

import triton
from triton import language as tl


@triton.jit
def pack_bf16(value, sign_mantissa):
    """Reassemble BF16 bits from an unbiased exponent and raw side byte."""
    return (
        (((value.to(tl.int32) + 127) & 255) << 7)
        | (sign_mantissa.to(tl.int32) & 0x7F)
        | ((sign_mantissa.to(tl.int32) & 0x80) << 8)
    )


@triton.jit
def decode_symbol(
    current, decode_table, center, FIRST_MASK: tl.constexpr,
    RARE_LENGTH: tl.constexpr,
):
    """Decode one Huffman symbol and return its exponent and consumed bit count."""
    first = tl.load(decode_table + (current & FIRST_MASK).to(tl.int32), cache_modifier='.ca')
    first_length = first & 255
    continuation = first_length == 0
    length = tl.where(continuation, RARE_LENGTH + 8, first_length)
    # Direct symbols already store the unbiased exponent in the shifted table.
    # Only escaped symbols carry a table index and need the inverse remap.
    symbol = ((current >> RARE_LENGTH) & 255).to(tl.int32)
    zero_delta = (-127 - center) & 255
    is_zero = symbol == 0
    delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
    delta = tl.where(delta >= 128, delta - 256, delta)
    escaped_value = tl.where(is_zero, -127, delta + center)
    value = tl.where(continuation, escaped_value, first >> 8)
    return value, length
