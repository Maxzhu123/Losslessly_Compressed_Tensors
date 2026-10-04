"""Center estimation and shifted Huffman tables shared across backends."""

import triton
from triton import language as tl

from ...codec.autotune import ESTIMATE_CENTER_AUTOTUNE_CONFIGS


@triton.jit
def _estimate_center_impl(
    source_bits, size,
    SAMPLE_SIZE: tl.constexpr, PRECOMPUTED: tl.constexpr,
    IGNORE_ZERO: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.int32)
    n = tl.zeros((BLOCK,), tl.int32)
    for i in range(0, SAMPLE_SIZE, BLOCK):
        idx = i + offsets
        mask = idx < SAMPLE_SIZE
        # One deterministic jittered sample per region avoids repeatedly
        # visiting the same columns when a fixed stride aligns with a matrix.
        # Integer hashing needs no RNG state, sample buffer, or extra launch.
        hashed = idx.to(tl.uint32) * 0x9E3779B9
        start = idx.to(tl.int64) * size // SAMPLE_SIZE
        end = (idx.to(tl.int64) + 1) * size // SAMPLE_SIZE
        jitter = (hashed.to(tl.uint64) * (end - start).to(tl.uint64)) >> 32
        pos = start + jitter.to(tl.int64)
        value = tl.load(source_bits + pos, mask=mask, other=0).to(tl.int32)
        if PRECOMPUTED:
            exp = value - 127
        else:
            exp = ((value >> 7) & 0xFF) - 127
        mask = mask & (exp >= -120)  # Exclude tiny values from centering only.
        if IGNORE_ZERO:
            nonzero = mask & (exp != -127)
            total += tl.where(nonzero, exp, 0)
            n += tl.where(nonzero, 1, 0)
        else:
            total += tl.where(mask, exp, 0)
            n += tl.where(mask, 1, 0)
    s = tl.sum(total, axis=0)
    n = tl.sum(n, axis=0)
    # If requested, ignore exact zeros when estimating the center.  This keeps
    # the center aligned with the nonzero component of zero-inflated
    # distributions.
    safe_n = tl.maximum(n, 1)
    center = tl.where(
        s >= 0,
        (s + safe_n // 2) // safe_n,
        -((-s + safe_n // 2) // safe_n),
    )
    center = tl.where(n > 0, center, 0)
    center = tl.minimum(tl.maximum(center, -128), 127)
    return center


@triton.autotune(
    configs=ESTIMATE_CENTER_AUTOTUNE_CONFIGS,
    key=["SAMPLE_SIZE"],
)
@triton.jit
def _estimate_center_kernel(
    source_bits, center_out, size,
    SAMPLE_SIZE: tl.constexpr, PRECOMPUTED: tl.constexpr,
    IGNORE_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    center = _estimate_center_impl(
        source_bits, size, SAMPLE_SIZE, PRECOMPUTED, IGNORE_ZERO, BLOCK,
    )
    tl.store(center_out, center)


@triton.jit
def _shift_encoding_table_impl(
    base_encode, center_value, shifted_encode, BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    zero_delta = (-127 - center_value) & 255
    raw_byte = idx
    exp = raw_byte - 127
    delta = (exp - center_value) & 255
    table_index = tl.where(
        exp == -127,
        0,
        delta + (delta < zero_delta).to(tl.int32),
    )
    packed = tl.load(base_encode + table_index).to(tl.uint32)
    tl.store(shifted_encode + raw_byte, packed)


@triton.jit
def _shift_encoding_table_kernel(
    base_encode, center, shifted_encode, BLOCK: tl.constexpr,
):
    """Create a raw-exponent-byte-indexed encode table for one center."""
    center_value = tl.load(center).to(tl.int32)
    _shift_encoding_table_impl(base_encode, center_value, shifted_encode, BLOCK)


@triton.jit
def _estimate_and_shift_encoding_table_kernel(
    source_bits, center_out, size, base_encode, shifted_encode,
    SAMPLE_SIZE: tl.constexpr, PRECOMPUTED: tl.constexpr,
    IGNORE_ZERO: tl.constexpr, BLOCK: tl.constexpr,
):
    """Reuse the sampled center directly when building the encoding table."""
    center = _estimate_center_impl(
        source_bits, size, SAMPLE_SIZE, PRECOMPUTED, IGNORE_ZERO, BLOCK,
    )
    tl.store(center_out, center)
    _shift_encoding_table_impl(base_encode, center, shifted_encode, BLOCK=256)


@triton.jit
def _shift_decoding_table_kernel(
    base_decode, center, shifted_decode,
    BLOCK: tl.constexpr,
):
    """Create a decode table that stores unbiased exponents directly."""
    idx = tl.arange(0, BLOCK)
    center_value = tl.load(center).to(tl.int32)
    zero_delta = (-127 - center_value) & 255
    packed = tl.load(base_decode + idx).to(tl.int32)
    length = packed & 255
    symbol = (packed >> 8) & 255
    is_zero = symbol == 0
    delta = tl.where(symbol == 0, 0, symbol - (symbol <= zero_delta).to(tl.int32))
    delta = tl.where(delta >= 128, delta - 256, delta)
    exponent = tl.where(is_zero, -127, delta + center_value)
    shifted = tl.where(length == 0, 0, length | (exponent << 8))
    tl.store(shifted_decode + idx, shifted)
