"""TileLang Huffman decoding with native autotuning."""

import torch
import tilelang
import tilelang.language as T

from .autotune import DECODE_AUTOTUNE_CONFIGS
from .tilelang_utils import silence_autotune


_PASS_CONFIGS = {"tl.disable_safe_memory_legalize": True, "tl.disable_warp_specialized": True}


silence_autotune()


# Integer intrinsics preserve every BF16 representation, including NaN payloads.
_DEVICE_SOURCE = r"""
__device__ __forceinline__ unsigned lct_decode_prefix(unsigned lo, unsigned hi, int shift) {
    return shift < 32 ? __funnelshift_r(lo, hi, shift) : hi >> (shift - 32);
}
__device__ __forceinline__ void lct_decode_store4(short* output, const short* source) {
    uint2 value = *reinterpret_cast<const uint2*>(source);
    __stcs(reinterpret_cast<uint2*>(output), value);
}
"""


@tilelang.autotune(configs=list(DECODE_AUTOTUNE_CONFIGS), warmup=3, rep=5)
@tilelang.jit(pass_configs=_PASS_CONFIGS, verbose=False)
def decode_kernel(encoded: torch.Tensor, side: torch.Tensor, base_table: torch.Tensor,
                  center: torch.Tensor, output: torch.Tensor,
                  block_symbols: int, lanes: int, steps: int, fixed_words: int, rare_length: int,
                  threads: int = 256, row_tile: int = 8, unroll: int = 4) -> None:
    """Autotune and launch the Huffman decoder in TileLang's eager style.

    The three runtime sizes come from the tensor shapes (``encoded``, ``side``
    and ``output``), so the tuner automatically keys on tensor shape/stride and
    selects a size-specific configuration. The caller owns ``output`` because
    the logical element count is not derivable from any single input tensor.
    """
    if lanes != 256 or threads not in (64, 128, 256) or fixed_words < 2:
        raise ValueError("TileLang decode requires 256 lanes and 64, 128, or 256 threads")
    if row_tile < 4 or row_tile % 2 or steps % row_tile or unroll < 1:
        raise ValueError("Decoder row tiles must divide the stream length and contain whole pairs")
    streams_per_thread = lanes // threads
    logical_numel, storage_numel, payload_numel = T.const("logical_numel, storage_numel, payload_numel")
    encoded: T.Tensor((payload_numel,), "int32")
    side: T.Tensor((storage_numel,), "uint8")
    base_table: T.Tensor((1024,), "int32")
    center: T.Tensor((1,), "int32")
    output: T.Tensor((logical_numel,), "int16")
    blocks = T.ceildiv(storage_numel, block_symbols)
    streams = blocks * lanes

    with T.Kernel(blocks, threads=threads) as block:
        T.import_source(_DEVICE_SOURCE)
        thread = T.get_thread_binding()
        lookup = T.alloc_shared((1024,), "int32")
        decoded = T.alloc_shared((row_tile, 256), "int16")
        word = T.alloc_local((streams_per_thread,), "int32")
        shift = T.alloc_local((streams_per_thread,), "int32")
        low = T.alloc_local((streams_per_thread,), "int32")
        high = T.alloc_local((streams_per_thread,), "int32")
        next_word = T.alloc_local((streams_per_thread,), "int32")
        prefix = T.alloc_local((streams_per_thread,), "uint32")
        sm = T.alloc_local((streams_per_thread,), "int32")
        center_value = center[0]
        zero_delta = (-127 - center_value) & 255

        # Translate the shared raw-symbol table once per block. Store biased
        # exponent bytes to avoid signed remapping in the common decode path.
        for i in T.unroll(1024 // threads):
            index = i * threads + thread
            packed = base_table[index]
            symbol = (packed >> 8) & 255
            raw = ((symbol - T.Cast("int32", symbol <= zero_delta) + center_value + 127) & 255)
            raw_exponent = raw & (-T.Cast("int32", symbol != 0))
            lookup[index] = (packed & 255) | (raw_exponent << 8)
        T.sync_threads()
        for owned in T.unroll(streams_per_thread):
            stream = block * lanes + thread + owned * threads
            word[owned] = 0
            shift[owned] = 0
            low[owned] = encoded[stream]
            high[owned] = encoded[streams + stream]
            next_word[owned] = encoded[T.min(2, fixed_words - 1) * streams + stream]

        # Specialize complete blocks separately from the final masked block.
        for path in T.unroll(2):
            if (((block + 1) * block_symbols <= logical_numel) == (path == 0)):
                for tile in T.serial(steps // row_tile):
                    for pair in T.unroll(row_tile // 2, unroll_factor=unroll if unroll < row_tile // 2 else None):
                        for member in T.unroll(2):
                            for owned in T.unroll(streams_per_thread):
                                lane = thread + owned * threads
                                if member == 0:
                                    prefix[owned] = T.call_extern("uint32", "__funnelshift_r", low[owned], high[owned], shift[owned])
                                else:
                                    prefix[owned] = T.call_extern("uint32", "lct_decode_prefix", low[owned], high[owned], shift[owned])
                                bits = prefix[owned]
                                first = lookup[T.Cast("int32", bits & 1023)]
                                length = first & 255
                                escaped_symbol = T.Cast("int32", (bits >> rare_length) & 255)
                                escaped_raw = ((escaped_symbol - T.Cast("int32", escaped_symbol <= zero_delta) + center_value + 127) & 255)
                                escaped_byte = escaped_raw & (-T.Cast("int32", escaped_symbol != 0))
                                raw_exponent = T.if_then_else(length == 0, escaped_byte, first >> 8)
                                consumed = T.if_then_else(length == 0, rare_length + 8, length)
                                step = tile * row_tile + pair * 2 + member
                                storage = block * block_symbols + step * lanes + lane
                                if path == 0:
                                    sm[owned] = T.Cast("int32", T.call_extern("uint8", "__ldcg", T.address_of(side[storage])))
                                else:
                                    sm[owned] = T.if_then_else(storage < storage_numel, T.Cast("int32", side[storage]), 0)
                                col = (lane + block * steps + step) & 255
                                decoded[pair * 2 + member, col] = T.Cast("int16", (raw_exponent << 7) | (sm[owned] & 127) | ((sm[owned] & 128) << 8))
                                shift[owned] = shift[owned] + consumed
                        for owned in T.unroll(streams_per_thread):
                            stream = block * lanes + thread + owned * threads
                            if shift[owned] >= 32:
                                low[owned] = high[owned]
                                high[owned] = next_word[owned]
                                word[owned] = word[owned] + 1
                                shift[owned] = shift[owned] - 32
                                # Retain the prefetched word until it is consumed;
                                # its replacement is needed at the next crossing.
                                next_word[owned] = encoded[T.min(word[owned] + 2, fixed_words - 1) * streams + stream]
                    T.sync_threads()
                    # Eight-byte aligned stores remove the global row swizzle.
                    # The rotated writes above place results in logical order.
                    for work in T.serial(row_tile * lanes // (4 * threads)):
                        flat = work * threads + thread
                        row = flat // 64
                        col_start = (flat % 64) * 4
                        out = (block * steps + tile * row_tile + row) * lanes + col_start
                        if path == 0 or out + 3 < logical_numel:
                            T.evaluate(T.call_extern("handle", "lct_decode_store4", T.address_of(output[out]), T.address_of(decoded[row, col_start])))
                        else:
                            for item in T.unroll(4):
                                if out + item < logical_numel:
                                    output[out + item] = decoded[row, col_start + item]
                    T.sync_threads()
