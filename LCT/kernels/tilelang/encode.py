"""TileLang fixed-payload Huffman encoding kernel."""

import torch
import tilelang
import tilelang.language as T

from .autotune import encode_autotune_configs
from .tilelang_utils import silence_autotune


silence_autotune()


# Tile loading, consumption, and shared-buffer reuse have explicit barriers.
_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.disable_warp_specialized": True,
    "tl.disable_thread_storage_sync": True,
}
_DEVICE_SOURCE = r"""
__device__ __forceinline__ void lct_encode_load4(short* dest, const short* source) {
    *reinterpret_cast<uint2*>(dest) = __ldcg(reinterpret_cast<const uint2*>(source));
}
__device__ __forceinline__ void lct_encode_load4_bytes(unsigned char* dest, const unsigned char* source) {
    *reinterpret_cast<unsigned*>(dest) = __ldcg(reinterpret_cast<const unsigned*>(source));
}
__device__ __forceinline__ void lct_encode_load8(short* dest, const short* source) {
    *reinterpret_cast<uint4*>(dest) = __ldcg(reinterpret_cast<const uint4*>(source));
}
__device__ __forceinline__ void lct_encode_load8_bytes(unsigned char* dest, const unsigned char* source) {
    *reinterpret_cast<uint2*>(dest) = __ldcg(reinterpret_cast<const uint2*>(source));
}
__device__ __forceinline__ unsigned lct_encode_side4(uint2 value) {
    // The permutation keeps bytes 0 and 2; complementary masks fold into LOP3.
    unsigned lo = (value.x & 0x7f7f7f7fu) | ((value.x >> 8) & 0x80808080u);
    unsigned hi = (value.y & 0x7f7f7f7fu) | ((value.y >> 8) & 0x80808080u);
    return __byte_perm(lo, hi, 0x6420);
}
__device__ __forceinline__ void lct_encode_stage4(
    unsigned char* exponents, unsigned char* side, const short* source, int col, int row_shift) {
    uint2 value = __ldcg(reinterpret_cast<const uint2*>(source));
    unsigned exp = __byte_perm(value.x >> 7, value.y >> 7, 0x6420);
    *reinterpret_cast<unsigned*>(exponents) = exp;
    unsigned current = lct_encode_side4(value);
    unsigned previous = __shfl_up_sync(0xffffffffu, current, 1);
    if ((threadIdx.x & 31) == 0) {
        previous = lct_encode_side4(__ldcg(reinterpret_cast<const uint2*>(source + (col == 0 ? 252 : -4))));
    }
    int rotation = (-row_shift) & 3;
    unsigned packed = rotation == 0 ? current : (current << (rotation * 8)) | (previous >> (32 - rotation * 8));
    int destination = ((col - row_shift) & 255) & ~3;
    __stcs(reinterpret_cast<unsigned*>(side + destination), packed);
}
__device__ __forceinline__ void lct_encode_stage8(
    unsigned char* exponents, unsigned char* side, const short* source, int col, int row_shift) {
    uint4 value = __ldcg(reinterpret_cast<const uint4*>(source));
    uint2 exp;
    exp.x = __byte_perm(value.x >> 7, value.y >> 7, 0x6420);
    exp.y = __byte_perm(value.z >> 7, value.w >> 7, 0x6420);
    *reinterpret_cast<uint2*>(exponents) = exp;
    unsigned lo = lct_encode_side4(make_uint2(value.x, value.y));
    unsigned hi = lct_encode_side4(make_uint2(value.z, value.w));
    int rotation = (-row_shift) & 7;
    int previous_lane = (threadIdx.x - 1) & 31;
    unsigned previous_lo = __shfl_sync(0xffffffffu, lo, previous_lane);
    unsigned previous_hi = __shfl_sync(0xffffffffu, hi, previous_lane);
    unsigned a = rotation < 4 ? previous_hi : previous_lo;
    unsigned b = rotation < 4 ? lo : previous_hi;
    unsigned c = rotation < 4 ? hi : lo;
    uint2 packed = make_uint2(__funnelshift_l(a, b, (rotation & 3) * 8),
                             __funnelshift_l(b, c, (rotation & 3) * 8));
    int destination = ((col - row_shift) & 255) & ~7;
    __stcs(reinterpret_cast<uint2*>(side + destination), packed);
}
"""


@tilelang.autotune(configs=encode_autotune_configs, warmup=3, rep=15)
@tilelang.jit(pass_configs=_PASS_CONFIGS, verbose=False)
def encode_kernel(source: torch.Tensor, side: torch.Tensor, encoded: torch.Tensor,
                  encode_table: torch.Tensor,
                  extra_starts: torch.Tensor, summaries: torch.Tensor,
                  logical_numel: int, block_symbols: int, lanes: int, steps: int, fixed_words: int,
                  precomputed: bool = False, write_summary: bool = True, vector_aligned: bool = False,
                  threads: int = 256, unroll: int = 4, row_tile: int = 16,
                  stage_fields: bool = True, vector_width: int = 8) -> None:
    """Pack paired codes using the shared raw-exponent-indexed shifted table."""
    if lanes != 256 or threads not in (64, 128, 256) or vector_width not in (4, 8):
        raise ValueError("Encoder requires 256 lanes, 64/128/256 threads, and 4/8-value transfers")
    if row_tile < 2 or row_tile % 2 or steps % row_tile or unroll < 1:
        raise ValueError("Encoder row tiles must divide the stream length and contain whole pairs")
    if vector_aligned and row_tile * lanes % (vector_width * threads):
        raise ValueError("Encoder row tiles must contain whole vector-loading groups")
    source_numel, storage_numel, payload_numel = T.const("source_numel, storage_numel, payload_numel")
    streams_numel, summary_numel = T.const("streams_numel, summary_numel")
    source_dtype = "uint8" if precomputed else "int16"
    summary_dtype = "int32" if write_summary else "uint8"
    source: T.Tensor((source_numel,), source_dtype)
    side: T.Tensor((storage_numel,), "uint8")
    encoded: T.Tensor((payload_numel,), "int32")
    encode_table: T.Tensor((256,), "int32")
    extra_starts: T.Tensor((streams_numel,), "uint8")
    summaries: T.Tensor((summary_numel,), summary_dtype)
    blocks = T.ceildiv(storage_numel, block_symbols)
    owners = lanes // threads
    tile_rows = row_tile
    packed_fields = stage_fields and vector_aligned and not precomputed
    tile_dtype = "uint8" if precomputed or packed_fields else "int16"

    with T.Kernel(blocks, threads=threads) as block:
        T.import_source(_DEVICE_SOURCE)
        thread = T.get_thread_binding()
        lookup = T.alloc_shared((256,), "uint32")
        source_tile = T.alloc_shared((row_tile, lanes), tile_dtype)
        scratch = T.alloc_shared((2 * (threads // 32),), "int32")
        word = T.alloc_local((owners,), "int32")
        shift = T.alloc_local((owners,), "int32")
        accumulator = T.alloc_local((owners,), "uint32")
        start = T.alloc_local((owners,), "int32")
        value0 = T.alloc_local((owners,), "int32")
        value1 = T.alloc_local((owners,), "int32")
        bad_count = T.alloc_local((1,), "int32")
        bad_bytes = T.alloc_local((1,), "int32")
        for i in T.unroll(256 // threads):
            index = i * threads + thread
            lookup[index] = T.Cast("uint32", encode_table[index])
        T.sync_threads()
        for owned in T.unroll(owners):
            word[owned] = 0
            shift[owned] = 0
            accumulator[owned] = 0
            start[owned] = 255

        for path in T.unroll(2):
            if (((block + 1) * block_symbols <= logical_numel) == (path == 0)):
                for tile in T.serial(steps // tile_rows):
                    if not vector_aligned:
                        # Contiguous tensor views can begin at an unaligned offset.
                        # Coalesced scalar loads preserve those views without copying.
                        for work in T.serial(row_tile * lanes // threads):
                            flat = work * threads + thread
                            row = flat // lanes
                            col = flat % lanes
                            position = block * block_symbols + (tile * row_tile + row) * lanes + col
                            source_tile[row, col] = 0
                            if path == 0 or position < logical_numel:
                                source_tile[row, col] = source[position]
                    for work in T.serial(row_tile * lanes // (vector_width * threads) if vector_aligned else 0):
                        flat = work * threads + thread
                        row = flat // (lanes // vector_width)
                        col = (flat % (lanes // vector_width)) * vector_width
                        position = block * block_symbols + (tile * row_tile + row) * lanes + col
                        if packed_fields:
                            if path == 0 or position - col + 255 < logical_numel:
                                storage_row = block * block_symbols + (tile * row_tile + row) * lanes
                                row_shift = (block * steps + tile * row_tile + row) & 255
                                if vector_width == 8:
                                    T.evaluate(T.call_extern("handle", "lct_encode_stage8", T.address_of(source_tile[row, col]), T.address_of(side[storage_row]), T.address_of(source[position]), col, row_shift))
                                else:
                                    T.evaluate(T.call_extern("handle", "lct_encode_stage4", T.address_of(source_tile[row, col]), T.address_of(side[storage_row]), T.address_of(source[position]), col, row_shift))
                            else:
                                for item in T.unroll(vector_width):
                                    source_tile[row, col + item] = 0
                                    if position + item < logical_numel:
                                        value = T.Cast("int32", source[position + item])
                                        source_tile[row, col + item] = T.Cast("uint8", (value >> 7) & 255)
                                        storage_col = (col + item - (block * steps + tile * row_tile + row)) & 255
                                        storage_row = block * block_symbols + (tile * row_tile + row) * lanes
                                        side[storage_row + storage_col] = T.Cast("uint8", (value & 127) | ((value >> 8) & 128))
                        else:
                            if path == 0 or position + vector_width - 1 < logical_numel:
                                if precomputed and vector_width == 8:
                                    T.evaluate(T.call_extern("handle", "lct_encode_load8_bytes", T.address_of(source_tile[row, col]), T.address_of(source[position])))
                                elif precomputed:
                                    T.evaluate(T.call_extern("handle", "lct_encode_load4_bytes", T.address_of(source_tile[row, col]), T.address_of(source[position])))
                                elif vector_width == 8:
                                    T.evaluate(T.call_extern("handle", "lct_encode_load8", T.address_of(source_tile[row, col]), T.address_of(source[position])))
                                else:
                                    T.evaluate(T.call_extern("handle", "lct_encode_load4", T.address_of(source_tile[row, col]), T.address_of(source[position])))
                            else:
                                for item in T.unroll(vector_width):
                                    source_tile[row, col + item] = 0
                                    if position + item < logical_numel:
                                        source_tile[row, col + item] = source[position + item]
                    T.sync_threads()
                    for pair in T.unroll(tile_rows // 2, unroll_factor=unroll if unroll < tile_rows // 2 else None):
                        for owned in T.unroll(owners):
                            # Side bytes are already staged for the vector path.
                            # Overflow tails need no further Huffman packing.
                            if (not packed_fields and not precomputed) or start[owned] == 255:
                                lane = thread + owned * threads
                                logical_row = block * steps + tile * tile_rows + pair * 2
                                input0 = logical_row * lanes + ((lane + logical_row) & 255)
                                input1 = (logical_row + 1) * lanes + ((lane + logical_row + 1) & 255)
                                storage0 = block * block_symbols + (tile * tile_rows + pair * 2) * lanes + lane
                                value0[owned] = T.Cast("int32", source_tile[pair * 2, (lane + logical_row) & 255])
                                value1[owned] = T.Cast("int32", source_tile[pair * 2 + 1, (lane + logical_row + 1) & 255])
                                if precomputed or packed_fields:
                                    byte0 = value0[owned] & 255
                                    byte1 = value1[owned] & 255
                                else:
                                    byte0 = (value0[owned] >> 7) & 255
                                    byte1 = (value1[owned] >> 7) & 255
                                    if path == 0 or input0 < logical_numel:
                                        side[storage0] = T.Cast("uint8", (value0[owned] & 127) | ((value0[owned] >> 8) & 128))
                                    if path == 0 or input1 < logical_numel:
                                        side[storage0 + lanes] = T.Cast("uint8", (value1[owned] & 127) | ((value1[owned] >> 8) & 128))
                                if start[owned] == 255:
                                    packed0 = T.if_then_else(path == 0 or input0 < logical_numel, lookup[byte0], T.uint32(0))
                                    packed1 = T.if_then_else(path == 0 or input1 < logical_numel, lookup[byte1], T.uint32(0))
                                    length0 = T.Cast("int32", packed0 >> 20)
                                    length = length0 + T.Cast("int32", packed1 >> 20)
                                    code = (packed0 & 0xfffff) | ((packed1 & 0xfffff) << length0)
                                    next_shift = shift[owned] + length
                                    crosses = next_shift >= 32
                                    first_overflow = (word[owned] >= fixed_words and length != 0) or (word[owned] == fixed_words - 1 and next_shift > 32)
                                    next_word = accumulator[owned] | (code << shift[owned])
                                    if first_overflow:
                                        start[owned] = tile * tile_rows + pair * 2
                                    if crosses:
                                        if word[owned] < fixed_words:
                                            encoded[word[owned] * streams_numel + block * lanes + lane] = T.Cast("int32", T.if_then_else(first_overflow, accumulator[owned], next_word))
                                        accumulator[owned] = T.if_then_else(shift[owned] == 0, T.uint32(0), code >> (32 - shift[owned]))
                                        word[owned] = word[owned] + 1
                                        shift[owned] = next_shift - 32
                                    else:
                                        accumulator[owned] = next_word
                                        shift[owned] = next_shift

                    # Finish lookup reads before reusing the tile or aliased scratch.
                    T.sync_threads()

        bad_count[0] = 0
        bad_bytes[0] = 0
        for owned in T.unroll(owners):
            lane = thread + owned * threads
            stream = block * lanes + lane
            if word[owned] < fixed_words and shift[owned] != 0:
                encoded[word[owned] * streams_numel + stream] = T.Cast("int32", accumulator[owned])
            extra_starts[stream] = T.Cast("uint8", start[owned])
            bad_count[0] = bad_count[0] + T.Cast("int32", start[owned] != 255)
            bad_bytes[0] = bad_bytes[0] + T.if_then_else(start[owned] != 255, steps - start[owned], 0)
        if write_summary:
            for reduction in T.unroll(5):
                bad_count[0] = bad_count[0] + T.shfl_down(bad_count[0], 16 >> reduction)
                bad_bytes[0] = bad_bytes[0] + T.shfl_down(bad_bytes[0], 16 >> reduction)
            warp = thread // 32
            if thread % 32 == 0:
                scratch[warp] = bad_count[0]
                scratch[threads // 32 + warp] = bad_bytes[0]
            T.sync_threads()
            if thread == 0:
                bad_count[0] = 0
                bad_bytes[0] = 0
                for warp_index in T.unroll(threads // 32):
                    bad_count[0] = bad_count[0] + scratch[warp_index]
                    bad_bytes[0] = bad_bytes[0] + scratch[threads // 32 + warp_index]
                summaries[block] = bad_count[0]
                summaries[blocks + block] = bad_bytes[0]
