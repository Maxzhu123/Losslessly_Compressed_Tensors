"""Launch shared codec kernels with GPU-specific tuning and one storage ABI."""

import torch
import triton

from ..compression.huffman_tables import FIRST_BITS, FIRST_MASK
from .autotune import DECODE_ON_DEMAND_CUTOFF
from .backends import FallbackStorage, backend_for


# ---------------------------------------------------------------------------
# Table preparation
# ---------------------------------------------------------------------------


def shift_encoding_table(
    encode_table: torch.Tensor, center: torch.Tensor, shifted_encode: torch.Tensor,
    decode_table: torch.Tensor, shifted_decode: torch.Tensor | None = None,
) -> None:
    """Build the raw-exponent encode table for one center.

    Emit the shifted decode table in the same launch for subsequent reads.
    """
    backend = backend_for(encode_table.device)
    if shifted_decode is None:
        shifted_decode = torch.empty_like(decode_table)
    backend.shift_encoding_table[(1,)](
        encode_table, center, shifted_encode, decode_table, shifted_decode,
        BLOCK=256,
    )


def estimate_and_shift_encoding_table(
    source: torch.Tensor, center: torch.Tensor, size: int,
    encode_table: torch.Tensor, shifted_encode: torch.Tensor,
    decode_table: torch.Tensor, shifted_decode: torch.Tensor | None,
    sample_size: int, precomputed: bool,
) -> None:
    """Sample the center from ``source`` and build the shifted encode table."""
    backend = backend_for(source.device)
    if shifted_decode is None:
        shifted_decode = torch.empty_like(decode_table)
    backend.estimate_and_shift_encoding_table[(1,)](
        source, center, size, encode_table, shifted_encode,
        decode_table, shifted_decode,
        SAMPLE_SIZE=sample_size, PRECOMPUTED=precomputed,
        BLOCK=4096, num_warps=4, num_stages=2,
    )


def shift_decoding_table(
    decode_table: torch.Tensor, center: torch.Tensor,
) -> torch.Tensor:
    """Return a center-shifted copy of ``decode_table``."""
    backend = backend_for(decode_table.device)
    shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=decode_table.device,
    )
    backend.shift_decoding_table[(1,)](
        decode_table, center, shifted_decode, BLOCK=1 << FIRST_BITS,
    )
    return shifted_decode


# ---------------------------------------------------------------------------
# Overflow metadata and tail storage
# ---------------------------------------------------------------------------


def prefix_allocate(
    summaries: torch.Tensor, counts: torch.Tensor, descriptor: torch.Tensor,
    buffer, blocks: int, buffered: bool, max_free_regions: int,
) -> None:
    """Scan the block summaries and reserve the shared fallback arena."""
    backend = backend_for(summaries.device)
    state = (
        (buffer._free_starts, buffer._free_sizes, buffer._free_count,
         buffer._lock, buffer._generation)
        if buffer is not None else (counts,) * 5
    )
    backend.prefix_allocate[(1,)](
        summaries, counts, descriptor, *state, blocks,
        BUFFERED=buffered, BLOCK=max(1, triton.next_power_of_2(blocks)),
        MAX_FREE_REGIONS=max_free_regions, num_warps=4, num_stages=2,
    )


def count_bad_streams(
    extra_starts: torch.Tensor, count: torch.Tensor, total: torch.Tensor,
    streams: int, steps: int,
) -> None:
    """Count overflow streams and bytes for the atomic compaction path."""
    backend = backend_for(extra_starts.device)
    grid = lambda meta: (triton.cdiv(streams, meta["BLOCK"]),)
    backend.count_bad_streams[grid](extra_starts, count, total, streams, steps)


def compact_bad_streams(
    extra_starts: torch.Tensor, storage: FallbackStorage,
    final_counts: torch.Tensor,
    bad_count: torch.Tensor, fallback_total: torch.Tensor,
    n_streams: int, steps: int, n_blocks: int, buffered: bool, lanes: int,
) -> None:
    """Compact the overflow list with the atomic path (no prefix summaries)."""
    backend = backend_for(extra_starts.device)
    backend.compact_bad_streams[(n_blocks,)](
        extra_starts, storage.lanes, storage.starts, storage.allocation,
        storage.blocks, final_counts, bad_count, fallback_total,
        n_streams, steps, n_blocks, BUFFERED=buffered, N_LANES=lanes,
    )


def compact_extra(
    source: torch.Tensor, storage: FallbackStorage, final_counts: torch.Tensor,
    n_blocks: int, n_elements: int, *,
    precomputed: bool, buffered: bool, logical_numel: int,
    block_symbols: int, lanes: int, steps: int,
    extra_starts: torch.Tensor, summaries: torch.Tensor, n_streams: int,
    build_meta: bool,
) -> None:
    """Copy the overflow exponents into the compact tail storage."""
    backend = backend_for(source.device)
    backend.compact_extra[(n_blocks,)](
        source, storage.lanes, storage.starts, storage.tail, storage.allocation,
        final_counts, storage.blocks, storage.base, n_blocks, n_elements,
        extra_starts, summaries, n_streams,
        BUFFERED=buffered, PRECOMPUTED=precomputed, LOGICAL_NUMEL=logical_numel,
        BUILD_META=build_meta,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
    )


def fallback_tail_args(storage: FallbackStorage) -> tuple:
    """Return the fallback argument tuple the selected decoder expects.

    The generator kernels ignore their descriptor when the tails are private, so
    a harmless tensor stands in for it rather than ``None``.
    """
    descriptor = (
        storage.allocation if storage.allocation is not None else storage.lanes
    )
    return (storage.lanes, storage.starts, storage.tail, descriptor,
            storage.count, storage.blocks, storage.base)


def scatter_fallback(
    storage: FallbackStorage, sign_mantissa: torch.Tensor, output: torch.Tensor,
    n_elements: int, logical_numel: int,
    block_symbols: int, lanes: int, steps: int,
) -> None:
    """Overwrite mapped decode output with the compact fallback tails."""
    backend = backend_for(output.device)
    blocks = triton.cdiv(n_elements, block_symbols)
    backend.scatter_fallback[(blocks,)](
        storage.lanes, storage.starts, storage.tail, storage.allocation,
        storage.count, storage.blocks, storage.base,
        sign_mantissa, output, n_elements,
        BUFFERED=storage.buffered, LOGICAL_NUMEL=logical_numel,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
    )


# ---------------------------------------------------------------------------
# Fused pointwise
# ---------------------------------------------------------------------------


def pointwise_dense(
    encoded: torch.Tensor, sign_mantissa: torch.Tensor, other: torch.Tensor,
    output: torch.Tensor, auxiliary: torch.Tensor, shifted_decode: torch.Tensor,
    n_elements: int, n_streams: int, center: torch.Tensor,
    alpha, beta, *, rare_length: int, logical_numel: int,
    block_symbols: int, lanes: int, steps: int, fixed_words: int,
    output_policy, operation_meta: dict,
) -> None:
    """Decode one compressed operand and apply the selected operation."""
    backend = backend_for(encoded.device)
    blocks = triton.cdiv(n_elements, block_symbols)
    backend.pointwise_dense[(blocks,)](
        encoded, sign_mantissa, other, output, auxiliary, shifted_decode,
        n_elements, n_streams, center, alpha, beta,
        LOGICAL_NUMEL=logical_numel,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        OUTPUT_POLICY=output_policy,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps, FIXED_WORDS=fixed_words,
        **operation_meta,
    )


def pointwise_fallback(
    storage: FallbackStorage, sign_mantissa: torch.Tensor, other: torch.Tensor,
    output: torch.Tensor, auxiliary: torch.Tensor, n_elements: int,
    alpha, beta, *, logical_numel: int,
    block_symbols: int, lanes: int, steps: int,
    output_policy, operation_meta: dict,
) -> None:
    """Correct the overflow rows of a fused pointwise result."""
    backend = backend_for(output.device)
    # The fallback kernels re-read the payload, so they take no RARE_LENGTH.
    meta = dict(
        OUTPUT_POLICY=output_policy, BUFFERED=storage.buffered,
        LOGICAL_NUMEL=logical_numel, BLOCK=block_symbols,
        N_LANES=lanes, N_STEPS=steps,
        **operation_meta,
    )
    blocks = triton.cdiv(n_elements, block_symbols)
    backend.pointwise_fallback[(blocks,)](
        storage.lanes, storage.starts, storage.tail, storage.allocation,
        storage.count, storage.blocks, storage.base,
        sign_mantissa, other, output, auxiliary, n_elements, alpha, beta,
        **meta,
    )


def dual_maps_init(
    a_stream_starts: torch.Tensor, a_stream_offsets: torch.Tensor,
    b_stream_starts: torch.Tensor, b_stream_offsets: torch.Tensor,
    a_count: torch.Tensor, b_count: torch.Tensor,
    n_streams: int, steps: int,
) -> None:
    """Initialize both operands' per-stream maps to the no-overflow default."""
    backend = backend_for(a_stream_starts.device)
    backend.dual_maps_init[(triton.cdiv(n_streams, 1024),)](
        a_stream_starts, a_stream_offsets, b_stream_starts, b_stream_offsets,
        a_count, b_count, n_streams, N_STEPS=steps, BLOCK=1024,
    )


def dual_tables_and_maps(
    a_decode_table: torch.Tensor, a_center: torch.Tensor, a_shifted: torch.Tensor,
    b_decode_table: torch.Tensor, b_center: torch.Tensor, b_shifted: torch.Tensor,
    a_storage: FallbackStorage, b_storage: FallbackStorage,
    a_stream_starts: torch.Tensor, a_stream_offsets: torch.Tensor,
    b_stream_starts: torch.Tensor, b_stream_offsets: torch.Tensor,
    blocks: int, *, lanes: int, steps: int, buffered: bool,
) -> None:
    """Shift both decode tables and expand both compact fallback lists."""
    backend = backend_for(a_shifted.device)
    backend.dual_tables_and_maps[(2 + blocks,)](
        a_decode_table, a_center, a_shifted,
        b_decode_table, b_center, b_shifted,
        a_storage.lanes, a_storage.starts, a_storage.allocation,
        a_storage.blocks, a_storage.count,
        a_stream_starts, a_stream_offsets,
        b_storage.lanes, b_storage.starts, b_storage.allocation,
        b_storage.blocks, b_storage.count,
        b_stream_starts, b_stream_offsets,
        N_SHIFT=2, BLOCK=1 << FIRST_BITS,
        N_STEPS=steps, N_LANES=lanes, BUFFERED=buffered,
    )


def dual_mapped(
    a_encoded: torch.Tensor, a_sign_mantissa: torch.Tensor,
    a_shifted: torch.Tensor, a_center: torch.Tensor,
    a_stream_starts: torch.Tensor, a_stream_offsets: torch.Tensor,
    a_storage: FallbackStorage,
    b_encoded: torch.Tensor, b_sign_mantissa: torch.Tensor,
    b_shifted: torch.Tensor, b_center: torch.Tensor,
    b_stream_starts: torch.Tensor, b_stream_offsets: torch.Tensor,
    b_storage: FallbackStorage,
    output: torch.Tensor, auxiliary: torch.Tensor, alpha,
    n_elements: int, n_streams: int, blocks: int, *,
    rare_length: int, logical_numel: int, buffered: bool,
    block_symbols: int, lanes: int, steps: int, fixed_words: int,
    output_policy,
) -> None:
    """Run the fused alpha * compA + compB kernel against both maps."""
    backend = backend_for(output.device)
    meta = dict(
        BUFFERED=buffered, OUTPUT_POLICY=output_policy, LOGICAL_NUMEL=logical_numel,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps, FIXED_WORDS=fixed_words,
    )
    backend.dual_mapped[(blocks,)](
        a_encoded, a_sign_mantissa, a_shifted, a_center,
        a_stream_starts, a_stream_offsets,
        a_storage.allocation, a_storage.count,
        a_storage.tail, a_storage.base,
        b_encoded, b_sign_mantissa, b_shifted, b_center,
        b_stream_starts, b_stream_offsets,
        b_storage.allocation, b_storage.count,
        b_storage.tail, b_storage.base,
        output, auxiliary, alpha, n_elements, n_streams,
        **meta,
    )


# ---------------------------------------------------------------------------
# Codec entry points
# ---------------------------------------------------------------------------


def encode_kernel(source: torch.Tensor, side: torch.Tensor, encoded: torch.Tensor,
                  encode_table: torch.Tensor, extra_starts: torch.Tensor, summaries: torch.Tensor,
                  size: int, streams: int, *, precomputed: bool, logical_numel: int,
                  block_symbols: int, lanes: int, steps: int, fixed_words: int,
                  write_summary: bool) -> None:
    """Launch the selected encoder using the shared shifted encoding table."""
    backend = backend_for(source.device)
    if backend.encode_tilelang:
        from .tilelang.tilelang_utils import launch_autotuned

        inputs = (source, side, encoded, encode_table, extra_starts, summaries)
        args = (*inputs, logical_numel, block_symbols, lanes, steps, fixed_words)
        vector_aligned = source.data_ptr() % 16 == 0 and (precomputed or side.data_ptr() % 8 == 0)
        kwargs = {"precomputed": precomputed, "write_summary": write_summary,
                  "vector_aligned": vector_aligned}
        launch_autotuned(backend.encode, inputs, args, kwargs)
    else:
        backend.encode[(triton.cdiv(size, block_symbols),)](
            source, side, encoded, encode_table, extra_starts, summaries, size, streams,
            PRECOMPUTED=precomputed, LOGICAL_NUMEL=logical_numel,
            FIXED_WORDS=fixed_words, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps, WRITE_SUMMARY=write_summary,
        )


def decode_kernel(encoded: torch.Tensor, side: torch.Tensor, output: torch.Tensor,
                  decode_table: torch.Tensor, center: torch.Tensor, size: int,
                  *, logical_numel: int, block_symbols: int, lanes: int, steps: int,
                  fixed_words: int, rare_length: int,
                  overflow: tuple | None = None,
                  shifted_decode: torch.Tensor | None = None) -> None:
    """Launch the selected decoder without allocating its output or restoring tails.

    ``overflow`` carries the backend's fallback argument tuple and the buffered
    flag, so the decoder can substitute overflowing rows in place. When it is
    ``None`` the fallback overlay is compiled out.
    """
    backend = backend_for(encoded.device)
    if backend.decode_tilelang:
        from .tilelang.tilelang_utils import launch_autotuned

        inputs = (encoded, side, decode_table, center, output)
        args = (*inputs, block_symbols, lanes, steps, fixed_words, rare_length)
        launch_autotuned(backend.decode, inputs, args, {})
        return
    if shifted_decode is None:
        shifted_decode = shift_decoding_table(decode_table, center)
    blocks = triton.cdiv(size, block_symbols)
    common = dict(
        LOGICAL_NUMEL=logical_numel,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps, FIXED_WORDS=fixed_words,
        ON_DEMAND=logical_numel > DECODE_ON_DEMAND_CUTOFF,
    )
    # A GPU-specific decoder may need one extra per-call table (the A100 escape
    # table); it is passed straight after the shifted decode table.
    aux = (
        (backend.decode_make_aux(center, shifted_decode),)
        if backend.decode_make_aux else ()
    )
    head = (encoded, side, output, shifted_decode, *aux, size, blocks * lanes, center)
    if overflow is None or backend.decode_overflow is None:
        # Plain decoder: keeping the original argument list keeps Triton's
        # per-launch argument packing off the critical path.
        backend.decode[(blocks,)](*head, **common)
    else:
        tail_args, buffered = overflow
        backend.decode_overflow[(blocks,)](*head, *tail_args, BUFFERED=buffered, **common)
