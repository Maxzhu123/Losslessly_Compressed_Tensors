"""Select and launch individual codec kernels into caller-owned buffers."""

import torch
import triton

from ..compression.huffman_tables import FIRST_BITS, FIRST_MASK
from .common.tables import _shift_decoding_table_kernel
from .backends import backend_for


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
                  fixed_words: int, rare_length: int) -> None:
    """Launch the selected decoder without allocating its output or restoring tails."""
    backend = backend_for(encoded.device)
    if backend.decode_tilelang:
        from .tilelang.tilelang_utils import launch_autotuned

        inputs = (encoded, side, decode_table, center, output)
        args = (*inputs, block_symbols, lanes, steps, fixed_words, rare_length)
        launch_autotuned(backend.decode, inputs, args, {})
    else:
        shifted_decode = torch.empty_like(decode_table)
        _shift_decoding_table_kernel[(1,)](
            decode_table, center, shifted_decode, BLOCK=1 << FIRST_BITS,
        )
        blocks = triton.cdiv(size, block_symbols)
        backend.decode[(blocks,)](
            encoded, side, output, shifted_decode, size, blocks * lanes, center,
            LOGICAL_NUMEL=logical_numel,
            FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
            BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps, FIXED_WORDS=fixed_words,
            ON_DEMAND=logical_numel > 600_000_000,
        )
