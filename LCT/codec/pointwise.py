"""Registry-driven pointwise operations for compressed tensors."""

from dataclasses import dataclass
import torch
import triton

from ..comp_tensor import CompressedTensor
from ..comp_format import StorageLayout
from ..compression.huffman_tables import FIRST_BITS, FIRST_MASK, get_distribution_tables
from .autotune import SCATTER_GRID_LIMIT
from .encode import encode as compress_dense, encode_components as compress_components
from .decode import decode as decode_dense
from .geometry import geometry
from ..kernels.common.tables import _shift_decoding_table_kernel
from ..kernels.generic.pointwise import (
    COMPRESSED_OUTPUT,
    DENSE_OUTPUT,
    pointwise_compressed_dense_fallback_kernel,
    pointwise_compressed_dense_kernel,
    add_op, multiply_op
)
from ..kernels.generic.pointwise_scalar import (
    pointwise_scalar_mul_add_dense_fallback_kernel,
    pointwise_scalar_mul_add_dense_kernel,
)
from ..kernels.generic.pointwise_scalar_dual import (
    _initialize_dual_maps_kernel,
    _prepare_dual_tables_and_maps_kernel,
    pointwise_scalar_mul_add_compressed_compressed_mapped_kernel,
)
from ..tensor_buffer import TensorBuffer


def _launch_pointwise_compressed_dense(data, other, operation, output_policy):
    """Launch the fixed-stream operation, then correct sparse fallback tails."""
    other = other.contiguous().view(-1)
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    _shift_decoding_table_kernel[(1,)](
        decode_table, data.center, shifted_decode,
        BLOCK=1 << FIRST_BITS,
    )
    block_symbols, lanes, steps, fixed_words = data.codec_geometry
    blocks = triton.cdiv(data.size, block_symbols)
    if output_policy == DENSE_OUTPUT:
        output = torch.empty(
            data.logical_numel,
            dtype=torch.bfloat16, device=data.data.device,
        )
        auxiliary = output
    else:
        output = torch.empty(
            data.size, dtype=torch.uint8, device=data.data.device
        )
        auxiliary = torch.empty_like(output)

    main_args = (
        data.data, data.sign_mantissa, other, output, auxiliary,
        shifted_decode, data.size, blocks * lanes, data.center,
    )
    main_meta = dict(
        OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    pointwise_compressed_dense_kernel[(blocks,)](
        *main_args, LOGICAL_NUMEL=data.logical_numel, **main_meta,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        fallback_args = (
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, other, output, auxiliary, data.size,
        )
        fallback_meta = dict(
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=True, TILE=64, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (min(triton.cdiv(blocks * lanes, 64), SCATTER_GRID_LIMIT),)
        pointwise_compressed_dense_fallback_kernel[fallback_grid](
            *fallback_args, LOGICAL_NUMEL=data.logical_numel, **fallback_meta,
        )
    elif data.offsets.numel():
        fallback_args = (
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            other, output, auxiliary, data.size,
        )
        fallback_meta = dict(
            OP=operation.triton_fn, OUTPUT_POLICY=output_policy,
            BUFFERED=False, TILE=64, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (min(triton.cdiv(data.offsets.numel(), 64), SCATTER_GRID_LIMIT),)
        pointwise_compressed_dense_fallback_kernel[fallback_grid](
            *fallback_args, LOGICAL_NUMEL=data.logical_numel, **fallback_meta,
        )
    return output, auxiliary


def _launch_scalar_mul_add_compressed_dense(
    data, other, alpha, output_policy, beta=None, alpha_is_one=False,
):
    """Launch the dedicated fused scalar multiply-add pointwise kernels."""
    other = other.contiguous().view(-1)
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    _shift_decoding_table_kernel[(1,)](
        decode_table, data.center, shifted_decode,
        BLOCK=1 << FIRST_BITS,
    )
    block_symbols, lanes, steps, fixed_words = data.codec_geometry
    blocks = triton.cdiv(data.size, block_symbols)
    if output_policy == DENSE_OUTPUT:
        output = torch.empty(
            data.logical_numel,
            dtype=torch.bfloat16, device=data.data.device,
        )
        auxiliary = output
    else:
        output = torch.empty(
            data.size, dtype=torch.uint8, device=data.data.device
        )
        auxiliary = torch.empty_like(output)

    main_args = (
        data.data, data.sign_mantissa, other, output, auxiliary,
        shifted_decode, data.size, blocks * lanes, data.center, alpha, beta,
    )
    main_meta = dict(
        SCALE_OTHER=beta is not None,
        ALPHA_IS_ONE=alpha_is_one,
        OUTPUT_POLICY=output_policy,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    pointwise_scalar_mul_add_dense_kernel[(blocks,)](
        *main_args, LOGICAL_NUMEL=data.logical_numel, **main_meta,
    )
    if data.fallback_descriptor is not None:
        metadata = data.fallback_buffer.view(torch.int32)
        fallback_args = (
            metadata, data.fallback_buffer, metadata, data.fallback_buffer, 0,
            metadata, data.fallback_descriptor, data.fallback_count,
            data.sign_mantissa, other, output, auxiliary, data.size, alpha, beta,
        )
        fallback_meta = dict(
            SCALE_OTHER=beta is not None,
            ALPHA_IS_ONE=alpha_is_one,
            OUTPUT_POLICY=output_policy,
            BUFFERED=True, TILE=64, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (min(triton.cdiv(blocks * lanes, 64), SCATTER_GRID_LIMIT),)
        pointwise_scalar_mul_add_dense_fallback_kernel[fallback_grid](
            *fallback_args, LOGICAL_NUMEL=data.logical_numel, **fallback_meta,
        )
    elif data.offsets.numel():
        fallback_args = (
            data.offsets, data.fallback_starts, data.fallback_offsets,
            data.fallback_buffer, data.fallback_base, data.offsets,
            data.offsets, data.fallback_count, data.sign_mantissa,
            other, output, auxiliary, data.size, alpha, beta,
        )
        fallback_meta = dict(
            SCALE_OTHER=beta is not None,
            ALPHA_IS_ONE=alpha_is_one,
            OUTPUT_POLICY=output_policy,
            BUFFERED=False, TILE=64, BLOCK=block_symbols,
            N_LANES=lanes, N_STEPS=steps,
        )
        fallback_grid = (min(triton.cdiv(data.offsets.numel(), 64), SCATTER_GRID_LIMIT),)
        pointwise_scalar_mul_add_dense_fallback_kernel[fallback_grid](
            *fallback_args, LOGICAL_NUMEL=data.logical_numel, **fallback_meta,
        )
    return output, auxiliary


def _launch_scalar_mul_add_compressed_compressed(
    data, other, alpha, output_policy,
):
    """Prepare fallback maps, then decode and compute both compressed operands."""
    _, a_decode_table, rare_length_a = get_distribution_tables(data.distribution)
    _, b_decode_table, rare_length_b = get_distribution_tables(other.distribution)
    if rare_length_a != rare_length_b:
        raise ValueError("two-compressed path requires matching RARE_LENGTH")

    a_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    b_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    block_symbols, lanes, steps, fixed_words = data.codec_geometry
    blocks = triton.cdiv(data.size, block_symbols)

    if output_policy == DENSE_OUTPUT:
        output = torch.empty(
            data.logical_numel,
            dtype=torch.bfloat16, device=data.data.device,
        )
        auxiliary = output
    else:
        output = torch.empty(
            data.size, dtype=torch.uint8, device=data.data.device
        )
        auxiliary = torch.empty_like(output)

    # The caller guarantees both operands use the same fallback storage mode.
    buffered = data.fallback_descriptor is not None
    a_metadata = data.fallback_buffer.view(torch.int32) if buffered else data.offsets
    b_metadata = other.fallback_buffer.view(torch.int32) if buffered else other.offsets
    a_bad_streams = a_metadata
    b_bad_streams = b_metadata
    a_bad_starts = data.fallback_buffer if buffered else data.fallback_starts
    b_bad_starts = other.fallback_buffer if buffered else other.fallback_starts
    a_fb_offsets = data.fallback_offsets if not buffered else a_metadata
    b_fb_offsets = other.fallback_offsets if not buffered else b_metadata
    a_descriptor = data.fallback_descriptor if buffered else data.offsets
    b_descriptor = other.fallback_descriptor if buffered else other.offsets

    streams = blocks * lanes

    # Direct maps avoid scanning the full fallback list in every codec block.
    # Initialize before scattering metadata so the launches cannot race.
    a_stream_starts = torch.empty(streams, dtype=torch.int32, device=data.data.device)
    a_stream_offsets = torch.empty(streams, dtype=torch.int32, device=data.data.device)
    b_stream_starts = torch.empty(streams, dtype=torch.int32, device=other.data.device)
    b_stream_offsets = torch.empty(streams, dtype=torch.int32, device=other.data.device)

    _initialize_dual_maps_kernel[(triton.cdiv(streams, 1024),)](
        a_stream_starts, a_stream_offsets, b_stream_starts, b_stream_offsets,
        data.fallback_count, other.fallback_count, streams,
        N_STEPS=steps, BLOCK=1024,
    )
    map_grid = (2 + min(triton.cdiv(streams, 1024), SCATTER_GRID_LIMIT),)
    _prepare_dual_tables_and_maps_kernel[map_grid](
        a_decode_table, data.center, a_shifted_decode,
        b_decode_table, other.center, b_shifted_decode,
        a_bad_streams, a_bad_starts, a_fb_offsets, a_metadata,
        a_descriptor, data.fallback_count,
        a_stream_starts, a_stream_offsets,
        b_bad_streams, b_bad_starts, b_fb_offsets, b_metadata,
        b_descriptor, other.fallback_count,
        b_stream_starts, b_stream_offsets,
        N_SHIFT=2,
        BLOCK=1 << FIRST_BITS,
        BUFFERED=buffered,
    )

    mapped_meta = dict(
        BUFFERED=buffered,
        OUTPUT_POLICY=output_policy,
        LOGICAL_NUMEL=data.logical_numel,
        FIRST_MASK=FIRST_MASK, RARE_LENGTH=rare_length_a,
        BLOCK=block_symbols, N_LANES=lanes, N_STEPS=steps,
        FIXED_WORDS=fixed_words,
    )
    pointwise_scalar_mul_add_compressed_compressed_mapped_kernel[(blocks,)](
        data.data, data.sign_mantissa, a_shifted_decode, data.center,
        a_stream_starts, a_stream_offsets,
        a_bad_streams, a_bad_starts, a_fb_offsets, a_metadata,
        a_descriptor, data.fallback_count,
        data.fallback_buffer, data.fallback_base,
        other.data, other.sign_mantissa, b_shifted_decode, other.center,
        b_stream_starts, b_stream_offsets,
        b_bad_streams, b_bad_starts, b_fb_offsets, b_metadata,
        b_descriptor, other.fallback_count,
        other.fallback_buffer, other.fallback_base,
        output, auxiliary, alpha,
        data.size, streams,
        **mapped_meta,
    )
    return output, auxiliary


def pointwise_scale_add_compressed(
    data: CompressedTensor,
    other: CompressedTensor,
    alpha: torch.Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """Apply ``alpha * data + other`` where both operands are compressed.

    Uses one fused decode/compute pass for blocked operands with the same
    geometry, including both private and buffered fallback storage.
    """
    same_layout = (
        data.layout == StorageLayout.COMPRESSED
        and other.layout == StorageLayout.COMPRESSED
    )
    same_buffering = (data.fallback_descriptor is None) == (other.fallback_descriptor is None)
    # The two-compressed path requires both operands to use the same fallback
    # storage mode; do not silently fall back to a mixed-buffer dense path.
    assert same_buffering, (
        "pointwise_scale_add_compressed requires both inputs to be buffer-backed "
        "or both to be non-buffer-backed"
    )
    if same_layout and data.codec_geometry == other.codec_geometry:
        result_distribution = distribution or data.distribution
        if (
            not dense_output
            and geometry(result_distribution) != data.codec_geometry
        ):
            raise ValueError(
                "matrix compressed output must preserve the input stream geometry"
            )
        policy = DENSE_OUTPUT if dense_output else COMPRESSED_OUTPUT
        values, auxiliary = _launch_scalar_mul_add_compressed_compressed(
            data, other, alpha, policy,
        )
        if dense_output:
            return values.reshape(data.shape)
        result = compress_components(
            auxiliary, values, data.size, result_distribution,
            buffer, data.shape, precomputed=True,
            logical_numel=data.logical_numel,
        )

        return result

    if other.layout == StorageLayout.RAW:
        other_dense = other.data.reshape(other.shape)
    else:
        other_dense = decode_dense(other)
    return pointwise_compressed_dense(
        data, other_dense, SCALAR_MUL_ADD, alpha=alpha,
        dense_output=dense_output, buffer=buffer, distribution=distribution,
    )


def pointwise_compressed_dense(
    data: CompressedTensor,
    other: torch.Tensor,
    operation: PointwiseOp,
    *,
    alpha=None,
    beta=None,
    alpha_is_one=False,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution=None,
) -> torch.Tensor | CompressedTensor:
    """ Apply a compressed + dense pointwise operation.
        For scalar_mul_add, optional beta scales the dense operand in FP32.
        alpha_is_one specializes away the alpha load and multiplication.
        If dense_output, returns a dense tensor. Otherwise, returns a compressed tensor.
    """
    # Both tensors must have the same shape.
    if tuple(other.shape) != data.shape:
        raise ValueError(f"{operation.name} expects tensors with the same shape")

    # Keep the input distribution/geometry unless the caller overrides it.
    result_distribution = distribution or data.distribution
    if operation.name == "scalar_mul_add" and alpha is None:
        raise ValueError("scalar_mul_add requires an alpha argument")
    if (
        data.layout == StorageLayout.COMPRESSED
        and not dense_output
        and geometry(result_distribution) != data.codec_geometry
    ):
        raise ValueError(
            "matrix compressed output must preserve the input stream geometry"
        )

    # Raw fast path: no codec streams, operate on dense payload directly.
    if data.layout == StorageLayout.RAW:
        if operation.name == "scalar_mul_add":
            if alpha_is_one:
                left = data.data.reshape(data.shape).float()
                if beta is None:
                    result = (left + other.float()).to(torch.bfloat16)
                else:
                    result = torch.addcmul(left, other.float(), beta).to(torch.bfloat16)
            elif beta is None:
                result = operation.torch_fn(data.data.reshape(data.shape), other, alpha)
            else:
                result = torch.addcmul(
                    other.float() * beta, data.data.reshape(data.shape).float(), alpha,
                ).to(torch.bfloat16)
        else:
            result = operation.torch_fn(data.data.reshape(data.shape), other)
        if dense_output:
            return result
        return compress_dense(result, result_distribution, buffer)

    # Launch the fused pointwise kernel and write dense or component output.
    policy = DENSE_OUTPUT if dense_output else COMPRESSED_OUTPUT
    if operation.name == "scalar_mul_add":
        values, auxiliary = _launch_scalar_mul_add_compressed_dense(
            data, other, alpha, policy, beta=beta, alpha_is_one=alpha_is_one,
        )
    else:
        values, auxiliary = _launch_pointwise_compressed_dense(
            data, other, operation, policy,
        )
    if dense_output:
        return values.reshape(data.shape)

    # Reuse the shared Huffman encoder on the fused kernel's exponent output.
    result = compress_components(
        auxiliary, values, data.size, result_distribution,
        buffer, data.shape, precomputed=True,
        logical_numel=data.logical_numel,
    )
    return result


@dataclass(frozen=True)
class PointwiseOp:
    """Pair an in-kernel operation with its PyTorch correctness reference."""
    name: str
    triton_fn: object
    torch_fn: object


ADD = PointwiseOp("add", add_op, torch.add)
MULTIPLY = PointwiseOp("multiply", multiply_op, torch.mul)
SCALAR_MUL_ADD = PointwiseOp(
    "scalar_mul_add",
    None,
    lambda left, right, alpha=1.0: (
        (left.float() * alpha + right.float()).to(torch.bfloat16)
    ),
)
POINTWISE_OPS = {
    operation.name: operation
    for operation in (ADD, MULTIPLY, SCALAR_MUL_ADD)
}
