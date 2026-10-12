"""Registry-driven pointwise operations for compressed tensors."""
from typing import TYPE_CHECKING
from dataclasses import dataclass
import torch
import triton

from ..comp_tensor import CompressedTensor
from ..comp_format import Distribution, StorageLayout
from ..compression.huffman_tables import FIRST_BITS, FIRST_MASK, get_distribution_tables
from .encode import encode as compress_dense, encode_components as compress_components
from .decode import (
    _storage as _fallback_storage,
    decode as decode_dense,
    has_fallback,
)
from .geometry import geometry
from ..kernels import dispatch
from ..kernels.generic.pointwise import (
    COMPRESSED_OUTPUT, DENSE_OUTPUT,
    add_op, multiply_op, _scaled_sum,
)
from ..tensor_buffer import TensorBuffer
if TYPE_CHECKING:
    from torch import Tensor
    from triton import language as tl


def _launch_pointwise_compressed_dense(
    data: CompressedTensor, other: Tensor,
    operation: PointwiseOp, output_policy: tl.constexpr,
    *,
    alpha: Tensor | None = None, beta: Tensor | None = None, alpha_is_one: bool = False,
):
    """ alpha * compA + beta * B
        Elementwise. alpha=1 and beta=1 special cases.
    """
    # Normalize arithmetic and unit scalars once for both kernels.
    scaled = operation.name == "scalar_mul_add"
    operation_args = (alpha, beta) if scaled else (None, None)
    operation_meta = dict(
        OP=_scaled_sum if scaled else operation.triton_fn,
        SCALED=scaled,
        ALPHA_IS_ONE=not scaled or alpha_is_one,
        BETA_IS_ONE=not scaled or beta is None,
    )

    # Prepare the decode table and flattened dense operand.
    other = other.contiguous().view(-1)
    _, decode_table, rare_length = get_distribution_tables(data.distribution)
    shifted_decode = data.decode_table
    if shifted_decode is None:
        shifted_decode = dispatch.shift_decoding_table(decode_table, data.center)
    block_symbols, lanes, steps, fixed_words = data.codec_geometry
    blocks = triton.cdiv(data.size, block_symbols)
    # Allocate dense values or the components consumed by the encoder.
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

    # Decode fixed streams and apply the selected operation.
    dispatch.pointwise_dense(
        data.data, data.sign_mantissa, other, output, auxiliary, shifted_decode,
        data.size, blocks * lanes, data.center, *operation_args,
        rare_length=rare_length, logical_numel=data.logical_numel,
        block_symbols=block_symbols, lanes=lanes, steps=steps,
        fixed_words=fixed_words, output_policy=output_policy,
        operation_meta=operation_meta,
    )
    # Correct overflow rows using the input's fallback storage layout.
    if has_fallback(data):
        dispatch.pointwise_fallback(
            _fallback_storage(data), data.sign_mantissa, other, output, auxiliary,
            data.size, *operation_args,
            logical_numel=data.logical_numel,
            block_symbols=block_symbols, lanes=lanes, steps=steps,
            output_policy=output_policy, operation_meta=operation_meta,
        )
    return output, auxiliary


def _launch_scalar_mul_add_compressed_compressed(
    data: CompressedTensor, other: CompressedTensor, alpha: Tensor,
    output_policy: tl.constexpr,
):
    """ alpha * compA + compB
        Elementwise. Dense BF16 values or components for recompression.
    """
    _, a_decode_table, rare_length_a = get_distribution_tables(data.distribution)
    _, b_decode_table, rare_length_b = get_distribution_tables(other.distribution)
    if rare_length_a != rare_length_b:
        raise ValueError("two-compressed path requires matching RARE_LENGTH")

    a_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=data.data.device,
    )
    b_shifted_decode = torch.empty(
        1 << FIRST_BITS, dtype=torch.int32, device=other.data.device,
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
    a_storage = _fallback_storage(data)
    b_storage = _fallback_storage(other)

    streams = blocks * lanes
    a_stream_starts = torch.empty(streams, dtype=torch.int32, device=data.data.device)
    a_stream_offsets = torch.empty(streams, dtype=torch.int32, device=data.data.device)
    b_stream_starts = torch.empty(streams, dtype=torch.int32, device=other.data.device)
    b_stream_offsets = torch.empty(streams, dtype=torch.int32, device=other.data.device)

    # Direct maps avoid scanning the full fallback list in every codec block.
    # Initialize before scattering metadata so the launches cannot race.
    dispatch.dual_maps_init(
        a_stream_starts, a_stream_offsets, b_stream_starts, b_stream_offsets,
        data.fallback_count, other.fallback_count, streams, steps,
    )
    dispatch.dual_tables_and_maps(
        a_decode_table, data.center, a_shifted_decode,
        b_decode_table, other.center, b_shifted_decode,
        a_storage, b_storage,
        a_stream_starts, a_stream_offsets, b_stream_starts, b_stream_offsets,
        blocks, lanes=lanes, steps=steps, buffered=buffered,
    )
    dispatch.dual_mapped(
        data.data, data.sign_mantissa, a_shifted_decode, data.center,
        a_stream_starts, a_stream_offsets, a_storage,
        other.data, other.sign_mantissa, b_shifted_decode, other.center,
        b_stream_starts, b_stream_offsets, b_storage,
        output, auxiliary, alpha, data.size, streams, blocks,
        rare_length=rare_length_a, logical_numel=data.logical_numel,
        buffered=buffered, block_symbols=block_symbols, lanes=lanes,
        steps=steps, fixed_words=fixed_words, output_policy=output_policy,
    )
    return output, auxiliary


def pointwise_scale_add_compressed(
    data: CompressedTensor, other: CompressedTensor, alpha: Tensor,
    *,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution: Distribution | None = None,
) -> Tensor | CompressedTensor:
    """ alpha * compA + compB
        Elementwise. Dense BF16 or compressed output.
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
    data: CompressedTensor,  other: Tensor, operation: PointwiseOp,
    *,
    alpha: Tensor | None = None, beta: Tensor | None = None, alpha_is_one: bool = False,
    dense_output: bool = True,
    buffer: TensorBuffer | None = None,
    distribution: Distribution | None = None,
) -> Tensor | CompressedTensor:
    """ alpha * compA + beta * B, or compA * B
        Elementwise. alpha=1 and beta=1 special cases.
        Dense BF16 or compressed output.
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
    values, auxiliary = _launch_pointwise_compressed_dense(
        data, other, operation, policy,
        alpha=alpha, beta=beta, alpha_is_one=alpha_is_one,
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


# Compute compA + B elementwise for additive updates.
ADD = PointwiseOp("add", add_op, torch.add)
# Compute compA * B elementwise for per-element scaling.
MULTIPLY = PointwiseOp("multiply", multiply_op, torch.mul)
# Compute alpha * compA + B or alpha * compA + beta * B for scaled updates.
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
