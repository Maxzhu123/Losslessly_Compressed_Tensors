"""Public lossless BF16 compression and decompression API."""
from typing import TYPE_CHECKING
import torch

from .comp_format import Distribution, StorageLayout
from .codec.encode import encode as compress_dense
from .codec.decode import decode as decode_dense
from .codec.pointwise import (
    ADD, MULTIPLY, SCALAR_MUL_ADD,
    pointwise_compressed_dense, pointwise_scale_add_compressed,
)
from .tensor_buffer import TensorBuffer

if TYPE_CHECKING:
    from torch import Tensor
    from .comp_tensor import CompressedTensor

@torch.compiler.disable
def compress(
    data: Tensor,
    distribution: Distribution | None = None,
    buffer: TensorBuffer | None = None,
    allow_raw: bool = False,
) -> CompressedTensor:
    """ Losslessly encode the exponent byte of a CUDA bfloat16 tensor."""
    if distribution is None:
        distribution = Distribution()
    return compress_dense(data.detach(), distribution, buffer, allow_raw=allow_raw)


@torch.compiler.disable
def decompress(data: CompressedTensor) -> Tensor:
    """ Decode a tensor produced by :func:`compress`."""
    if data.layout == StorageLayout.COMPRESSED:
        return decode_dense(data)
    return data.data.reshape(data.shape)


def A_compBT(A: Tensor, B_comp: CompressedTensor) -> Tensor:
    """ Compute A @ B_comp.T."""
    if A.ndim != 2 or A.shape[1] != B_comp.shape[1]:
        raise ValueError("activation and weight inner dimensions must match")
    return A.contiguous() @ decompress(B_comp).T


def A_compB(A: Tensor, B_comp: CompressedTensor) -> Tensor:
    """ Compute A @ B_comp."""
    if A.ndim != 2 or A.shape[1] != B_comp.shape[0]:
        raise ValueError("activation and weight inner dimensions must match")
    return A.contiguous() @ decompress(B_comp)


def compA_add_B(
    A_comp: CompressedTensor, B: Tensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """ Compute A + B where A is compressed."""
    return pointwise_compressed_dense(
        A_comp, B, ADD, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def compA_mul_B(
    A_comp: CompressedTensor, B: Tensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """ Compute A * B elementwise, where A is sparse."""
    return pointwise_compressed_dense(
        A_comp, B, MULTIPLY, dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def a_compA_add_B(
    A_comp: CompressedTensor, a: Tensor, B: Tensor,
    *,
    alpha_is_one: bool = False,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """Compute ``a * A_comp + B`` with a fused pointwise operation.

    alpha_is_one=True treats a as 1 without reading its value on the GPU.
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    return pointwise_compressed_dense(
        A_comp, B, SCALAR_MUL_ADD, alpha=a, alpha_is_one=alpha_is_one,
        dense_output=dense_output,
        buffer=buffer, distribution=distribution,
    )


def a_compA_add_b_B(
    A_comp: CompressedTensor, a: Tensor, B: Tensor, b: Tensor,
    *,
    alpha_is_one: bool = False,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """Compute ``a * A_comp + b * B`` without materializing either scaled operand.

    Scalars are CUDA float32 tensors. Multiply B by b in FP32, then use FP32
    fused multiply-add for a * A + scaled_B, rounding the result to BF16.
    alpha_is_one=True treats a as 1 and uses FMA(B, b, A) instead, removing
    the alpha load and redundant multiplication at compile time.
    """
    for name, scalar in (("alpha", a), ("beta", b)):
        if not isinstance(scalar, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if scalar.numel() != 1 or scalar.dtype != torch.float32 or scalar.device != A_comp.data.device:
            raise ValueError(f"{name} must be a float32 scalar on the compressed tensor's device")
    return pointwise_compressed_dense(
        A_comp, B, SCALAR_MUL_ADD, alpha=a, beta=b, alpha_is_one=alpha_is_one,
        dense_output=dense_output, buffer=buffer, distribution=distribution,
    )


def a_compA_add_compB(
    A_comp: CompressedTensor, a: Tensor, B_comp: CompressedTensor,
    *,
    dense_output: bool = True, buffer: TensorBuffer | None = None, distribution=None,
) -> Tensor | CompressedTensor:
    """Compute ``a * A_comp + B_comp`` with both operands compressed.

    This is the fused sparse-update entry point. It uses the one-pass fused
    matrix kernel, which decodes both compressed operands directly and supports
    private and buffered fallback storage.
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    return pointwise_scale_add_compressed(
        A_comp, B_comp, a,
        dense_output=dense_output, buffer=buffer, distribution=distribution,
    )
