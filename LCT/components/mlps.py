from typing import TYPE_CHECKING, cast

import torch
from torch.autograd import Function

from LCT.LCTensor import LCTTensor
from LCT.components.dist_configs import act_dist, act_relu_dist

if TYPE_CHECKING:
    from LCT.tensor_buffer import TensorBuffer
    from torch import Tensor


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=None, buffer=None, compressed=False):
        if eps is None:
            eps = torch.finfo(x.dtype).eps

        y, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], None, eps,
        )

        if compressed:
            x = LCTTensor(x, buffer=buffer, dist=act_dist)
        ctx.save_for_backward(x, rstd)
        ctx.compressed = compressed
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, rstd = ctx.saved_tensors
        if ctx.compressed:
            x = x.decompress_free()
        grad_x, grad_weight = torch.ops.aten._fused_rms_norm_backward.default(
            grad_output, x, [x.shape[-1]], rstd, None, [True, False]
        )
        return grad_x, None, None, None


class RMSFFN(Function):
    """RMSNorm followed by FFN, without caching the normalized input.

    Save the original input, reciprocal RMS, weights, and ReLU activation.
    Reconstruct the normalized input in backward for the W1 gradient.
    """

    @staticmethod
    def forward(ctx, x, W1, W2, buffer: TensorBuffer|None=None,
                compressed: bool=False, eps=None):
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        normalized, rstd = torch.ops.aten._fused_rms_norm.default(
            x, [x.shape[-1]], None, eps,
        )
        h = (normalized @ W1.T).relu_()
        del normalized
        output = h @ W2.T

        if compressed:
            x = LCTTensor(x, buffer=buffer, dist=act_dist)
            h = LCTTensor(h, buffer=buffer, dist=act_relu_dist)
        ctx.save_for_backward(x, W1, W2, h, rstd)
        ctx.compressed = compressed
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, h, rstd = ctx.saved_tensors
        if ctx.compressed:
            x = x.decompress_free()
            h = h.decompress_free()
        # The local tensors now suffice. In particular, release the saved
        # compressed storage before allocating dense backward intermediates.
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_W2 = grad_output.T @ h if ctx.needs_input_grad[2] else None
        grad_x = grad_W1 = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_h = grad_output @ W2
            grad_z = torch.ops.aten.threshold_backward.grad_input(
                grad_h, h, 0, grad_input=grad_h,
            )
            del h, grad_h
            if ctx.needs_input_grad[1]:
                # Multiply in rstd's precision, writing directly to the input
                # dtype without materializing full-size float32 temporaries.
                normalized = torch.empty_like(x)
                torch.mul(x, rstd, out=normalized)
                grad_W1 = grad_z.T @ normalized
                del normalized
            if ctx.needs_input_grad[0]:
                grad_normalized = grad_z @ W1
                del grad_z
                grad_x, _ = torch.ops.aten._fused_rms_norm_backward.default(
                    grad_normalized, x, [x.shape[-1]], rstd, None, [True, False],
                )
        return grad_x, grad_W1, grad_W2, None, None, None


class FFN(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.
    """

    @staticmethod
    def forward(ctx, x, W1, W2, buffer: TensorBuffer|None=None, compressed: bool=False):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        h = z.relu_()
        output = h @ W2.T

        if compressed:
            x = LCTTensor(x, buffer=buffer, dist=act_dist)
            h = LCTTensor(h, buffer=buffer, dist=act_relu_dist)

        ctx.save_for_backward(x, W1, W2, h)
        ctx.compressed = compressed
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        needs_x = ctx.needs_input_grad[0]

        x, W1, W2, h = ctx.saved_tensors

        if ctx.compressed:
            x = x.decompress_free()
            h = h.decompress_free()

        grad_z = grad_output @ W2
        grad_W2 = grad_output.T @ h

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, h, 0, grad_input=grad_z)

        del h, grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        if needs_x:
            grad_x = grad_preact @ W1
        else:
            grad_x = None
        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None
