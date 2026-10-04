"""Opaque native RMSNorm kernels for compiled dense layer regions."""
import torch
from torch import Tensor


@torch.library.custom_op("lct::rms_norm", mutates_args=(), device_types="cuda")
def rms_norm(x: Tensor, gain: Tensor | None) -> tuple[Tensor, Tensor]:
    return torch.ops.aten._fused_rms_norm.default(x, [x.shape[-1]], gain, None)


@rms_norm.register_fake
def _rms_norm_fake(x, gain):
    return torch.ops.aten._fused_rms_norm.default(x, [x.shape[-1]], gain, None)


@torch.library.custom_op(
    "lct::rms_norm_backward", mutates_args=(), device_types="cuda",
    schema="(Tensor grad, Tensor x, Tensor rstd, Tensor? gain) -> (Tensor, Tensor?)",
)
def rms_norm_backward(grad: Tensor, x: Tensor, rstd: Tensor,
                      gain: Tensor | None) -> tuple[Tensor, Tensor | None]:
    return torch.ops.aten._fused_rms_norm_backward.default(
        grad, x, [x.shape[-1]], rstd, gain, [True, gain is not None])


@rms_norm_backward.register_fake
def _rms_norm_backward_fake(grad, x, rstd, gain):
    return torch.ops.aten._fused_rms_norm_backward.default(
        grad, x, [x.shape[-1]], rstd, gain, [True, gain is not None])


def _rms_norm_setup_context(ctx, inputs, output):
    x, gain = inputs
    _, rstd = output
    ctx.save_for_backward(x, rstd, gain)
    ctx.mark_non_differentiable(rstd)


def _rms_norm_autograd(ctx, grad, grad_rstd):
    x, rstd, gain = ctx.saved_tensors
    return rms_norm_backward(grad, x, rstd, gain)


rms_norm.register_autograd(_rms_norm_autograd, setup_context=_rms_norm_setup_context)
