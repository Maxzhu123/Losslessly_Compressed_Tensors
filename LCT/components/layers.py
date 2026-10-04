"""BF16 trainable matrices with optional persistent LCT weight storage."""
import torch
from torch import nn
import torch.nn.functional as F

from LCT.LCTensor import LCTTensor
from LCT.compress import compress, decompress
from LCT.components.dist_configs import weight_dist, act_dist, act_relu_dist
from LCT.components.ops import rms_norm, rms_norm_backward


def dense_weight(weight):
    return weight.decompress() if isinstance(weight, LCTTensor) else weight


@torch.compile(fullgraph=True)
def _restore_normalized(x, rstd, gain):
    # Match fused RMSNorm: apply the gain in FP32, then round once to BF16.
    return (x.float() * rstd * gain.float()).to(x.dtype).reshape(-1, x.shape[-1])


class RMSLinear(torch.autograd.Function):
    """RMSNorm followed by one linear projection, caching only the norm input."""

    @staticmethod
    @torch.compile()
    def rms_linear(x, norm_weight, W_dense, bias):
        """RMSNorm + linear projection with optional input compression."""
        # RMS + linear projection
        normalized, rstd = rms_norm(x, norm_weight.to(x.dtype))

        output = F.linear(normalized, W_dense, bias.to(x.dtype))
        return output , rstd

    @staticmethod
    # @torch.compile()
    def rms_linear_backward(norm_weight, grad, W_dense, rstd, x, bias_dtype):
        gain = norm_weight.to(x.dtype)
        grad = grad.reshape(-1, grad.shape[-1])
        grad_normalized = grad @ W_dense
        grad_x, grad_gain = rms_norm_backward(grad_normalized.reshape(x.shape), x, rstd, gain)
        del grad_normalized
        normalized = _restore_normalized(x, rstd, gain)
        grad_weight = grad.T @ normalized
        grad_bias = grad.sum(0).to(bias_dtype)
        return grad_x, grad_weight, grad_gain.to(norm_weight.dtype), grad_bias

    @staticmethod
    def forward(ctx, x, weight, norm_weight, bias, buffer=None, compressed=False,
                distribution=None, min_elements=65536, decoded_weight=None):
        ctx.compressed = compressed and x.numel() >= min_elements
        W_dense = dense_weight(weight) if decoded_weight is None else decoded_weight

        output, rstd = RMSLinear.rms_linear(x, norm_weight, W_dense, bias)

        # Save for autograd
        ctx.input = compress(x, distribution=distribution or act_dist, buffer=buffer) if ctx.compressed else x
        ctx.save_for_backward(norm_weight, rstd, weight if decoded_weight is None else decoded_weight)
        ctx.bias_dtype = bias.dtype
        return output

    @staticmethod
    def backward(ctx, grad):
        norm_weight, rstd, weight = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None

        W_dense = dense_weight(weight)
        grads = RMSLinear.rms_linear_backward(norm_weight, grad, W_dense, rstd, x, ctx.bias_dtype)
        return grads + (None,) * (len(ctx.needs_input_grad) - 4)


class _RMSNormQKV(torch.autograd.Function):
    """One shared RMSNorm and input cache for the three Q/K/V projections."""

    @staticmethod
    @torch.compile()
    def rms_qkv(x, norm_weight, weights, biases):
        gain = norm_weight.to(x.dtype)
        normalized, rstd = rms_norm(x, gain)
        outputs = tuple(F.linear(normalized, w, b.to(x.dtype))
                        for w, b in zip(weights, biases))
        return outputs, rstd

    @staticmethod
    @torch.compile()
    def rms_qkv_backward(norm_weight, grad_outputs, weights, rstd, x, bias_dtypes):
        gain = norm_weight.to(x.dtype)
        grads = [grad.reshape(-1, grad.shape[-1]) for grad in grad_outputs]
        # Accumulate projection gradients before the shared norm backward.
        grad_normalized = grads[-1] @ weights[-1]
        for grad, weight in zip(grads[-2::-1], weights[-2::-1]):
            grad_normalized.add_(grad @ weight)
        grad_x, grad_gain = rms_norm_backward(grad_normalized.reshape(x.shape), x, rstd, gain)
        grad_gain = grad_gain.to(norm_weight.dtype)
        del grad_normalized

        # Reuse the saved reciprocal RMS instead of repeating the reduction.
        normalized = _restore_normalized(x, rstd, gain)
        parameter_grads = []
        for grad, bias_dtype in zip(grads, bias_dtypes):
            parameter_grads.extend((grad.T @ normalized, grad.sum(0).to(bias_dtype)))
        return grad_x, grad_gain, *parameter_grads

    @staticmethod
    def forward(ctx, x, norm_weight, buffer, compressed, distribution,
                q_weight, q_bias, k_weight, k_bias, v_weight, v_bias):
        weights = (q_weight, k_weight, v_weight)
        biases = (q_bias, k_bias, v_bias)
        dense_weights = tuple(dense_weight(w) for w in weights)
        outputs, rstd = _RMSNormQKV.rms_qkv(x, norm_weight, dense_weights, biases)

        ctx.compressed = compressed
        ctx.input = compress(x, distribution=distribution, buffer=buffer) if compressed else x
        ctx.save_for_backward(norm_weight, rstd, *weights)
        ctx.bias_dtypes = tuple(b.dtype for b in biases)
        return outputs

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        norm_weight, rstd, *weights = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None

        dense_weights = tuple(dense_weight(w) for w in weights)
        grad_x, grad_gain, *parameter_grads = _RMSNormQKV.rms_qkv_backward(
            norm_weight, (grad_q, grad_k, grad_v), dense_weights, rstd, x, ctx.bias_dtypes)
        return grad_x, grad_gain, None, None, None, *parameter_grads


def rms_norm_qkv(x, norm_weight, q, k, v, *, buffer=None, compressed=False,
                 distribution=None, min_elements=65536):
    """Normalize once and project to Q/K/V, sharing the cached input."""
    return _RMSNormQKV.apply(x, norm_weight, buffer, compressed and x.numel() >= min_elements,
                             distribution or act_dist, q.weight, q.bias, k.weight, k.bias,
                             v.weight, v.bias)


class RMSNormFunction(torch.autograd.Function):
    """RMSNorm with an explicitly owned, optionally compressed input cache."""

    @staticmethod
    @torch.compile()
    def norm_forward(x, gains):
        gain = gains.to(x.dtype) if gains is not None else None
        return rms_norm(x, gain)

    @staticmethod
    @torch.compile()
    def norm_backward(grad, x, rstd, gains):
        gain = gains.to(x.dtype) if gains is not None else None
        grad_x, grad_gains = rms_norm_backward(grad, x, rstd, gain)
        if grad_gains is not None:
            grad_gains = grad_gains.to(gains.dtype)
        return grad_x, grad_gains

    @staticmethod
    def forward(ctx, x, gains, buffer, compressed, distribution, min_elements):
        output, rstd = RMSNormFunction.norm_forward(x, gains)
        ctx.compressed = compressed and x.numel() >= min_elements
        if ctx.compressed:
            ctx.input = compress(x, distribution=distribution or act_dist, buffer=buffer)
        else:
            ctx.input = x
        ctx.save_for_backward(gains, rstd)
        return output

    @staticmethod
    def backward(ctx, grad):
        gains, rstd = ctx.saved_tensors
        x = decompress(ctx.input) if ctx.compressed else ctx.input
        if ctx.compressed:
            ctx.input.free()
        ctx.input = None
        return RMSNormFunction.norm_backward(grad, x, rstd, gains) + (None, None, None, None)


class Relu2Linear(torch.autograd.Function):
    """Save ReLU(z) once; reconstruct its square for the projection gradient."""

    @staticmethod
    @torch.compile()
    def relu2_linear(z, W_dense, bias):
        relu = z.relu()
        output = F.linear(relu.square(), W_dense, bias.to(z.dtype))
        return output, relu

    @staticmethod
    @torch.compile()
    def relu2_linear_backward(grad, W_dense, relu, bias_dtype):
        g = grad.reshape(-1, grad.shape[-1])
        grad_z = (g @ W_dense).reshape(relu.shape)
        grad_z.mul_(2 * relu)
        grad_z.masked_fill_(relu == 0, 0)
        grad_weight = g.T @ relu.square().reshape(-1, relu.shape[-1])
        grad_bias = g.sum(0).to(bias_dtype)
        return grad_z, grad_weight, grad_bias

    @staticmethod
    def forward(ctx, z, weight, bias, buffer=None, compressed=False,
                distribution=None, min_elements=65536):
        ctx.compressed = compressed and z.numel() >= min_elements
        W_dense = dense_weight(weight)
        output, relu = Relu2Linear.relu2_linear(z, W_dense, bias)

        if ctx.compressed:
            ctx.relu = compress(relu, distribution=distribution or act_relu_dist, buffer=buffer)
        else:
            ctx.relu = relu
        ctx.save_for_backward(weight)
        ctx.bias_dtype = bias.dtype
        return output

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        relu = decompress(ctx.relu) if ctx.compressed else ctx.relu
        if ctx.compressed:
            ctx.relu.free()
        ctx.relu = None

        W_dense = dense_weight(weight)
        return Relu2Linear.relu2_linear_backward(grad, W_dense, relu, ctx.bias_dtype) + (None, None, None, None)


class _AttentionState:
    def __init__(self, tensors, compressed, buffer, distribution):
        self.compressed = compressed
        self.values = []
        for tensor in tensors:
            if compressed:
                # The trainer supplies BHTD views of contiguous BTHD storage.
                self.values.append(compress(tensor.transpose(1, 2),
                                            distribution=distribution, buffer=buffer))
            else:
                self.values.append(tensor)

    def unpack(self):
        if self.compressed:
            return [decompress(value).transpose(1, 2) for value in self.values]
        return self.values

    def close(self):
        if self.compressed:
            for value in self.values:
                value.free()


class FlashAttention(torch.autograd.Function):
    """Native flash kernels with an explicit Q/K/V/output cache, without hooks.

    Inputs are BHTD; the projected output is BTD. Causal, scale 0.12, no dropout,
    matching the nanoGPT trainer. FP32 statistics and RNG state stay unchanged.
    """

    @staticmethod
    @torch.compile()
    def attention_forward(q, k, v, W_dense, bias):
        out, lse, cq, ck, max_q, max_k, rng, unused, _ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, 0.0, True, False, scale=0.12))
        hidden = out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        output = F.linear(hidden, W_dense, bias.to(out.dtype))
        return output, out, (lse, cq, ck, max_q, max_k, rng, unused)

    @staticmethod
    # @torch.compile()
    def attention_backward(grad_output, q, k, v, out, W_dense, auxiliary, bias_dtype):
        batch, heads, length, head_dim = q.shape
        g = grad_output.reshape(-1, grad_output.shape[-1])
        hidden = out.transpose(1, 2).reshape(-1, heads * head_dim)
        grad_weight = g.T @ hidden
        grad_bias = g.sum(0).to(bias_dtype)
        grad_attention = (g @ W_dense).reshape(
            batch, length, heads, head_dim).transpose(1, 2)
        lse, cq, ck, max_q, max_k, rng, unused = auxiliary
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            grad_attention, q, k, v, out, lse, cq, ck, max_q, max_k,
            0.0, True, rng, unused, scale=0.12)
        return dq, dk, dv, grad_weight, grad_bias

    @staticmethod
    def forward(ctx, q, k, v, weight, bias, buffer, compressed, distribution, min_elements):
        W_dense = dense_weight(weight)
        output, out, ctx.auxiliary = FlashAttention.attention_forward(q, k, v, W_dense, bias)
        ctx.weight = weight
        ctx.bias_dtype = bias.dtype
        ctx.state = _AttentionState((q, k, v, out),
                                    compressed and q.numel() >= min_elements,
                                    buffer, distribution)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out = ctx.state.unpack()
        W_dense = dense_weight(ctx.weight)
        grads = FlashAttention.attention_backward(
            grad_output, q, k, v, out, W_dense, ctx.auxiliary, ctx.bias_dtype)
        ctx.state.close()
        ctx.state = ctx.auxiliary = ctx.weight = None
        return grads + (None, None, None, None)


def compress_weight(module, buffer=None, distribution=None):
    """Call after initialization, device placement, and checkpoint loading."""
    if isinstance(module.weight, LCTTensor):
        return
    packed = LCTTensor(module.weight, buffer=buffer, dist=distribution or weight_dist)
    del module._parameters['weight']
    module.weight = packed


class Linear(nn.Linear):
    """Parameter holder for the functional projection operators."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self.weight = nn.Parameter(self.weight.to(torch.bfloat16))

    def compress_weight(self, buffer=None, distribution=None):
        compress_weight(self, buffer, distribution)


class Embedding(torch.autograd.Function):
    """Functional embedding lookup with dense or compressed weights."""

    @staticmethod
    @torch.compile()
    def embed(indices, W_dense):
        return F.embedding(indices, W_dense)

    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.num_embeddings = weight.shape[0]
        W_dense = dense_weight(weight)
        return Embedding.embed(indices, W_dense)

    @staticmethod
    @torch.compile()
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return None, torch.ops.aten.embedding_dense_backward(
            grad, indices, ctx.num_embeddings, -1, False)


def named_trainable_tensors(model):
    """Include compressed leaves, which cannot be wrapped in nn.Parameter."""
    yield from model.named_parameters()
    for name, module in model.named_modules():
        if isinstance(module, (Linear, nn.Embedding)) and isinstance(module.weight, LCTTensor):
            yield (name + '.' if name else '') + 'weight', module.weight
