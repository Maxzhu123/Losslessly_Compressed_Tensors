import math

import torch
from torch import Tensor

from .comp_format import Distribution
from .comp_tensor import CompressedTensor
from .compress import compress, decompress, compA_add_B, a_compA_add_B, a_compA_add_b_B, a_compA_add_compB
from .tensor_buffer import TensorBuffer


class LCTTensor(Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl
    x: CompressedTensor

    @staticmethod
    def __new__(cls, x, buffer: TensorBuffer|None, dist: Distribution):
        assert x.dtype == torch.bfloat16
        assert x.device.type == "cuda", f"x.device={x.device}"
        return torch.Tensor._make_wrapper_subclass(
            cls,
            x.shape, dtype=x.dtype, device=x.device,
            # autograd belongs to the inner representation.
            requires_grad=x.requires_grad,
        )

    def __init__(self, x: Tensor, buffer: TensorBuffer|None, dist: Distribution):
        self.x: CompressedTensor = compress(x, buffer=buffer, distribution=dist)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.t.default:
            x = args[0]
            return decompress(x.x).T
        if func is torch.ops.aten.permute.default:
            x = args[0]
            return decompress(x.x).T

        if func is torch.ops.aten.mm.default:
            a, b = args
            if isinstance(a, LCTTensor):
                a = decompress(a.x)
            if isinstance(b, LCTTensor):
                b = decompress(b.x)
            return a @ b
        raise NotImplementedError(f"{func} not implemented for LCTTensor")

    @property
    def nbytes(self):
        return self.x.memory_size()

    @property
    def dense_nbytes(self):
        return math.prod(self.x.shape) * 2 # bfloat16 is 2 bytes

    @property
    def layout(self):
        """Return the storage layout of this compressed tensor."""
        return "LCT compressed"

    def __repr__(self):
        return f"LCTTensor({self.x})"

    def add_(self, update: Tensor):
        """ Inplace add,
            x <- x + update
        """
        prev = self.x
        self.x = compA_add_B(self.x, update,
                             dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        prev.free()

    def mul_add_(self, alpha: Tensor, update: Tensor, beta: Tensor | None = None,
                 *, alpha_is_one: bool = False):
        """ Inplace multiply-add,
            x <- alpha * x + beta * update (beta defaults to 1).
            alpha_is_one specializes alpha to 1 without reading its tensor.
        """
        prev = self.x
        if beta is None:
            self.x = a_compA_add_B(prev, alpha, update,
                                   alpha_is_one=alpha_is_one,
                                   dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        else:
            self.x = a_compA_add_b_B(prev, alpha, update, beta,
                                     alpha_is_one=alpha_is_one,
                                     dense_output=False, distribution=prev.distribution, buffer=prev.buffer)
        prev.free()

    def add_comp_(self, update_comp: LCTTensor, alpha: Tensor):
        """ Inplace add with another compressed tensor,
            x <- x + alpha * update_comp
        """
        prev = self.x
        self.x = a_compA_add_compB(update_comp.x, alpha, prev,
            dense_output=False, buffer=prev.buffer, distribution=prev.distribution,
        )
        prev.free()

    def decompress(self) -> Tensor:
        return decompress(self.x)

    def decompress_free(self) -> Tensor:
        """ Decompresses the compressed tensor and frees the compressed representation."""
        x_dense = self.decompress()
        self.x.free()
        return x_dense

    def free(self):
        self.x.free()
