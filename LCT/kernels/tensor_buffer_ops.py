"""Opaque allocator operations for torch.compile."""
import torch


@torch.library.custom_op("lct::tensor_buffer_free", mutates_args=("storage", "descriptor"))
def tensor_buffer_free(storage: torch.Tensor, descriptor: torch.Tensor,
                       max_free_regions: int) -> torch.Tensor:
    from .tensor_buffer import _free_storage_kernel

    # Keep one storage argument for alias tracking; derive pointers in the kernel.
    status = torch.empty(1, dtype=torch.int32, device=storage.device)
    _free_storage_kernel[(1,)](
        storage, descriptor, status, MAX_FREE_REGIONS=max_free_regions,
    )
    return status


@tensor_buffer_free.register_fake
def _tensor_buffer_free_fake(storage, descriptor, max_free_regions):
    return torch.empty(1, dtype=torch.int32, device=storage.device)
