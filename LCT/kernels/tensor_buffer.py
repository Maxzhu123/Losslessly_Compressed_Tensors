"""Triton kernels for the GPU-resident descriptor-based first-fit allocator."""

import triton
from triton import language as tl


# Status codes written to a descriptor's slot 2 by allocate/free.
_STATUS_OK = 0
_STATUS_OUT_OF_MEMORY = 1
_STATUS_INVALID_REQUEST = 2
_STATUS_FREE_LIST_FULL = 3
_STATUS_INVALID_FREE = 4
_STATUS_FREED = 5


@triton.jit
def _lock(lock):
    while tl.atomic_cas(lock, 0, 1) != 0:
        pass


@triton.jit
def _unlock(lock):
    tl.atomic_xchg(lock, 0)


@triton.jit
def _reset_kernel(
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    capacity_bytes,
    MAX_FREE_REGIONS: tl.constexpr,
):
    indices = tl.arange(0, MAX_FREE_REGIONS)
    tl.store(free_starts + indices, 0)
    tl.store(free_sizes + indices, 0)
    tl.store(free_sizes, capacity_bytes)
    tl.store(free_count, 1)
    tl.store(lock, 0)
    tl.store(generation, tl.load(generation) + 1)


@triton.jit
def _allocate_kernel(
    requested_bytes,
    item_count,
    descriptor,
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    ALIGNMENT: tl.constexpr,
    ITEM_BYTES: tl.constexpr,
    MAX_FREE_REGIONS: tl.constexpr,
):
    _lock(lock)

    request = tl.load(requested_bytes).to(tl.int32)
    if ITEM_BYTES:
        request += tl.load(item_count).to(tl.int32) * ITEM_BYTES
    current_generation = tl.load(generation).to(tl.int32)
    size = ((request + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    count = tl.load(free_count).to(tl.int32)
    indices = tl.arange(0, MAX_FREE_REGIONS)
    active = indices < count
    starts = tl.load(free_starts + indices, mask=active, other=0)
    sizes = tl.load(free_sizes + indices, mask=active, other=0)
    candidates = tl.where(active & (sizes >= size), indices, MAX_FREE_REGIONS)
    slot = tl.min(candidates, axis=0)
    success = (request > 0) & (slot < count)

    start = tl.load(free_starts + slot, mask=success, other=0)
    region_size = tl.load(free_sizes + slot, mask=success, other=0)
    exact = success & (region_size == size)
    partial = success & ~exact

    next_indices = indices + 1
    shift_left = exact & (indices >= slot) & (next_indices < count)
    next_starts = tl.load(free_starts + next_indices, mask=shift_left, other=0)
    next_sizes = tl.load(free_sizes + next_indices, mask=shift_left, other=0)
    # The shift overlaps its source across warps; finish all reads first.
    tl.debug_barrier()
    tl.store(free_starts + indices, next_starts, mask=shift_left)
    tl.store(free_sizes + indices, next_sizes, mask=shift_left)
    tl.store(free_count, count - 1, mask=exact)

    tl.store(free_starts + slot, start + size, mask=partial)
    tl.store(free_sizes + slot, region_size - size, mask=partial)
    tl.store(descriptor, tl.where(success, start, -1))
    tl.store(descriptor + 1, tl.where(success, size, 0))
    tl.store(
        descriptor + 2,
        tl.where(request <= 0, 2, tl.where(success, 0, 1)),
    )
    tl.store(descriptor + 3, current_generation)
    _unlock(lock)


@triton.jit
def _free_kernel(
    descriptor,
    free_starts,
    free_sizes,
    free_count,
    lock,
    generation,
    status_out,
    MAX_FREE_REGIONS: tl.constexpr,
):
    _lock(lock)

    offset = tl.load(descriptor).to(tl.int32)
    size = tl.load(descriptor + 1).to(tl.int32)
    allocation_status = tl.load(descriptor + 2).to(tl.int32)
    allocation_generation = tl.load(descriptor + 3).to(tl.int32)
    current_generation = tl.load(generation).to(tl.int32)
    count = tl.load(free_count).to(tl.int32)
    valid = (
        (allocation_status == 0)
        & (allocation_generation == current_generation)
        & (offset >= 0)
        & (size > 0)
    )

    indices = tl.arange(0, MAX_FREE_REGIONS)
    active = indices < count
    starts = tl.load(free_starts + indices, mask=active, other=0)
    sizes = tl.load(free_sizes + indices, mask=active, other=0)
    insert = tl.sum((active & (starts < offset)).to(tl.int32), axis=0)

    has_previous = insert > 0
    has_next = insert < count
    previous_index = insert - 1
    previous_start = tl.load(
        free_starts + previous_index, mask=has_previous, other=0
    )
    previous_size = tl.load(
        free_sizes + previous_index, mask=has_previous, other=0
    )
    next_start = tl.load(free_starts + insert, mask=has_next, other=0)
    next_size = tl.load(free_sizes + insert, mask=has_next, other=0)
    merge_previous = has_previous & (previous_start + previous_size == offset)
    merge_next = has_next & (offset + size == next_start)
    insert_new = valid & ~merge_previous & ~merge_next & (count < MAX_FREE_REGIONS)
    previous_only = valid & merge_previous & ~merge_next
    next_only = valid & ~merge_previous & merge_next
    both = valid & merge_previous & merge_next

    # Keep the snapshot intact until every warp has read its entries.
    tl.debug_barrier()
    tl.store(free_sizes + previous_index, previous_size + size, mask=previous_only)
    tl.store(free_starts + insert, offset, mask=next_only)
    tl.store(free_sizes + insert, next_size + size, mask=next_only)
    tl.store(
        free_sizes + previous_index,
        previous_size + size + next_size,
        mask=both,
    )

    next_indices = indices + 1
    shift_left = both & (indices >= insert) & (next_indices < count)
    shifted_starts = tl.load(free_starts + next_indices, mask=shift_left, other=0)
    shifted_sizes = tl.load(free_sizes + next_indices, mask=shift_left, other=0)
    tl.debug_barrier()
    tl.store(free_starts + indices, shifted_starts, mask=shift_left)
    tl.store(free_sizes + indices, shifted_sizes, mask=shift_left)
    tl.store(free_count, count - 1, mask=both)

    shift_right = insert_new & (indices >= insert) & (indices < count)
    tl.store(free_starts + indices + 1, starts, mask=shift_right)
    tl.store(free_sizes + indices + 1, sizes, mask=shift_right)
    tl.store(free_starts + insert, offset, mask=insert_new)
    tl.store(free_sizes + insert, size, mask=insert_new)
    tl.store(free_count, count + 1, mask=insert_new)

    free_list_full = valid & ~merge_previous & ~merge_next & (
        count >= MAX_FREE_REGIONS
    )
    tl.store(descriptor + 2, 5, mask=valid & ~free_list_full)
    tl.store(
        status_out,
        tl.where(~valid, 4, tl.where(free_list_full, 3, 0)),
    )
    _unlock(lock)


@triton.jit
def _free_storage_kernel(
    storage, descriptor, status, MAX_FREE_REGIONS: tl.constexpr,
):
    """Derive metadata pointers on the GPU instead of constructing host views."""
    metadata = storage.to(tl.pointer_type(tl.int32))
    _free_kernel(
        descriptor, metadata, metadata + MAX_FREE_REGIONS,
        metadata + 2 * MAX_FREE_REGIONS, metadata + 2 * MAX_FREE_REGIONS + 1,
        metadata + 2 * MAX_FREE_REGIONS + 2, status, MAX_FREE_REGIONS,
    )

