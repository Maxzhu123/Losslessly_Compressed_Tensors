"""GPU-resident first-fit allocator for descriptor-based CUDA buffers."""

from dataclasses import dataclass
import torch

from .kernels.tensor_buffer import _allocate_kernel, _reset_kernel
from .kernels.tensor_buffer_ops import tensor_buffer_free


_ALIGNMENT = 16


@dataclass(frozen=True)
class Allocation:
    """A CUDA ``[offset, aligned_size, status, generation]`` descriptor."""

    descriptor: torch.Tensor
    owner: object

    @property
    def offset(self) -> torch.Tensor:
        return self.descriptor[0]

    @property
    def size(self) -> torch.Tensor:
        return self.descriptor[1]

    @property
    def status(self) -> torch.Tensor:
        return self.descriptor[2]

    @property
    def generation(self) -> torch.Tensor:
        return self.descriptor[3]


class TensorBuffer:
    """CUDA-resident first-fit allocator returning device-side descriptors.

    Callers pass the payload base and returned device offsets directly to
    kernels; dynamically sized Python tensor views are intentionally avoided.
    """

    def __init__(
        self, capacity_bytes: int, *,
        max_free_regions: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        """ max_free_regions: Maximum number of free regions to track in the allocator"""
        if capacity_bytes > torch.iinfo(torch.int32).max:
            raise ValueError("capacity_bytes must fit in int32")
        if max_free_regions <= 0 or max_free_regions & (max_free_regions - 1):
            raise ValueError("max_free_regions must be a positive power of two")

        self.capacity_bytes = int(capacity_bytes)
        self.max_free_regions = int(max_free_regions)
        self.device = torch.device(device or "cuda")
        self._metadata_bytes = self._align(8 * self.max_free_regions + 12)
        self._storage = torch.zeros(
            self._metadata_bytes + self.capacity_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        self.device = self._storage.device
        self._data = self._storage.narrow(
            0, self._metadata_bytes, self.capacity_bytes
        )
        self._free_starts = self._storage.narrow(
            0, 0, self.max_free_regions * 4
        ).view(torch.int32)
        self._free_sizes = self._storage.narrow(
            0, self.max_free_regions * 4, self.max_free_regions * 4
        ).view(torch.int32)
        self._free_count = self._storage.narrow(
            0, self.max_free_regions * 8, 4
        ).view(torch.int32)
        self._lock = self._storage.narrow(
            0, self.max_free_regions * 8 + 4, 4
        ).view(torch.int32)
        self._generation = self._storage.narrow(
            0, self.max_free_regions * 8 + 8, 4
        ).view(torch.int32)
        self.reset()

    @staticmethod
    def _align(nbytes: int) -> int:
        return (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT

    @property
    def data(self) -> torch.Tensor:
        return self._data

    def allocate(self, nbytes: torch.Tensor) -> Allocation:
        """Reserve a one-element CUDA integer request without host sync."""
        descriptor = torch.empty(4, dtype=torch.int32, device=self.device)
        _allocate_kernel[(1,)](
            nbytes, nbytes, descriptor,
            self._free_starts, self._free_sizes, self._free_count,
            self._lock, self._generation, ALIGNMENT=_ALIGNMENT,
            ITEM_BYTES=0, MAX_FREE_REGIONS=self.max_free_regions,
        )
        return Allocation(descriptor, self)

    def allocate_with_items(
        self,
        payload_bytes: torch.Tensor,
        item_count: torch.Tensor,
        item_bytes: int,
    ) -> Allocation:
        """Reserve ``payload_bytes + item_count * item_bytes`` asynchronously."""
        if item_bytes <= 0:
            raise ValueError("item_bytes must be positive")
        descriptor = torch.empty(4, dtype=torch.int32, device=self.device)
        _allocate_kernel[(1,)](
            payload_bytes, item_count, descriptor,
            self._free_starts, self._free_sizes, self._free_count,
            self._lock, self._generation, ALIGNMENT=_ALIGNMENT,
            ITEM_BYTES=int(item_bytes), MAX_FREE_REGIONS=self.max_free_regions,
        )
        return Allocation(descriptor, self)

    def free(self, allocation: Allocation) -> torch.Tensor:
        """Return an allocation and asynchronously return a device status."""
        if allocation.owner is not self:
            raise ValueError("allocation belongs to a different allocator")
        if allocation.descriptor.device != self.device:
            raise ValueError("allocation must be on the allocator device")
        return tensor_buffer_free(self._storage, allocation.descriptor, self.max_free_regions)

    def reset(self) -> None:
        """Reset asynchronously; previously returned descriptors become stale."""
        _reset_kernel[(1,)](
            self._free_starts, self._free_sizes,  self._free_count,
            self._lock, self._generation, self.capacity_bytes,
            MAX_FREE_REGIONS=self.max_free_regions,
        )


def _free_regions_snapshot(
    buffer: TensorBuffer,
) -> list[tuple[int, int]]:
    """Synchronize and copy the device allocator's free-list metadata."""
    metadata = buffer._storage.narrow(0, 0, buffer._metadata_bytes).view(
        torch.int32
    ).cpu()
    count = int(metadata[2 * buffer.max_free_regions].item())
    if not 0 <= count <= buffer.max_free_regions:
        raise RuntimeError(f"corrupt free-list count: {count}")
    return [
        (
            int(metadata[index].item()),
            int(metadata[buffer.max_free_regions + index].item()),
        )
        for index in range(count)
    ]


def verify_buffer(buffer: TensorBuffer) -> None:
    """Synchronize and validate the allocator's free-list invariants."""
    previous_end = 0
    free_bytes = 0
    for start, size in _free_regions_snapshot(buffer):
        if size <= 0 or start < previous_end or start + size > buffer.capacity_bytes:
            raise RuntimeError(f"invalid free region: [{start}, {start + size})")
        if start == previous_end and previous_end != 0:
            raise RuntimeError("adjacent free regions were not coalesced")
        previous_end = start + size
        free_bytes += size
    if free_bytes > buffer.capacity_bytes:
        raise RuntimeError("free regions exceed payload capacity")


def visualize_buffer(buffer: TensorBuffer, width: int = 80) -> str:
    """Synchronize and return an ASCII map of payload usage."""
    if buffer is None:
        return ""
    if width < 8:
        raise ValueError("width must be at least 8")
    free_regions = _free_regions_snapshot(buffer)
    free_bytes = sum(size for _, size in free_regions)
    used_bytes = buffer.capacity_bytes - free_bytes
    cells = ["#"] * width
    for start, size in free_regions:
        left = start * width // buffer.capacity_bytes
        right = (start + size) * width + buffer.capacity_bytes - 1
        right //= buffer.capacity_bytes
        for cell in range(left, min(width, max(left + 1, right))):
            cells[cell] = "."
    ranges = [f"[{start}, {start + size})" for start, size in free_regions]
    if len(ranges) > 8:
        ranges = [*ranges[:4], "...", *ranges[-4:]]
    return "\n".join(
        (
            f"payload: {used_bytes}/{buffer.capacity_bytes} B used "
            f"({used_bytes / buffer.capacity_bytes:.1%}); "
            f"{len(free_regions)} free region(s)",
            f"0 [{''.join(cells)}] {buffer.capacity_bytes}",
            "# used  . free",
            f"free: {', '.join(ranges) or 'none'}",
        )
    )
