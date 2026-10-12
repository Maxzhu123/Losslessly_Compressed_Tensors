from typing import TYPE_CHECKING
from dataclasses import dataclass, field, fields
import math
import torch

from .comp_format import Distribution, DistType, NoiseLevel, StorageLayout

if TYPE_CHECKING:
    from .tensor_buffer import TensorBuffer


@dataclass(frozen=True)
class CompressedTensor:
    data: torch.Tensor  # Fixed exponent payload or source fallback.
    size: int  # Number of elements represented by codec storage, including padding.
    sign_mantissa: torch.Tensor | None = None  # Raw sign + 7-bit mantissa.
    # Legacy constructor slots; new tensors use compact lane/start metadata.
    offsets: torch.Tensor | None = None
    fallback_starts: torch.Tensor | None = None  # First fallback step per stream.
    fallback_offsets: torch.Tensor | None = None
    fallback_buffer: torch.Tensor | None = None  # Shared or private tail storage.
    fallback_descriptor: torch.Tensor | None = None  # Device allocator descriptor.
    buffer: TensorBuffer | None = None
    fallback_base: int = 0
    fallback_count: torch.Tensor | None = None
    fallback_used: torch.Tensor | None = None
    distribution: Distribution = field(
        default_factory=Distribution
    )  # Huffman table selector.
    center: torch.Tensor | int = 0  # CUDA center scalar used by encode/decode.
    shape: tuple[int, ...] = ()  # Original tensor shape.
    layout: StorageLayout = StorageLayout.RAW
    stream_geometry: tuple[int, int, int, int] | None = None
    # One overflow format on every GPU: uint8 lane/start pairs and one
    # int32 count/first-entry/tail-base triple per storage block.
    fallback_lanes: torch.Tensor | None = None
    fallback_blocks: torch.Tensor | None = None
    decode_table: torch.Tensor | None = None  # Shifted table, built at encode time.

    @property
    def codec_geometry(self) -> tuple[int, int, int, int]:
        """Stored layout, independent of the GPU used to read this tensor."""
        if self.layout == StorageLayout.COMPRESSED and self.fallback_blocks is None:
            raise ValueError("legacy compressed layout: decode with the original codec and recompress")
        if self.stream_geometry is not None:
            return self.stream_geometry
        # Tensors created before GPU-specific geometry use the original layout.
        from .codec.geometry import geometry

        if (
            self.distribution.family == DistType.GAUSSIAN
            and self.distribution.noise_level == NoiseLevel.CLEAN
        ):
            return 65536, 256, 256, 24
        return geometry(self.distribution)

    def memory_size(self) -> int:
        """Return GPU allocation bytes owned by one compressed tensor.
        """
        allocations: dict[tuple[torch.device, int], int] = {}
        for f in fields(self):
            tensor = getattr(self, f.name)
            if isinstance(tensor, torch.Tensor):
                if f.name == "fallback_buffer" and self.fallback_descriptor is not None:
                    continue
                storage = tensor.untyped_storage()
                key = (tensor.device, storage.data_ptr())
                allocations[key] = storage.nbytes()
        total = sum(allocations.values())
        if self.fallback_descriptor is not None:
            total += int(self.fallback_descriptor[1].item())
        return total

    def memory_buffer_size(self):
        total = 0
        if self.fallback_descriptor is not None:
            total += int(self.fallback_descriptor[1].item())
        return total

    def free(self) -> None:
        """Release this tensor's fallback allocation, if buffer-backed."""
        if self.fallback_descriptor is None or self.buffer is None:
            return
        from .tensor_buffer import Allocation

        self.buffer.free(Allocation(self.fallback_descriptor, self.buffer))

    @property
    def logical_numel(self) -> int:
        """Return the number of elements exposed by the logical tensor shape."""
        return math.prod(self.shape)

    @property
    def storage_numel(self) -> int:
        """Return the element count represented by compressed storage."""
        return self.size
