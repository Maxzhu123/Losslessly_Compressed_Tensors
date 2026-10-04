"""Distribution and GPU-specific geometry of the compressed storage format."""

import torch

from ..comp_format import Distribution, DistType, NoiseLevel
from ..kernels.device import device_profile

BLOCK_SYMBOLS = 65536
LANES = 256
LANE_BITS = 800

if device_profile(torch.device("cuda")).model == "a100_80gb_pcie":
    GAUSSIAN_BLOCK_SYMBOLS = 32768
    GAUSSIAN_LANE_BITS = 400 - 16
else:
    GAUSSIAN_BLOCK_SYMBOLS = BLOCK_SYMBOLS
    GAUSSIAN_LANE_BITS = LANE_BITS - 32


def geometry(distribution: Distribution) -> tuple[int, int, int, int]:
    """Return ``(block_symbols, lanes, steps, fixed_words)`` for new storage.

    GPU-specific constants are selected once at import for this process.
    Readers use the geometry saved on the compressed tensor.
    """
    steps = BLOCK_SYMBOLS // LANES
    if distribution.noise_level == NoiseLevel.CLEAN:
        block_symbols = BLOCK_SYMBOLS
        lane_bits = LANE_BITS
        if distribution.family == DistType.GAUSSIAN:
            block_symbols = GAUSSIAN_BLOCK_SYMBOLS
            steps = block_symbols // LANES
            lane_bits = GAUSSIAN_LANE_BITS
    elif distribution.noise_level == NoiseLevel.SPARSE:
        block_symbols = BLOCK_SYMBOLS
        lane_bits = 5 * steps // 2
    elif distribution.noise_level == NoiseLevel.MEDIUM:
        block_symbols = BLOCK_SYMBOLS // 2
        steps //= 2
        lane_bits = LANE_BITS - 6 * 32
    elif distribution.noise_level == NoiseLevel.HIGH:
        block_symbols = BLOCK_SYMBOLS // 2
        steps //= 2
        lane_bits = LANE_BITS - 3 * 32
    else:
        raise ValueError(f"Unknown noise level: {distribution.noise_level}")
    return block_symbols, LANES, steps, lane_bits // 32
