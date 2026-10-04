import math
from dataclasses import dataclass
from enum import Enum, auto


class NoiseLevel(Enum):
    CLEAN = auto()   # Default fixed-stream Huffman layout.
    MEDIUM = auto()  # Larger fixed payload for medium-noise data.
    HIGH = auto()    # Smaller blocks with extra payload for high-noise data.
    SPARSE = auto()  # 2.5 fixed payload bits per exponent, plus overflow storage.


class DistType(Enum):
    EMPIRICAL = "empirical"
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    GAMMA = "gamma"


class StorageLayout(Enum):
    """ If the tensor is compressed or raw format."""
    RAW = auto()
    COMPRESSED = auto()


@dataclass
class Distribution:
    """Codec distribution selector.

    ``family`` is a :class:`DistType` enum member. ``param`` is the source
    value scale: empirical-distribution scale, Gaussian standard deviation, or
    Laplace scale, and defaults to 1. ``mean`` is the source distribution's
    location and defaults to 0. ``zero_prob`` is the probability of an exact
    zero and defaults to 0. ``None`` means no extra zero point mass; the
    distribution's natural zero probability is used. The first two parameters
    are rounded to the nearest 0.25 and ``zero_prob`` to the nearest 0.05 so
    the table cache stays small.
    """

    family: DistType = DistType.EMPIRICAL
    param: float = 1.0
    mean: float = 0.0
    noise_level: NoiseLevel = NoiseLevel.CLEAN
    zero_prob: float | None = 0.0

    def __post_init__(self):
        if not isinstance(self.family, DistType):
            raise TypeError("family must be a DistType member")
        if not isinstance(self.noise_level, NoiseLevel):
            raise TypeError("noise_level must be a NoiseLevel member")
        param = float(self.param)
        mean = float(self.mean)
        zero_prob = self.zero_prob
        if zero_prob is None:
            # No explicit zero inflation requested: use the distribution's
            # natural zero probability, i.e. no added point mass.
            zero_prob = 0.0
        zero_prob = float(zero_prob)
        if not math.isfinite(param) or param <= 0.0:
            raise ValueError("distribution parameter must be finite and positive")
        if not math.isfinite(mean):
            raise ValueError("distribution mean must be finite")
        if not math.isfinite(zero_prob) or not (0.0 <= zero_prob <= 1.0):
            raise ValueError("zero_prob must be finite and between 0 and 1")
        self.param = float(max(0.25, round(param / 0.25) * 0.25))
        self.mean = float(round(mean / 0.25) * 0.25)
        self.zero_prob = round(max(0.0, min(1.0, zero_prob)), 2)

    def __str__(self) -> str:
        label = "std" if self.family == DistType.GAUSSIAN else "scale"
        return (
            f"{self.family.value}/{label}={self.param}/mean={self.mean}/"
            f"{self.noise_level.name.lower()}"
        )
