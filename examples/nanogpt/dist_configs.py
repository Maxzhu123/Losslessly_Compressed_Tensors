"""nanoGPT codebooks, independent of the MLP experiment's LCT defaults."""
from LCT.comp_format import DistType, Distribution, NoiseLevel

weight_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN, zero_prob=0.01)
momentum_dist = Distribution(DistType.EMPIRICAL, noise_level=NoiseLevel.CLEAN, zero_prob=0.02)
act_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.CLEAN)
act_relu_dist = Distribution(DistType.LAPLACE, noise_level=NoiseLevel.SPARSE, zero_prob=0.5)
