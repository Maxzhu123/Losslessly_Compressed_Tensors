from LCT.comp_format import DistType, Distribution, NoiseLevel

# Shared default weight distribution.
weight_dist = Distribution(
    DistType.LAPLACE,
    noise_level=NoiseLevel.CLEAN,
)

# Momentum distribution
momentum_dist = Distribution(
    DistType.EMPIRICAL,
    noise_level=NoiseLevel.CLEAN,
    zero_prob=0.02,
)

# Momentum distribution
momentum_first_dist = Distribution(
    DistType.EMPIRICAL,
    noise_level=NoiseLevel.CLEAN,
    zero_prob=0.5,
)


# Activation distribution
act_dist = Distribution(
    DistType.LAPLACE,
    noise_level=NoiseLevel.CLEAN,
)

# relu activation distribution
act_relu_dist = Distribution(
    DistType.LAPLACE,
    noise_level=NoiseLevel.CLEAN,
    zero_prob=0.5,
)
