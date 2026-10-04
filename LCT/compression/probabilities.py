"""Exponent probabilities induced by the benchmark value distributions."""

import math
import numpy as np
from scipy.integrate import quad


def _shifted_magnitude_cdf(base_magnitude_cdf, mean):
    """Return the magnitude CDF after shifting a symmetric distribution."""
    mean = float(mean)

    def symmetric_cdf(value):
        if math.isinf(value):
            return 1.0 if value > 0.0 else 0.0
        magnitude_probability = base_magnitude_cdf(abs(value))
        if value >= 0.0:
            return 0.5 * (1.0 + magnitude_probability)
        return 0.5 * (1.0 - magnitude_probability)

    def magnitude_cdf(value):
        if math.isinf(value):
            return 1.0
        return symmetric_cdf(value - mean) - symmetric_cdf(-value - mean)

    return magnitude_cdf


def _shifted_magnitude_pdf(base_magnitude_pdf, mean):
    """Return the magnitude PDF after shifting a symmetric distribution."""
    mean = float(mean)

    def magnitude_pdf(value):
        # For a symmetric signed distribution, the magnitude density is the
        # sum of the two folded-side densities after the shift.
        return 0.5 * (
            base_magnitude_pdf(abs(value - mean))
            + base_magnitude_pdf(abs(value + mean))
        )

    return magnitude_pdf


def _interval_probability(pdf, low, high):
    """Integrate a magnitude PDF over the BF16 magnitude bin [low, high]."""
    if high == low:
        return 0.0
    return quad(pdf, low, high, epsabs=1e-12, epsrel=1e-10, limit=200)[0]


def _exponent_probabilities(magnitude_pdf, zero_prob=0.0):
    """Map a value-magnitude CDF onto the codec's centered exponent symbols.

    The center is computed from the continuous/nonzero component only.  Any
    zero point mass is then placed at the known offset
    ``(-127 - center) & 255``.  This keeps the center stable under changes to
    ``zero_prob`` while still giving exact zeros a short Huffman code when
    they are common.
    """
    raw = np.zeros(256, dtype=np.float64)
    for exponent in range(-127, 129):
        low = 0.0 if exponent == -127 else math.ldexp(1.0, exponent)
        high = math.inf if exponent == 128 else math.ldexp(1.0, exponent + 1)
        raw[exponent & 255] = _interval_probability(magnitude_pdf, low, high)
    raw /= raw.sum()

    exponents = np.arange(-127, 129)
    mean = float(np.dot(exponents, raw[exponents & 255]))
    center = math.floor(mean + 0.5) if mean >= 0.0 else -math.floor(-mean + 0.5)
    centered = np.zeros(256, dtype=np.float64)
    for exponent in exponents:
        centered[(exponent - center) & 255] = raw[exponent & 255]

    if zero_prob:
        if not (0.0 <= zero_prob <= 1.0):
            raise ValueError("zero_prob must be between 0 and 1")
        centered = (1.0 - zero_prob) * centered
        centered[(-127 - center) & 255] += zero_prob

    # Reserve table index 0 for exact zero.  All other exponents use the
    # remaining 255 table indices: centered deltas less than the zero delta
    # shift up by one, while deltas greater than it keep their value.
    zero_delta = (-127 - center) & 255
    probabilities = np.zeros(256, dtype=np.float64)
    for delta, probability in enumerate(centered):
        if delta == zero_delta:
            table_index = 0
        elif delta < zero_delta:
            table_index = delta + 1
        else:
            table_index = delta
        probabilities[table_index] += probability

    return probabilities


def gamma_probabilities(shape: float = 0.82, scale: float = 2.43):
    """Return the Gamma magnitude PDF.

    Hardcoded defaults match the best Gamma fit found for the current
    ReLU-style activation data.
    """
    shape = float(shape)
    scale = float(scale)
    log_norm = shape * math.log(scale) + math.lgamma(shape)

    def magnitude_pdf(value):
        value = abs(value)

        return math.exp(
            (shape - 1.0) * math.log(value) - value / scale - log_norm
        )

    return magnitude_pdf


def gaussian_probabilities(std: float = 1.0, mean: float = 0.0):
    """Return the Gaussian value-magnitude PDF."""
    sigma = float(std)

    def base_magnitude_pdf(value):
        return (
            2.0 / (sigma * math.sqrt(2.0 * math.pi))
            * math.exp(-0.5 * (value / sigma) ** 2)
        )

    magnitude_pdf = _shifted_magnitude_pdf(base_magnitude_pdf, mean)
    return magnitude_pdf


def laplace_probabilities(scale: float = 1.0, mean: float = 0.0):
    """Return the Laplace value-magnitude PDF."""
    s = float(scale)

    def base_magnitude_pdf(value):
        return math.exp(-value / s) / s

    magnitude_pdf = _shifted_magnitude_pdf(base_magnitude_pdf, mean)
    return magnitude_pdf


def empirical_probabilities(scale: float = 1.0, mean: float = 0.0):
    """Return the empirical value-magnitude PDF."""
    s = float(scale)
    tail_probability = 0.05
    tail_alpha = 2.8
    tail_start = -s * math.log(tail_probability)

    def base_magnitude_pdf(value):
        if value <= tail_start:
            return math.exp(-value / s) / s
        # Derivative of the power-law tail survival.
        return (
            tail_probability * (tail_alpha - 1.0) / tail_start
            * (tail_start / value) ** tail_alpha
        )

    magnitude_pdf = _shifted_magnitude_pdf(base_magnitude_pdf, mean)
    return magnitude_pdf
