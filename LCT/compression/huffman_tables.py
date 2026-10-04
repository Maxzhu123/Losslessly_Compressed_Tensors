import numpy as np
from functools import lru_cache
import torch

from .probabilities import (
    _exponent_probabilities,
    empirical_probabilities,
    gamma_probabilities,
    gaussian_probabilities,
    laplace_probabilities,
)
from ..comp_format import DistType, Distribution

FIRST_BITS = 10
FIRST_MASK = (1 << FIRST_BITS) - 1


def _length_limited_huffman_lengths(
    weights,
    max_length=FIRST_BITS,
):
    """Optimal length-limited Huffman code lengths via Package-Merge."""
    weights = np.asarray(weights, dtype=np.longdouble)
    n = len(weights)

    if n == 0:
        return np.empty(0, dtype=np.int16)
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative and not all zero")
    if n > (1 << max_length):
        raise ValueError("Too many symbols for requested maximum length")

    if n <= 2:
        return np.ones(n, dtype=np.int16)

    # Package-Merge works with ascending weights.
    order = np.argsort(weights, kind="stable")
    original = weights[order].tolist()
    previous = original

    rows = []

    for _ in range(max_length - 1):
        m = len(previous) & ~1
        packages = [
            previous[i] + previous[i + 1]
            for i in range(0, m, 2)
        ]

        current = []
        is_package = []

        i = j = 0
        while i < n and j < len(packages):
            if original[i] <= packages[j]:
                current.append(original[i])
                is_package.append(False)
                i += 1
            else:
                current.append(packages[j])
                is_package.append(True)
                j += 1

        if i < n:
            current.extend(original[i:])
            is_package.extend([False] * (n - i))

        if j < len(packages):
            current.extend(packages[j:])
            is_package.extend([True] * (len(packages) - j))

        rows.append(is_package)
        previous = current

    lengths_sorted = np.zeros(n, dtype=np.int16)
    num_analyze = 2 * n - 2

    for is_package in reversed(rows):
        num_merged = 0
        symbol = 0

        for packaged in is_package[:num_analyze]:
            if packaged:
                num_merged += 1
            else:
                lengths_sorted[symbol] += 1
                symbol += 1

        num_analyze = 2 * num_merged

    lengths_sorted[:num_analyze] += 1

    lengths = np.empty(n, dtype=np.int16)
    lengths[order] = lengths_sorted
    return lengths


def _optimal_escape_solution(probabilities, max_length=FIRST_BITS, max_esc_length=8):
    """Exact optimal escape-cutoff solution using Package-Merge.

    ``max_esc_length`` is accepted for API compatibility.  The package-merge
    implementation uses a single global max length; when an ESC cap is given
    we use it as the global cap so rare codes also stay within the existing
    decoder constraints.
    """
    if max_esc_length is not None:
        max_length = min(max_length, max_esc_length)
    p = np.asarray(probabilities, dtype=np.longdouble)
    p /= p.sum()
    n = len(p)

    order = np.argsort(-p, kind="stable")
    ps = p[order]

    tail = np.zeros(n + 1, dtype=np.longdouble)
    tail[1:] = np.cumsum(ps[::-1], dtype=np.longdouble)

    # No ESC case.
    lengths = _length_limited_huffman_lengths(ps, max_length)
    best_cost = float(np.dot(ps, lengths.astype(np.longdouble)))
    best_k = 0
    best_lengths = lengths
    best_pesc = 0.0

    for k in range(1, n):
        num_direct = n - k
        p_esc = float(tail[k])

        weights = np.empty(num_direct + 1, dtype=np.longdouble)
        weights[:num_direct] = ps[:num_direct]
        weights[-1] = tail[k]

        lengths = _length_limited_huffman_lengths(weights, max_length)

        cost = float(
            np.dot(weights, lengths.astype(np.longdouble)) + 8.0 * tail[k]
        )

        if cost < best_cost:
            best_cost = cost
            best_k = k
            best_lengths = lengths.copy()
            best_pesc = p_esc

    if best_k == 0:
        direct_indices = order
        escaped_indices = np.empty(0, dtype=np.int64)
        direct_lengths = best_lengths
        esc_length = 0
    else:
        num_direct = n - best_k
        direct_indices = order[:num_direct]
        escaped_indices = order[num_direct:]
        direct_lengths = best_lengths[:num_direct]
        esc_length = int(best_lengths[-1])

    return {
        "expected_bits": best_cost,
        "k": best_k,
        "direct_indices": direct_indices,
        "escaped_indices": escaped_indices,
        "direct_lengths": direct_lengths,
        "esc_length": esc_length,
        "escape_probability": best_pesc,
    }


def _canonical_codes(lengths):
    """Build canonical prefix codes from a length vector."""
    n = len(lengths)
    max_len = max(lengths) if n else 0
    counts = [0] * (max_len + 1)
    for length in lengths:
        counts[length] += 1
    next_code = [0] * (max_len + 1)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + counts[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for idx in sorted(range(n), key=lambda i: (lengths[i], i)):
        length = lengths[idx]
        codes[idx] = (next_code[length], length)
        next_code[length] += 1
    return codes


def _reverse_bits(value, length):
    result = 0
    for _ in range(length):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _build_huffman_tables_from_lengths(probabilities, max_length=FIRST_BITS, max_esc_length=8):
    """Exact optimal escape-cutoff Huffman tables with a fixed zero symbol.

    Table index 0 is reserved for exact zero.  The remaining table indices are
    assigned to the 255 nonzero centered exponent deltas.  The returned encode
    table is indexed by this table-index space; the kernels perform the small
    fixed-zero remapping in registers.
    """
    solution = _optimal_escape_solution(
        probabilities, max_length=max_length, max_esc_length=max_esc_length
    )
    direct_indices = solution["direct_indices"]
    escaped_indices = solution["escaped_indices"]
    direct_lengths = solution["direct_lengths"]
    esc_length = solution["esc_length"]

    # Lengths for all direct symbols plus ESC as the last symbol.
    all_lengths = np.concatenate([direct_lengths, [esc_length]])
    codes = _canonical_codes(all_lengths)
    esc_code, _ = codes[len(direct_lengths)]  # ESC is the last symbol

    # Build codewords for the 256 table indices.
    codewords = {}
    for pos, raw in enumerate(direct_indices):
        codewords[raw] = codes[pos]
    for raw in escaped_indices:
        codewords[raw] = (
            (esc_code << 8) | _reverse_bits(raw & 255, 8),
            esc_length + 8,
        )

    encode = []
    for raw in range(256):
        code, length = codewords[raw]
        encode.append(_reverse_bits(code, length) | (length << 20))

    decode = [0] * (1 << FIRST_BITS)
    for symbol, (code, length) in codewords.items():
        reversed_code = _reverse_bits(code, length)
        packed = length | (symbol << 8)
        if length <= FIRST_BITS:
            for suffix in range(1 << (FIRST_BITS - length)):
                decode[reversed_code | (suffix << length)] = packed
        else:
            decode[reversed_code & FIRST_MASK] = 0

    return encode, decode, esc_length


def get_distribution_tables(dist: Distribution):
    """Return cached tables for a distribution, independent of noise level."""
    if not isinstance(dist, Distribution):
        raise TypeError("distribution must be a Distribution instance")
    return _get_distribution_tables(
        dist.family, dist.param, dist.mean, dist.zero_prob,
    )


@lru_cache(maxsize=None)
def _get_distribution_tables(
    family: DistType,
    param: float,
    mean: float,
    zero_prob: float,
):
    if family == DistType.EMPIRICAL:
        magnitude_cdf = empirical_probabilities(param, mean)
    elif family == DistType.GAUSSIAN:
        magnitude_cdf = gaussian_probabilities(param, mean)
    elif family == DistType.LAPLACE:
        magnitude_cdf = laplace_probabilities(param, mean)
    elif family == DistType.GAMMA:
        # Best hardcoded Gamma fit for the observed activation distribution.
        magnitude_cdf = gamma_probabilities()
    else:  # pragma: no cover - guarded by Distribution validation
        raise ValueError(f"unknown distribution family: {family!r}")
    probabilities = _exponent_probabilities(magnitude_cdf, zero_prob)

    encode, decode, rare_length = _build_huffman_tables_from_lengths(
        probabilities, max_length=FIRST_BITS, max_esc_length=8,
    )
    decode_tensor = torch.tensor(decode, dtype=torch.int32, device="cuda")
    encode_tensor = torch.tensor(encode, dtype=torch.int32, device="cuda")
    return encode_tensor, decode_tensor, rare_length
