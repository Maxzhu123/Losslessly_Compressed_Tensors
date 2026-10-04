"""Distribution-aware BF16 exponent-compression benchmark."""

import math
import time
import sys
from pathlib import Path
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LCT.compress import compress, decompress
from LCT.tensor_buffer import TensorBuffer
from LCT.comp_format import DistType, Distribution, NoiseLevel

SIZE_WEIGHTS = {50_000_000: 3, 200_000_000: 1}
WARMUP = 20
ITERS = 100


def make_empirical(n: int, scale: float = 0.5, seed: int = 0) -> torch.Tensor:
    """Sample signed values with an exponential body and power-law tail."""
    G = torch.Generator(device="cuda").manual_seed(seed)
    tail_probability = 0.05
    tail_alpha = 2.8
    tail_start = -scale * math.log(tail_probability)
    u = torch.rand(n, device="cuda", dtype=torch.float32, generator=G)
    body = u < (1.0 - tail_probability)
    values = torch.empty_like(u)
    values[body] = -scale * torch.log1p(-u[body])
    values[~body] = tail_start * (
        tail_probability / (1.0 - u[~body])
    ) ** (1.0 / (tail_alpha - 1.0))
    signs = torch.randint(
        0, 2, (n,), device="cuda", dtype=torch.int8, generator=G
    ).to(torch.float32)
    return (values * (signs * 2.0 - 1.0)).to(torch.bfloat16)


def make_gaussian_values(
    n: int, mean: float = 0.0, std: float = 2.0, seed: int = 0,
) -> torch.Tensor:
    G = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(n, device="cuda", dtype=torch.float32, generator=G)
    return values * std + mean


def make_gaussian(
    n: int, mean: float = 0.0, std: float = 2.0, seed: int = 0,
) -> torch.Tensor:
    return make_gaussian_values(n, mean=mean, std=std, seed=seed).to(torch.bfloat16)


def make_laplace(
    n: int, scale: float = 1.5, seed: int = 0,
) -> torch.Tensor:
    G = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.rand(n, device="cuda", dtype=torch.float32, generator=G) - 0.5
    values = -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())
    return values.to(torch.bfloat16)


def make_localized_noise(
    n: int, noise_fraction: float = 0.2, seed: int = 0,
) -> torch.Tensor:
    """Gaussian values with a contiguous uniform-value noise region."""
    values = make_gaussian_values(n, mean=0.0, std=2.0, seed=seed)
    start = n // 2
    end = min(n, start + int(n * noise_fraction))
    G = torch.Generator(device="cuda").manual_seed(seed + 1)
    values[start:end] = torch.empty(
        end - start, device="cuda", dtype=torch.float32
    ).uniform_(-32.0, 32.0, generator=G)
    return values.to(torch.bfloat16)


def _bf16_ratio(exponent_ratio: float) -> float:
    """Convert an exponent-stream ratio to a total BF16 storage ratio."""
    return (1.0 + exponent_ratio) / 2.0


# Each case: (source/codec_family/noise_level, max_total_bf16_ratio)
CASES: list[tuple[str, float]] = [
    ("gaussian/gaussian/clean", _bf16_ratio(0.4)),
    ("laplace/laplace/medium", _bf16_ratio(0.61)),
    ("gaussian/empirical/clean", _bf16_ratio(0.5)),
    ("laplace/gaussian/clean", _bf16_ratio(0.66)),
    ("localized/empirical/high", _bf16_ratio(0.78)),
]


def make_data(name: str, n: int, distribution: Distribution | None = None) -> torch.Tensor:
    if name not in {case[0] for case in CASES}:
        raise ValueError(f"unknown case: {name}")

    source = name.split("/", 1)[0]
    if source == "empirical":
        scale = distribution.param if distribution is not None else 0.5
        return make_empirical(n, scale)
    if source == "gaussian":
        return make_gaussian(n)
    if source == "laplace":
        return make_laplace(n)
    if source == "shifted_gaussian":
        return make_gaussian(n, mean=50.0)
    if source == "localized":
        return make_localized_noise(n)
    raise ValueError(f"unknown case: {name}")


def run_case(
    name: str, n: int, max_ratio: float, buffer: TensorBuffer,
) -> float:
    _, family, noise = name.split("/")
    distribution = Distribution(DistType(family), noise_level=NoiseLevel[noise.upper()])
    x = make_data(name, n, distribution)

    # Correctness pass: allocate, decode, then release the buffer regions.
    compressed = compress(x, distribution=distribution, buffer=buffer)
    restored = decompress(compressed)
    assert torch.equal(x, restored), f"roundtrip mismatch: {name}"
    compressed.free()

    for _ in range(WARMUP):
        compressed = compress(x, distribution=distribution, buffer=buffer)
        restored = decompress(compressed)
        compressed.free()

    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(ITERS):
        compressed = compress(x, distribution=distribution, buffer=buffer)
        # Keep the final result for decoding and the compression-ratio check.
        if i != ITERS - 1:
            compressed.free()
    torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - start) / ITERS * 1000.0

    start = time.perf_counter()
    for _ in range(ITERS):
        restored = decompress(compressed)
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) / ITERS * 1000.0
    elapsed_ms = encode_ms + decode_ms

    ratio = compressed.memory_size() / x.nbytes
    assert ratio <= max_ratio, (
        f"{name} n={n}: ratio {ratio:.4f} exceeds {max_ratio:.4f}"
    )

    print(
        f"{name:32s} n={n / 1e6:6.0f}M  "
        f"encode={encode_ms:7.3f} ms  "
        f"decode={decode_ms:7.3f} ms  "
        f"total={elapsed_ms:7.3f} ms"
    )
    compressed.free()
    del x, compressed, restored
    torch.cuda.empty_cache()
    return elapsed_ms


def main() -> None:
    print(f"SHAPES = {list(SIZE_WEIGHTS)}")

    # Persistent buffer used by the codec for fallback storage.
    buffer = TensorBuffer(
        max(SIZE_WEIGHTS) + 64 * 1024 * 1024,
        device="cuda",
    )

    weighted_time = 0.0
    total_time = 0.0
    total_weight = 0
    for n, weight in SIZE_WEIGHTS.items():
        for name, max_ratio in CASES:
            elapsed_ms = run_case(name, n, max_ratio, buffer)
            weighted_time += weight * elapsed_ms
            total_time += elapsed_ms
            total_weight += weight


    print("Passed")
    task_count = len(CASES) * len(SIZE_WEIGHTS)
    print(f"Average time per task: {total_time / task_count:.5g}ms")
    print(f"Final time: {weighted_time / total_weight:.5g}ms")


if __name__ == "__main__":
    main()
