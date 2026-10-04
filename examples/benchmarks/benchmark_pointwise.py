"""Run with python -m examples.benchmarks.pointwise_ops; edit settings below."""
import csv
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import torch
from tabulate import tabulate

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.benchmarks.benchmark_lct import make_empirical, make_gaussian, make_laplace
from LCT.comp_format import DistType, Distribution
from LCT.comp_tensor import CompressedTensor
from LCT.compress import (
    compress, decompress, compA_add_B, compA_mul_B,
    a_compA_add_B, a_compA_add_b_B, a_compA_add_compB,
)
from LCT.tensor_buffer import TensorBuffer


def release(result):
    if isinstance(result, CompressedTensor):
        result.free()


def two_scale_add(left, right, alpha, beta):
    """Dense baseline with the same FP32 rounding order as the fused kernel."""
    return torch.addcmul(right.float() * beta, left.float(), alpha).to(torch.bfloat16)


def check_result(result, reference, shape):
    """Check every output bit; construct dense references in small chunks."""
    actual = decompress(result) if isinstance(result, CompressedTensor) else result
    assert actual.shape == shape and actual.dtype == torch.bfloat16
    for start in range(0, actual.numel(), 1024 ** 2):
        stop = min(start + 1024 ** 2, actual.numel())
        expected = reference(start, stop)
        assert torch.equal(actual[start:stop].view(torch.int16), expected.view(torch.int16)), (
            f"Bitwise mismatch in elements {start}:{stop}"
        )


def measure(functions, reference, shape, warmup, iterations):
    for _, function in functions:
        result = function()  # First-use compilation and validation are not timed.
        try:
            check_result(result, reference, shape)
        finally:
            release(result)
            del result
        for _ in range(warmup):
            result = function()
            torch.cuda.synchronize()
            release(result)
            del result

    times = {name: [] for name, _ in functions}
    for iteration in range(iterations):
        # Rotate and reverse execution order so each variant takes every position.
        shift = iteration % len(functions)
        ordered = functions[shift:] + functions[:shift]
        if (iteration // len(functions)) % 2:
            ordered = ordered[::-1]
        for name, function in ordered:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) * 1000)
            release(result)  # Buffer release is outside timing.
            del result
    return {
        name: (mean(samples), stdev(samples) / sqrt(len(samples))
               if len(samples) > 1 else float("nan"))
        for name, samples in times.items()
    }


@torch.no_grad()
def main():
    # Edit these settings before running. Each dense operand has tensor_bytes.
    tensor_bytes = 2**29
    family = DistType.GAUSSIAN  # GAUSSIAN, EMPIRICAL, LAPLACE, or GAMMA
    warmup, iterations = 3, 50
    seed = 0
    gaussian_std = 2.0
    empirical_scale = 0.5
    laplace_scale = 1.5
    gamma_shape, gamma_scale = 0.82, 2.43
    scale = -0.5
    beta_scale = -0.02  # Dense-operand scale for a_compA_add_b_B.
    multiplier_mean, multiplier_std = 1.0, 0.01  # Multiplication-only operand.

    if tensor_bytes <= 0 or tensor_bytes % 2:
        raise ValueError("tensor_bytes must be positive and divisible by 2 for BF16")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if not torch.cuda.is_available():
        raise RuntimeError("LCT requires CUDA")
    elements = tensor_bytes // 2
    parameters = {DistType.GAUSSIAN: gaussian_std, DistType.EMPIRICAL: empirical_scale,
                  DistType.LAPLACE: laplace_scale, DistType.GAMMA: 1.0}
    distribution = Distribution(family, param=parameters[family])

    def make_data(data_seed):
        if family == DistType.GAUSSIAN:
            return make_gaussian(elements, std=distribution.param, seed=data_seed)
        if family == DistType.EMPIRICAL:
            return make_empirical(elements, scale=distribution.param, seed=data_seed)
        if family == DistType.LAPLACE:
            return make_laplace(elements, scale=distribution.param, seed=data_seed)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(data_seed)
            gamma = torch.distributions.Gamma(torch.tensor(gamma_shape, device="cuda"),
                                              torch.tensor(1 / gamma_scale, device="cuda"))
            return gamma.sample((elements,)).to(torch.bfloat16)

    a, b = make_data(seed), make_data(seed + 1)
    multiplier = make_gaussian(elements, mean=multiplier_mean, std=multiplier_std, seed=seed + 2)
    alpha = torch.tensor([scale], device="cuda", dtype=torch.float32)
    beta = torch.tensor([beta_scale], device="cuda", dtype=torch.float32)
    capacity = (elements + 64 * 1024 ** 2 + 15) // 16 * 16
    a_buffer, b_buffer, out_buffer = [TensorBuffer(capacity, device="cuda") for _ in range(3)]
    a_comp = compress(a, distribution, a_buffer)
    b_comp = compress(b, distribution, b_buffer)
    check_result(a_comp, lambda start, stop: a[start:stop], a.shape)
    check_result(b_comp, lambda start, stop: b[start:stop], b.shape)

    reference_a = lambda start, stop: a[start:stop]
    reference_add = lambda start, stop: a[start:stop] + b[start:stop]
    reference_mul = lambda start, stop: a[start:stop] * multiplier[start:stop]
    reference_scale_add = lambda start, stop: (
        a[start:stop].float() * alpha + b[start:stop].float()
    ).to(torch.bfloat16)

    def naive(operation, dense_output, other, compressed_other=False):
        left = decompress(a_comp)
        right = decompress(b_comp) if compressed_other else other
        result = operation(left, right)
        if dense_output:
            return result
        return compress(result, distribution, out_buffer)

    # One scaled-add operation, without materialized FP32 intermediates.
    scale_add = lambda left, right: torch.add(right, left, alpha=scale)
    two_scale = lambda left, right: two_scale_add(left, right, alpha, beta)
    reference_two_scale = lambda start, stop: two_scale(a[start:stop], b[start:stop])
    cases = [
        ("compress", "compressed", [("standalone", lambda: compress(a, distribution, out_buffer))], reference_a),
        ("decompress", "dense", [("standalone", lambda: decompress(a_comp))], reference_a),
    ]
    for dense in (True, False):
        output = "dense" if dense else "compressed"
        kwargs = dict(dense_output=dense, buffer=out_buffer, distribution=distribution)
        operations = [
            ("compA_add_B", output, lambda kw=kwargs: compA_add_B(a_comp, b, **kw), reference_add),
            ("compA_mul_B", output, lambda kw=kwargs: compA_mul_B(a_comp, multiplier, **kw), reference_mul),
            ("a_compA_add_B", output, lambda kw=kwargs: a_compA_add_B(a_comp, alpha, b, **kw), reference_scale_add),
            ("a_compA_add_compB", output, lambda kw=kwargs: a_compA_add_compB(a_comp, alpha, b_comp, **kw), reference_scale_add),
            ("a_compA_add_b_B", output, lambda kw=kwargs: a_compA_add_b_B(a_comp, alpha, b, beta, **kw), reference_two_scale),
        ]
        for (name, output_kind, fused, reference), operation in zip(
            operations, (torch.add, torch.mul, scale_add, scale_add, two_scale)
        ):
            other = multiplier if name == "compA_mul_B" else b
            baseline = lambda op=operation, d=dense, rhs=other, dual=name == "a_compA_add_compB": naive(op, d, rhs, dual)
            # Pure dense always returns BF16 dense output, even when compared
            # against the compressed-output paths. It performs no codec work.
            dense_baseline = lambda op=operation, rhs=other: op(a, rhs)
            cases.append((name, output_kind, [
                ("fused", fused), ("naive", baseline), ("dense", dense_baseline),
            ], reference))

    size_label = f"{tensor_bytes / 1024 ** 3:g}gib"
    output = Path(__file__).with_name("results") / f"lct_fused_vs_naive_vs_dense_{family.value}_{size_label}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table = []
    try:
        with output.open("a", newline="") as file:
            writer = csv.writer(file)
            # LCT is the fused path, or standalone compress/decompress.
            if file.tell() == 0:
                writer.writerow(("operation", "tensor_bytes", "output", "lct_mean_ms", "lct_sem_ms",
                                 "naive_mean_ms", "naive_sem_ms", "dense_mean_ms", "dense_sem_ms", "speedup"))
            for name, output_kind, functions, reference in cases:
                results = measure(functions, reference, a.shape, warmup, iterations)
                timings = {
                    variant: f"{avg:.3f} ± {sem:.3f}"
                    for variant, (avg, sem) in results.items()
                }
                timing_line = " | ".join(f"{variant}: {timing} ms" for variant, timing in timings.items())
                print(f"{name} ({output_kind}) | {timing_line}", flush=True)
                table.append((name, output_kind,
                              timings.get("fused", timings.get("standalone", "-")),
                              timings.get("naive", "-"), timings.get("dense", "-")))
                lct_mean, lct_sem = results.get("fused", results.get("standalone"))
                naive_mean, naive_sem = results.get("naive", ("", ""))
                dense_mean, dense_sem = results.get("dense", ("", ""))
                speedup = ""
                if "fused" in results:
                    speedup = naive_mean / lct_mean
                writer.writerow((name, a.nbytes, output_kind, lct_mean, lct_sem,
                                 naive_mean, naive_sem, dense_mean, dense_sem, speedup))
                file.flush()
    finally:
        a_comp.free()
        b_comp.free()
    print(tabulate(table, headers=("Operation", "Output", "LCT (ms)", "Naive (ms)", "Dense (ms)"),
                   tablefmt="simple", colalign=("left", "left", "right", "right", "right")), flush=True)
    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    main()
