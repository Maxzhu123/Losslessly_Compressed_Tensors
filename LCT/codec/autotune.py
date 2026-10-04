"""Triton launch configurations and bounded-grid limits."""

import torch
import triton

from ..kernels.device import device_profile


ESTIMATE_CENTER_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=2),
]
ENCODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=1, maxnreg=32),
    triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=3)
]
COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=3),
]
# Absolute-row overflow processing is the default on every supported GPU.
COMPACT_EXTRA_AUTOTUNE_CONFIGS = [
    triton.Config({"TILE": 32, "ROW_TILE": 16}, num_warps=4, num_stages=2),
]
POINTWISE_FALLBACK_AUTOTUNE_CONFIGS = [
    triton.Config({"ROW_TILE": 16}, num_warps=4, num_stages=2),
]
DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=5, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=1, maxnreg=48),
    triton.Config({}, num_warps=2, num_stages=1),
]
DUAL_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=3),
]

DENSE_SCATTER_AUTOTUNE_CONFIGS = [
    triton.Config({"ROW_TILE": 16}, num_warps=4, num_stages=2),
]

_gpu = device_profile(torch.device("cuda"))
if "a100" in _gpu.model:
    ENCODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=9, maxnreg=29),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    ]
elif _gpu.model == "geforce_rtx_3070":
    # Winners observed on prepare.py's 10M, 50M and 200M cases.
    ENCODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1, maxnreg=32),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
        triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({}, num_warps=8, num_stages=3, maxnreg=32),
    ]
    DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=5, maxnreg=64),
        triton.Config({}, num_warps=4, num_stages=1, maxnreg=48),
        triton.Config({}, num_warps=4, num_stages=2, maxnreg=48),
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=32),
    ]
elif _gpu.model == "geforce_rtx_4090":
    ENCODE_AUTOTUNE_CONFIGS = [ *ENCODE_AUTOTUNE_CONFIGS,
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    DECODE_AUTOTUNE_CONFIGS = [ *DECODE_AUTOTUNE_CONFIGS,
        triton.Config({}, num_warps=4, num_stages=2),
    ]


# Bound the single-program prefix scan and persistent overflow grids.
SUMMARY_BLOCK_LIMIT = 8192
COMPACT_GRID_LIMIT = 820
SCATTER_GRID_LIMIT = 410

if _gpu.model == "geforce_rtx_4090":
    COMPACT_GRID_LIMIT = 4096
    SCATTER_GRID_LIMIT = 8192
