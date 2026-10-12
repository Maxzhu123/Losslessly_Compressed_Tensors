"""All Triton launch configurations: defaults followed by GPU overrides.

Kernel decorators import these lists directly. Selection happens once when this
module is imported, using the current GPU or config.kernel_backend for experiments.
"""

import torch
import triton

from .. import config
from .device import device_profile


# Portable defaults.
ENCODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=1, maxnreg=32),
    triton.Config({}, num_warps=8, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=3),
]
DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=5, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=2, maxnreg=64),
    triton.Config({}, num_warps=2, num_stages=3, maxnreg=64),
    triton.Config({}, num_warps=4, num_stages=1, maxnreg=48),
    triton.Config({}, num_warps=2, num_stages=1),
]
DECODE_OVERFLOW_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
DUAL_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=8, num_stages=3),
]
COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=3),
]
COMPACT_BLOCK_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=2),
]
COMPACT_EXTRA_AUTOTUNE_CONFIGS = [
    triton.Config({"ROW_TILE": 8}, num_warps=4, num_stages=2),
]
DENSE_SCATTER_AUTOTUNE_CONFIGS = [
    triton.Config({"ROW_TILE": 16}, num_warps=4, num_stages=2),
]
POINTWISE_FALLBACK_AUTOTUNE_CONFIGS = [
    triton.Config({"ROW_TILE": 16}, num_warps=4, num_stages=2),
]

SUMMARY_BLOCK_LIMIT = 8192
COMPACT_GRID_LIMIT = 820
SCATTER_GRID_LIMIT = 410
# Logical elements above which decoding loads payload words on demand.
DECODE_ON_DEMAND_CUTOFF = 1_000_000_000

# Overrides are selected before any kernel's @triton.autotune decorator runs.
GPU_MODEL = device_profile(torch.device("cuda")).model
TUNING_GPU = config.kernel_backend or GPU_MODEL

if "a100" in TUNING_GPU:
    ENCODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=2, maxnreg=32),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=3),
    ]
    DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ]
    DECODE_OVERFLOW_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK": 4096}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=2),
    ]
    COMPACT_EXTRA_AUTOTUNE_CONFIGS = [
        triton.Config({"ROW_TILE": 8}, num_warps=4, num_stages=2),
        triton.Config({"ROW_TILE": 8}, num_warps=2, num_stages=2),
    ]
elif TUNING_GPU in ("geforce_rtx_3070", "3070"):
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
    DECODE_OVERFLOW_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
elif TUNING_GPU in ("geforce_rtx_3090", "3090"):
    DECODE_ON_DEMAND_CUTOFF = 1_000_000_000
    ENCODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
    ]
    DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1, maxnreg=48),
    ]
    DECODE_OVERFLOW_AUTOTUNE_CONFIGS = [
        *DECODE_AUTOTUNE_CONFIGS,
        triton.Config({}, num_warps=8, num_stages=1),
    ]
    POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    DUAL_DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ]
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=2),
    ]
elif TUNING_GPU in ("geforce_rtx_4090", "4090"):
    ENCODE_AUTOTUNE_CONFIGS = [
        *ENCODE_AUTOTUNE_CONFIGS,
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    DECODE_AUTOTUNE_CONFIGS = [
        *DECODE_AUTOTUNE_CONFIGS,
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    DECODE_OVERFLOW_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    COMPACT_GRID_LIMIT = 4096
    SCATTER_GRID_LIMIT = 8192
elif TUNING_GPU in ("geforce_rtx_5090", "5090"):
    ENCODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=8, num_stages=2),
    ]
    DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=4, num_stages=1),
    ]
    DECODE_OVERFLOW_AUTOTUNE_CONFIGS = [
        triton.Config(
            {"PARALLEL_MAP": False, "EMPTY_FAST_PATH": False},
            num_warps=4, num_stages=1, maxnreg=48,
        ),
    ]
    POINTWISE_DENSE_AUTOTUNE_CONFIGS = DECODE_AUTOTUNE_CONFIGS
    DUAL_DECODE_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    COMPACT_BAD_STREAMS_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK": 1024}, num_warps=1, num_stages=2),
    ]
elif config.kernel_backend and TUNING_GPU != "generic":
    raise ValueError(f"unknown autotune GPU override: {TUNING_GPU!r}")
