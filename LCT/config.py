"""Explicit settings for LCT codecs and kernels.

Edit these defaults or assign attributes before importing LCT's codec, kernel,
or tensor modules::

    from LCT import config
    config.use_tilelang = True
    from LCT.LCTensor import LCTTensor

Lane mapping, geometry, and tuning profiles are selected at import time.
Keep the lane mapping fixed while compressed tensors are in use.
"""

from typing import Literal

# Even XOR masks keep adjacent lane pairs together while mixing columns.
# Linear mapping requires statistically interchangeable columns.
lane_mapping: Literal["swizzle", "linear"] = "swizzle"

# Empty detects the GPU; "generic" selects portable Triton tuning defaults.
# GPU model keys and 3070/3090/4090/5090 aliases select a tuning override.
kernel_backend: str = ""

# Retune TileLang on each program run while retaining compiled kernels.
reset_cache: bool = False

# TileLang must be installed when enabled.
use_tilelang: bool = False

# Show TileLang autotuning progress and logs.
tilelang_verbose: bool = False
