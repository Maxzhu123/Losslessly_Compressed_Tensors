"""Shared TileLang logging and representative-input tuning utilities."""

import logging
from typing import Any

import torch
from tilelang.autotuner import set_autotune_inputs
from tilelang.autotuner.tuner import _normalize_value
from tilelang.env import env

from ... import config


class _QuietTqdm:
    """No-op stand-in for the tuner's ``tqdm``, silencing bars and writes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.n = 0
        self.total = kwargs.get("total", 0)

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, *args: object, **kwargs: object) -> None:
        pass

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_QuietTqdm":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    @staticmethod
    def write(*args: object, **kwargs: object) -> None:
        pass


def configure_autotune() -> None:
    """Apply LCT's cache and logging settings before creating kernel tuners."""
    # TileLang's runtime override leaves the process environment untouched.
    env.TILELANG_AUTO_TUNING_DISABLE_CACHE = "1" if config.reset_cache else "0"
    if config.tilelang_verbose:
        return

    logging.getLogger("tilelang.autotuner.tuner").setLevel(logging.ERROR)
    from tilelang.autotuner import tuner as _tuner

    _tuner.tqdm = _QuietTqdm


def launch_autotuned(kernel: Any, inputs: tuple[torch.Tensor, ...], args: tuple[object, ...],
                     kwargs: dict[str, object]) -> None:
    """Tune on real inputs once, then directly launch the native cached winner."""
    key = (_normalize_value(args, sort_dict_items=True),
           _normalize_value(kwargs, sort_dict_items=True))
    cached = kernel._tuner_cache.get(key)
    if cached is None:
        with set_autotune_inputs(*inputs):
            compiled = kernel.compile(*args, **kwargs)
    else:
        compiled = cached[0]
    compiled(*inputs)
