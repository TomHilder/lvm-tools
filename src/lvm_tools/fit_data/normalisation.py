"""normalisation.py - data normalisation for data preparation."""

from functools import partial
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

NORM_PADDING = 0.01


NormaliseStrategy = Literal[None, "max only", "98 only", "extrema", "1σ", "2σ", "3σ", "padded"]


def calc_normalisation(data: ArrayLike, strategy: NormaliseStrategy) -> tuple[float, float]:
    offset = 0.0
    scale = 1.0
    if strategy is None:
        pass
    elif strategy == "max only":
        scale = np.nanmax(data)
    elif strategy == "98 only":
        scale = np.nanpercentile(data, 98)
    elif strategy == "extrema":
        offset = np.nanmin(data)
        scale = np.nanmax(data) - offset
    elif strategy in ("1σ", "2σ", "3σ"):
        offset = np.nanmean(data)
        scale = 2.0 * int(strategy[0]) * np.nanstd(data)
    elif strategy == "padded":
        data_range = np.nanmax(data) - np.nanmin(data)
        offset = np.nanmin(data) - NORM_PADDING * data_range
        scale = (1 + 2 * NORM_PADDING) * data_range
    else:
        raise ValueError(f"Unknown normalisation strategy: {strategy}")
    return float(offset), float(scale)


def normalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    return (data - offset) / scale


def denormalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    return data * scale + offset


def get_norm_funcs(offset: float, scale: float) -> Callable:
    return (partial(f, offset=offset, scale=scale) for f in (normalise, denormalise))
