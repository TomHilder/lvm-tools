"""data_config.py - Objects for specifying configuration of data processing before fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from lvm_lib.data.tile import LVMTileLike

BAD_SPAXEL_THRESHOLD = -0.1e-13
NORM_PADDING = 0.05


ExcludeStrategy = Literal[None, "pixel", "spaxel"]
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
    return offset, scale


def normalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    return (data - offset) / scale


def denormalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    return data * scale + offset


@dataclass(frozen=True)
class DataConfig:
    # Data truncation ranges (aka choose data of interest)
    λ_range: tuple[float, float] = (-np.inf, np.inf)
    α_range: tuple[float, float] = (-np.inf, np.inf)
    δ_range: tuple[float, float] = (-np.inf, np.inf)
    # Bad data ranges and strategies (aka exclude bad data)
    F_bad_strategy: ExcludeStrategy = "spaxel"
    F_bad_range: tuple[float, float] = (-np.inf, np.inf)
    nans_strategy: ExcludeStrategy = "pixel"
    # Handling of flagged data
    fibre_status_include: tuple[int] = (0, 1, 2, 3, 4, 5)
    apply_mask: bool = True
    # Normalisation
    normalise_F_strategy: NormaliseStrategy = "max only"
    normalise_F_offset: float = 0.0
    normalise_F_scale: float = 1.0
    normalise_αδ_strategy: NormaliseStrategy = "padded"
    normalise_αδ_offset: float = 0.0
    normalise_αδ_scale: float = 1.0

    def __post_init__(self) -> None:
        # Validate the config
        # - ranges must be strictly increasing
        # - strategies exists
        # (assume normalisation was already calculated)
        pass

    @staticmethod
    def default() -> DataConfig:
        return DataConfig()

    @staticmethod
    def from_tiles(tiles: LVMTileLike, **overrides) -> DataConfig:
        calculated_config = DataConfig(...).to_dict()
        overrides_applied = calculated_config | overrides
        return DataConfig.from_dict(overrides_applied)

    @staticmethod
    def from_dict(config: dict) -> DataConfig:
        if len(config) != len(DataConfig.default().to_dict()):
            raise ValueError("config has the wrong number of entries.")
        return DataConfig(**config)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _validate_range(x_range: tuple[float, float]) -> None:
        if x_range[1] < x_range[0]:
            raise ValueError("Requested data range restriction has max < min.")
