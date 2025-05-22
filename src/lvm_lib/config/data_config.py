"""data_config.py - Objects for specifying configuration of data processing before fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

from lvm_lib.config.data_config_calc import bounding_square
from lvm_lib.data.tile import LVMTileLike

BAD_SPAXEL_THRESHOLD = -0.1e-13
NORM_PADDING = 0.05


ExcludeStrategy = Literal[None, "pixel", "spaxel"]
NormaliseStrategy = Literal[None, "max only", "98 only", "extrema", "1σ", "2σ", "3σ", "padded"]

FibreStatus = Literal[0, 1, 2, 3]  # I have no idea what these mean, but they're in the data


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
    fibre_status_include: tuple[FibreStatus] = (0,)
    apply_mask: bool = True
    # Normalisation
    normalise_F_strategy: NormaliseStrategy = "max only"
    normalise_F_offset: float = 0.0
    normalise_F_scale: float = 1.0
    normalise_αδ_strategy: NormaliseStrategy = "padded"
    normalise_αδ_offset: float = 0.0
    normalise_αδ_scale: float = 1.0

    def __post_init__(self) -> None:
        self._validate_range(self.λ_range)
        self._validate_range(self.α_range)
        self._validate_range(self.δ_range)
        self._validate_excl_strategy(self.F_bad_strategy)
        self._validate_range(self.F_bad_range)
        self._validate_excl_strategy(self.nans_strategy)
        self._validate_fib_status_incl(self.fibre_status_include)
        self._validate_apply_mask(self.apply_mask)
        self._validate_norm_strategy(self.normalise_F_strategy)
        self._validate_norm_strategy(self.normalise_αδ_strategy)
        self._validate_offset(self.normalise_F_offset)
        self._validate_scale(self.normalise_F_scale)
        self._validate_norm_strategy(self.normalise_αδ_strategy)
        self._validate_offset(self.normalise_αδ_offset)
        self._validate_scale(self.normalise_αδ_scale)

    @staticmethod
    def default() -> DataConfig:
        return DataConfig()

    @staticmethod
    def from_tiles(tiles: LVMTileLike, **overrides) -> DataConfig:
        # λ_range cannot be set automatically
        # α_range and δ_range we typically want a square region that contains all the spaxels
        α_range, δ_range = bounding_square(
            tiles.data["ra"].min(),
            tiles.data["ra"].max(),
            tiles.data["dec"].min(),
            tiles.data["dec"].max(),
        )
        # F_bad range we typicalling exclude whole spaxels below BAD_SPAXEL_THRESHOLD
        F_bad_range = (BAD_SPAXEL_THRESHOLD, np.inf)
        #

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

    # TODO: move validation functions to a separate module

    @staticmethod
    def _validate_range(x_range: tuple[float, float]) -> None:
        if not isinstance(x_range, tuple):
            raise TypeError("Data range must be in a tuple.")
        if len(x_range) != 2:
            raise ValueError("Data range must be a tuple with exactly two values (min, max).")
        if x_range[1] < x_range[0]:
            raise ValueError("Requested data range restriction has max < min.")

    @staticmethod
    def _validate_excl_strategy(strategy: ExcludeStrategy) -> None:
        if strategy not in get_args(ExcludeStrategy):
            raise ValueError(f"Unknown exclusion strategy: {strategy}")

    @staticmethod
    def _validate_norm_strategy(strategy: NormaliseStrategy) -> None:
        if strategy not in get_args(NormaliseStrategy):
            raise ValueError(f"Unknown normalisation strategy: {strategy}")

    @staticmethod
    def _validate_fib_status_incl(fibre_status_include: tuple[FibreStatus]) -> None:
        if not isinstance(fibre_status_include, tuple):
            raise TypeError("fibre_status_include must be a tuple.")
        for fs in fibre_status_include:
            if fs not in get_args(FibreStatus):
                raise ValueError(f"Unknown fibre status: {fs}")

    @staticmethod
    def _validate_offset(offset: float) -> None:
        if not isinstance(offset, float):
            raise TypeError("offset must be float.")
        if not np.isfinite(offset):
            raise Exception("Bad offset (nan or infty).")

    @staticmethod
    def _validate_scale(scale: float) -> None:
        if not isinstance(scale, float):
            raise TypeError("scale must be float.")
        if not np.isfinite(scale):
            raise Exception("Bad scale (nan or infty).")
        if scale <= 0:
            raise Exception("Scale is not positive, but it must be.")

    @staticmethod
    def _validate_apply_mask(apply_mask: bool) -> None:
        if not isinstance(apply_mask, bool):
            raise TypeError("apply_mask must be a boolean.")
