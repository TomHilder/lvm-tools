"""data_config.py - Objects for specifying configuration of data processing before fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import get_args

import numpy as np

from lvm_lib.data.tile import LVMTileLike
from lvm_lib.fit_data.clipping import bounding_square, clip_dataset
from lvm_lib.fit_data.filtering import (
    BAD_FLUX_THRESHOLD,
    ExcludeStrategy,
    FibreStatus,
    filter_dataset,
)
from lvm_lib.fit_data.normalisation import NormaliseStrategy, calc_normalisation


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration object for data processing before fitting.

    args:
        λ_range: tuple[float, float] - Wavelength range to include.
        α_range: tuple[float, float] - Right Ascension range to include.
        δ_range: tuple[float, float] - Declination range to include.
        nans_strategy: ExcludeStrategy - Strategy for handling NaN values.
        F_bad_strategy: ExcludeStrategy - Strategy for handling bad flux values. For "pixel", the flux range is applied to each pixel. For "spaxel", the flux range is applied to the median of all pixels in a spaxel.
        F_range: tuple[float, float] - Flux range to include.
        fibre_status_include: tuple[FibreStatus] - Fibre status values to include.
        apply_mask: bool - Whether to apply a mask to the data.
        normalise_F_strategy: NormaliseStrategy - Strategy for normalising flux data.
        normalise_F_offset: float - Offset for normalising flux data.
        normalise_F_scale: float - Scale for normalising flux data.
        normalise_αδ_strategy: NormaliseStrategy - Strategy for normalising α and δ data.
        normalise_αδ_offset: float - Offset for normalising α and δ data.
        normalise_αδ_scale: float - Scale for normalising α and δ data.
    """

    # Data clipping ranges (aka choose data of interest)
    λ_range: tuple[float, float] = (-np.inf, np.inf)
    α_range: tuple[float, float] = (-np.inf, np.inf)
    δ_range: tuple[float, float] = (-np.inf, np.inf)
    # Bad data ranges and strategies (aka exclude bad data)
    nans_strategy: ExcludeStrategy = "pixel"
    F_bad_strategy: ExcludeStrategy = "spaxel"
    F_range: tuple[float, float] = (BAD_FLUX_THRESHOLD, np.inf)
    # Handling of flagged data
    fibre_status_include: tuple[FibreStatus] = (0,)
    apply_mask: bool = True
    # Normalisation
    normalise_F_strategy: NormaliseStrategy = "max only"
    normalise_F_offset: float = 0.0
    normalise_F_scale: float = 1.0
    normalise_αδ_strategy: NormaliseStrategy = "padded"
    normalise_α_offset: float = 0.0
    normalise_α_scale: float = 1.0
    normalise_δ_offset: float = 0.0
    normalise_δ_scale: float = 1.0

    def __post_init__(self) -> None:
        self._validate_range(self.λ_range)
        self._validate_range(self.α_range)
        self._validate_range(self.δ_range)
        self._validate_excl_strategy(self.nans_strategy)
        self._validate_excl_strategy(self.F_bad_strategy)
        self._validate_range(self.F_range)
        self._validate_fib_status_incl(self.fibre_status_include)
        self._validate_apply_mask(self.apply_mask)
        self._validate_norm_strategy(self.normalise_F_strategy)
        self._validate_norm_strategy(self.normalise_αδ_strategy)
        self._validate_offset(self.normalise_F_offset)
        self._validate_scale(self.normalise_F_scale)
        self._validate_norm_strategy(self.normalise_αδ_strategy)
        self._validate_offset(self.normalise_α_offset)
        self._validate_scale(self.normalise_α_scale)
        self._validate_offset(self.normalise_δ_offset)
        self._validate_scale(self.normalise_δ_scale)

    @staticmethod
    def default() -> DataConfig:
        return DataConfig()

    @staticmethod
    def from_tiles(
        tiles: LVMTileLike, λ_range: tuple[float, float] = (-np.inf, np.inf), **overrides
    ) -> DataConfig:
        # λ_range cannot be set automatically
        # α_range and δ_range we typically want a square region that contains all the spaxels
        α_range, δ_range = bounding_square(
            tiles.data["ra"].min(),
            tiles.data["ra"].max(),
            tiles.data["dec"].min(),
            tiles.data["dec"].max(),
        )

        # Instantiate a data config with calc'd + default + overrides
        config = DataConfig(
            λ_range=λ_range,
            α_range=α_range,
            δ_range=δ_range,
            **overrides,
        )

        # Clip then filter the data
        ds = clip_dataset(tiles.data, config.λ_range, config.α_range, config.δ_range)
        ds = filter_dataset(
            tiles.data,
            config.nans_strategy,
            config.F_bad_strategy,
            config.F_range,
            config.fibre_status_include,
            config.apply_mask,
        )

        # Calculate the normalisation parameters
        normalise_F_offset, normalise_F_scale = calc_normalisation(
            ds["flux"].values, config.normalise_F_strategy
        )
        normalise_α_offset, normalise_α_scale = calc_normalisation(
            ds["ra"].values, config.normalise_αδ_strategy
        )
        normalise_δ_offset, normalise_δ_scale = calc_normalisation(
            ds["dec"].values, config.normalise_αδ_strategy
        )

        # Update the config with the calculated values
        norm_overrides = {
            "normalise_F_offset": normalise_F_offset,
            "normalise_F_scale": normalise_F_scale,
            "normalise_α_offset": normalise_α_offset,
            "normalise_α_scale": normalise_α_scale,
            "normalise_δ_offset": normalise_δ_offset,
            "normalise_δ_scale": normalise_δ_scale,
        }

        # Merge partial config + norm + user overrides, with user overrides taking precedence
        config_dict = DataConfig(...).to_dict()
        new_config_dict = config_dict | norm_overrides | overrides
        return DataConfig.from_dict(new_config_dict)

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
