"""data_config.py - Objects for specifying configuration of data processing before fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lvm_tools.config.validation import (
    validate_apply_mask,
    validate_excl_strategy,
    validate_fib_status_incl,
    validate_norm_strategy,
    validate_offset,
    validate_range,
    validate_scale,
)
from lvm_tools.data.tile import LVMTileLike
from lvm_tools.fit_data.filtering import (
    BAD_FLUX_THRESHOLD,
    ExcludeStrategy,
    FibreStatus,
)
from lvm_tools.fit_data.normalisation import NormaliseStrategy
from lvm_tools.fit_data.processing import get_normalisations, get_αδ_ranges, process_tile_data


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
        validate_range(self.λ_range)
        validate_range(self.α_range)
        validate_range(self.δ_range)
        validate_excl_strategy(self.nans_strategy)
        validate_excl_strategy(self.F_bad_strategy)
        validate_range(self.F_range)
        validate_fib_status_incl(self.fibre_status_include)
        validate_apply_mask(self.apply_mask)
        validate_norm_strategy(self.normalise_F_strategy)
        validate_norm_strategy(self.normalise_αδ_strategy)
        validate_offset(self.normalise_F_offset)
        validate_scale(self.normalise_F_scale)
        validate_norm_strategy(self.normalise_αδ_strategy)
        validate_offset(self.normalise_α_offset)
        validate_scale(self.normalise_α_scale)
        validate_offset(self.normalise_δ_offset)
        validate_scale(self.normalise_δ_scale)

    @staticmethod
    def default() -> DataConfig:
        return DataConfig()

    @staticmethod
    def from_tiles(
        tiles: LVMTileLike,
        λ_range: tuple[float, float] = (-np.inf, np.inf),
        **overrides,
    ) -> DataConfig:
        # λ_range cannot be set automatically
        α_range, δ_range = get_αδ_ranges(tiles)

        # Instantiate a data config with calc'd + default + overrides
        config_dict = DataConfig(λ_range=λ_range, α_range=α_range, δ_range=δ_range).to_dict()
        config = DataConfig.from_dict(config_dict | overrides)

        # Clip and filter the data
        ds = process_tile_data(tiles, config)

        # Calculate the normalisation parameters
        (
            (normalise_F_offset, normalise_F_scale),
            (normalise_α_offset, normalise_α_scale),
            (normalise_δ_offset, normalise_δ_scale),
        ) = get_normalisations(ds, config)

        # We want a square domain in the α, δ plane
        norm_αδ_scale = max(normalise_α_scale, normalise_δ_scale)

        # Update the config with the calculated values
        norm_overrides = {
            "normalise_F_offset": normalise_F_offset,
            "normalise_F_scale": normalise_F_scale,
            "normalise_α_offset": normalise_α_offset,
            "normalise_α_scale": norm_αδ_scale,
            "normalise_δ_offset": normalise_δ_offset,
            "normalise_δ_scale": norm_αδ_scale,
        }

        # Merge partial config + norm + user overrides, with user overrides taking precedence
        return DataConfig.from_dict(config.to_dict() | norm_overrides | overrides)

    @staticmethod
    def from_dict(config: dict) -> DataConfig:
        if len(config) != len(DataConfig.default().to_dict()):
            raise ValueError("config has the wrong number of entries.")
        return DataConfig(**config)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        def format_tuple(t):
            """Format tuple with floats to 2 decimal places or scientific notation if very small."""
            formatted = []
            for x in t:
                if isinstance(x, float):
                    if abs(x) < 1e-3 and x != 0:
                        formatted.append(f"{x:.2e}")
                    else:
                        formatted.append(f"{x:.2f}")
                else:
                    formatted.append(str(x))
            return f"({', '.join(formatted)})"

        def format_float(f):
            """Format float to 2 decimal places, or scientific notation if very small."""
            if abs(f) < 1e-3 and f != 0:
                return f"{f:.2e}"
            else:
                return f"{f:.2f}"

        lines = [f"{self.__class__.__name__}("]

        lines = [f"{self.__class__.__name__} ({hex(id(self))}):"]

        pad = 26

        # Data clipping ranges
        lines.append("    Data clipping ranges:")
        lines.append(f"        {'λ_range:':{pad}}{format_tuple(self.λ_range)}")
        lines.append(f"        {'α_range:':{pad}}{format_tuple(self.α_range)}")
        lines.append(f"        {'δ_range:':{pad}}{format_tuple(self.δ_range)}")

        # Bad data handling
        lines.append("    Bad data handling:")
        lines.append(f"        {'nans_strategy:':{pad}}'{self.nans_strategy}'")
        lines.append(f"        {'F_bad_strategy:':{pad}}'{self.F_bad_strategy}'")
        lines.append(f"        {'F_range:':{pad}}{format_tuple(self.F_range)}")

        # Flagged data handling
        lines.append("    Flagged data handling:")
        lines.append(f"        {'fibre_status_include:':{pad}}{self.fibre_status_include}")
        lines.append(f"        {'apply_mask:':{pad}}{self.apply_mask}")

        # Flux normalisation
        lines.append("    Flux normalisation:")
        lines.append(f"        {'normalise_F_strategy:':{pad}}'{self.normalise_F_strategy}'")
        lines.append(
            f"        {'normalise_F_offset:':{pad}}{format_float(self.normalise_F_offset)}"
        )
        lines.append(f"        {'normalise_F_scale:':{pad}}{format_float(self.normalise_F_scale)}")

        # Coordinate normalisation
        lines.append("    Coordinate normalisation:")
        lines.append(f"        {'normalise_αδ_strategy:':{pad}}'{self.normalise_αδ_strategy}'")
        lines.append(
            f"        {'normalise_α_offset:':{pad}}{format_float(self.normalise_α_offset)}"
        )
        lines.append(f"        {'normalise_α_scale:':{pad}}{format_float(self.normalise_α_scale)}")
        lines.append(
            f"        {'normalise_δ_offset:':{pad}}{format_float(self.normalise_δ_offset)}"
        )
        lines.append(f"        {'normalise_δ_scale:':{pad}}{format_float(self.normalise_δ_scale)}")

        return "\n".join(lines)
