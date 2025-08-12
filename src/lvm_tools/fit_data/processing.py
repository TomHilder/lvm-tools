"""processing.py - convenience wrappers for filtering and clipping given LVMTileLike and DataConfig."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

from xarray import Dataset

if TYPE_CHECKING:
    from lvm_tools.config.data_config import DataConfig
from lvm_tools.data.tile import LVMTileLike
from lvm_tools.fit_data.clipping import bounding_square, clip_dataset
from lvm_tools.fit_data.filtering import filter_dataset
from lvm_tools.fit_data.normalisation import calc_normalisation, get_norm_funcs


def clip_data(tile_data: Dataset, config: DataConfig) -> Dataset:
    return clip_dataset(
        tile_data,
        config.λ_range,
        config.α_range,
        config.δ_range,
    )


def filter_tile_data(tile_data: Dataset, config: DataConfig) -> Dataset:
    return filter_dataset(
        tile_data,
        config.nans_strategy,
        config.F_bad_strategy,
        config.F_range,
        config.fibre_status_include,
        config.apply_mask,
    )


def process_tile_data(tiles: LVMTileLike, config: DataConfig) -> Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = clip_data(tiles.data, config)
        return filter_tile_data(ds, config)


def get_αδ_ranges(tiles: LVMTileLike) -> tuple[tuple[float, float], tuple[float, float]]:
    return bounding_square(
        tiles.data["ra"].values.min(),
        tiles.data["ra"].values.max(),
        tiles.data["dec"].values.min(),
        tiles.data["dec"].values.max(),
    )


def get_normalisations(
    ds: Dataset, config: DataConfig
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            calc_normalisation(ds["flux"].values, config.normalise_F_strategy),
            calc_normalisation(ds["ra"].values, config.normalise_αδ_strategy),
            calc_normalisation(ds["dec"].values, config.normalise_αδ_strategy),
        )


def get_normalisation_functions(
    config: DataConfig,
) -> tuple[tuple[Callable, Callable], tuple[Callable, Callable], tuple[Callable, Callable]]:
    return (
        *get_norm_funcs(config.normalise_F_offset, config.normalise_F_scale),
        *get_norm_funcs(0.0, config.normalise_F_scale**-2),
        *get_norm_funcs(config.normalise_α_offset, config.normalise_α_scale),
        *get_norm_funcs(config.normalise_δ_offset, config.normalise_δ_scale),
    )


def flatten_tile_coord(ds: Dataset) -> Dataset:
    """Flatten the tile and spaxel coordinates into a single coordinate."""
    return ds.stack(flat_spaxel=("tile", "spaxel")).reset_index("flat_spaxel")
