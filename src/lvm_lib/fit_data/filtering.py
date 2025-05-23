"""filtering.py - data filtering for data preparation."""

import warnings
from typing import Literal

import numpy as np
from xarray import DataArray, Dataset

BAD_FLUX_THRESHOLD = -0.1e-13


ExcludeStrategy = Literal[None, "pixel", "spaxel"]
FibreStatus = Literal[0, 1, 2, 3]  # I have no idea what these mean, but they're in the data


def get_where_nan(arr: DataArray) -> DataArray:
    return arr.isnull()


def get_where_bad(arr: DataArray, bad_range: tuple[float, float]) -> DataArray:
    return ~(arr > bad_range[0]) & (arr < bad_range[1])


def get_where_bad_median(arr: DataArray, bad_range: tuple[float, float]) -> DataArray:
    where_bad_median_l = arr.median(dim="wavelength") < bad_range[0]
    where_bad_median_u = arr.median(dim="wavelength") > bad_range[1]
    all_nan = arr.isnull().all(dim="wavelength")
    return where_bad_median_l | where_bad_median_u | all_nan


def get_where_badfib(fib_stat_arr: DataArray, fibre_status_incl: tuple[FibreStatus]) -> DataArray:
    return ~fib_stat_arr.isin(fibre_status_incl)


def get_where_mask(arr_mask: DataArray) -> DataArray:
    return arr_mask == 1


def combine_wheres(list_where: list[DataArray]) -> DataArray:
    combined_where = list_where[0]
    for where in list_where[1:]:
        combined_where = combined_where | where
    return combined_where


def filter_dataset(
    data: Dataset,
    nans_strategy: ExcludeStrategy,
    F_bad_strategy: ExcludeStrategy,
    F_bad_range: tuple[float, float],
    fibre_status_include: tuple[FibreStatus],
    apply_mask: bool,
) -> Dataset:
    where_bad = []

    # Nans
    if nans_strategy == "pixel":
        pass  # no action needed
    elif nans_strategy == "spaxel":
        where_bad.append(get_where_nan(data["flux"]).any(dim="wavelength"))
    else:
        raise ValueError(f"Unknown nans strategy: {nans_strategy}")

    # Fluxes
    if F_bad_strategy == "pixel":
        where_bad.append(get_where_bad(data["flux"], F_bad_range))
    elif F_bad_strategy == "spaxel":
        where_bad.append(get_where_bad_median(data["flux"], F_bad_range))
    else:
        raise ValueError(f"Unknown bad flux strategy: {F_bad_strategy}")

    # Bad fibres
    where_bad.append(get_where_badfib(data["fibre_status"], fibre_status_include))

    # Filter using mask
    if apply_mask:
        where_bad.append(get_where_mask(data["mask"]))

    return data.where(~combine_wheres(where_bad))


def filter_inspector(
    data: Dataset,
    F_bad_range: tuple[float, float],
    fibre_status_include: tuple[FibreStatus],
):
    # TODO: maybe plots instead of printing?

    # ignore warnings about median of all nans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # nans:
        where_nan = get_where_nan(data["flux"])
        n_nans = int(np.sum(where_nan))
        n_spaxels_nan = int(np.sum(where_nan.any(dim="wavelength")))

        # bad flux (per pix):
        where_Fbad = get_where_bad(data["flux"], F_bad_range)
        n_Fbad = int(np.sum(where_Fbad))

        # bad flux (per spaxel):
        where_Fbad_median = get_where_bad_median(data["flux"], F_bad_range)
        n_Fbad_median = int(np.sum(where_Fbad_median))

        # fibre status:
        where_badfib = get_where_badfib(data["fibre_status"], fibre_status_include)
        n_spaxels_badfib = int(np.sum(where_badfib))

        # mask:
        where_mask = get_where_mask(data["mask"])
        n_mask = int(np.sum(where_mask))
        n_spaxels_mask = int(np.sum(where_mask.any(dim="wavelength")))

        # anything is bad
        where_anybad = where_nan | where_Fbad | where_badfib | where_mask
        n_anybad = int(np.sum(where_anybad))

        where_anybad_spaxel = where_nan | where_Fbad_median | where_badfib | where_mask
        n_spaxels_anybad = int(np.sum(where_anybad_spaxel.any(dim="wavelength")))

    return {
        "nans": (n_nans, n_spaxels_nan),
        "bad flux": (n_Fbad, None),
        "bad flux median": (None, n_Fbad_median),
        "fibre status": (None, n_spaxels_badfib),
        "mask": (n_mask, n_spaxels_mask),
        "any bad": (n_anybad, n_spaxels_anybad),
    }
