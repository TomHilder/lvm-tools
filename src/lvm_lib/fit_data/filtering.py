"""filtering.py - data filtering for data preparation."""

import warnings
from typing import Literal

import numpy as np
from xarray import Dataset

BAD_FLUX_THRESHOLD = -0.1e-13


ExcludeStrategy = Literal[None, "pixel", "spaxel"]
FibreStatus = Literal[0, 1, 2, 3]  # I have no idea what these mean, but they're in the data


def filter(
    data: Dataset,
    nans_strategy: ExcludeStrategy,
) -> Dataset:
    # Filter nans using strategy
    # Filter bad fluxes using strategy
    # Filter by fibre status
    # Filter using mask
    return data


def filter_inspector(
    data: Dataset,
    F_bad_range: tuple[float, float],
    fibre_status_include: tuple[FibreStatus],
):
    # TODO: maybe plots instead of printing?

    # Inspect the data to see what filters are needed
    # The user can use this before building config and FitData objects

    # nans:
    where_nan = data["flux"].isnull()
    n_nans = int(np.sum(where_nan))
    n_spaxels_nan = int(np.sum(where_nan.any(dim="wavelength")))

    # bad flux (per pix):
    where_Fbad = ~(data["flux"] > F_bad_range[0]) & (data["flux"] < F_bad_range[1])
    n_Fbad = int(np.sum(where_Fbad))
    n_spaxels_Fbad = int(np.sum(where_Fbad.any(dim="wavelength")))

    # bad flux (per spaxel):
    # ignore warnings about median of all nans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        where_Fbad_median_l = data["flux"].median(dim="wavelength") < F_bad_range[0]
        where_Fbad_median_u = data["flux"].median(dim="wavelength") > F_bad_range[1]
        all_nan = data["flux"].isnull().all(dim="wavelength")
        where_Fbad_median = where_Fbad_median_l | where_Fbad_median_u | all_nan
        n_Fbad_median = int(np.sum(where_Fbad_median))

    # fibre status:
    where_badfib = ~data["fibre_status"].isin(fibre_status_include)
    n_spaxels_badfib = int(np.sum(where_badfib))

    # mask:
    where_mask = data["mask"] == 1
    n_mask = int(np.sum(where_mask))
    n_spaxels_mask = int(np.sum(where_mask.any(dim="wavelength")))

    # anything is bad
    where_anybad = where_nan | where_Fbad | where_badfib | where_mask
    n_anybad = int(np.sum(where_anybad))
    n_spaxels_anybad = int(np.sum(where_anybad.any(dim="wavelength")))

    return {
        "nans": (n_nans, n_spaxels_nan),
        "bad flux": (n_Fbad, n_spaxels_Fbad),
        "bad flux median": (n_Fbad_median,),
        "fibre status": (n_spaxels_badfib,),
        "mask": (n_mask, n_spaxels_mask),
        "any bad": (n_anybad, n_spaxels_anybad),
    }
