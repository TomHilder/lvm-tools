# fit_data.py
# Written by Thomas Hilder

from dataclasses import dataclass

import astropy.units as unit
import numpy as np
from numpy.typing import NDArray

from lvm_lib.tile import FLUX_UNIT, Tile

BAD_SPAXEL_THRESHOLD = -0.1e-13 * FLUX_UNIT


def get_property(tiles: list[Tile], property_name: str) -> list:
    stack = np.stack([getattr(tile, property_name) for tile in tiles])
    return stack


def get_property_flat(tiles: list[Tile], property_name: str) -> NDArray:
    return get_property(tiles, property_name).flatten()


@dataclass(frozen=True)
class DataConfig:
    λ_range: tuple[float, float]
    α_range: tuple[float, float]
    δ_range: tuple[float, float]
    F_bad_thresh: float = -np.inf
    remove_nans: bool = True


@dataclass(frozen=True)
class FitData:
    tiles = list[Tile]

    def __post_init__(self):
        # Brainstorm of checks:
        # - Check the units are correct
        # - Check the wavelengths are the same in all tiles
        # - Check the shapes are correct (between attributes and between tiles)



    def get_arrays(self, data_spec: DataSpec) -> tuple[NDArray, NDArray]:
        # Does all the filtering and stuff required as specified in the data spec
        # Then returns all the arrays
        pass


class FitDataOld:
    def __init__(self, lvm_tiles: list[LVMTile], line: str, n_lambda: int) -> None:
        # Check that the line exists
        if line not in LINE_CENTRES.keys():
            raise Exception("Unknown line.")
        # Get line centre index
        line_centre = LINE_CENTRES[line]

        # Save tile(s)
        self._tiles = lvm_tiles

        # Check that the wavelengths are the same in all tiles
        for tile in self._tiles:
            if not np.allclose(tile.wavelength, self._tiles[0].wavelength, rtol=1e-8):
                raise Exception("Not all tiles contain the same wavelengths (bad)")
            else:
                pass

        # Finding the index of the line of interest
        self.i_line = np.argmin(np.abs(self._tiles[0].wavelength - line_centre))
        self.i_lower = self.i_line - n_lambda // 2
        self.i_upper = self.i_line + n_lambda // 2

        # Get normalisation for coords
        ra_vals = get_property_flat(self._tiles, "ra")
        dec_vals = get_property_flat(self._tiles, "dec")
        ra_range, dec_range = map(np.ptp, (ra_vals, dec_vals))
        self.coord_norm = max(ra_range, dec_range)
        self.ra_min = np.nanmin(ra_vals)
        self.dec_min = np.nanmin(dec_vals)

        # Get normalisation for flux and bad spaxels
        fluxes = get_property(self._tiles, "flux")
        self.flux_norm = np.nanmax(fluxes.flatten())

        # Get bad spaxels
        fluxes = np.moveaxis(fluxes, 0, 1)
        med_fluxes = np.nanmedian(fluxes, axis=0).flatten()
        self.i_bad_spaxels = np.where(med_fluxes < BAD_SPAXEL_THRESHOLD)[0]

    @property
    def ra_all(self) -> NDArray:
        ra_ = get_property_flat(self._tiles, "ra")
        return (ra_ - self.ra_min) / self.coord_norm

    @property
    def ra(self) -> NDArray:
        return np.delete(self.ra_all, self.i_bad_spaxels)

    @property
    def dec_all(self) -> NDArray:
        dec_ = get_property_flat(self._tiles, "dec")
        return (dec_ - self.dec_min) / self.coord_norm

    @property
    def dec(self) -> NDArray:
        return np.delete(self.dec_all, self.i_bad_spaxels)

    @property
    def flux_all(self) -> NDArray:
        flux_ = get_property(self._tiles, "flux")
        flux_ = flux_[:, self.i_lower : self.i_upper, :]
        flux_ = np.moveaxis(flux_, 0, 1)
        flux_ = flux_.reshape((flux_.shape[0], -1))
        return flux_ / self.flux_norm

    @property
    def flux(self) -> NDArray:
        return np.delete(self.flux_all, self.i_bad_spaxels, axis=1)

    @property
    def i_var_all(self) -> NDArray:
        i_var_ = get_property(self._tiles, "i_var")
        i_var_ = i_var_[:, self.i_lower : self.i_upper, :]
        i_var_ = np.moveaxis(i_var_, 0, 1)
        i_var_ = i_var_.reshape((i_var_.shape[0], -1))
        return i_var_ * self.flux_norm**2

    @property
    def i_var(self) -> NDArray:
        return np.delete(self.i_var_all, self.i_bad_spaxels, axis=1)

    @property
    def wavelength(self) -> NDArray:
        return self._tiles[0].wavelength[self.i_lower : self.i_upper]

    @property
    def lsf_sigma_all(self) -> NDArray:
        lsf_ = get_property(self._tiles, "lsf_sigma")
        lsf_ = lsf_[:, self.i_lower : self.i_upper, :]
        lsf_ = np.moveaxis(lsf_, 0, 1)
        lsf_ = lsf_.reshape((lsf_.shape[0], -1))
        return lsf_

    @property
    def lsf_sigma(self) -> NDArray:
        return np.delete(self.lsf_sigma_all, self.i_bad_spaxels, axis=1)

    @property
    def lsf_sigma_at_line(self) -> NDArray:
        return self.lsf_sigma[self.i_line - self.i_lower, :]

    @property
    def average_lsf_sigma(self) -> NDArray:
        return np.nanmean(self.lsf_sigma_at_line)

    @property
    def flat_flux(self) -> NDArray:
        return self.flux.flatten()

    @property
    def flat_i_var(self) -> NDArray:
        return self.i_var.flatten()

    @property
    def flat_lsf_sigma(self) -> NDArray:
        return self.lsf_sigma.flatten()

    @staticmethod
    def find_nan_inds(arr: NDArray) -> NDArray:
        return np.where(np.logical_not(np.isfinite(arr)))[0]

    # This one doesn't work as intended right now but also I don't need it
    # @property
    # def nan_inds(self) -> NDArray:
    #     return self.find_nan_inds(self.flux)

    @property
    def flat_nan_inds(self) -> NDArray:
        return self.find_nan_inds(self.flat_flux)

    @property
    def flat_flux_no_nans(self) -> NDArray:
        flat_flux_ = self.flat_flux
        flat_flux_[self.flat_nan_inds] = 0.0
        return flat_flux_

    @property
    def flat_i_var_no_nans(self) -> NDArray:
        flat_i_var_ = self.flat_i_var
        flat_i_var_[self.flat_nan_inds] = 0.0
        return flat_i_var_

    @property
    def n_spaxels_fit(self) -> NDArray:
        return len(self.ra)
