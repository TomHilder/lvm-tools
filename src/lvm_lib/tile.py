# tile.np
# Written by Thomas Hilder

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList
from numpy.typing import NDArray

# Conversions between FWHM and Gaussian sigma
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_TO_SIGMA = 1.0 / SIGMA_TO_FWHM

# Physical units for the data
FLUX_UNIT = u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1
WAVELENGTH_UNIT = u.angstrom
SKY_UNIT = u.degree


@dataclass(frozen=True)
class Tile:
    # Metadata
    file_path: Path
    tile_id: int
    fibre_ids: NDArray
    # Dimensions
    n_spaxels: int
    n_wavelengths: int
    # Science data
    ra: NDArray
    dec: NDArray
    flux: NDArray
    i_var: NDArray
    wavelength: NDArray
    lsf_sigma: NDArray

    # Constructor method
    @classmethod
    def from_file(cls, drp_file: Path | str) -> Tile:
        file = Path(drp_file)
        if not file.exists():
            raise FileNotFoundError("Could not find DRP file.")
        with fits.open(file) as hdul:
            ra, dec, flux, i_var, wave, lsf = Tile.get_science_data(hdul)
        return cls(
            file,
            np.copy(ra),
            np.copy(dec),
            np.copy(flux),
            np.copy(i_var),
            np.copy(wave),
            np.copy(lsf),
        )

    @staticmethod
    def get_science_data(hdulist: HDUList) -> tuple[NDArray]:
        # Flux and inverse variance
        flux = hdulist[1].data.T
        ivar = hdulist[2].data.T
        # Wavelengths
        wave = hdulist[4].data
        # LSF
        lsf = hdulist[5].data.T
        # Get science indices
        slitmap = hdulist[-1].data
        science_inds = np.where(slitmap.field("targettype") == "science")[0]
        return (
            slitmap.field("ra")[science_inds] * SKY_UNIT,
            slitmap.field("dec")[science_inds] * SKY_UNIT,
            flux[:, science_inds] * FLUX_UNIT,
            ivar[:, science_inds] * FLUX_UNIT**-2,
            wave * WAVELENGTH_UNIT,
            lsf[:, science_inds] * FWHM_TO_SIGMA * WAVELENGTH_UNIT,
        )
