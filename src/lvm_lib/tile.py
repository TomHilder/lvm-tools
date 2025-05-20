"""tile.py - Tile class for LVM data processing"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import FITS_rec, HDUList
from astropy.units import Unit
from numpy.typing import NDArray
from xarray import Dataset

from lvm_lib.helper import daskify_native, numpyfy_native

# Conversions between FWHM and Gaussian sigma
SIGMA_TO_FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_TO_SIGMA: float = 1.0 / SIGMA_TO_FWHM

# Physical units for the data
FLUX_UNIT: Unit = u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1
SPECTRAL_UNIT: Unit = u.angstrom
SPATIAL_UNIT: Unit = u.degree
WAVELENGTH_UNIT: Unit = u.angstrom

# Default chunk size for Dask arrays
CHUNKSIZE: str = "auto"


def get_science_inds(slitmap: FITS_rec) -> NDArray:
    return np.where(slitmap.field("targettype") == "science")[0]


@dataclass(frozen=True)
class LVMTileMeta:
    filename: str
    tile_id: int
    exp_num: int
    drp_ver: str


@dataclass(frozen=True)
class LVMTile:
    data: Dataset
    meta: LVMTileMeta

    @classmethod
    def from_file(cls, drp_file: Path | str) -> LVMTile:
        file = Path(drp_file)
        if not file.exists():
            raise FileNotFoundError("Could not find DRP file.")

        with fits.open(file, memmap=True) as hdul:
            tile_id, exp_num, drp_ver = cls.get_metadata(hdul)
            (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status) = (
                cls.get_science_data(hdul)
            )

        # Conver the lsf from full width at half maximum (FWHM) to sigma
        lsf *= FWHM_TO_SIGMA

        # Common dimensions for cube data
        pixel_dims = ("tile", "spaxel", "wavelength")
        spaxel_dims = ("tile", "spaxel")

        # Assemble data into xarray Dataset, containing both dask arrays and numpy arrays
        data = Dataset(
            data_vars={
                "flux": (pixel_dims, flux[None, :, :], {"units": str(FLUX_UNIT)}),
                "i_var": (pixel_dims, i_var[None, :, :], {"units": str(FLUX_UNIT**-2)}),
                "lsf_sigma": (pixel_dims, lsf[None, :, :], {"units": str(SPECTRAL_UNIT)}),
                "mask": (pixel_dims, mask[None, :, :]),
            },
            coords={
                # Main dimensions/coordinates
                "tile": ("tile", [tile_id]),
                "spaxel": ("spaxel", np.arange(len(fibre_id))),
                "wavelength": ("wavelength", wave, {"units": str(WAVELENGTH_UNIT)}),
                # More coordinates
                "ra": (spaxel_dims, ra[None, :], {"units": str(SPATIAL_UNIT)}),
                "dec": (spaxel_dims, dec[None, :], {"units": str(SPATIAL_UNIT)}),
                "fibre_id": (spaxel_dims, fibre_id[None, :]),
                "fibre_status": (spaxel_dims, fibre_status[None, :]),
            },
        )

        # Assemble metadata
        meta = LVMTileMeta(
            filename=file.name,
            tile_id=tile_id,
            exp_num=exp_num,
            drp_ver=drp_ver,
        )

        return cls(data=data, meta=meta)

    @staticmethod
    def get_science_data(drp_hdulist: HDUList) -> tuple[tuple[da.Array], tuple[NDArray]]:
        slitmap = drp_hdulist[-1].data
        science_inds = get_science_inds(slitmap)
        # Lazily load cubes
        flux = daskify_native(drp_hdulist[1].data, CHUNKSIZE)[science_inds, :]
        i_var = daskify_native(drp_hdulist[2].data, CHUNKSIZE)[science_inds, :]
        mask = daskify_native(drp_hdulist[3].data, CHUNKSIZE)[science_inds, :]
        lsf = daskify_native(drp_hdulist[5].data, CHUNKSIZE)[science_inds, :]
        # Eagerly coordinates
        wave = numpyfy_native(drp_hdulist[4].data)
        ra = numpyfy_native((slitmap["ra"])[science_inds])
        dec = numpyfy_native((slitmap["dec"])[science_inds])
        fibre_id = numpyfy_native((slitmap["fiberid"])[science_inds])
        fibre_status = numpyfy_native((slitmap["fibstatus"])[science_inds])
        return (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status)

    @staticmethod
    def get_metadata(drp_hdulist: HDUList) -> tuple[int, int, str]:
        tile_id = int(drp_hdulist[0].header["OBJECT"].split("=")[1])
        exp_num = int(drp_hdulist[0].header["EXPOSURE"])
        drp_ver = str(drp_hdulist[0].header["DRPVER"])
        return tile_id, exp_num, drp_ver
