"""tile.py - Tile classes for LVM data processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import astropy.units as u  # type: ignore[import]
import dask.array as da
import numpy as np
from astropy.io import fits  # type: ignore[import]
from astropy.io.fits import FITS_rec, HDUList  # type: ignore[import]
from astropy.units import Unit  # type: ignore[import]
from numpy.typing import NDArray
from xarray import Dataset, concat

from lvm_tools.data.coordinates import get_mjd
from lvm_tools.data.helper import daskify_native, numpyfy_native, summarize_with_units

# Conversions between FWHM and Gaussian sigma
SIGMA_TO_FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_TO_SIGMA: float = 1.0 / SIGMA_TO_FWHM

# Physical units for the data
FLUX_UNIT: Unit = u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1
SPECTRAL_UNIT: Unit = u.angstrom
SPATIAL_UNIT: Unit = u.degree

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
            tile_id, exp_num, drp_ver, mjd = cls.get_metadata(hdul)
            (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status, ifu_label) = (
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
                "tile": ("tile", [exp_num]),
                "spaxel": ("spaxel", np.arange(len(fibre_id))),
                "wavelength": ("wavelength", wave, {"units": str(SPECTRAL_UNIT)}),
                # More coordinates
                "mjd": ("tile", [mjd], {"units": "day"}),
                "ra": (spaxel_dims, ra[None, :], {"units": str(SPATIAL_UNIT)}),
                "dec": (spaxel_dims, dec[None, :], {"units": str(SPATIAL_UNIT)}),
                "fibre_id": (spaxel_dims, fibre_id[None, :]),
                "ifu_label": (spaxel_dims, ifu_label[None, :]),
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

    def __repr__(self) -> str:
        prefix = f"LVMTile ({hex(id(self))}):"
        prefix += f"\n    Filename:        {self.meta.filename}"
        prefix += f"\n    Exposure:        {self.meta.exp_num}"
        prefix += f"\n    DRP version:     {self.meta.drp_ver}"
        prefix += f"\n    Tile ID:         {self.meta.tile_id}"
        return f"{prefix}\n{summarize_with_units(self.data)}"

    @staticmethod
    def get_science_data(drp_hdulist: HDUList) -> tuple[tuple, tuple]:
        slitmap = drp_hdulist[-1].data
        science_inds = get_science_inds(slitmap)
        # Lazily load cubes
        flux: da.Array = daskify_native(drp_hdulist[1].data, CHUNKSIZE)[science_inds, :]
        i_var: da.Array = daskify_native(drp_hdulist[2].data, CHUNKSIZE)[science_inds, :]
        mask: da.Array = daskify_native(drp_hdulist[3].data, CHUNKSIZE)[science_inds, :]
        lsf: da.Array = daskify_native(drp_hdulist[5].data, CHUNKSIZE)[science_inds, :]
        # Eagerly coordinates
        wave: NDArray = numpyfy_native(drp_hdulist[4].data)
        ra: NDArray = numpyfy_native((slitmap["ra"])[science_inds])
        dec: NDArray = numpyfy_native((slitmap["dec"])[science_inds])
        fibre_id: NDArray = numpyfy_native((slitmap["fiberid"])[science_inds])
        fibre_status: NDArray = numpyfy_native((slitmap["fibstatus"])[science_inds])
        ifu_label: NDArray = numpyfy_native((slitmap["ifulabel"])[science_inds])
        return (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status, ifu_label)

    @staticmethod
    def get_metadata(drp_hdulist: HDUList) -> tuple[int, int, str]:
        try:
            tile_id = int(drp_hdulist[0].header["OBJECT"].split("=")[1])
        except IndexError:
            try:
                tile_id = int(drp_hdulist[0].header["OBJECT"])
            except ValueError:
                tile_id = str(drp_hdulist[0].header["OBJECT"])
        exp_num = int(drp_hdulist[0].header["EXPOSURE"])
        drp_ver = str(drp_hdulist[0].header["DRPVER"])
        mjd = float(get_mjd(drp_hdulist[0].header))
        return tile_id, exp_num, drp_ver, mjd


@dataclass(frozen=True)
class LVMTileCollection:
    data: Dataset
    meta: Mapping[int, LVMTileMeta]

    @classmethod
    def from_tiles(cls, tiles: list[LVMTile]) -> LVMTileCollection:
        # Concatenate tile datasets along the 'tile' dimension
        combined_data = concat([tile.data for tile in tiles], dim="tile")

        # Construct metadata dictionary keyed by exposure number
        meta_dict = {tile.meta.exp_num: tile.meta for tile in tiles}

        return cls(data=combined_data, meta=meta_dict)

    def __repr__(self) -> str:
        prefix = f"LVMTileCollection ({hex(id(self))}):"
        prefix += f"\n    Tiles:            {len(self.meta)}"
        prefix += f"\n    Exposures:        {set(meta.exp_num for meta in self.meta.values())}"
        prefix += f"\n    DRP versions:     {set(meta.drp_ver for meta in self.meta.values())}"
        prefix += f"\n    Tile IDs:         {set(meta.tile_id for meta in self.meta.values())}"
        return f"{prefix}\n{summarize_with_units(self.data)}"


LVMTileLike = LVMTile | LVMTileCollection
