"""fit_data.py - FitData class for holding data ready to be fitted."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import numpy as np
from jax.numpy import pi as π
from xarray import DataArray, Dataset

from lvm_tools.physical_properties.barycentric_corr import get_v_barycentric

if TYPE_CHECKING:
    from jaxtyping import Array as JaxArray
    from spectracles.model.data import SpatialDataLVM

try:
    from spectracles.model.data import SpatialDataLVM
except Exception as e:
    _import_err = e
    msg = (
        "Could not import SpatialDataLVM from spectracles.model.data. "
        "Install the optional dependency: pip install spectracles"
    )

    class SpatialDataLVM:  # stub preserves the API shape
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(msg) from _import_err

    import warnings

    warnings.warn(msg, ImportWarning)


def to_π_domain(x):
    # return x * 2 * π - π
    return x * 2 * np.pi - np.pi


def from_π_domain(x):
    return (x + π) / (2 * π)


def to_jax_array(arr: DataArray, dtype=np.float64) -> JaxArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return jnp.array(arr, dtype=dtype)


@dataclass(frozen=True)
class FitData:
    processed_data: Dataset
    normalise_flux: Callable
    predict_flux: Callable
    normalise_ivar: Callable
    predict_ivar: Callable
    normalise_α: Callable
    _predict_α: Callable
    normalise_δ: Callable
    _predict_δ: Callable

    @property
    def _flux(self) -> JaxArray:
        return self.normalise_flux(to_jax_array(self.processed_data["flux"].values))

    @property
    def flux(self) -> JaxArray:
        return jnp.nan_to_num(self._flux)

    @property
    def _i_var(self) -> JaxArray:
        return self.normalise_ivar(to_jax_array(self.processed_data["i_var"].values))

    @property
    def i_var(self) -> JaxArray:
        return jnp.nan_to_num(self._i_var, nan=1e-4)

    @property
    def _u_flux(self) -> JaxArray:
        return self._i_var**-0.5

    @property
    def u_flux(self) -> JaxArray:
        return jnp.nan_to_num(self._u_flux, nan=1e2)

    @property
    def α(self) -> JaxArray:
        return to_π_domain(self.normalise_α(to_jax_array(self.processed_data["ra"].values)))

    @property
    def δ(self) -> JaxArray:
        return to_π_domain(self.normalise_δ(to_jax_array(self.processed_data["dec"].values)))

    def predict_α(self, x: JaxArray) -> JaxArray:
        return self._predict_α(from_π_domain(x))

    def predict_δ(self, x: JaxArray) -> JaxArray:
        return self._predict_δ(from_π_domain(x))

    @property
    def αδ_data(self) -> SpatialDataLVM:
        return SpatialDataLVM(self.α, self.δ, self.spaxel_idx, self.tile_idx, self.ifu_idx)

    @property
    def λ(self) -> JaxArray:
        return to_jax_array(self.processed_data["wavelength"].values)

    @property
    def _lsf_σ(self) -> JaxArray:
        return to_jax_array(self.processed_data["lsf_sigma"].values)

    @property
    def lsf_σ(self) -> JaxArray:
        median_lsf_σ = jnp.nanmedian(self._lsf_σ)
        return jnp.nan_to_num(self._lsf_σ, nan=median_lsf_σ)

    @property
    def mjd(self) -> JaxArray:
        return to_jax_array(self.processed_data["mjd"].values)

    @property
    def mask(self) -> JaxArray:
        return ~jnp.isnan(self._flux)

    @property
    def λ_idx(self) -> JaxArray:
        return jnp.arange(len(self.λ), dtype=np.int64)

    @property
    def spaxel_idx(self) -> JaxArray:
        return jnp.arange(len(self.α), dtype=np.int64)

    @property
    def tile_idx(self) -> JaxArray:
        tile = to_jax_array(self.processed_data["tile"].values, dtype=np.int64)
        return jnp.unique(tile, return_inverse=True)[1]

    @property
    def ifu_idx(self) -> JaxArray:
        ifu = self.processed_data["ifu_label"].values
        return to_jax_array(np.unique(ifu, return_inverse=True)[1], dtype=np.int64)

    @property
    def v_bary(self) -> JaxArray:
        return to_jax_array(
            get_v_barycentric(
                mjd=self.mjd,
                α=self.predict_α(self.α),
                δ=self.predict_δ(self.δ),
                unit="km/s",
            )
        )

    def __repr__(self):
        # TODO: add something here
        raise NotImplementedError
