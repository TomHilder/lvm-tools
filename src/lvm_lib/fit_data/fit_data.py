"""fit_data.py - FitData classm for holding data ready to be fitted."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array as JaxArray
from xarray import DataArray, Dataset


def to_jax_array(arr: DataArray) -> JaxArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return jnp.array(arr, dtype=jnp.float64)


@dataclass(frozen=True)
class FitData:
    processed_data: Dataset
    normalise_flux: Callable
    predict_flux: Callable
    normalise_ivar: Callable
    predict_ivar: Callable
    normalise_α: Callable
    predict_α: Callable
    normalise_δ: Callable
    predict_δ: Callable

    @property
    def flux(self) -> JaxArray:
        return self.normalise_flux(to_jax_array(self.processed_data["flux"].values))

    @property
    def i_var(self) -> JaxArray:
        return self.normalise_ivar(to_jax_array(self.processed_data["i_var"].values))

    @property
    def α(self) -> JaxArray:
        return self.normalise_α(to_jax_array(self.processed_data["ra"].values))

    @property
    def δ(self) -> JaxArray:
        return self.normalise_δ(to_jax_array(self.processed_data["dec"].values))

    @property
    def λ(self) -> JaxArray:
        return to_jax_array(self.processed_data["wavelength"].values)

    def __repr__(self):
        # TODO: add something here
        raise NotImplementedError
