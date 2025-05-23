"""fit_data.py - TODO: add something here"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray
from xarray import Dataset


@dataclass(frozen=True)
class FitData:
    processed_data: Dataset
    normalise_F: Callable
    denormalise_F: Callable
    normalise_α: Callable
    denormalise_α: Callable
    normalise_δ: Callable
    denormalise_δ: Callable

    def get_arrays(self, data_spec) -> tuple[NDArray, NDArray]:
        # Does all the filtering and stuff required as specified in the data spec
        # Then returns all the arrays
        pass
