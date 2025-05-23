from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from lvm_lib.data.tile import LVMTileLike


@dataclass(frozen=True)
class FitData:
    tiles = LVMTileLike

    def __post_init__(self):
        # Brainstorm of checks:
        # - Check the units are correct
        # - Check the wavelengths are the same in all tiles
        # - Check the shapes are correct (between attributes and between tiles)
        pass

    def get_arrays(self, data_spec) -> tuple[NDArray, NDArray]:
        # Does all the filtering and stuff required as specified in the data spec
        # Then returns all the arrays
        pass
