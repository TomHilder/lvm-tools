"""builder.py - FitDataBuilder class for constructing FitData with reproducibility."""

import json
from dataclasses import dataclass
from hashlib import sha256

from lvm_lib.config.data_config import DataConfig
from lvm_lib.data.tile import LVMTileLike
from lvm_lib.fit_data.fit_data import FitData
from lvm_lib.fit_data.processing import (
    flatten_tile_coord,
    get_normalisation_functions,
    process_tile_data,
)


@dataclass(frozen=True)
class FitDataBuilder:
    tiles: LVMTileLike
    config: DataConfig

    def build(self) -> FitData:
        return FitData(
            flatten_tile_coord(process_tile_data(self.tiles, self.config)),
            *get_normalisation_functions(self.config),
        )

    def hash(self) -> str:
        data = {
            "config": json.dumps(self.config.to_dict(), sort_keys=True),
            "tiles": ...,  # tile meta data
        }
        serialised = json.dumps(data, sort_keys=True)
        return sha256(serialised.encode()).hexdigest()
