"""builder.py - FitDataBuilder class for constructing FitData with reproducibility."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from lvm_tools.config.data_config import DataConfig
from lvm_tools.data.tile import LVMTile, LVMTileLike
from lvm_tools.fit_data.fit_data import FitData
from lvm_tools.fit_data.processing import (
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
            "config": json.dumps(self._configdict, sort_keys=True),
            "tiles": json.dumps(self._metadict, sort_keys=True),
        }
        serialised = json.dumps(data, sort_keys=True)
        return sha256(serialised.encode()).hexdigest()

    @property
    def _configdict(self) -> dict:
        return self.config.to_dict()

    @property
    def _metadict(self) -> dict:
        # Ensure meta is always a dict of LVMTileMeta
        if isinstance(self.tiles, LVMTile):
            meta = {self.tiles.meta.exp_num: self.tiles.meta.copy()}
        else:
            meta = self.tiles.meta.copy()

        for key, value in meta.items():
            meta[key] = asdict(value)

        return meta
