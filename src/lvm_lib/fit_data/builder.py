"""builder.py - FitDataBuilder class for constructing FitData with reproducibility."""

import json
import warnings
from dataclasses import dataclass
from hashlib import sha256

from lvm_lib.config.data_config import DataConfig
from lvm_lib.data.tile import LVMTileLike
from lvm_lib.fit_data.clipping import clip_dataset
from lvm_lib.fit_data.filtering import filter_dataset
from lvm_lib.fit_data.fit_data import FitData
from lvm_lib.fit_data.normalisation import get_norm_funcs


@dataclass(frozen=True)
class FitDataBuilder:
    tiles: LVMTileLike
    config: DataConfig

    def build(self) -> FitData:
        # Clip and filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds = clip_dataset(
                self.tiles.data,
                self.config.λ_range,
                self.config.α_range,
                self.config.δ_range,
            )
            ds = filter_dataset(
                self.tiles.data,
                self.config.nans_strategy,
                self.config.F_bad_strategy,
                self.config.F_range,
                self.config.fibre_status_include,
                self.config.apply_mask,
            )

        # Build forward and reverse transformations for normalisation
        norm_F, denorm_F = get_norm_funcs(
            self.config.normalise_F_offset,
            self.config.normalise_F_scale,
        )
        norm_α, denorm_α = get_norm_funcs(
            self.config.normalise_α_offset,
            self.config.normalise_α_scale,
        )
        norm_δ, denorm_δ = get_norm_funcs(
            self.config.normalise_δ_offset,
            self.config.normalise_δ_scale,
        )

        return None  # FitData(...)

    def hash(self) -> str:
        data = {
            "config": json.dumps(self.config.to_dict(), sort_keys=True),
            "tiles": ...,  # tile meta data
        }
        serialised = json.dumps(data, sort_keys=True)
        return sha256(serialised.encode()).hexdigest()
