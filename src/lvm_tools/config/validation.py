"""validation.py - Validation functions for data processing configuration."""

from typing import get_args

import numpy as np

from lvm_tools.fit_data.filtering import ExcludeStrategy, FibreStatus
from lvm_tools.fit_data.normalisation import NormaliseStrategy


def validate_range(x_range: tuple[float, float]) -> None:
    # if not isinstance(x_range, tuple):
    # raise TypeError("Data range must be in a tuple.")
    if len(x_range) != 2:
        raise ValueError(
            "Data range must be a tuple with exactly two values (min, max)."
        )
    if x_range[1] < x_range[0]:
        raise ValueError("Requested data range restriction has max < min.")


def validate_excl_strategy(strategy: ExcludeStrategy) -> None:
    if strategy not in get_args(ExcludeStrategy):
        raise ValueError(f"Unknown exclusion strategy: {strategy}")


def validate_norm_strategy(strategy: NormaliseStrategy) -> None:
    if strategy not in get_args(NormaliseStrategy):
        raise ValueError(f"Unknown normalisation strategy: {strategy}")


def validate_fib_status_incl(fibre_status_include: tuple[FibreStatus]) -> None:
    # if not isinstance(fibre_status_include, tuple):
    # raise TypeError("fibre_status_include must be a tuple.")
    for fs in fibre_status_include:
        if fs not in get_args(FibreStatus):
            raise ValueError(f"Unknown fibre status: {fs}")


def validate_offset(offset: float) -> None:
    if not isinstance(offset, (float, np.floating)):
        raise TypeError("offset must be float.")
    if not np.isfinite(offset):
        raise Exception("Bad offset (nan or infty).")


def validate_scale(scale: float) -> None:
    if not isinstance(scale, (float, np.floating)):
        raise TypeError("scale must be float.")
    if not np.isfinite(scale):
        raise Exception("Bad scale (nan or infty).")
    if scale <= 0:
        raise Exception("Scale is not positive, but it must be.")


def validate_apply_mask(apply_mask: bool) -> None:
    if not isinstance(apply_mask, bool):
        raise TypeError("apply_mask must be a boolean.")
