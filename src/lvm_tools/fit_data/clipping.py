"""clipping.py - data clipping for data preparation."""

from xarray import DataArray, Dataset


def bounding_square(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    t_range = 1.01 * max(x_max - x_min, y_max - y_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_min_ = x_mid - t_range / 2
    x_max_ = x_mid + t_range / 2
    y_min_ = y_mid - t_range / 2
    y_max_ = y_mid + t_range / 2
    return (x_min_, x_max_), (y_min_, y_max_)


def slice_mask(arr: DataArray, x_min: float, x_max: float) -> DataArray:
    return (arr >= x_min) & (arr <= x_max)


def clip_dataset(
    data: Dataset,
    λ_range: tuple[float, float],
    α_range: tuple[float, float],
    δ_range: tuple[float, float],
) -> Dataset:
    # Clip to wavelength range (simple since wavelength is an indexed coordinate)
    data = data.sel(wavelength=slice(*λ_range))
    # Clip to ra, dec range. Less simple since spaxel is the indexed coordinate
    α_slice = slice_mask(data["ra"], *α_range)
    δ_slice = slice_mask(data["dec"], *δ_range)
    return data.where(α_slice & δ_slice, drop=True)
