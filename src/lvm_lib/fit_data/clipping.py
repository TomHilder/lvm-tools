"""clipping.py - data clipping for data preparation."""

from xarray import DataArray, Dataset


def slice_mask(arr: DataArray, x_min: float, x_max: float) -> DataArray:
    return (arr >= x_min) & (arr <= x_max)


def clip(
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
