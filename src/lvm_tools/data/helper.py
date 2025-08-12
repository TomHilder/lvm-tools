"""helper.py - helper functions for LVM data processing."""

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset


def daskify_native(array: ArrayLike, chunks: str | int | tuple) -> da.Array:
    """Convert input to a Dask array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return da.from_array(arr, chunks)  # type: ignore[no-any-return]


def numpyfy_native(array: ArrayLike) -> NDArray:
    """Convert input to a NumPy array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return arr


def summarize_with_units(ds: Dataset) -> str:
    lines = [f"    Data size:        {ds.nbytes // 1024**2}MB"]
    lines.append(f"    Dimensions:       {', '.join(f'{k}: {v}' for k, v in ds.sizes.items())}")

    # Coordinates
    lines.append("    Coordinates:")
    for name, coord in ds.coords.items():
        dims = f"({', '.join(coord.sizes)})"
        dtype = str(coord.dtype)
        size = f"{coord.nbytes // 1024}kB"
        units = coord.attrs.get("units", "")
        lines.append(f"        {name:<13} {dims:<28} {dtype:<8} {size:<6} [{units}]")

    # Data variables
    lines.append("    Data:")
    for name, var in ds.data_vars.items():
        dims = f"({', '.join(var.sizes)})"
        dtype = str(var.dtype)
        size = f"{var.nbytes // 1024 // 1024}MB"
        if hasattr(var.data, "chunks") and var.data.chunks is not None:
            chunks = f"DaskArray<chunksize={var.data.chunks}>"
        else:
            chunks = "NDArray"
        units = var.attrs.get("units", "")
        lines.append(f"        {name:<13} {dims:<28} {dtype:<8} {size:<6} {chunks} [{units}]")

    return "\n".join(lines)


def convert_sci_to_int(arr: ArrayLike) -> NDArray:
    mapping = {"Sci1": 0, "Sci2": 1, "Sci3": 2}
    return np.array([mapping[item] for item in arr], dtype=int)
