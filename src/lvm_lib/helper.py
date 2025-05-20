"""helper.py - helper functions for LVM data processing"""

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike


def daskify_native(array: ArrayLike, chunks: str | int | tuple) -> da.Array:
    """Convert input to a Dask array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return da.from_array(arr, chunks)  # type: ignore[no-any-return]


def numpyfy_native(array: ArrayLike) -> np.ndarray:
    """Convert input to a NumPy array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return arr
