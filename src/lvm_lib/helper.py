"""helper.py - Helper functions for LVM data processing"""

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike


def daskify_native(array: ArrayLike, chunks: int | tuple) -> da.Array:
    """Convert input to a Dask array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return da.from_array(arr, chunks)


def numpyfy_native(array: ArrayLike) -> np.ndarray:
    """Convert input to a NumPy array with native byte order."""
    arr = np.asarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return arr
