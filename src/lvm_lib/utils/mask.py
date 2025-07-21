import numpy as np
from scipy.spatial import cKDTree


def mask_near_points(xgrid, ygrid, xpoints, ypoints, threshold=None):
    """
    Generate a boolean mask for a 2D grid, where True means the grid cell is close to at least one (xpoint, ypoint).

    Parameters
    ----------
    xgrid : 1D array
        Grid coordinates along the x-axis (must be monotonically increasing).
    ygrid : 1D array
        Grid coordinates along the y-axis (must be monotonically increasing).
    xpoints : 1D array
        X-coordinates of the data points.
    ypoints : 1D array
        Y-coordinates of the data points.
    threshold : float, optional
        Maximum distance from a grid cell center to be considered "near" a data point.
        If None, uses 1.5 × max(mean grid spacing in x and y).

    Returns
    -------
    mask : 2D boolean array
        Mask array with shape (len(ygrid), len(xgrid)), where True means "keep" (near a point).
    """
    xx, yy = np.meshgrid(xgrid, ygrid, indexing="xy")
    grid_centers = np.column_stack([xx.ravel(), yy.ravel()])

    tree = cKDTree(np.column_stack([xpoints, ypoints]))
    dists, _ = tree.query(grid_centers, k=1)

    if threshold is None:
        dx = np.mean(np.diff(xgrid))
        dy = np.mean(np.diff(ygrid))
        threshold = 1.5 * max(dx, dy)

    mask = (dists < threshold).reshape(xx.shape)
    return mask
