"""data_config_calc.py - functions for calculating the auto-config from tile data."""


def bounding_square(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    t_range = max(x_max - x_min, y_max - y_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_min_ = x_mid - t_range / 2
    x_max_ = x_mid + t_range / 2
    y_min_ = y_mid - t_range / 2
    y_max_ = y_mid + t_range / 2
    return (x_min_, x_max_), (y_min_, y_max_)
