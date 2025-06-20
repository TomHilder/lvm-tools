from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy.typing import ArrayLike, NDArray


# def get_v_barycentric(fit_data: FitData, unit="km/s") -> NDArray:
def get_v_barycentric(mjd: ArrayLike, α: ArrayLike, δ: ArrayLike, unit="km/s") -> NDArray:
    times = Time(mjd, format="mjd")
    coords = SkyCoord(ra=α, dec=δ, obstime=times, unit="deg", frame="icrs")
    location = EarthLocation.of_site("Las Campanas Observatory")
    return coords.radial_velocity_correction("barycentric", location=location).to_value(unit)
