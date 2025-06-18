from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy.typing import NDArray

from lvm_lib.fit_data.fit_data import FitData


def get_v_barycentric(fit_data: FitData, unit="km/s") -> NDArray:
    times = Time(fit_data.mjd, format="mjd")
    coords = SkyCoord(ra=fit_data.α, dec=fit_data.δ, obstime=times, unit="deg", frame="icrs")
    location = EarthLocation.of_site("Las Campanas Observatory")
    return coords.radial_velocity_correction("barycentric", location=location).to_value(unit)
