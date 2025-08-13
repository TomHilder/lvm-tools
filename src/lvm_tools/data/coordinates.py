"""coordinates.py - observation time and location extraction for LVM data processing."""

from astropy.coordinates import EarthLocation
from astropy.io.fits import Header
from astropy.time import Time


def get_mjd(header: Header) -> float:
    if "INTSTART" in header and "INTEND" in header:
        start_time = Time(header["INTSTART"], format="isot")
        end_time = Time(header["INTEND"], format="isot")
        mid_time = start_time + (end_time - start_time) / 2
        return mid_time.mjd
    else:
        raise ValueError("Could not find sufficient time information in header")


def get_observatory_code(header: Header) -> str:
    observatory = header.get("OBSERVAT", "").strip()
    known_observatories = ["LCO"]
    if observatory in known_observatories:
        return observatory
    else:
        raise ValueError(
            f"Unknown observatory: {observatory}. Known observatories: {known_observatories}"
        )


def get_observatory_location(observatory: str) -> EarthLocation:
    if observatory == "LCO":
        return EarthLocation.of_site("Las Campanas Observatory")
    else:
        raise ValueError(f"Unknown observatory: {observatory}")
