#!/usr/bin/env python3
"""
Build comets_orbit_ephem.json from JPL Horizons orbital elements + ephemerides.

Key fixes vs previous version:
- r_au is now read from Horizons column 'r'
- phase_deg is now read from Horizons column 'alpha'
- vmag is actually computed using M1, K1, r_au, and delta_au
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from astroquery.jplhorizons import Horizons
from astropy.time import Time


# -------------------------------------------------
# Configuration
# -------------------------------------------------

OBSERVER_CODE = "500"   # Geocenter
EPHEM_DAYS = 15         # Number of days of ephemeris to generate


@dataclass
class PhotometryModel:
    M1: Optional[float]
    K1: Optional[float]
    model: str = "M1 + 5*log10(delta_au) + K1*log10(r_au)"


@dataclass
class EphemerisPoint:
    epoch_utc: str
    r_au: Optional[float]
    delta_au: Optional[float]
    phase_deg: Optional[float]
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    vmag: Optional[float]
    ra_jnow_deg: Optional[float]
    dec_jnow_deg: Optional[float]
    photometry: Dict[str, Any]


@dataclass
class OrbitInfo:
    epoch_jd_tdb: float
    frame: str
    type: str
    q_au: float
    e: float
    i_deg: float
    Omega_deg: float
    omega_deg: float
    Tp_jd_tdb: float
    solution: str
    reference: str
    a_au: Optional[float] = None
    Q_au: Optional[float] = None
    period_years: Optional[float] = None
    period_days: Optional[float] = None
    n_deg_per_day: Optional[float] = None


@dataclass
class CometOutput:
    id: str
    epoch_utc: str
    orbit: Dict[str, Any]
    ephemeris_15d: List[Dict[str, Any]]
    cobs_mag: Optional[float]
    name_full: str
    display_name: str


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def today_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def jd_to_utc_iso(jd: float) -> str:
    t = Time(jd, format="jd", scale="tdb")
    # Horizons returns TDB; convert to UTC for output
    return t.utc.isot + "Z"


# -------------------------------------------------
# Horizons queries
# -------------------------------------------------

def fetch_orbit_from_horizons(comet_id: str) -> OrbitInfo:
    """
    Fetch osculating orbit elements from Horizons.
    """
    obj = Horizons(
        id=comet_id,
        id_type="designation",
        location=OBSERVER_CODE,
        epochs=None,
    )
    orb_table = obj.orbital_elements()

    row = orb_table[0]

    q_au = float(row["q"])
    e = float(row["e"])
    i_deg = float(row["incl"])
    Omega_deg = float(row["Omega"])
    omega_deg = float(row["w"])
    Tp_jd_tdb = float(row["Tp"])
    epoch_jd_tdb = float(row["epoch"])

    # Compute derived quantities only for bound (elliptic) orbits (e < 1)
    a_au: Optional[float] = None
    Q_au: Optional[float] = None
    period_years: Optional[float] = None
    period_days: Optional[float] = None
    n_deg_per_day: Optional[float] = None

    if e < 1.0:
        a_au = q_au / (1.0 - e)
        Q_au = a_au * (1.0 + e)
        # Kepler's 3rd law in years if a_au in AU
        period_years = a_au ** 1.5
        period_days = period_years * 365.25
        if period_days > 0:
            n_deg_per_day = 360.0 / period_days

    orbit = OrbitInfo(
        epoch_jd_tdb=epoch_jd_tdb,
        frame="ecliptic J2000",
        type="comet",
        q_au=q_au,
        e=e,
        i_deg=i_deg,
        Omega_deg=Omega_deg,
        omega_deg=omega_deg,
        Tp_jd_tdb=Tp_jd_tdb,
        solution="osculating",
        reference="JPL SBDB (via Horizons)",
        a_au=a_au,
        Q_au=Q_au,
        period_years=period_years,
        period_days=period_days,
        n_deg_per_day=n_deg_per_day,
    )

    return orbit


def build_ephemeris_for_comet(
    comet_id: str,
    M1: Optional[float],
    K1: Optional[float],
    days: int = EPHEM_DAYS,
) -> List[EphemerisPoint]:
    """
    Build a list of EphemerisPoint objects for the next `days` days.

    *** FIXED ***
    - r_au taken from Horizons column 'r'
    - phase_deg taken from Horizons column 'alpha'
    - vmag computed when r_au and delta_au are present
    """

    start = today_utc()
    epochs = [
        Time(start + timedelta(days=i), scale="utc").jd
        for i in range(days)
    ]

    obj = Horizons(
        id=comet_id,
        id_type="designation",
        location=OBSERVER_CODE,
        epochs=epochs,
    )

    # quantities='1' is enough to get r, delta, RA, DEC, alpha, etc.
    eph = obj.ephemerides(quantities="1")

    ephem_points: List[EphemerisPoint] = []

    for row in eph:
        # Raw values from Horizons
        r_val = float(row["r"])
        delta_val = float(row["delta"])
        alpha_val = float(row["alpha"])
        ra_val = float(row["RA"])     # degrees J2000
        dec_val = float(row["DEC"])   # degrees J2000

        # Convert NaNs to None just in case
        def clean(x: float) -> Optional[float]:
            return None if (x is None or math.isnan(x)) else float(x)

        r_au = clean(r_val)
        delta_au = clean(delta_val)
        phase_deg = clean(alpha_val)

        # --- FIXED: actually compute vmag when we have r_au and delta_au ---
        vmag: Optional[float] = None
        if (
            r_au is not None
            and delta_au is not None
            and M1 is not None
            and K1 is not None
        ):
            # Standard comet total magnitude law
            vmag = M1 + 5.0 * math.log10(delta_au) + K1 * math.log10(r_au)

        phot = PhotometryModel(M1=M1, K1=K1)

        point = EphemerisPoint(
            epoch_utc=jd_to_utc_iso(float(row["datetime_jd"])),
            r_au=r_au,
            delta_au=delta_au,
            phase_deg=phase_deg,
            ra_deg=ra_val,
            dec_deg=dec_val,
            vmag=vmag,
            # If you ever want true JNOW, you’d precess here.
            ra_jnow_deg=ra_val,
            dec_jnow_deg=dec_val,
            photometry=asdict(phot),
        )

        ephem_points.append(point)

    return ephem_points


# -------------------------------------------------
# Top-level builder
# -------------------------------------------------

def build_comets_orbit_ephem(
    comets: List[Dict[str, Any]],
    days: int = EPHEM_DAYS,
    observer_code: str = OBSERVER_CODE,
) -> Dict[str, Any]:
    """
    comets: list of dicts with at least:
        {
          "id": "C/2025 K1",
          "name_full": "C/2025 K1 (ATLAS)",
          "display_name": "C/2025 K1 (ATLAS)",
          "M1": 14.1,
          "K1": 4.5,
          "cobs_mag": 9.9
        }
    """

    now = today_utc()

    items: List[Dict[str, Any]] = []
    for c in comets:
        comet_id = c["id"]
        name_full = c.get("name_full", comet_id)
        display_name = c.get("display_name", name_full)
        M1 = c.get("M1")
        K1 = c.get("K1")
        cobs_mag = c.get("cobs_mag")

        orbit = fetch_orbit_from_horizons(comet_id)
        ephem_points = build_ephemeris_for_comet(
            comet_id=comet_id,
            M1=M1,
            K1=K1,
            days=days,
        )

        comet_output = CometOutput(
            id=comet_id,
            epoch_utc=utc_iso(now),
            orbit=asdict(orbit),
            ephemeris_15d=[asdict(p) for p in ephem_points],
            cobs_mag=cobs_mag,
            name_full=name_full,
            display_name=display_name,
        )

        items.append(asdict(comet_output))

    out: Dict[str, Any] = {
        "generated_utc": utc_iso(now),
        "observer": observer_code,
        "days": days,
        "items": items,
    }

    return out


# -------------------------------------------------
# Script entry point
# -------------------------------------------------

def main() -> None:
    # TODO: replace this with however you currently gather your comets
    # (e.g. your COBS-based JSON/logic). This is only an example list
    # using values visible in your current output file.
    comets_config: List[Dict[str, Any]] = [
        {
            "id": "C/2025 K1",
            "name_full": "C/2025 K1 (ATLAS)",
            "display_name": "C/2025 K1 (ATLAS)",
            "M1": 14.1,
            "K1": 4.5,
            "cobs_mag": 9.9,
        },
        {
            "id": "3I",
            "name_full": "3I/ATLAS",
            "display_name": "3I/ATLAS",
            "M1": 12.1,
            "K1": 4.75,
            "cobs_mag": 10.1,
        },
        {
            "id": "C/2025 T1",
            "name_full": "C/2025 T1 (ATLAS)",
            "display_name": "C/2025 T1 (ATLAS)",
            "M1": 11.3,
            "K1": 35.75,
            "cobs_mag": 10.1,
        },
        # Add C/2025 R2 and others here, with their M1 / K1 / cobs_mag
    ]

    result = build_comets_orbit_ephem(comets_config, days=EPHEM_DAYS)

    with open("comets_orbit_ephem.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
