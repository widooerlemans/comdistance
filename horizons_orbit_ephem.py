#!/usr/bin/env python3
"""
Generate data/comets_orbit_ephem.json for the brightest comets.

- Uses the same COBS-driven list and brightness filter as horizons_pull.py
- For each comet:
    * Fetches osculating elements from JPL SBDB via sbdb_elements()
      and augments them with a, Q, orbital period, mean motion.
    * Builds a 15-day daily ephemeris via JPL Horizons with RA/DEC,
      r, delta, phase, and (predicted) magnitude.

This script is intentionally separate from horizons_pull.py so the
existing pipeline remains unchanged.
"""

import json
import math
import time
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

from horizons_pull import (
    load_cobs_designations,
    now_iso,
    OBSERVER,
    BRIGHT_LIMIT_ENV,
    BRIGHT_LIMIT_DEFAULT,
    try_float_env,
    _sort_key,
    _row_to_payload_with_photometry,
    resolve_ambiguous_to_record_id,
    sbdb_elements,
    horizons_elements,
    QUANTITIES,  # still imported, but not used here on purpose
)

OUT_JSON_PATH = Path("data/comets_orbit_ephem.json")
DAYS = 15
PAUSE_S = 0.25  # small courtesy delay between Horizons calls


# ---------- orbit helpers ----------

...
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def sbdb_orbit_extended(label: str) -> Optional[Dict[str, Any]]:
    """
    Get orbital elements for `label`:

    1. Try JPL SBDB via sbdb_elements(label)
    2. Fallback to Horizons' own osculating elements via horizons_elements(label)

    Returns:
        A dict of orbital elements extended with:
          - a_au: semi-major axis [au]
          - Q_au: apoapsis distance [au]
          - period_yr: orbital period [years], if computable
          - n_deg_per_day: mean motion [deg/day], if computable
    """
    base = None

    try:
        base = sbdb_elements(label)
    except Exception as e:
        print(f"[orbit] SBDB elements failed for {label}: {e!r}")

    if not base:
        try:
            base = horizons_elements(label)
        except Exception as e:
            print(f"[orbit] Horizons elements failed for {label}: {e!r}")

    if not base:
        return None

    # Extend with a, Q, period, mean motion if we have enough to do so
    try:
        q = _safe_float(base.get("q_au"))
        e = _safe_float(base.get("e"))
        if q is not None and e is not None and e != 1.0:
            a = q / (1.0 - e)
            base["a_au"] = a
            base["Q_au"] = a * (1.0 + e)
            # Kepler's third law (a^3 ~ P^2), period in years
            if a > 0:
                period_yr = math.sqrt(a * a * a)
                base["period_yr"] = period_yr
                # mean motion in deg/day: 360 degrees per period
                base["n_deg_per_day"] = 360.0 / (period_yr * 365.25)
    except Exception as e:
        print(f"[orbit] Could not extend elements for {label}: {e!r}")

    return base


# ---------- ephemeris helpers ----------

def build_ephemeris_span(designation: str, observer: str, days: int = DAYS) -> List[Dict[str, Any]]:
    """
    Build a multi-day ephemeris with RA/DEC, r, delta, phase, V, v_pred.

    We query Horizons at 1-day spacing starting at *now*.

    IMPORTANT:
    - We request *apparent* RA/Dec from Horizons (aberrations='apparent'),
      so RA/Dec are equator-of-date (JNow), not J2000.
    - We then store those explicitly as ra_jnow_deg / dec_jnow_deg in the JSON.
    """
    jd0 = Time.now().jd
    epochs = [jd0 + float(i) for i in range(days)]

    # Use the same ambiguity-resolution logic as horizons_pull.py
    rec_id = resolve_ambiguous_to_record_id(designation)
    if rec_id:
        id_value = rec_id
        id_type = "smallbody"
    else:
        id_value = designation
        id_type = "designation"

    # Create Horizons object for this target and epoch set.
    obj = Horizons(
        id=id_value,
        id_type=id_type,
        location=observer,
        epochs=epochs,
    )

    # Request apparent-of-date RA/Dec (equator-of-date) with extra precision.
    # RA/DEC here are apparent-of-date because of aberrations="apparent".
    eph = obj.ephemerides(aberrations="apparent", extra_precision=True)

    out: List[Dict[str, Any]] = []
    for row in eph:
        # Try to read r_au (heliocentric distance) directly from the Horizons ephemeris.
        try:
            r_au = float(row["r"])
        except Exception:
            r_au = None

        core = _row_to_payload_with_photometry(row, r_au=r_au, delta_vec_au=None)

        # Force ra_jnow_deg / dec_jnow_deg from apparent RA/DEC.
        try:
            ra_app = float(row["RA"])
            dec_app = float(row["DEC"])
            core["ra_jnow_deg"] = ra_app
            core["dec_jnow_deg"] = dec_app
        except Exception:
            # If anything goes wrong, leave whatever _row_to_payload_with_photometry set.
            pass

        # datetime_jd is TDB in Horizons output; convert to UTC safely.
        jd = float(row["datetime_jd"])
        t_tdb = Time(jd, format="jd", scale="tdb")
        t_utc = t_tdb.utc
        dt = t_utc.to_datetime(timezone=timezone.utc)
        epoch_iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        entry: Dict[str, Any] = {"epoch_utc": epoch_iso}
        entry.update(core)
        out.append(entry)

    return out


def fetch_orbit_and_ephem(comet_id: str, observer: str) -> Dict[str, Any]:
    """
    For a comet designation:
      - fetch extended orbit from SBDB/Horizons
      - build 15-day Horizons ephemeris
      - derive v_pred_now from the first ephemeris row (if available)
    """
    item: Dict[str, Any] = {"id": comet_id}
    orbit = sbdb_orbit_extended(comet_id)
    if orbit:
        item["orbit"] = orbit

    error = None
    ephem: List[Dict[str, Any]] = []
    try:
        ephem = build_ephemeris_span(comet_id, observer, days=DAYS)
    except Exception as e:
        error = f"Horizons ephemeris failed: {e!r}"
        print(f"[orbit_ephem] {error}")

    if ephem:
        item["ephemeris_15d"] = ephem
    if error and not ephem:
        item["error"] = error

    # v_pred_now from the first ephemeris row if available
    if ephem:
        first = ephem[0]
        vpred = first.get("v_pred") or first.get("vmag")
        if vpred is not None:
            try:
                item["v_pred_now"] = float(vpred)
            except Exception:
                pass

    return item


# ---------- main ----------

def main() -> None:
    # Load the same COBS designations list used by horizons_pull.py
    cobs_map = load_cobs_designations(Path("data/cobs_list.json"))
    debug_first_names = cobs_map.pop("_debug_first_names", [])
    debug_counts = cobs_map.pop("_debug_counts", {})
    fullname_map = cobs_map.pop("_fullname_map", {})

    comet_ids: List[str] = sorted(cobs_map.keys())
    print(f"[orbit_ephem] Loaded {len(comet_ids)} COBS designations; sample: {comet_ids[:10]}")

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        print(f"[orbit_ephem] Fetching {cid} ...")
        item = fetch_orbit_and_ephem(cid, OBSERVER)

        # Attach observed magnitude from COBS
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]

        # Attach pretty full name if present
        if cid in fullname_map:
            item["display_name"] = fullname_map[cid]

        results.append(item)
        time.sleep(PAUSE_S)

    # Sort for stable output (by brightness, then name)
    results.sort(key=_sort_key)

    payload: Dict[str, Any] = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "bright_limit_used": try_float_env(BRIGHT_LIMIT_ENV, BRIGHT_LIMIT_DEFAULT),
        "items": results,
        "debug_first_names": debug_first_names,
        "debug_counts": debug_counts,
    }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[orbit_ephem] Wrote {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
