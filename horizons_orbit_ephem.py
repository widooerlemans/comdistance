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
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
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


# ---------- COBS ↔ Horizons ID mapping helpers ----------

# Hard-coded overrides for known weird cases.
# NOTE: we'll normalize COBS IDs before looking them up here.
SPECIAL_ID_ALIASES: Dict[str, str] = {
    # Interstellar object 3I/ATLAS:
    # If COBS ever gives you a zero-padded code like 0003I/003I, we can
    # map that to the proper Horizons/SBDB designation. In your *current*
    # working environment this may not be needed for 3I, but we leave it
    # here as a safety net.
    "3I": "C/2025 N1 (ATLAS)",
    "0003I": "C/2025 N1 (ATLAS)",
    "003I": "C/2025 N1 (ATLAS)",

    # Example for a periodic comet; only used if the raw ID matches:
    "210P": "210P/Christensen",
}


def _strip_leading_zeros_in_interstellar(code: str) -> str:
    """
    Normalize interstellar-style IDs like '0003I' / '003I' / '3I' → '3I'.

    Only touches patterns ending in 'I' with digits in front.
    Other comet designations remain unchanged.
    """
    if not code:
        return code
    code = code.strip().upper()
    m = re.fullmatch(r"0*(\d+I)", code)
    if m:
        return m.group(1)  # e.g. '0003I' → '3I'
    return code


def normalize_cobs_code(raw_code: Optional[str]) -> str:
    """
    Produce a key we can reliably look up in SPECIAL_ID_ALIASES.

    - Trim whitespace
    - Uppercase
    - Apply the interstellar zero-stripping helper
    """
    if not raw_code:
        return ""
    code = raw_code.strip().upper()
    code = _strip_leading_zeros_in_interstellar(code)
    return code


def map_cobs_id_to_horizons_target(raw_id: str) -> str:
    """
    Map a COBS MPC/name-style ID to the Horizons target string.

    For almost all objects this is just an identity mapping, but it
    allows us to fix known mismatches like 0003I → C/2025 N1 (ATLAS).
    """
    key = normalize_cobs_code(raw_id)
    if key in SPECIAL_ID_ALIASES:
        return SPECIAL_ID_ALIASES[key]
    return raw_id


# ---------- orbit helpers ----------

def _fnum(x) -> Optional[float]:
    """Parse a float, returning None on error/NaN."""
    try:
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
    2. Fallback to Horizons elements(idspec)
    3. Enrich with:
        - a_au, Q_au
        - period_days, period_years
        - n_deg_per_day (mean motion)
    """
    # 1) SBDB
    base = sbdb_elements(label)

    # 2) Fallback to Horizons elements if SBDB didn't work
    if not base:
        rec_id = resolve_ambiguous_to_record_id(label)
        idspec = rec_id or label
        base = horizons_elements(idspec)

    if not base:
        return None

    out = dict(base)  # shallow copy we can mutate

    # Normalize basic orbital quantities
    e = _fnum(out.get("e"))
    q = _fnum(out.get("q_au") or out.get("q"))
    a = _fnum(out.get("a_au") or out.get("a"))

    # Only derive extras for bound (elliptic) orbits
    if (e is not None) and (e < 1.0):
        # Derive a from q and e if needed
        if (a is None) and (q is not None):
            try:
                a = q / (1.0 - e)
                out["a_au"] = a
            except ZeroDivisionError:
                pass

        # Derive q from a and e if needed
        if (q is None) and (a is not None):
            q = a * (1.0 - e)
            out["q_au"] = q

        # Aphelion distance
        if a is not None:
            Q = a * (1.0 + e)
            out["Q_au"] = Q

            # Kepler's third law: P(yrs)^2 = a^3, with a in AU
            try:
                period_years = math.sqrt(a ** 3)
                period_days = period_years * 365.25
                out["period_years"] = period_years
                out["period_days"] = period_days
                out["n_deg_per_day"] = 360.0 / period_days
            except Exception:
                pass

    # Default labels for clarity in UI/tooltips
    out.setdefault("solution", "osculating")
    out.setdefault("reference", "JPL SBDB (via Horizons)")

    return out


# ---------- ephemeris helpers ----------

def build_ephemeris_span(designation: str, observer: str, days: int = DAYS) -> List[Dict[str, Any]]:
    """
    Build a multi-day ephemeris with RA/DEC, r, delta, phase, V, v_pred.

    We query Horizons at 1-day spacing starting at *now*.
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

    # IMPORTANT: let Horizons return the full default ephemeris
    # so that r, alpha (phase angle), V, m1, k1 are all available.
    obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=epochs)
    eph = obj.ephemerides()  # <- no quantities=QUANTITIES here on purpose

    out: List[Dict[str, Any]] = []
    for row in eph:
        # Try to read r_au directly from the Horizons ephemeris
        try:
            r_au = float(row["r"])
        except Exception:
            r_au = None

        core = _row_to_payload_with_photometry(row, r_au=r_au, delta_vec_au=None)

        # datetime_jd is TDB in Horizons output; convert to UTC safely
        jd = float(row["datetime_jd"])
        t_tdb = Time(jd, format="jd", scale="tdb")
        t_utc = t_tdb.utc

        # --- compute JNow (equinox-of-date) from J2000 ---
        ra = core.get("ra_deg")
        dec = core.get("dec_deg")
        if (ra is not None) and (dec is not None):
            try:
                # Treat Horizons RA/DEC as FK5 at J2000 equinox
                c_j2000 = SkyCoord(
                    ra * u.deg,
                    dec * u.deg,
                    frame=FK5(equinox=Time("J2000"))
                )
                # Transform to FK5 with equinox at the current observation time
                c_jnow = c_j2000.transform_to(FK5(equinox=t_utc))

                core["ra_jnow_deg"] = float(c_jnow.ra.deg)
                core["dec_jnow_deg"] = float(c_jnow.dec.deg)
            except Exception:
                # If anything goes wrong, keep whatever core already had
                pass
        # --- END JNow block ---

        dt = t_utc.to_datetime(timezone=timezone.utc)
        epoch_iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        entry: Dict[str, Any] = {"epoch_utc": epoch_iso}
        entry.update(core)
        out.append(entry)

    return out


def fetch_orbit_and_ephem(comet_id: str, observer: str) -> Dict[str, Any]:
    """
    For a comet designation from COBS:

      - Map it (if needed) to the Horizons/SBDB target string via
        map_cobs_id_to_horizons_target().
      - Fetch extended orbit from SBDB/Horizons using that target.
      - Build 15-day Horizons ephemeris using that target.
      - Derive v_pred_now from the first ephemeris row (if available).

    We keep the original COBS ID in the 'id' field, and store the
    Horizons label in 'horizons_target' for transparency.
    """
    horizons_target = map_cobs_id_to_horizons_target(comet_id)

    orbit = sbdb_orbit_extended(horizons_target)
    ephem: List[Dict[str, Any]] = []
    error: Optional[str] = None

    try:
        ephem = build_ephemeris_span(horizons_target, observer, days=DAYS)
    except Exception as e:
        error = str(e)

    item: Dict[str, Any] = {
        "id": comet_id,  # original COBS ID
        "horizons_target": horizons_target,  # what we actually fed to Horizons/SBDB
        "epoch_utc": now_iso(),
    }

    if orbit:
        item["orbit"] = orbit
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
            item["name_full"] = fullname_map[cid]

        # display_name fallback
        item["display_name"] = item.get("name_full") or item["id"]

        results.append(item)
        time.sleep(PAUSE_S)

    # ---- brightness filter: COBS OR predicted-now ----
    limit = try_float_env(BRIGHT_LIMIT_ENV)
    if limit is None:
        limit = BRIGHT_LIMIT_DEFAULT

    before = len(results)
    filtered: List[Dict[str, Any]] = []

    for it in results:
        obs = it.get("cobs_mag")
        pred = it.get("v_pred_now")

        def _ok(v: Any) -> bool:
            try:
                return (v is not None) and (float(v) <= limit)
            except Exception:
                return False

        if _ok(obs) or _ok(pred):
            filtered.append(it)

    results = filtered
    print(f"[orbit_ephem] Brightness filter (<= {limit}) kept {len(results)}/{before}")

    # Sort by same key as main script
    results.sort(key=_sort_key)

    # Truncate to 15 brightest
    if len(results) > 15:
        results = results[:15]
        print(f"[orbit_ephem] Truncated to top 15 brightest")

    payload: Dict[str, Any] = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "days": DAYS,
        "items": results,
        "script": "horizons_orbit_ephem.py",
        "filter": {
            "mode": "cobs_or_pred_now",
            "bright_limit": limit,
            "max_items": 15,
        },
        "_debug_first_names": debug_first_names,
        "_debug_counts": debug_counts,
        "units": {
            "angles": "deg",
            "dist": "au",
            "epoch_time_scale": "TDB",
            "frame": "ecliptic J2000",
            "period_days": "days",
            "period_years": "Julian years",
            "n_deg_per_day": "deg/day",
        },
    }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[orbit_ephem] Wrote {OUT_JSON_PATH} with {len(results)} items")


if __name__ == "__main__":
    main()
