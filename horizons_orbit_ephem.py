#!/usr/bin/env python3
"""
Generate data/comets_orbit_ephem.json for the brightest comets.

- Uses the same COBS-driven list and brightness filter as horizons_pull.py
- For each comet:
    * Fetches osculating elements from JPL SBDB via sbdb_elements()
      (fallback: Horizons elements()) and augments them with:
          - a_au, Q_au
          - period_days, period_years
          - n_deg_per_day
    * Builds a 15-day daily ephemeris via JPL Horizons with RA/DEC,
      r, delta, phase, and predicted magnitude (v_pred).

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
    QUANTITIES,
)

OUT_JSON_PATH = Path("data/comets_orbit_ephem.json")
DAYS = 15
PAUSE_S = 0.25  # small courtesy delay between Horizons calls


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

    obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=epochs)
    eph = obj.ephemerides(quantities=QUANTITIES)

    out: List[Dict[str, Any]] = []
    for row in eph:
        # r_au from Horizons ephemerides (*** FIXED: keep this value! ***)
        try:
            r_au = float(row["r"])
        except Exception:
            r_au = None  # <- now correctly indented under except

        core = _row_to_payload_with_photometry(row, r_au=r_au, delta_vec_au=None)

        # datetime_jd is TDB in Horizons output; convert to UTC safely
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
    orbit = sbdb_orbit_extended(comet_id)
    ephem: List[Dict[str, Any]] = []
    error: Optional[str] = None

    try:
        ephem = build_ephemeris_span(comet_id, observer, days=DAYS)
    except Exception as e:
        error = str(e)

    item: Dict[str, Any] = {
        "id": comet_id,
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
