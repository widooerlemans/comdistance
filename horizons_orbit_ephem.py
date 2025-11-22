#!/usr/bin/env python3
"""
Generate data/comets_orbit_ephem.json for the brightest comets.

This version:

- Uses its *own* load_cobs_designations() that calls the global COBS
  Comet List API (no location / observer filter) and applies the
  BRIGHT_LIMIT magnitude cut there.
- Does NOT use the old location-based cobs_list.json produced by
  horizons_pull.py. The cobs_list_path argument is only used as a
  debug snapshot output.
- For each comet that passes the COBS brightness cut:
    * Fetches osculating elements from JPL SBDB via sbdb_elements()
      and augments them with a, Q, orbital period, mean motion.
    * Builds a 15-day daily ephemeris via JPL Horizons with RA/DEC,
      r, delta, phase, and (predicted) magnitude.
    * Computes JNow (equinox-of-date) RA/Dec from the Horizons J2000 values.
- After building all items, applies the same brightness filter
  (COBS mag OR v_pred_now <= BRIGHT_LIMIT), sorts by brightness,
  and truncates to the 15 brightest.
"""

import json
import math
import time
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astroquery.jplhorizons import Horizons

from horizons_pull import (
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
    QUANTITIES,  # intentionally not used here in ephemeris
)

OUT_JSON_PATH = Path("data/comets_orbit_ephem.json")
DAYS = 15
PAUSE_S = 0.25  # small courtesy delay between Horizons calls


# ---------- COBS global list loader ----------

def load_cobs_designations(cobs_list_path: Path) -> Dict[str, Any]:
    """
    Fetch a *global* comet list from the COBS Comet List API (no location filter).

    - Uses BRIGHT_LIMIT (e.g. 15.0) as a maximum allowed current magnitude.
    - Returns a dict with:
        - keys = MPC designations (e.g. "CK25A060" or "3I")
        - values = current magnitude (float)
        - plus:
            "_debug_first_names": sample of full names
            "_debug_counts": basic stats
            "_fullname_map": mapping MPC -> full name

    NOTE: The cobs_list_path argument is used only to write a debug
    snapshot of the raw API response, so you can inspect what COBS
    returned. It is NOT read as an input list and has nothing to do
    with the old, location-based cobs_list.json.
    """
    limit_mag = try_float_env(BRIGHT_LIMIT_ENV)
    if limit_mag is None:
        limit_mag = BRIGHT_LIMIT_DEFAULT
    print(f"[orbit_ephem] COBS global list: using BRIGHT_LIMIT={limit_mag}")

    base_url = "https://cobs.si/api/comet_list.api"

    # Use the BRIGHT_LIMIT as the current-magnitude cutoff for the API.
    # COBS expects an integer here; we round up a little for safety.
    api_mag_limit = int(math.ceil(limit_mag))
    params_base = {
        "format": "json",
        "cur-mag": str(api_mag_limit),
    }

    cobs_map: Dict[str, float] = {}
    fullname_map: Dict[str, str] = {}
    debug_counts: Dict[str, int] = {
        "total_objects": 0,
        "with_mpc_name": 0,
        "within_mag_limit": 0,
        "pages_fetched": 0,
    }

    all_objects: List[Dict[str, Any]] = []  # for optional snapshot
    last_info: Dict[str, Any] = {}

    page = 1
    while True:
        params = dict(params_base)
        params["page"] = str(page)
        print(f"[orbit_ephem] Fetching COBS comet_list.api page {page} ...")

        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        info = data.get("info", {})
        objects = data.get("objects", [])
        last_info = info
        debug_counts["pages_fetched"] += 1
        debug_counts["total_objects"] += len(objects)
        all_objects.extend(objects)

        for obj in objects:
            # MPC designation field name per COBS docs
            mpc_name = obj.get("mpc_name") or obj.get("mpc") or obj.get("name")
            if not mpc_name:
                continue

            debug_counts["with_mpc_name"] += 1

            # Use current magnitude if available, fallback to cur_mag
            cur_mag = obj.get("current_mag", obj.get("cur_mag"))
            try:
                mag_val = float(cur_mag)
            except (TypeError, ValueError):
                continue

            # Apply the *same* BRIGHT_LIMIT cut here
            if mag_val > limit_mag:
                continue

            debug_counts["within_mag_limit"] += 1

            # Keep the "best" (brightest) value if duplicated
            if (mpc_name not in cobs_map) or (mag_val < cobs_map[mpc_name]):
                cobs_map[mpc_name] = mag_val
                fullname_map[mpc_name] = obj.get("fullname") or obj.get("name", mpc_name)

        total_pages = int(info.get("pages", 1) or 1)
        if page >= total_pages:
            break
        page += 1

    # Build debug helper fields expected by main()
    comet_ids = sorted(cobs_map.keys())
    debug_first_names = [fullname_map[cid] for cid in comet_ids[:10]]

    result: Dict[str, Any] = dict(cobs_map)
    result["_debug_first_names"] = debug_first_names
    result["_debug_counts"] = debug_counts
    result["_fullname_map"] = fullname_map

    print(
        f"[orbit_ephem] COBS global list: "
        f"{debug_counts['total_objects']} objects across "
        f"{debug_counts['pages_fetched']} pages; "
        f"{len(comet_ids)} unique MPC IDs within cur-mag <= {api_mag_limit}"
    )

    # Optional: write a snapshot of what we fetched so you can inspect it
    try:
        cobs_list_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "info": last_info,
            "objects": all_objects,
            "signature": {
                "source": "COBS Comet List API (global)",
                "bright_limit": limit_mag,
                "api_cur_mag_limit": api_mag_limit,
                "fetched_utc": now_iso(),
            },
        }
        cobs_list_path.write_text(json.dumps(snapshot, indent=2))
        print(f"[orbit_ephem] Wrote COBS snapshot to {cobs_list_path}")
    except Exception as e:
        print(f"[orbit_ephem] Warning: could not write COBS snapshot: {e}")

    return result


# ---------- COBS ↔ Horizons ID mapping helpers ----------

def map_cobs_id_to_horizons_target(raw_id: str) -> str:
    """
    Map a COBS MPC/name-style ID to the Horizons target string.

    We keep this extremely conservative on purpose:

    - For *almost all* objects this is just an identity mapping.
    - For 3I/ATLAS, if COBS ever gives a zero-padded code like "0003I"
      or "003I", we map that to plain "3I" and let the Horizons /
      SBDB logic resolve it. We do NOT try to guess "3I/ATLAS" or
      "C/2025 N1 (ATLAS)" here.
    """
    if not raw_id:
        return raw_id

    rid = raw_id.strip()

    # Minimal special case: zero-padded 3I → "3I"
    if rid in ("0003I", "003I"):
        return "3I"

    return rid


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

    # Let Horizons return the full default ephemeris so that r, alpha
    # (phase angle), V, m1, k1 are all available.
    obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=epochs)
    eph = obj.ephemerides()

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
    # COBS global designations (no location dependence)
    cobs_map = load_cobs_designations(Path("data/cobs_list_global_snapshot.json"))
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
