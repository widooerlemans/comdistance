#!/usr/bin/env python3
"""
Build data/comets_orbit_ephem.json for the brightest comets.

- Uses the same COBS-driven list + brightness filter (<= BRIGHT_LIMIT)
- For each comet, queries:
    * JPL SBDB for extended orbital elements
    * Horizons for a 15-day RA/DEC + brightness ephemeris

Output (simplified):

{
  "generated_utc": "...",
  "observer": "500",
  "days": 15,
  "filter": {...},
  "items": [
    {
      "id": "C/2025 K1",
      "display_name": "C/2025 K1 (ATLAS)",
      "cobs_mag": 9.9,
      "v_pred_now": 12.44,
      "orbit": {
        "epoch_jd_tdb": ...,
        "frame": "ecliptic J2000",
        "type": "comet",
        "e": ...,
        "q_au": ...,
        "Q_au": ...,
        "a_au": ...,
        "i_deg": ...,
        "Omega_deg": ...,
        "omega_deg": ...,
        "Tp_jd_tdb": ...,
        "M_deg": ...,
        "period_days": ...,
        "period_years": ...,
        "n_deg_per_day": ...,
        "solution": "...",
        "reference": "JPL SBDB (via Horizons)"
      },
      "ephemeris_15d": [
        {
          "epoch_utc": "...",
          "r_au": ...,
          "delta_au": ...,
          "phase_deg": ...,
          "ra_deg": ...,
          "dec_deg": ...,
          "vmag": ...,
          "v_pred": ...
        },
        ...
      ]
    },
    ...
  ]
}
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from astropy.time import Time
from astroquery.jplhorizons import Horizons

# Reuse helpers & config from the existing script
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
)

OUTPATH = "data/comets_orbit_ephem.json"
OUT_JSON_PATH = Path(OUTPATH)
DAYS = 15
PAUSE_S = 0.2

# Use the same quantities as the main script:
# r, delta, alpha, RA, DEC, V (+ comet photom where available)
QUANTITIES = "1,3,4,20,21,31"

SBDB_TIMEOUT_S = 25


# ---------- JPL SBDB extended orbit ----------

def _fnum(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _pick(d: dict, *names):
    for k in names:
        if k in d and d[k] is not None:
            return d[k]
    return None


def sbdb_orbit_extended(label: str) -> Optional[Dict[str, Any]]:
    """
    Extended SBDB orbit query.

    Returns a dict with:
      e, q_au, Q_au, a_au, i_deg, Omega_deg, omega_deg, Tp_jd_tdb,
      M_deg, period_days, period_years, n_deg_per_day, epoch_jd_tdb, frame, type
    """
    if not label:
        return None

    url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={label}&full-prec=true"
    try:
        r = requests.get(url, timeout=SBDB_TIMEOUT_S)
        if r.status_code != 200:
            return None
        js = r.json()

        orb = js.get("orbit") or {}
        if isinstance(orb, dict) and isinstance(orb.get("elements"), dict):
            elems = orb["elements"]
        else:
            elems = orb
        if not isinstance(elems, dict):
            return None

        e = _fnum(_pick(elems, "e"))
        q = _fnum(_pick(elems, "q"))
        a = _fnum(_pick(elems, "a"))
        Q = _fnum(_pick(elems, "Q", "ad"))
        inc = _fnum(_pick(elems, "incl", "i"))
        node = _fnum(_pick(elems, "node", "om", "Omega"))
        argp = _fnum(_pick(elems, "argp", "w", "omega"))
        M = _fnum(_pick(elems, "ma", "M", "mean_anomaly"))
        tp = _fnum(_pick(elems, "tp_tdb", "tp"))
        ep = _fnum(_pick(elems, "epoch_tdb", "epoch", "epoch_jd"))
        per_days = _fnum(_pick(elems, "per", "per_d"))
        per_years = _fnum(_pick(elems, "per_y"))
        n_deg = _fnum(_pick(elems, "n", "n_ave"))

        # Compute missing a / Q / period if possible for bound orbits
        if a is None and q is not None and e is not None and e < 1.0:
            try:
                a = q / (1.0 - e)
            except ZeroDivisionError:
                pass

        if Q is None and a is not None and e is not None and e < 1.0:
            Q = a * (1.0 + e)

        if per_days is None and a is not None and a > 0 and e is not None and e < 1.0:
            # Kepler's 3rd law: P(years)^2 = a^3, with a in AU
            try:
                per_years = math.sqrt(a ** 3) if per_years is None else per_years
                per_days = per_years * 365.25
            except Exception:
                pass
        elif per_years is None and per_days is not None:
            per_years = per_days / 365.25

        if n_deg is None and per_days is not None and per_days > 0:
            n_deg = 360.0 / per_days

        if ep is None or e is None or inc is None or node is None or argp is None:
            return None

        out = {
            "epoch_jd_tdb": ep,
            "frame": "ecliptic J2000",
            "e": e,
            "q_au": q,
            "Q_au": Q,
            "a_au": a,
            "i_deg": inc,
            "Omega_deg": node,
            "omega_deg": argp,
            "Tp_jd_tdb": tp,
            "M_deg": M,
            "period_days": per_days,
            "period_years": per_years,
            "n_deg_per_day": n_deg,
        }

        # Label type heuristically
        typ = "comet"
        try:
            object_type = (js.get("object", {}) or {}).get("orbit_class", {}).get("name", "")
            if "asteroid" in str(object_type).lower():
                typ = "asteroid"
        except Exception:
            pass
        out["type"] = typ

        # Solution/reference metadata
        out["solution"] = (orb.get("solution") or js.get("orbit", {}).get("solution") or "osculating")
        out["reference"] = "JPL SBDB (via Horizons)"

        return out
    except Exception:
        return None


# ---------- 15-day ephemeris via Horizons ----------

def build_ephemeris_span(designation: str, observer: str, days: int = DAYS) -> List[Dict[str, Any]]:
    """
    Build a 15-day ephemeris with RA/DEC, r, delta, phase, V, v_pred.
    """
    jd0 = Time.now().jd
    epochs = [jd0 + float(i) for i in range(days)]

    # Handle ambiguous designations the same way as horizons_pull.py
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
    for i, row in enumerate(eph):
        # r comes directly from Horizons ephemerides
        try:
            r_au = float(row["r"])
        except Exception:
            r_au = None

        core = _row_to_payload_with_photometry(row, r_au=r_au, delta_vec_au=None)

        jd = float(row["datetime_jd"])
        t = Time(jd, format="jd", scale="tdb").to_datetime(timezone.utc)
        epoch_iso = t.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        entry = {"epoch_utc": epoch_iso}
        entry.update(core)
        out.append(entry)

    return out


# ---------- Per-comet wrapper ----------

def fetch_orbit_and_ephem(comet_id: str, observer: str) -> Dict[str, Any]:
    """
    For a comet designation:

    - Gets extended orbit from SBDB
    - Builds 15-day Horizons ephemeris
    """
    orbit = sbdb_orbit_extended(comet_id)
    ephem = []
    error = None

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

    # Derive a "now" predicted magnitude from the first ephem row if present
    if ephem:
        first = ephem[0]
        vpred = first.get("v_pred") or first.get("vmag")
        if vpred is not None:
            try:
                item["v_pred_now"] = float(vpred)
            except Exception:
                pass

    return item


# ---------- Main ----------

def main():
    # Load COBS map exactly like the existing script
    cobs_map = load_cobs_designations(Path("data/cobs_list.json"))
    debug_first_names = cobs_map.pop("_debug_first_names", [])
    debug_counts = cobs_map.pop("_debug_counts", {})
    fullname_map = cobs_map.pop("_fullname_map", {})

    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else []
    print(f"[orbit_ephem] Loaded {len(comet_ids)} COBS designations; sample: {comet_ids[:10]}")

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        print(f"[orbit_ephem] Fetching {cid} ...")
        item = fetch_orbit_and_ephem(cid, OBSERVER)

        # Attach observed magnitude from COBS
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]

        # Attach pretty full name if present in COBS
        if cid in fullname_map:
            item["name_full"] = fullname_map[cid]

        # Display name fallback
        item["display_name"] = item.get("name_full") or item["id"]

        results.append(item)
        time.sleep(PAUSE_S)

    # Brightness filter (same logic as horizons_pull.py)
    limit = try_float_env(BRIGHT_LIMIT_ENV)
    if limit is None:
        limit = BRIGHT_LIMIT_DEFAULT

    before = len(results)
    filtered: List[Dict[str, Any]] = []
    for it in results:
        obs = it.get("cobs_mag")
        # Use v_pred_now as the theoretical brightness proxy
        pred = it.get("v_pred_now")

        def _leq(v, lim):
            try:
                return v is not None and float(v) <= lim
            except Exception:
                return False

        if _leq(obs, limit) or _leq(pred, limit):
            filtered.append(it)

    results = filtered
    print(f"[orbit_ephem] Brightness filter (<= {limit}) kept {len(results)}/{before}")

    # Sort by observed brightness, then predicted, same as original
    results.sort(key=_sort_key)

    # Optionally restrict to the 15 brightest
    if len(results) > 15:
        results = results[:15]
        print(f"[orbit_ephem] Truncated to top 15 brightest")

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "days": DAYS,
        "items": results,
        "script": "horizons_orbit_ephem.py",
        "filter": {
            "mode": "cobs_or_pred",
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
