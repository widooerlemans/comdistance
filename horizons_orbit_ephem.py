#!/usr/bin/env python3
"""
Standalone generator for data/comets_orbit_ephem.json for the brightest comets.

Key points:

- Uses its *own* load_cobs_designations() that calls the global COBS
  Comet List API (no location / observer filter) and applies the
  BRIGHT_LIMIT magnitude cut there.
- For each comet that passes the COBS brightness cut:
    * Fetches osculating elements from JPL SBDB via sbdb_elements()
      and augments them with a, Q, orbital period, mean motion.
    * Builds a 15-day daily ephemeris via JPL Horizons with RA/DEC,
      r, delta, phase, and (predicted) magnitude.
    * Computes JNow (equinox-of-date) RA/Dec from the Horizons J2000 values.
- After building all items, applies a brightness filter
  (COBS mag OR v_pred_now <= BRIGHT_LIMIT), sorts by brightness,
  and truncates to the 15 brightest.

Normalization:

- Strip leading zeros from interstellar-style IDs like "0003I" → "3I"
- For packed MPC codes like "K10B020" we use a specific Horizons alias mapping.
"""

import json
import math
import time
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from astropy.time import Time
# UPDATED IMPORT: Added get_constellation
from astropy.coordinates import SkyCoord, FK5, get_constellation
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
    QUANTITIES,
)

OUT_JSON_PATH = Path("data/comets_orbit_ephem.json")
DAYS = 15
PAUSE_S = 0.25
COBS_SNAPSHOT_PATH = Path("data/cobs_list_global_snapshot.json")

def load_cobs_designations(cobs_list_path: Path) -> Dict[str, Any]:
    limit_mag = try_float_env(BRIGHT_LIMIT_ENV)
    if limit_mag is None:
        limit_mag = BRIGHT_LIMIT_DEFAULT
    print(f"[orbit_ephem] COBS global list: using BRIGHT_LIMIT={limit_mag}")

    base_url = "https://cobs.si/api/comet_list.api"
    api_mag_limit = int(math.ceil(limit_mag))
    params_base = {"format": "json", "cur-mag": str(api_mag_limit)}

    cobs_map: Dict[str, float] = {}
    fullname_map: Dict[str, str] = {}
    debug_counts: Dict[str, int] = {
        "total_objects": 0,
        "with_mpc_name": 0,
        "within_mag_limit": 0,
        "pages_fetched": 0,
    }

    all_objects: List[Dict[str, Any]] = []
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
            mpc_name = obj.get("mpc_name") or obj.get("mpc") or obj.get("name")
            if not mpc_name:
                continue

            debug_counts["with_mpc_name"] += 1
            cur_mag = obj.get("current_mag", obj.get("cur_mag"))
            try:
                mag_val = float(cur_mag)
            except (TypeError, ValueError):
                continue

            if mag_val > limit_mag:
                continue

            debug_counts["within_mag_limit"] += 1

            if (mpc_name not in cobs_map) or (mag_val < cobs_map[mpc_name]):
                cobs_map[mpc_name] = mag_val
                fullname_map[mpc_name] = obj.get("fullname") or obj.get("name", mpc_name)

        total_pages = int(info.get("pages", 1) or 1)
        if page >= total_pages:
            break
        page += 1

    comet_ids = sorted(cobs_map.keys())
    debug_first_names = [fullname_map[cid] for cid in comet_ids[:10]]

    result: Dict[str, Any] = dict(cobs_map)
    result["_debug_first_names"] = debug_first_names
    result["_debug_counts"] = debug_counts
    result["_fullname_map"] = fullname_map

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
    except Exception as e:
        print(f"[orbit_ephem] Warning: could not write COBS snapshot: {e}")

    return result

SPECIAL_HORIZONS_ALIASES: Dict[str, str] = {
    "K10B020": "90001394",
    "141P-B": "141P",
}

def _strip_leading_zeros_in_interstellar(code: str) -> str:
    if not code:
        return code
    code = code.strip().upper()
    m = re.fullmatch(r"0*(\d+I)", code)
    if m:
        return m.group(1)
    return code

def map_cobs_id_to_horizons_target(raw_id: str) -> str:
    if not raw_id:
        return raw_id
    rid = raw_id.strip().upper()
    rid = _strip_leading_zeros_in_interstellar(rid)
    if rid in SPECIAL_HORIZONS_ALIASES:
        return SPECIAL_HORIZONS_ALIASES[rid]
    return rid

def _fnum(x) -> Optional[float]:
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
    base = sbdb_elements(label)
    if not base:
        rec_id = resolve_ambiguous_to_record_id(label)
        idspec = rec_id or label
        base = horizons_elements(idspec)

    if not base:
        return None

    out = dict(base)
    e = _fnum(out.get("e"))
    q = _fnum(out.get("q_au") or out.get("q"))
    a = _fnum(out.get("a_au") or out.get("a"))

    if (e is not None) and (e < 1.0):
        if (a is None) and (q is not None):
            try:
                a = q / (1.0 - e)
                out["a_au"] = a
            except ZeroDivisionError:
                pass

        if (q is None) and (a is not None):
            q = a * (1.0 - e)
            out["q_au"] = q

        if a is not None:
            Q = a * (1.0 + e)
            out["Q_au"] = Q
            try:
                period_years = math.sqrt(a ** 3)
                period_days = period_years * 365.25
                out["period_years"] = period_years
                out["period_days"] = period_days
                out["n_deg_per_day"] = 360.0 / period_days
            except Exception:
                pass

    out.setdefault("solution", "osculating")
    out.setdefault("reference", "JPL SBDB (via Horizons)")
    return out

def _get_horizons_ephemerides_for_label(
    label: str,
    observer: str,
    epochs: List[float],
):
    rec_id = resolve_ambiguous_to_record_id(label)
    candidates: List[Tuple[str, str]] = []
    if rec_id:
        candidates.append((rec_id, "smallbody"))
        candidates.append((rec_id, "id"))
        candidates.append((rec_id, "designation"))
    candidates.append((label, "designation"))
    candidates.append((label, "smallbody"))
    candidates.append((label, "id"))

    last_error: Optional[Exception] = None
    for obj_id, id_type in candidates:
        try:
            obj = Horizons(id=obj_id, id_type=id_type, location=observer, epochs=epochs)
            return obj.ephemerides(), obj_id, id_type
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Horizons ephemerides failed for {label!r}")

def build_ephemeris_span(
    primary_label: str,
    observer: str,
    days: int = DAYS,
    alt_label: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    jd0 = Time.now().jd
    epochs = [jd0 + float(i) for i in range(days)]

    eph = None
    label_used = primary_label
    horizons_name: Optional[str] = None

    try:
        eph_primary, used_id, used_type = _get_horizons_ephemerides_for_label(
            primary_label, observer, epochs
        )
        eph = eph_primary
        label_used = used_id
        try:
            horizons_name = eph_primary.meta.get("targetname") or str(eph_primary["targetname"][0])
        except Exception:
            horizons_name = None
    except Exception:
        pass

    if eph is None and alt_label:
        try:
            eph_alt, used_id, used_type = _get_horizons_ephemerides_for_label(
                alt_label, observer, epochs
            )
            eph = eph_alt
            label_used = used_id
            try:
                horizons_name = eph_alt.meta.get("targetname") or str(eph_alt["targetname"][0])
            except Exception:
                pass
        except Exception:
            pass

    if eph is None:
        raise RuntimeError(f"Horizons ephemerides resolution failed for {primary_label}")

    out: List[Dict[str, Any]] = []
    for row in eph:
        try:
            r_au = float(row["r"])
        except Exception:
            r_au = None

        core = _row_to_payload_with_photometry(row, r_au=r_au, delta_vec_au=None)
        jd = float(row["datetime_jd"])
        t_utc = Time(jd, format="jd", scale="tdb").utc

        ra = core.get("ra_deg")
        dec = core.get("dec_deg")
        if (ra is not None) and (dec is not None):
            try:
                c_j2000 = SkyCoord(
                    ra * u.deg,
                    dec * u.deg,
                    frame=FK5(equinox=Time("J2000"))
                )
                c_jnow = c_j2000.transform_to(FK5(equinox=t_utc))
                core["ra_jnow_deg"] = float(c_jnow.ra.deg)
                core["dec_jnow_deg"] = float(c_jnow.dec.deg)
                # UPDATED: Added constellation lookup
                core["constellation"] = get_constellation(c_j2000)
            except Exception:
                pass

        dt = t_utc.to_datetime(timezone=timezone.utc)
        epoch_iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        entry: Dict[str, Any] = {"epoch_utc": epoch_iso}
        entry.update(core)
        out.append(entry)

    return out, label_used, horizons_name

def fetch_orbit_and_ephem(
    comet_id: str,
    observer: str,
    full_name: Optional[str] = None,
) -> Dict[str, Any]:
    primary_label = map_cobs_id_to_horizons_target(comet_id)
    orbit = sbdb_orbit_extended(primary_label)
    if not orbit and full_name and full_name != primary_label:
        orbit = sbdb_orbit_extended(full_name)

    ephem: List[Dict[str, Any]] = []
    error: Optional[str] = None
    label_used_for_ephem: str = primary_label
    horizons_name: Optional[str] = None

    try:
        ephem, label_used_for_ephem, horizons_name = build_ephemeris_span(
            primary_label, observer, days=DAYS, alt_label=full_name
        )
    except Exception as e:
        error = str(e)

    item: Dict[str, Any] = {
        "id": comet_id,
        "horizons_target": label_used_for_ephem,
        "epoch_utc": now_iso(),
    }

    if orbit:
        item["orbit"] = orbit
    if ephem:
        item["ephemeris_15d"] = ephem
    if error and not ephem:
        item["error"] = error
    if horizons_name:
        item["horizons_name"] = horizons_name

    if ephem:
        first = ephem[0]
        vpred = first.get("v_pred") or first.get("vmag")
        if vpred is not None:
            try:
                item["v_pred_now"] = float(vpred)
            except Exception:
                pass

    return item

def main() -> None:
    cobs_map = load_cobs_designations(COBS_SNAPSHOT_PATH)
    debug_first_names = cobs_map.pop("_debug_first_names", [])
    debug_counts = cobs_map.pop("_debug_counts", {})
    fullname_map = cobs_map.pop("_fullname_map", {})

    comet_ids: List[str] = sorted(cobs_map.keys())
    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        print(f"[orbit_ephem] Fetching {cid} ...")
        full_name = fullname_map.get(cid)
        item = fetch_orbit_and_ephem(cid, OBSERVER, full_name=full_name)

        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]
        if full_name:
            item["name_full"] = full_name

        display_name = item.get("name_full") or item["id"]
        hname = item.get("horizons_name")
        if isinstance(hname, str):
            hname_stripped = hname.strip()
            if re.match(r"^\d+P/", hname_stripped):
                display_name = hname_stripped
        item["display_name"] = display_name
        results.append(item)
        time.sleep(PAUSE_S)

    limit = try_float_env(BRIGHT_LIMIT_ENV) or BRIGHT_LIMIT_DEFAULT
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
        if (_ok(obs) or _ok(pred)) and "error" not in it:
            filtered.append(it)

    results = filtered
    results.sort(key=_sort_key)

    if len(results) > 15:
        results = results[:15]

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
