#!/usr/bin/env python3
"""
Standalone generator for data/comets_orbit_ephem.json for the brightest comets.

Key points:

- Uses its *own* load_cobs_designations() that calls the global COBS
  Comet List API (no location / observer filter) and applies the
  BRIGHT_LIMIT magnitude cut there.
- Does NOT depend on the old, location-based cobs_list.json from
  horizons_pull.py. The cobs_list_path argument is only used as a
  debug snapshot output (fresh each run).
- For each comet that passes the COBS brightness cut:
    * Fetches osculating elements from JPL SBDB via sbdb_elements()
      and augments them with a, Q, orbital period, mean motion.
    * Builds a 15-day daily ephemeris via JPL Horizons with RA/DEC,
      r, delta, phase, and (predicted) magnitude.
    * Computes JNow (equinox-of-date) RA/Dec from the Horizons J2000 values.
- After building all items, applies a brightness filter
  (COBS mag OR v_pred_now <= BRIGHT_LIMIT), sorts by brightness,
  and truncates to the 15 brightest.

No per-comet manual mapping is used. The only normalization is:

- Strip leading zeros from interstellar-style IDs like "0003I" → "3I"
  (applied generically to any NNNI pattern).
- For packed MPC codes like "K10B020" where Horizons doesn't recognise
  the short code, we automatically fall back to using the full COBS
  name "P/2010 B2 (WISE)" when querying Horizons, without hard-coding
  that name in the script.
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
COBS_SNAPSHOT_PATH = Path("data/cobs_list_global_snapshot.json")


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

            # Apply the BRIGHT_LIMIT cut here
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


# ---------- COBS ↔ Horizons ID normalization ----------

# Specific Horizons alias for periodic comets where COBS uses the
# provisional / packed designation but Horizons only knows the
# numbered periodic one.
SPECIAL_HORIZONS_ALIASES: Dict[str, str] = {
    # P/2010 B2 (WISE) – COBS uses packed code K10B020.
    # Use the numeric Horizons ID (Rec #) instead of the name.
    "K10B020": "90001394",
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


def map_cobs_id_to_horizons_target(raw_id: str) -> str:
    """
    Map a COBS MPC/name-style ID to the Horizons target string.

    - Trim + uppercase
    - Strip leading zeros for generic interstellar patterns (NNNI)
    - If we have a known Horizons alias (e.g. K10B020 → 412P/WISE),
      use that.
    - Otherwise return the normalized ID unchanged.
    """
    if not raw_id:
        return raw_id

    rid = raw_id.strip().upper()
    rid = _strip_leading_zeros_in_interstellar(rid)

    # Check for special Horizons aliases (like K10B020 → 412P/WISE)
    if rid in SPECIAL_HORIZONS_ALIASES:
        return SPECIAL_HORIZONS_ALIASES[rid]

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


# ---------- Horizons ephemeris helper with generic fallback ----------

def _get_horizons_ephemerides_for_label(
    label: str,
    observer: str,
    epochs: List[float],
):
    """
    Core helper: given a *single* label string, try the standard
    Horizons resolution path (rec_id + a couple of id_types).
    """
    errors: List[Exception] = []

    # 1) Try with rec_id if available
    rec_id = resolve_ambiguous_to_record_id(label)
    candidates: List[Tuple[str, str]] = []
    if rec_id:
        candidates.append((rec_id, "smallbody"))
        candidates.append((rec_id, "id"))
        candidates.append((rec_id, "designation"))

    # 2) Try the label itself under several id_types
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
            errors.append(e)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Horizons ephemerides failed for {label!r} with no detail")


def build_ephemeris_span(
    primary_label: str,
    observer: str,
    days: int = DAYS,
    alt_label: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build a multi-day ephemeris with RA/DEC, r, delta, phase, V, v_pred.

    We query Horizons at 1-day spacing starting at *now*.

    - primary_label: main ID from COBS (after normalization)
    - alt_label: optional full name from COBS, e.g. "P/2010 B2 (WISE)"

    This will try the primary label first; if that fails, it will try
    the alt_label (if provided). The first one that works wins, and the
    function returns (ephemeris_list, label_used_for_horizons).
    """
    jd0 = Time.now().jd
    epochs = [jd0 + float(i) for i in range(days)]

    errors: List[str] = []
    eph = None
    label_used = primary_label

    # Try primary label
    try:
        eph_primary, used_id, used_type = _get_horizons_ephemerides_for_label(
            primary_label, observer, epochs
        )
        eph = eph_primary
        label_used = used_id  # what we actually passed to Horizons
    except Exception as e:
        errors.append(f"{primary_label}: {e}")

    # If that failed and we have an alternate label, try that
    if eph is None and alt_label:
        try:
            eph_alt, used_id, used_type = _get_horizons_ephemerides_for_label(
                alt_label, observer, epochs
            )
            eph = eph_alt
            label_used = used_id
        except Exception as e:
            errors.append(f"{alt_label}: {e}")

    if eph is None:
        raise RuntimeError(
            "Horizons ephemerides resolution failed for labels: " + "; ".join(errors)
        )

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

    return out, label_used


# ---------- per-comet wrapper ----------

def fetch_orbit_and_ephem(
    comet_id: str,
    observer: str,
    full_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For a comet designation from COBS:

      - Normalize the COBS ID via map_cobs_id_to_horizons_target().
      - Fetch extended orbit from SBDB/Horizons using that primary label,
        with fallback to the full name if needed.
      - Build 15-day Horizons ephemeris using the same labels, with a
        generic fallback from primary_label -> full_name.
      - Derive v_pred_now from the first ephemeris row (if available).

    We keep the original COBS ID in the 'id' field, and store the
    actual Horizons label we ended up using in 'horizons_target' for
    transparency. No manual per-comet name mapping is used.
    """
    primary_label = map_cobs_id_to_horizons_target(comet_id)

    # --- orbit ---
    orbit = sbdb_orbit_extended(primary_label)
    if not orbit and full_name and full_name != primary_label:
        orbit = sbdb_orbit_extended(full_name)

    # --- ephemeris ---
    ephem: List[Dict[str, Any]] = []
    error: Optional[str] = None
    label_used_for_ephem: str = primary_label

    try:
        ephem, label_used_for_ephem = build_ephemeris_span(
            primary_label, observer, days=DAYS, alt_label=full_name
        )
    except Exception as e:
        error = str(e)

    item: Dict[str, Any] = {
        "id": comet_id,  # original COBS ID (e.g. "K10B020")
        "horizons_target": label_used_for_ephem,  # what we actually fed to Horizons
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
    cobs_map = load_cobs_designations(COBS_SNAPSHOT_PATH)
    debug_first_names = cobs_map.pop("_debug_first_names", [])
    debug_counts = cobs_map.pop("_debug_counts", {})
    fullname_map = cobs_map.pop("_fullname_map", {})

    comet_ids: List[str] = sorted(cobs_map.keys())
    print(f"[orbit_ephem] Loaded {len(comet_ids)} COBS designations; sample: {comet_ids[:10]}")

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        print(f"[orbit_ephem] Fetching {cid} ...")
        full_name = fullname_map.get(cid)
        item = fetch_orbit_and_ephem(cid, OBSERVER, full_name=full_name)

        # Attach observed magnitude from COBS
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]

        # Attach pretty full name if present
        if full_name:
            item["name_full"] = full_name

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


