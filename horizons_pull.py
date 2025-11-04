#!/usr/bin/env python3
"""
Fetch comet distances (and brightness estimates) from JPL Horizons
and write data/comets_ephem.json

What this script does:
- Queries "now" using epochs=[JD] (list) to avoid TLIST/WLDINI issues.
- Resolves ambiguous periodic comet designations (e.g., 2P/12P/13P) by picking
  the most recent apparition (favor last N years) and re-querying by numeric id.
- Uses observer ephemeris for RA/DEC/Δ. Computes r (heliocentric) and phase angle
  from state vectors so values are present even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude:
      v_pred = M1 + 5*log10(Δ) + k1*log10(r)
- Optional brightness filter to keep only comets roughly ≤ mag 15–16.
"""

import json
import time
import re
import math
from math import acos, degrees
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 7  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter; swap to your site dict later if you want
YEARS_WINDOW = 6                 # pick the most recent apparition (prefer within N years)
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V (we still compute r/alpha ourselves)
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"

# Optional: only keep comets with predicted magnitude <= this limit
# Set to None to keep everything.
BRIGHT_LIMIT = 15.5

# Hand list to produce useful output right now (designations!).
# You can freely add e.g. "C/2024 S4", "C/2025 A1", "P/2021 A3", etc.
COMETS: List[str] = [
    "2P",
    "12P",
    "13P",
    "C/2023 A3",
    # add more designations below if you like
    # "C/2024 S4",
    # "C/2025 A1",
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Parse ambiguity table lines like: "90000091    2022    2P   2P   Encke"
_ROW = re.compile(r"^\s*(?P<rec>9\d{7})\s+(?P<epoch>\d{4})\s+")


def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    """Choose the most recent acceptable record id from Horizons' ambiguity table."""
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1

    # Prefer within the window first
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rec = m.group("rec")
        epoch = int(m.group("epoch"))
        if epoch >= now_year - years_window and epoch > best_epoch:
            best, best_epoch = rec, epoch
    if best:
        return best

    # Else: newest overall
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rec = m.group("rec")
        epoch = int(m.group("epoch"))
        if epoch > best_epoch:
            best, best_epoch = rec, epoch
    return best


def resolve_ambiguous_to_record_id(designation: str) -> Optional[str]:
    """If designation is ambiguous, pick the most recent apparition's numeric record id."""
    try:
        jd_now = Time.now().jd
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now])\
            .ephemerides(quantities="1")
        return None  # not ambiguous
    except Exception as e:
        msg = str(e)
        if "Ambiguous target name" not in msg:
            return None
        return _pick_recent_record(msg, YEARS_WINDOW)


def _query_ephem(id_value: str, id_type: str, observer, jd_now: float, try_again: bool = True):
    """Observer ephemeris query with a tiny retry for Horizons hiccups."""
    try:
        obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
        return obj.ephemerides(quantities=QUANTITIES)
    except Exception as e:
        msg = str(e)
        if try_again and ("no TLIST" in msg or "WLDINI" in msg):
            time.sleep(0.8)
            obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
            return obj.ephemerides(quantities=QUANTITIES)
        raise


def _query_vectors(location: str, id_value: str, id_type: str, jd_now: float):
    """Vectors query (returns state vectors). location '@10'=Sun, '@399'=Earth geocenter."""
    obj = Horizons(id=id_value, id_type=id_type, location=location, epochs=[jd_now])
    return obj.vectors()


def _vec_norm(x, y, z):
    return (x*x + y*y + z*z) ** 0.5


def _phase_from_vectors(v_sun_row, v_earth_row):
    # Both rows have x,y,z in AU for comet position relative to the center body
    sx, sy, sz = float(v_sun_row["x"]), float(v_sun_row["y"]), float(v_sun_row["z"])     # Sun->comet
    ex, ey, ez = float(v_earth_row["x"]), float(v_earth_row["y"]), float(v_earth_row["z"])  # Earth->comet
    rn = _vec_norm(sx, sy, sz)      # heliocentric distance r (AU)
    dn = _vec_norm(ex, ey, ez)      # geocentric distance Δ (AU)
    dot = sx*ex + sy*ey + sz*ez
    c = max(-1.0, min(1.0, dot / (rn * dn)))
    return degrees(acos(c)), rn, dn


def _colmap(cols) -> Dict[str, str]:
    return {c.lower(): c for c in cols}


def _get_optional_float(row, cmap: Dict[str, str], key_lower: str) -> Optional[float]:
    k = cmap.get(key_lower)
    if not k:
        return None
    try:
        return float(row[k])
    except Exception:
        return None


def _row_to_payload_with_photometry(row, r_au: Optional[float], delta_vec_au: Optional[float]) -> Dict[str, Any]:
    """
    Build a payload from an ephemeris row, case-insensitively reading columns.
    If ephemeris 'delta' is missing, fall back to vector delta.
    If 'V' is missing but M1/k1 are present, compute v_pred = M1 + 5*log10(Δ) + k1*log10(r).
    """
    cols = getattr(row, "colnames", None) or row.table.colnames
    cmap = _colmap(cols)

    ra   = _get_optional_float(row, cmap, "ra")
    dec  = _get_optional_float(row, cmap, "dec")
    delt = _get_optional_float(row, cmap, "delta")
    alpha= _get_optional_float(row, cmap, "alpha")
    vmag = _get_optional_float(row, cmap, "v")
    M1   = _get_optional_float(row, cmap, "m1")
    k1   = _get_optional_float(row, cmap, "k1")

    # prefer ephemeris delta if present; else use vector delta
    delta_au = delt if delt is not None else delta_vec_au

    out = {
        "r_au": r_au,
        "delta_au": delta_au,
        "phase_deg": alpha,  # may be None; we fill from vectors if needed
        "ra_deg": ra,
        "dec_deg": dec,
        "vmag": vmag,        # may be None
    }

    # If V is missing but M1/k1 are present, compute predicted magnitude
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    # Debug aid: if critical bits are missing, expose available columns
    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/delta from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # 1) Try by designation
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]

        # vectors for r & phase (try designation, fall back to record id if needed)
        try:
            v_sun = _query_vectors("@10", comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "epoch_utc": now_iso(), **core}
        except Exception:
            rec_id = resolve_ambiguous_to_record_id(comet_id)
            if rec_id is None:
                raise
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}

    except Exception as e1:
        # 2) If ephemeris failed, resolve and retry entirely by record id
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now)
            row = eph[0]
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}
        except Exception as e2:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": f"{e1} | retry:{e2}"}


def main():
    results: List[Dict[str, Any]] = []
    for cid in COMETS:
        results.append(fetch_one(cid, OBSERVER))
        time.sleep(PAUSE_S)

    # Optional: filter on predicted brightness so your UI shows likely-visible comets
    if BRIGHT_LIMIT is not None:
        filtered = []
        for it in results:
            vpred = it.get("v_pred")
            # keep if we have a prediction and it's bright enough
            if (vpred is not None) and (vpred <= BRIGHT_LIMIT):
                filtered.append(it)
        results = filtered

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": BRIGHT_LIMIT,
        "count": len(results),
        "items": results,
    }
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
