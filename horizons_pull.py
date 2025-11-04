#!/usr/bin/env python3
"""
Fetch comet distances from JPL Horizons and write data/comets_ephem.json

- Query at 'now' using epochs=[JD].
- Resolve ambiguous periodic comet designations (e.g., 2P, 12P, 13P) by picking
  the most recent apparition (favoring last N years) and retrying via the
  unique numeric Horizons record id.
- Use observer ephemeris for RA/DEC/delta. Compute r (heliocentric) and phase
  angle from state vectors (Sun->comet and Earth->comet), so we always have
  consistent values even if Horizons omits r/alpha columns.
"""

import json
import time
import re
from math import acos, degrees
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 5  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter; you can swap to your site dict later
YEARS_WINDOW = 6                 # choose most recent apparition, prefer within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V (we still compute r/alpha ourselves)
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"

# Minimal fixed list to validate pipeline;
# we’ll swap to a COBS-driven dynamic list after this works.
COMETS: List[str] = [
    "2P",
    "13P",
    "C/2023 A3",
    # "12P",  # you can enable this; resolver handles ambiguity
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
    rn = _vec_norm(sx, sy, sz)     # heliocentric distance r
    dn = _vec_norm(ex, ey, ez)     # geocentric distance delta (should match ephemeris delta)
    dot = sx*ex + sy*ey + sz*ez
    c = max(-1.0, min(1.0, dot / (rn * dn)))
    return degrees(acos(c)), rn, dn


def _colmap(cols) -> Dict[str, str]:
    """Map lowercase -> actual column name for case-insensitive access."""
    return {c.lower(): c for c in cols}


def _get_float(row, cmap: Dict[str, str], key_lower: str) -> Optional[float]:
    """Safely fetch a float from a table row using case-insensitive key."""
    k = cmap.get(key_lower)
    if not k:
        return None
    try:
        return float(row[k])
    except Exception:
        return None


def _row_to_payload(row) -> Dict[str, Any]:
    # astropy Table Row
    cols = getattr(row, "colnames", None) or row.table.colnames
    cmap = _colmap(cols)
    out = {
        "r_au":      _get_float(row, cmap, "r"),       # might be None (not always included)
        "delta_au":  _get_float(row, cmap, "delta"),
        "phase_deg": _get_float(row, cmap, "alpha"),   # might be None
        "ra_deg":    _get_float(row, cmap, "ra"),
        "dec_deg":   _get_float(row, cmap, "dec"),
        "vmag":      _get_float(row, cmap, "v"),
    }
    if any(v is None for v in out.values()):
        out["_cols"] = list(cmap.values())
    return out


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/delta from ephemeris, and r/phase from vectors."""
    jd_now = Time.now().jd

    # 1) Try by designation
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
        core = _row_to_payload(row)

        # Get vectors to compute r and phase angle
        try:
            v_sun = _query_vectors("@10", comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
        except Exception:
            # designation may be ambiguous for P/ comets; resolve to record id
            rec_id = resolve_ambiguous_to_record_id(comet_id)
            if rec_id is None:
                raise
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            core["horizons_id"] = rec_id

        alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
        core["r_au"] = r_au if core.get("r_au") is None else core["r_au"]
        core["delta_au"] = core["delta_au"] if core.get("delta_au") is not None else delta_vec_au
        core["phase_deg"] = alpha_deg if core.get("phase_deg") is None else core["phase_deg"]

        return {"id": comet_id, "epoch_utc": now_iso(), **core}

    except Exception as e1:
        # 2) If ephemeris itself failed (often due to ambiguity), resolve and retry entirely
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}

        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now)
            row = eph[0]
            core = _row_to_payload(row)

            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)

            core["r_au"] = r_au if core.get("r_au") is None else core["r_au"]
            core["delta_au"] = core["delta_au"] if core.get("delta_au") is not None else delta_vec_au
            core["phase_deg"] = alpha_deg if core.get("phase_deg") is None else core["phase_deg"]

            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}
        except Exception as e2:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": f"{e1} | retry:{e2}"}


def main():
    results: List[Dict[str, Any]] = []
    for cid in COMETS:
        results.append(fetch_one(cid, OBSERVER))
        time.sleep(PAUSE_S)

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "items": results,
    }
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
