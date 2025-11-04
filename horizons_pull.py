#!/usr/bin/env python3
"""
Fetch comet distances from JPL Horizons and write data/comets_ephem.json

- Query at 'now' using a list with one Julian Date: epochs=[JD].
- If a periodic comet designation is ambiguous, parse the ambiguity table and
  choose the most recent apparition (preferably within last N years), then
  retry using the unique numeric record id.
- Column access is robust: we map ephemeris column names case-insensitively and
  include available column names in errors if something is missing.
"""

import json
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer apparitions within this many years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"

COMETS: List[str] = [
    "2P",
    "13P",
    "C/2023 A3",
    # "12P",  # you can re-enable; resolver will handle it
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Parse ambiguity table lines like: "90000091    2022    2P   2P   Encke"
_ROW = re.compile(r"^\s*(?P<rec>9\d{7})\s+(?P<epoch>\d{4})\s+")


def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1

    # Prefer within the recent window
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
    """One-shot query with a tiny retry if Horizons glitches."""
    try:
        obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
        eph = obj.ephemerides(quantities=QUANTITIES)
        return eph
    except Exception as e:
        # Retry once after a short pause if it's the TLIST/WLDINI kind of hiccup
        msg = str(e)
        if try_again and ("no TLIST" in msg or "WLDINI" in msg):
            time.sleep(0.8)
            obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
            return obj.ephemerides(quantities=QUANTITIES)
        raise


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
    cols = getattr(row, "colnames", None) or row.table.colnames  # astropy table row
    cmap = _colmap(cols)
    out = {
        "r_au":    _get_float(row, cmap, "r"),
        "delta_au":_get_float(row, cmap, "delta"),
        "phase_deg":_get_float(row, cmap, "alpha"),
        "ra_deg":  _get_float(row, cmap, "ra"),
        "dec_deg": _get_float(row, cmap, "dec"),
        "vmag":    _get_float(row, cmap, "v"),
    }
    # If some keys are missing, include what columns we actually got (debug aid)
    if any(v is None for v in out.values()):
        out["_cols"] = list(cmap.values())
    return out


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch single-epoch ephemeris at 'now' (JD list) for one comet."""
    jd_now = Time.now().jd
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
        core = _row_to_payload(row)
        if core["r_au"] is None or core["delta_au"] is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": "missing columns", **core}
        return {"id": comet_id, "epoch_utc": now_iso(), **core}
    except Exception as e1:
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now)
            row = eph[0]
            core = _row_to_payload(row)
            if core["r_au"] is None or core["delta_au"] is None:
                return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(),
                        "error": "missing columns", **core}
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
        "items": results,
    }
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
