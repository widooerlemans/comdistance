#!/usr/bin/env python3
"""
Fetch comet distances from JPL Horizons and write data/comets_ephem.json

- Uses designations first (e.g., "2P", "13P", "C/2023 A3").
- If designation is ambiguous (multiple apparitions), parse Horizons' table and
  pick the most recent apparition (preferably within last N years), then retry
  using the unique numeric record id.
- Uses a single-epoch query at "now" by passing a Julian Date (JD) to `epochs`.
"""

import json
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time   # <-- important: to produce JD
from astroquery.jplhorizons import Horizons

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter (robust). Later you can use your site dict.
YEARS_WINDOW = 6                 # prefer apparitions within last N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"

# Small fixed list to confirm pipeline; we’ll swap to COBS later
COMETS: List[str] = [
    "2P",          # Encke
    "13P",         # Olbers
    "C/2023 A3",   # Tsuchinshan-ATLAS
    # "12P",       # You can add this; resolver handles ambiguity
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Parse ambiguity table lines like:
#   90000091    2022    2P   2P   Encke
_ROW = re.compile(r"""
    ^\s*(?P<rec>9\d{7})\s+   # numeric record id, e.g., 90000091
    (?P<epoch>\d{4})\s+      # epoch year, e.g., 2022
""", re.VERBOSE)


def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    """Choose the most recent acceptable record id from Horizons' ambiguity table."""
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1

    # Prefer inside window
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

    # Else newest overall
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
    """
    Try designation expecting ambiguity; parse the table and pick the most recent
    apparition (within YEARS_WINDOW if possible). Return numeric record id or None.
    """
    try:
        # Intentionally call with designation to trigger ambiguity for P/ comets
        jd_now = Time.now().jd
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=jd_now)\
            .ephemerides(quantities="1")
        return None  # not ambiguous
    except Exception as e:
        msg = str(e)
        if "Ambiguous target name" not in msg:
            return None
        return _pick_recent_record(msg, YEARS_WINDOW)


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch single-epoch ephemeris at 'now' (JD) for one comet."""
    jd_now = Time.now().jd  # <-- this is the key fix
    try:
        obj = Horizons(id=comet_id, id_type="designation", location=observer, epochs=jd_now)
        eph = obj.ephemerides(quantities=QUANTITIES)
        row = eph[0]
        return {
            "id": comet_id,
            "epoch_utc": now_iso(),
            "r_au": float(row["r"]),
            "delta_au": float(row["delta"]),
            "phase_deg": float(row["alpha"]),
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
            "vmag": float(row["V"]),
        }
    except Exception as e1:
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            obj = Horizons(id=rec_id, id_type="smallbody", location=observer, epochs=jd_now)
            eph = obj.ephemerides(quantities=QUANTITIES)
            row = eph[0]
            return {
                "id": comet_id,         # keep designation for readability
                "horizons_id": rec_id,  # expose unique record used
                "epoch_utc": now_iso(),
                "r_au": float(row["r"]),
                "delta_au": float(row["delta"]),
                "phase_deg": float(row["alpha"]),
                "ra_deg": float(row["RA"]),
                "dec_deg": float(row["DEC"]),
                "vmag": float(row["V"]),
            }
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
