#!/usr/bin/env python3
"""
Fetch comet distances from JPL Horizons and write data/comets_ephem.json

- Uses designations first (e.g., "2P", "13P", "C/2023 A3").
- If Horizons says a periodic comet designation is AMBIGUOUS (multiple apparitions),
  we parse the ambiguity table and pick the most recent apparition, preferably within
  a recent-year window (e.g., last 6 years), and retry using that unique numeric record id.
- Output JSON is compact and easy for your frontend to read.
"""

import json
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astroquery.jplhorizons import Horizons

# ---------- CONFIG ----------
# Robust default: geocenter. (You can later switch to your site dict if you want.)
OBSERVER = "500"  # or: {"lon": 5.1214, "lat": 52.0907, "elevation": 10}

# How far back you're willing to accept an apparition (years).
# If no apparition is within this window, we pick the newest overall.
YEARS_WINDOW = 6

# Which quantities to return from Horizons:
# 1=r (heliocentric AU), 3=delta (observer AU), 4=alpha (deg),
# 20=RA (deg), 21=DEC (deg), 31=V magnitude (approx).
QUANTITIES = "1,3,4,20,21,31"

# Be nice to the service
PAUSE_S = 0.3

# Output path
OUTPATH = "data/comets_ephem.json"

# Initial fixed list for a first working run. You can replace this later
# with a dynamic list (e.g., from COBS Planner API). Keep DESIGNATIONS only.
COMETS: List[str] = [
    "2P",          # Encke
    "13P",         # Olbers
    "C/2023 A3",   # Tsuchinshan-ATLAS
    # You can add "12P" now; the resolver will handle ambiguity:
    # "12P",
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Matches the “Record #  Epoch-yr …” lines in Horizons’ ambiguity table.
# Example line:
#   90000091    2022    2P   2P   Encke
_ROW = re.compile(r"""
    ^\s*(?P<rec>9\d{7})\s+   # numeric record id, e.g. 90000091
    (?P<epoch>\d{4})\s+      # epoch year, e.g. 2022
""", re.VERBOSE)


def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    """
    Parse Horizons ambiguity table text and choose the best record id:
    - Prefer any record whose Epoch-yr >= (now_year - years_window), picking the newest of those.
    - If none meet the window, pick the newest overall.
    Returns the record id as a string, or None if none found.
    """
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1

    # First pass: look for candidates within the window.
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

    # Second pass: pick the newest overall if nothing matched the window.
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
    Try the designation in Horizons expecting an ambiguity; when ambiguous,
    parse the returned table and choose the most recent apparition (within window if possible).
    Return the unique numeric record id (string), or None if not resolvable.
    """
    try:
        # This call intentionally tries the designation first.
        # If it doesn't raise, target wasn't ambiguous -> nothing to resolve.
        Horizons(id=designation, id_type="designation", location=OBSERVER).ephemerides(quantities="1")
        return None
    except Exception as e:
        msg = str(e)
        if "Ambiguous target name" not in msg:
            return None
        return _pick_recent_record(msg, YEARS_WINDOW)


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """
    Fetch a single comet's ephemeris (r, delta, etc.).
    1) Try designation directly.
    2) If ambiguous, resolve to numeric record id for the most recent apparition, then retry.
    """
    epoch = datetime.utcnow()

    # 1) Try designation
    try:
        obj = Horizons(id=comet_id, id_type="designation", location=observer, epochs=epoch)
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
        # 2) If ambiguous, resolve and retry using unique record id
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            obj = Horizons(id=rec_id, id_type="smallbody", location=observer, epochs=epoch)
            eph = obj.ephemerides(quantities=QUANTITIES)
            row = eph[0]
            return {
                "id": comet_id,         # keep the human-friendly designation
                "horizons_id": rec_id,  # also expose which unique record was used
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
