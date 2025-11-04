#!/usr/bin/env python3
import json, time, re
from datetime import datetime, timezone
from typing import List, Dict, Any
from astroquery.jplhorizons import Horizons

# Use geocenter for reliability (you can switch to your site later)
OBSERVER = "500"

OUTPATH = "data/comets_ephem.json"
QUANTITIES = "1,3,4,20,21,31"  # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3

# Fixed list for the first successful run — *designations only*
COMETS = ["2P", "13P", "2P", "C/2023 A3"]

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# (kept for later when we switch to COBS names; safe to leave in now)
_designation_pat = re.compile(r"""
    ^(
        \d+\s*P                 # e.g. 12P or '12 P'
       |[PCADX]/\d{4}\s+[A-Z]\d+  # e.g. C/2023 A3, P/1995 O1, A/2017 U1, D/1993 F2, X/...
    )
""", re.IGNORECASE | re.VERBOSE)

def to_designation(s: str) -> str:
    s = s.strip()
    m = _designation_pat.match(s)
    if m: return re.sub(r"\s+", " ", m.group(1)).upper()
    m = re.match(r"^\s*(\d+)\s*P\s*/", s, re.IGNORECASE)
    if m: return (m.group(1) + "P").upper()
    m = re.search(r"\(([^)]+)\)", s)
    if m: return to_designation(m.group(1))
    return s

def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    epoch = datetime.utcnow()
    desig = to_designation(comet_id)
    try:
        obj = Horizons(id=desig, id_type="designation", location=observer, epochs=epoch)
        eph = obj.ephemerides(quantities=QUANTITIES)
        row = eph[0]
        return {
            "id": desig,
            "epoch_utc": now_iso(),
            "r_au": float(row["r"]),
            "delta_au": float(row["delta"]),
            "phase_deg": float(row["alpha"]),
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
            "vmag": float(row["V"]),
        }
    except Exception as e:
        return {"id": desig, "epoch_utc": now_iso(), "error": str(e)}

def main():
    results: List[Dict[str, Any]] = []
    for cid in COMETS:
        results.append(fetch_one(cid, OBSERVER))
        time.sleep(PAUSE_S)
    payload = {"generated_utc": now_iso(), "observer": OBSERVER, "items": results}
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
