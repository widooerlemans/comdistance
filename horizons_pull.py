#!/usr/bin/env python3
import json, time
from datetime import datetime, timezone
from typing import List, Dict, Any
from astroquery.jplhorizons import Horizons

# Horizons-friendly IDs (edit as you like)
COMETS = ["12P", "13P", "2P", "C/2023 A3"]

# Start with geocenter for reliability. Later you can use your site:
# OBSERVER = {"lon": 5.1214, "lat": 52.0907, "elevation": 10}
OBSERVER = "500"  # geocenter

OUTPATH = "data/comets_ephem.json"
QUANTITIES = "1,3,4,20,21,31"  # r, delta, alpha, RA, DEC, V

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    epoch = datetime.utcnow()
    try:
        obj = Horizons(id=comet_id, id_type="smallbody", location=observer, epochs=epoch)
        eph = obj.ephemerides(quantities=QUANTITIES)
        row = eph[0]
        return {
            "id": comet_id,
            "epoch_utc": now_iso(),
            "r_au": float(row["r"]),         # Sun–comet (AU)
            "delta_au": float(row["delta"]), # Observer–comet (AU)
            "phase_deg": float(row["alpha"]),
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
            "vmag": float(row["V"]),
        }
    except Exception as e:
        return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e)}

def main():
    results: List[Dict[str, Any]] = []
    for cid in COMETS:
        results.append(fetch_one(cid, OBSERVER))
        time.sleep(0.3)
    payload = {"generated_utc": now_iso(), "observer": OBSERVER, "items": results}
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()