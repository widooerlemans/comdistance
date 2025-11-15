#!/usr/bin/env python3
"""
augment_tracks_beta.py  (v2)
---------------------------
Reads:  data/comets_ephem.json        (your current daily file)
Writes: data/comets_ephem_beta.json   (beta file with hourly 'track' from Horizons)

ENV (optional):
  OBSERVER=500            # geocenter or your MPC site code (e.g. "Z23")
  ENABLE_COMET_TRACKS=1   # set 0/false to skip adding tracks
  EPHEM_SPAN_DAYS=2       # how far into the future to sample
  EPHEM_STEP_MIN=60       # cadence in minutes (e.g., 30 or 60)
  LIMIT_N=0               # limit number of comets for testing (0 = all)
"""
import os, json, sys, time, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from astroquery.jplhorizons import Horizons

IN_PATH   = Path("data/comets_ephem.json")
BETA_PATH = Path("data/comets_ephem_beta.json")

OBSERVER = os.environ.get("OBSERVER", "500")
ENABLE   = os.environ.get("ENABLE_COMET_TRACKS", "1").lower() not in ("0","false","no","")
SPAN_D   = float(os.environ.get("EPHEM_SPAN_DAYS", "2"))
STEP_M   = int(float(os.environ.get("EPHEM_STEP_MIN", "60")))
LIMIT_N  = int(os.environ.get("LIMIT_N", "0"))  # 0 = all

QUANTITIES = "1,3,4,20,21,31"  # r, delta, phase(alpha), RA, DEC, V

# When Horizons says "Ambiguous target name", it prints a table like:
#   9xxxxxxx 2024 ...  -> we pick the record with the most recent epoch (largest year)
AMBIG_LINE = re.compile(r"^\s*(9\d{7})\s+(\d{4})\s+", re.MULTILINE)

def horizons_ephem(id_value: str, id_type: Optional[str], start_dt, span_days, step_min):
    stop_dt = start_dt + timedelta(days=span_days)
    step_str = f"{step_min} m" if step_min < 60 else f"{int(step_min/60)} h"
    obj = Horizons(
        id=id_value,
        id_type=id_type,
        location=OBSERVER,
        epochs={
            "start": start_dt.strftime("%Y-%m-%d %H:%M"),
            "stop" : stop_dt.strftime("%Y-%m-%d %H:%M"),
            "step" : step_str,
        },
    )
    return obj.ephemerides(quantities=QUANTITIES)

def build_track_from_eph(eph):
    out: List[Dict[str, Any]] = []
    for row in eph:
        rec = {
            "epoch_utc": str(row["datetime_str"]),
            "ra_deg":    float(row["RA"]),
            "dec_deg":   float(row["DEC"]),
            "r_au":      float(row["r"]),
            "delta_au":  float(row["delta"]),
            "phase_deg": float(row["alpha"]),
        }
        try:
            rec["vmag"] = float(row["V"])
        except Exception:
            pass
        out.append(rec)
    return out

def resolve_and_track(item: Dict[str, Any], start_dt):
    # Try several identifiers: designation first, then display name, and let Horizons guess.
    candidates = []
    desig = item.get("id") or item.get("designation")
    name  = item.get("display_name") or item.get("name_full") or item.get("name")
    if desig: candidates.append((desig, "designation"))
    if name and name != desig: candidates.append((name, "designation"))
    if desig: candidates.append((desig, None))  # let Horizons infer

    last_error = None
    for id_value, id_type in candidates:
        try:
            eph = horizons_ephem(id_value, id_type, start_dt, SPAN_D, STEP_M)
            return build_track_from_eph(eph)
        except Exception as e:
            msg = str(e)
            last_error = msg
            if "Ambiguous target name" in msg:
                # Parse Horizons' ambiguity list and pick the most recent record id.
                best_id, best_epoch = None, -1
                for rec_id, epoch in AMBIG_LINE.findall(msg):
                    ep = int(epoch)
                    if ep > best_epoch:
                        best_epoch = ep
                        best_id = rec_id
                if best_id:
                    try:
                        eph2 = horizons_ephem(best_id, "smallbody", start_dt, SPAN_D, STEP_M)
                        return build_track_from_eph(eph2)
                    except Exception as e2:
                        last_error = f"{msg} | fallback smallbody {best_id} failed: {e2}"
    raise RuntimeError(last_error or "Unknown Horizons error")

def main():
    if not ENABLE:
        print("[beta v2] Tracks disabled via ENABLE_COMET_TRACKS=0; nothing to do.")
        return

    if not IN_PATH.exists():
        print(f"[beta v2] ERROR: {IN_PATH} not found. Run your existing daily job first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        print("[beta v2] ERROR: Unexpected JSON structure; 'items' must be a list.", file=sys.stderr)
        sys.exit(1)

    if LIMIT_N and LIMIT_N > 0:
        items = items[:LIMIT_N]

    start_dt = datetime.now(timezone.utc)
    updated = 0
    for item in items:
        label = item.get("display_name") or item.get("id") or item.get("designation") or "?"
        print(f"[beta v2] Building track for: {label} ...", flush=True)
        try:
            item["track"] = resolve_and_track(item, start_dt)
            updated += 1
            print(f"[beta v2]   OK: {len(item['track'])} points", flush=True)
            time.sleep(0.2)  # courtesy to Horizons
        except Exception as e:
            item["track_error"] = str(e)
            print(f"[beta v2]   ERROR: {item['track_error']}", flush=True)

    out_payload = {
        **{k: v for k, v in data.items() if k != "items"},
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observer": os.environ.get("OBSERVER", data.get("observer", "500")),
        "items": items,
        "beta": True,
        "track_span_days": SPAN_D,
        "track_step_min": STEP_M
    }
    BETA_PATH.parent.mkdir(parents=True, exist_ok=True)
    BETA_PATH.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[beta v2] Wrote {BETA_PATH} with {updated}/{len(items)} tracks.", flush=True)

if __name__ == "__main__":
    main()
