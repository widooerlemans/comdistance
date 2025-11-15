#!/usr/bin/env python3
"""
augment_tracks_beta.py
----------------------
Create a *beta* JSON with high‑cadence Horizons tracks for each comet, without
touching your existing production JSON.

Reads:  data/comets_ephem.json        (your current daily file)
Writes: data/comets_ephem_beta.json   (NEW beta file for testing)

Usage (manual GitHub Action): python augment_tracks_beta.py

ENV (optional):
  OBSERVER=500            # geocenter or MPC site code (e.g. "Z23")
  ENABLE_COMET_TRACKS=1   # set 0/false to skip adding tracks
  EPHEM_SPAN_DAYS=2       # how far into the future to sample
  EPHEM_STEP_MIN=60       # cadence in minutes (e.g., 30 or 60)
"""
import os, json, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from astroquery.jplhorizons import Horizons

IN_PATH   = Path("data/comets_ephem.json")
BETA_PATH = Path("data/comets_ephem_beta.json")

OBSERVER = os.environ.get("OBSERVER", "500")
ENABLE   = os.environ.get("ENABLE_COMET_TRACKS", "1").lower() not in ("0","false","no","")
SPAN_D   = float(os.environ.get("EPHEM_SPAN_DAYS", "2"))
STEP_M   = int(float(os.environ.get("EPHEM_STEP_MIN", "60")))

QUANTITIES = "1,3,4,20,21,31"  # r, delta, alpha, RA, DEC, V

def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[augment_tracks_beta] ERROR: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def horizons_track(id_value: str, id_type: str, observer: str,
                   start_dt: datetime, span_days: float, step_min: int) -> List[Dict[str, Any]]:
    stop_dt = start_dt + timedelta(days=span_days)
    step_str = f"{step_min} m" if step_min < 60 else f"{int(step_min/60)} h"
    obj = Horizons(
        id=id_value,
        id_type=id_type,
        location=observer,
        epochs={
            "start": start_dt.strftime("%Y-%m-%d %H:%M"),
            "stop" : stop_dt.strftime("%Y-%m-%d %H:%M"),
            "step" : step_str,
        },
    )
    eph = obj.ephemerides(quantities=QUANTITIES)
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

def main():
    if not ENABLE:
        print("[augment_tracks_beta] Tracks disabled via ENABLE_COMET_TRACKS=0; nothing to do.")
        return

    if not IN_PATH.exists():
        print(f"[augment_tracks_beta] ERROR: {IN_PATH} not found. Run your existing daily job first.", file=sys.stderr)
        sys.exit(1)

    data = load_json(IN_PATH)
    items = data.get("items", [])
    if not isinstance(items, list):
        print("[augment_tracks_beta] ERROR: Unexpected JSON structure; 'items' must be a list.", file=sys.stderr)
        sys.exit(1)

    start_dt = datetime.now(timezone.utc)
    updated = 0
    for item in items:
        # prefer numeric Horizons record id if present
        id_value = item.get("horizons_id") or item.get("id") or item.get("designation")
        id_type  = "smallbody" if item.get("horizons_id") else "designation"
        if not id_value:
            item["track_error"] = "missing id"
            continue
        try:
            item["track"] = horizons_track(
                id_value=id_value,
                id_type=id_type,
                observer=OBSERVER,
                start_dt=start_dt,
                span_days=SPAN_D,
                step_min=STEP_M,
            )
            updated += 1
            time.sleep(0.2)  # be nice to Horizons
        except Exception as e:
            item["track_error"] = str(e)

    # Stamp and write BETA file (do NOT overwrite the production JSON)
    out_payload = {
        **{k: v for k, v in data.items() if k != "items"},
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observer": os.environ.get("OBSERVER", data.get("observer", "500")),
        "items": items,
        "beta": True,
        "track_span_days": SPAN_D,
        "track_step_min": STEP_M
    }
    save_json(BETA_PATH, out_payload)
    print(f"[augment_tracks_beta] Wrote {BETA_PATH} with {updated}/{len(items)} tracks "
          f"(span={SPAN_D}d, step={STEP_M}m, observer={OBSERVER}).")

if __name__ == "__main__":
    main()
