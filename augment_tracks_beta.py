#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augment_tracks_beta.py  (v2.0.3)

Build short (±EPHEM_SPAN_DAYS) hourly tracks for comets using JPL Horizons,
and append them into data/comets_ephem_beta.json.

Hotfixes in this version:
- Do NOT restrict `quantities`; let astroquery request the full default set,
  so columns like 'r' and 'delta' are always present.
- Always prefer `horizons_id` (record number) when available to avoid ambiguity.
- Robust column access and clear per-target error messages.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from astroquery.jplhorizons import Horizons

DATA_IN  = "data/comets_ephem.json"
DATA_OUT = "data/comets_ephem_beta.json"

# Env toggles (all optional)
SPAN_DAYS   = float(os.getenv("EPHEM_SPAN_DAYS", "2"))   # total span in days
STEP_MIN    = int(os.getenv("EPHEM_STEP_MIN", "60"))     # step size in minutes
OBSERVER    = os.getenv("OBSERVER", "500")               # '500' = geocenter; can be topocentric code
LIMIT_N     = int(os.getenv("LIMIT_N", "0"))             # 0 means no limit

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _dtfmt(dt: datetime) -> str:
    # Horizons wants UTC calendar strings
    return dt.strftime("%Y-%m-%d %H:%M")

def _split_span(span_days: float) -> (str, str, str):
    """Return (start_utc, stop_utc, step_str) around 'now' with given span and step."""
    now = datetime.now(timezone.utc)
    half = timedelta(days=span_days/2.0)
    start = now - half
    stop  = now + half
    step_str = f"{STEP_MIN}m"
    return _dtfmt(start), _dtfmt(stop), step_str

def _target_for(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return Horizons target selector dict.
    Prefer numeric horizons record when available to avoid ambiguity.
    """
    if "horizons_id" in item and item["horizons_id"]:
        # Numeric record number
        return {"id": str(item["horizons_id"]), "id_type": "smallbody"}
    # Fall back to item['id'] (e.g., 'C/2025 A6', '29P', etc.)
    return {"id": item["id"], "id_type": "smallbody"}

def horizons_ephem(item: Dict[str, Any],
                   start_utc: str, stop_utc: str, step_str: str,
                   location: str) -> List[Dict[str, Any]]:
    """
    Query Horizons and return a list of dict points containing:
    time_utc, ra_deg, dec_deg, delta_au, r_au
    """
    tgt = _target_for(item)
    obj = Horizons(
        id=tgt["id"],
        id_type=tgt["id_type"],
        location=location,
        epochs={"start": start_utc, "stop": stop_utc, "step": step_str},
    )

    # IMPORTANT: do not restrict `quantities` – default includes everything we need.
    eph = obj.ephemerides(refsystem="J2000")  # returns an astropy Table

    # Column names we rely on (astroquery uses simplified names):
    # 'datetime_str', 'RA', 'DEC', 'delta', 'r'
    required = ["datetime_str", "RA", "DEC", "delta", "r"]
    for col in required:
        if col not in eph.colnames:
            raise KeyError(f"Missing column '{col}' in Horizons table for {tgt['id']}")

    points: List[Dict[str, Any]] = []
    for row in eph:
        points.append({
            "time_utc": str(row["datetime_str"]),
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
            "delta_au": float(row["delta"]),
            "r_au": float(row["r"]),
        })
    return points

def build_tracks(base: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "generated_utc": _utc_now_iso(),
        "observer": str(OBSERVER),
        "script_version": 203,
        "beta": True,
        "track_span_days": SPAN_DAYS,
        "track_step_min": STEP_MIN,
        "items": [],
    }

    items = base.get("items", [])
    if LIMIT_N > 0:
        items = items[:LIMIT_N]

    start_utc, stop_utc, step_str = _split_span(SPAN_DAYS)

    ok, total = 0, 0
    for item in items:
        total += 1
        item_copy = dict(item)  # don’t mutate original
        try:
            track = horizons_ephem(item_copy, start_utc, stop_utc, step_str, OBSERVER)
            item_copy["track"] = track
            ok += 1
            print(f"[beta v2.0.3] OK: {item_copy.get('display_name', item_copy.get('id'))} → {len(track)} points")
        except Exception as e:
            # Leave a crisp message
            msg = str(e)
            item_copy["track_error"] = msg
            print(f"[beta v2.0.3] ERROR: {item_copy.get('display_name', item_copy.get('id'))}: {msg}")

        out["items"].append(item_copy)

    print(f"[beta v2.0.3] Wrote {ok}/{total} tracks (span={SPAN_DAYS}d, step={STEP_MIN}m, observer={OBSERVER}).")
    return out

def main() -> None:
    with open(DATA_IN, "r", encoding="utf-8") as f:
        base = json.load(f)

    result = build_tracks(base)

    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
