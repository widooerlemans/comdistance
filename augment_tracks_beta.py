#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_tracks_beta.py — v21.1 (safe columns)

Reads data/comets_ephem.json, queries JPL Horizons for hourly RA/Dec tracks
over ±EPHEM_SPAN_DAYS (default 2 days) with EPHEM_STEP_MIN spacing (default 60m),
and writes data/comets_ephem_beta.json.

Changes from the earlier runnable version:
- No longer raises KeyError when Horizons omits 'r' or 'delta' (the "'r'" errors you saw).
- We require only datetime_str, RA, DEC; r/delta are optional.
- Keeps the same env vars and output shape so your workflow & HTML don’t need edits.
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from astroquery.jplhorizons import Horizons

DATA_IN  = "data/comets_ephem.json"
DATA_OUT = "data/comets_ephem_beta.json"

SPAN_DAYS = float(os.getenv("EPHEM_SPAN_DAYS", "2"))
STEP_MIN  = int(os.getenv("EPHEM_STEP_MIN", "60"))
OBSERVER  = os.getenv("OBSERVER", "500")
LIMIT_N   = int(os.getenv("LIMIT_N", "0"))

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _dtfmt(dt: datetime) -> str:
    # Horizons expects "YYYY-MM-DD HH:MM"
    return dt.strftime("%Y-%m-%d %H:%M")

def _time_window() -> tuple[str, str, str]:
    now = datetime.now(timezone.utc)
    half = timedelta(days=SPAN_DAYS / 2.0)
    start = _dtfmt(now - half)
    stop  = _dtfmt(now + half)
    step  = f"{STEP_MIN}m"
    return start, stop, step

def _target_for(item: Dict[str, Any]) -> dict:
    # Prefer numeric Horizons small-body ID if present, else use designation
    hid = str(item.get("horizons_id") or "").strip()
    if hid:
        return {"id": hid, "id_type": "smallbody"}
    # Fall back to the comet designation/name string
    return {"id": str(item.get("id") or item.get("name_full") or ""), "id_type": "smallbody"}

def fetch_track(item: Dict[str, Any], start_utc: str, stop_utc: str, step_str: str) -> List[Dict[str, Any]]:
    tgt = _target_for(item)
    obj = Horizons(
        id=tgt["id"],
        id_type=tgt["id_type"],
        location=str(OBSERVER),
        epochs={"start": start_utc, "stop": stop_utc, "step": step_str},
    )

    # Default ephemerides often include RA/DEC always; r/delta may be absent for odd cases.
    eph = obj.ephemerides(refsystem="J2000")

    # Require only datetime_str, RA, DEC
    for col in ("datetime_str", "RA", "DEC"):
        if col not in eph.colnames:
            raise KeyError(f"Missing '{col}' for target {tgt['id']}")

    has_r     = "r" in eph.colnames
    has_delta = "delta" in eph.colnames

    out: List[Dict[str, Any]] = []
    for row in eph:
        rec = {
            "time_utc": str(row["datetime_str"]),
            "ra_deg": float(row["RA"]),
            "dec_deg": float(row["DEC"]),
        }
        if has_r:
            rec["r_au"] = float(row["r"])
        if has_delta:
            rec["delta_au"] = float(row["delta"])
        out.append(rec)
    return out

def main() -> None:
    with open(DATA_IN, "r", encoding="utf-8") as f:
        base = json.load(f)

    items = list(base.get("items", []))
    if LIMIT_N > 0:
        items = items[:LIMIT_N]

    start_utc, stop_utc, step_str = _time_window()

    result = {
        "generated_utc": _utc_now_iso(),
        "observer": str(OBSERVER),
        "script_version": 21,  # keep the same version tag your JSON showed
        "beta": True,
        "track_span_days": SPAN_DAYS,
        "track_step_min": STEP_MIN,
        "items": [],
    }

    ok = 0
    total = 0
    for item in items:
        total += 1
        copy = dict(item)
        try:
            track = fetch_track(copy, start_utc, stop_utc, step_str)
            copy["track"] = track
            ok += 1
            print(f"[augment_tracks_beta] OK: {copy.get('display_name', copy.get('id'))} → {len(track)} points")
        except Exception as e:
            copy["track_error"] = str(e)
            print(f"[augment_tracks_beta] ERROR: {copy.get('display_name', copy.get('id'))}: {e}")
        result["items"].append(copy)

    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[augment_tracks_beta] Wrote {DATA_OUT} with {ok}/{total} tracks (span={SPAN_DAYS}d, step={STEP_MIN}m, observer={OBSERVER}).")

if __name__ == "__main__":
    main()
