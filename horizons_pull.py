#!/usr/bin/env python3
# horizons_pull.py
#
# Outputs: data/comets_ephem.json
#
# Keeps your original schema and adds name fields. Uses:
# - COBS planner list (data/cobs_list.json) as candidates + observed magnitudes
# - JPL Horizons (astroquery) for ephemerides at current time (observer 500)
#
# Env:
#   BRIGHT_LIMIT (float)  : filter by observed magnitude <= BRIGHT_LIMIT, default 15.0

from __future__ import annotations
import os, json, sys, math, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.coordinates import Angle

DATA_DIR = Path("data")
COBS_FILE = DATA_DIR / "cobs_list.json"
OUT_FILE = DATA_DIR / "comets_ephem.json"

BRIGHT_LIMIT = float(os.environ.get("BRIGHT_LIMIT", "15"))
OBSERVER_CODE = "500"  # geocentric
YEARS_WINDOW = 6
SCRIPT_VERSION = 18  # bump

# ---------- helpers for names ----------

PAREN = re.compile(r"\(([^)]+)\)\s*$")

def suffix_from_full(full: str | None) -> Optional[str]:
    if not full:
        return None
    m = PAREN.search(full)
    if m:  # "C/2025 A6 (Lemmon)"
        s = (m.group(1) or "").strip()
        return s or None
    if "/" in full and "(" not in full:  # "24P/Schaumasse" or "3I/ATLAS"
        s = full.split("/", 1)[1].strip()
        return s or None
    return None

def canonical_full(desig: str, full: Optional[str]) -> str:
    """Build the preferred display form given a designation and a COBS full name."""
    if full:
        suf = suffix_from_full(full)
        if suf:
            if ("/" in desig) and ("(" not in desig):
                return f"{desig}/{suf}"   # 24P/Schaumasse, 3I/ATLAS
            else:
                return f"{desig} ({suf})" # C/2025 A6 (Lemmon)
        return full.strip()
    return desig

# ---------- COBS ingest ----------

def load_cobs() -> Dict[str, Dict[str, Any]]:
    """
    Return a map keyed by simple designation (e.g., 'C/2025 A6', '24P', '3I')
    with fields: magnitude, comet_fullname, etc.
    """
    m: Dict[str, Dict[str, Any]] = {}
    if not COBS_FILE.exists():
        print("WARN: COBS list not found, proceeding with empty set", file=sys.stderr)
        return m

    try:
        cobs = json.loads(COBS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARN: failed parsing {COBS_FILE}: {e}", file=sys.stderr)
        return m

    for o in cobs.get("comet_list", []):
        desig = (o.get("comet_name") or "").strip()
        if not desig:
            continue
        m[desig] = {
            "cobs_mag": o.get("magnitude"),
            "cobs_fullname": (o.get("comet_fullname") or "").strip() or None,
            "cobs_type": o.get("comet_type"),
            "cobs_mpc_name": o.get("mpc_name"),
            # useful extra fields you already use in UI:
            "best_time": o.get("best_time"),
            "best_ra": o.get("best_ra"),
            "best_dec": o.get("best_dec"),
            "best_alt": o.get("best_alt"),
            "trend": o.get("trend"),
            "constellation": o.get("constelation"),
        }
    return m

# ---------- Horizons query ----------

def horizons_ephem_for(desig: str) -> Optional[Dict[str, Any]]:
    """
    Query Horizons for one target designation at the current UTC time, observer 500.
    Returns dict with fields matching your original output (ra_deg, dec_deg, r_au, delta_au, phase_deg, v_pred, etc.)
    """
    try:
        now = datetime.now(timezone.utc)
        # Horizons accepts 'id' like 'C/2025 A6', '24P', '3I'
        obj = Horizons(id=desig, location=OBSERVER_CODE, epochs=now.timestamp()*1e3)  # epochs in JD millis? safer: use dict time
        # Better: pass ISO time through epochs={'start':..., 'stop':..., 'step':'1m'} but we only need one row.
        # Using a single epoch as JD: astroquery allows float JD; convert:
        # JD = now timestamp / 86400 + 2440587.5
        jd = (now.timestamp() / 86400.0) + 2440587.5
        obj = Horizons(id=desig, location=OBSERVER_CODE, epochs=jd)
        eph = obj.ephemerides()
        if len(eph) == 0:
            return None
        row = eph[0]

        # Distances (au)
        r_au = float(row["r"])          # heliocentric distance
        delta_au = float(row["delta"])  # observer distance

        # Phase (deg)
        phase_deg = float(row["alpha"])

        # RA/Dec (sexagesimal strings in Horizons), convert to degrees
        ra_str = str(row["RA"])   # e.g. "11 19 47.50" or hh:mm:ss, Horizons uses hours
        dec_str = str(row["DEC"])

        # Robust conversion
        ra_deg = Angle(ra_str, unit="hourangle").degree
        dec_deg = Angle(dec_str, unit="deg").degree

        # Predicted V magnitude from Horizons
        v_pred = None
        try:
            v_pred = float(row["V"])
        except Exception:
            v_pred = None

        # Horizon's targetname sometimes useful as 'horizons_id' surrogate for periodic comets
        horizons_id = None
        try:
            # row['targetname'] looks like '24P/Schaumasse'
            horizons_id = str(row["targetname"])
        except Exception:
            pass

        return {
            "r_au": r_au,
            "delta_au": delta_au,
            "phase_deg": phase_deg,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "ra_jnow_deg": ra_deg,   # close enough for our purposes
            "dec_jnow_deg": dec_deg,
            "v_pred": v_pred,
            "horizons_id": horizons_id if horizons_id and "/" in horizons_id else None,
        }
    except Exception as e:
        print(f"WARN: Horizons failed for {desig}: {e}", file=sys.stderr)
        return None

# ---------- main build ----------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cobs_map = load_cobs()
    # pick candidates with observed magnitude <= BRIGHT_LIMIT (and that have a numeric magnitude)
    candidates: List[str] = []
    for desig, info in cobs_map.items():
        m = info.get("cobs_mag")
        if isinstance(m, (int, float)):
            if m <= BRIGHT_LIMIT:
                candidates.append(desig)
    # stable order by observed brightness, then designation
    candidates.sort(key=lambda d: (cobs_map[d].get("cobs_mag") if isinstance(cobs_map[d].get("cobs_mag"), (int, float)) else 99.0, d))

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    items_out: List[Dict[str, Any]] = []
    debug_first = []
    packed_unpacked = 0
    fragments = 0
    plain = 0

    for desig in candidates:
        info = cobs_map[desig]
        cobs_mag = info.get("cobs_mag")

        # count buckets (for your debug)
        if "-" in desig and desig.endswith(("-A", "-B", "-C")):
            fragments += 1
        elif any(k in desig for k in ("P/", "C/", "I/")) or "/" in desig:
            packed_unpacked += 1
        else:
            plain += 1

        # Horizons data
        eph = horizons_ephem_for(desig)

        # Assemble item in your original shape
        item: Dict[str, Any] = {
            "id": desig,
            "epoch_utc": now_utc,
            "r_au": eph.get("r_au") if eph else None,
            "delta_au": eph.get("delta_au") if eph else None,
            "phase_deg": eph.get("phase_deg") if eph else None,
            "ra_deg": eph.get("ra_deg") if eph else None,
            "dec_deg": eph.get("dec_deg") if eph else None,
            "vmag": None,  # preserve your original null
            "ra_jnow_deg": eph.get("ra_jnow_deg") if eph else None,
            "dec_jnow_deg": eph.get("dec_jnow_deg") if eph else None,
            "v_pred": eph.get("v_pred") if eph else None,
            "cobs_mag": cobs_mag,
            "mag_diff_pred_minus_obs": ( (eph.get("v_pred") - cobs_mag) if (eph and isinstance(cobs_mag, (int, float)) and isinstance(eph.get("v_pred"), (int, float))) else None ),
        }

        # Drop None horizons_id to match your earlier style (only present for some)
        if eph and eph.get("horizons_id"):
            item["horizons_id"] = eph["horizons_id"]

        # Carry through helpful COBS bits you already showed in your debug outputs
        # (They won't break your existing consumers and are nice for UI)
        item.update({
            "trend": info.get("trend"),
            "constellation": info.get("constellation"),
        })

        # --- Name augmentation (the only *new* thing you asked for) ---
        full_from_cobs = info.get("cobs_fullname")  # e.g., "C/2025 A6 (Lemmon)" or "24P/Schaumasse" or "3I/ATLAS"
        display = canonical_full(desig, full_from_cobs)
        suf = suffix_from_full(full_from_cobs)
        # For slash style we still want name_suffix taken from right-hand part:
        if not suf and isinstance(full_from_cobs, str) and "/" in full_from_cobs and "(" not in full_from_cobs:
            partsuf = full_from_cobs.split("/", 1)[1].strip()
            suf = partsuf or None

        # Add the three fields (non-destructive)
        if suf is not None:
            item["name_suffix"] = suf
        item["name_full"] = display
        item["display_name"] = display
        # --------------------------------------------------------------

        items_out.append(item)
        if len(debug_first) < 12:
            # MPC packed code list in your old debug — reuse mpc_name if present; fall back to desig
            debug_first.append(info.get("cobs_mpc_name") or desig)

    out: Dict[str, Any] = {
        "generated_utc": now_utc,
        "observer": OBSERVER_CODE,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": float(BRIGHT_LIMIT),
        "sorted_by": "cobs_mag asc, then v_pred asc",
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
        },
        "cobs_designations": len(cobs_map),
        "cobs_used": True,
        "debug_first_cobs_names": debug_first,
        "debug_counts": {
            "packed_unpacked": packed_unpacked,
            "fragments": fragments,
            "plain": plain,
        },
        "count": len(items_out),
        "items": items_out,
    }

    # Write compact JSON to match your previous style
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT_FILE} with {len(items_out)} items (bright_limit <= {BRIGHT_LIMIT})")

if __name__ == "__main__":
    main()
