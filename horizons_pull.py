#!/usr/bin/env python3
# horizons_pull.py — preserves your original fields; adds display_name built from COBS fullname

from __future__ import annotations
import os, json, sys, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from astroquery.jplhorizons import Horizons
from astropy.coordinates import Angle

DATA_DIR = Path("data")
COBS_FILE = DATA_DIR / "cobs_list.json"
OUT_FILE = DATA_DIR / "comets_ephem.json"

BRIGHT_LIMIT = float(os.environ.get("BRIGHT_LIMIT", "15"))
OBSERVER_CODE = "500"
YEARS_WINDOW = 6
SCRIPT_VERSION = 18

PAREN = re.compile(r"\(([^)]+)\)\s*$")

def suffix_from_full(full: Optional[str]) -> Optional[str]:
    if not full:
        return None
    m = PAREN.search(full)
    if m:
        s = (m.group(1) or "").strip()
        return s or None
    if "/" in full and "(" not in full:
        # handle "3I/ATLAS" → "ATLAS"
        return full.split("/", 1)[1].strip() or None
    return None

def canonical_full(desig: str, full: Optional[str]) -> str:
    """
    Build the user-facing full name with the designation first:
    - if full = "C/2025 A6 (Lemmon)" → return "C/2025 A6 (Lemmon)"
    - if full = "3I/ATLAS"            → return "3I/ATLAS"
    - if full missing but suffix known → "C/2025 A6 (Suffix)" or "3I/Suffix"
    - else fallback to desig
    """
    if full:
        suf = suffix_from_full(full)
        if suf:
            if "/" in desig and "(" not in desig:
                # e.g., "3I" + "ATLAS" → "3I/ATLAS"
                return f"{desig}/{suf}"
            else:
                return f"{desig} ({suf})"
        return full.strip()
    return desig

def load_cobs() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    if not COBS_FILE.exists():
        return m
    try:
        cobs = json.loads(COBS_FILE.read_text(encoding="utf-8"))
    except Exception:
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
            "trend": o.get("trend"),
            "constellation": o.get("constelation"),
        }
    return m

def horizons_ephem_for(desig: str) -> Optional[Dict[str, Any]]:
    try:
        now = datetime.now(timezone.utc)
        jd = (now.timestamp() / 86400.0) + 2440587.5
        obj = Horizons(id=desig, location=OBSERVER_CODE, epochs=jd)
        eph = obj.ephemerides()
        if len(eph) == 0:
            return None
        row = eph[0]
        r_au = float(row["r"])
        delta_au = float(row["delta"])
        phase_deg = float(row["alpha"])
        ra_deg = Angle(str(row["RA"]), unit="hourangle").degree
        dec_deg = Angle(str(row["DEC"]), unit="deg").degree
        try:
            v_pred = float(row["V"])
        except Exception:
            v_pred = None
        horizons_id = None
        try:
            horizons_id = str(row["targetname"])
        except Exception:
            pass
        return {
            "r_au": r_au,
            "delta_au": delta_au,
            "phase_deg": phase_deg,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "ra_jnow_deg": ra_deg,
            "dec_jnow_deg": dec_deg,
            "v_pred": v_pred,
            "horizons_id": horizons_id if horizons_id and "/" in horizons_id else None,
        }
    except Exception as e:
        print(f"WARN: Horizons failed for {desig}: {e}", file=sys.stderr)
        return None

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cobs_map = load_cobs()

    # Build candidate list using your brightness filter
    candidates: List[str] = []
    for desig, info in cobs_map.items():
        m = info.get("cobs_mag")
        if isinstance(m, (int, float)) and m <= BRIGHT_LIMIT:
            candidates.append(desig)

    candidates.sort(key=lambda d: ((cobs_map[d].get("cobs_mag") if isinstance(cobs_map[d].get("cobs_mag"), (int, float)) else 99.0), d))

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    items_out: List[Dict[str, Any]] = []
    debug_first = []
    packed_unpacked = fragments = plain = 0

    for desig in candidates:
        info = cobs_map[desig]
        cobs_mag = info.get("cobs_mag")

        if "-" in desig and desig.endswith(("-A", "-B", "-C")):
            fragments += 1
        elif any(k in desig for k in ("P/", "C/", "I/")) or "/" in desig:
            packed_unpacked += 1
        else:
            plain += 1

        eph = horizons_ephem_for(desig)

        item: Dict[str, Any] = {
            "id": desig,
            "epoch_utc": now_utc,
            "r_au": eph.get("r_au") if eph else None,
            "delta_au": eph.get("delta_au") if eph else None,
            "phase_deg": eph.get("phase_deg") if eph else None,
            "ra_deg": eph.get("ra_deg") if eph else None,
            "dec_deg": eph.get("dec_deg") if eph else None,
            "vmag": None,
            "ra_jnow_deg": eph.get("ra_jnow_deg") if eph else None,
            "dec_jnow_deg": eph.get("dec_jnow_deg") if eph else None,
            "v_pred": eph.get("v_pred") if eph else None,
            "cobs_mag": cobs_mag,
            "mag_diff_pred_minus_obs": ((eph.get("v_pred") - cobs_mag) if (eph and isinstance(cobs_mag, (int, float)) and isinstance(eph.get("v_pred"), (int, float))) else None),
            "trend": info.get("trend"),
            "constellation": info.get("constellation"),
        }

        if eph and eph.get("horizons_id"):
            item["horizons_id"] = eph["horizons_id"]

        # Names (from COBS full name)
        full_from_cobs = info.get("cobs_fullname")
        display = canonical_full(desig, full_from_cobs)
        suf = suffix_from_full(full_from_cobs)
        if not suf and isinstance(full_from_cobs, str) and "/" in full_from_cobs and "(" not in full_from_cobs:
            suf = full_from_cobs.split("/", 1)[1].strip() or None

        if suf is not None:
            item["name_suffix"] = suf
        item["name_full"] = display
        item["display_name"] = display

        items_out.append(item)
        if len(debug_first) < 12:
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

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT_FILE} with {len(items_out)} items")

if __name__ == "__main__":
    main()
