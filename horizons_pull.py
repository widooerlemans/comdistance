#!/usr/bin/env python3
"""
horizons_pull.py — build a JSON list of comets visible from COBS + names from JPL Horizons

- Reads COBS planner JSON from: data/cobs_list.json
- Filters by observed magnitude (<= BRIGHT_LIMIT, default 15)
- Tries to enrich each comet with a full name from JPL Horizons (`targetname`)
- Guarantees human-facing naming as: "Designation (Suffix)" or "65P/Gunn" etc.
- Writes: data/comets_ephem.json

Dependencies (installed in your workflow):
    pip install numpy astropy astroquery
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# astroquery is optional at runtime: script runs even if Horizons fails
try:
    from astroquery.jplhorizons import Horizons  # type: ignore
    _HAS_HORIZONS = True
except Exception:
    _HAS_HORIZONS = False


BRIGHT_LIMIT = float(os.environ.get("BRIGHT_LIMIT", "15").strip() or 15)

COBS_PATH = os.path.join("data", "cobs_list.json")
OUT_PATH = os.path.join("data", "comets_ephem.json")


# ----------------------------- helpers: parsing names ----------------------------- #

_DESIG_RE = re.compile(
    r"""^\s*(?:
            (?P<cat>[CPDXAI])/             # C/ P/ D/ (and A/, X/, I/ just in case)
            (?P<year>\d{4})\s*
            (?P<half>[A-Za-z]{1,2}\d*)     # e.g. A6, R2, J3, QE78
         |
            (?P<num>\d{1,3})(?P<ptype>[CPDXAI])  # 65P, 141P, etc. (with optional -fragment)
            (?:-(?P<frag>[A-Za-z]))?
        )""",
    re.VERBOSE
)

def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _designation_from_fields(comet_name: str | None, mpc_name: str | None, comet_fullname: str | None) -> str:
    """
    Prefer the clean, human designation from COBS `comet_name` (usually like "C/2025 A6").
    Fallbacks try to extract from fullname or even from MPC packed id if needed.
    """
    name = _clean(comet_name)
    if name:
        # Often already a clean designation
        return name

    full = _clean(comet_fullname)
    if full:
        # Examples:
        # "C/2025 A6 (Lemmon)"  -> "C/2025 A6"
        # "65P/Gunn"            -> "65P"
        # "141P-B/Machholz"     -> "141P-B"
        m = _DESIG_RE.search(full)
        if m:
            if m.group("cat"):
                return f"{m.group('cat')}/{m.group('year')} {m.group('half')}"
            else:
                desig = f"{m.group('num')}{m.group('ptype')}"
                if m.group("frag"):
                    desig = f"{desig}-{m.group('frag')}"
                return desig

    # Very last resort: try to turn something like "0065P" into "65P"
    mpc = _clean(mpc_name)
    if mpc and len(mpc) >= 5 and mpc[-1].isalpha():
        # e.g. 0065P -> 65P ; 0145P -> 145P
        try:
            return f"{int(mpc[:-1])}{mpc[-1]}"
        except Exception:
            pass

    # If completely unknown, give back whatever we had
    return name or full or mpc or "Unknown"


def _suffix_from_fullname(fullname: str | None) -> Optional[str]:
    """
    Pull the discoverer/short name from a fullname.
      - "C/2025 A6 (Lemmon)"          -> "Lemmon"
      - "65P/Gunn"                     -> "Gunn"
      - "141P-B/Machholz"              -> "Machholz"
      - "C/2023 A3 (Tsuchinshan-ATLAS)"-> "Tsuchinshan-ATLAS"
    """
    full = _clean(fullname)
    if not full:
        return None

    # Parenthetical first
    m = re.search(r"\(([^)]+)\)", full)
    if m:
        return _clean(m.group(1))

    # Slash form: 65P/Gunn , 141P-B/Machholz
    if "/" in full:
        parts = full.split("/", 1)
        if len(parts) == 2:
            return _clean(parts[1])

    return None


def _display_name_from_parts(desig: str | None, suffix: str | None, fullname: str | None) -> str:
    """
    Build a human-facing name with designation first, then discoverer in parentheses.
    Rules:
      - If fullname already contains the designation *first*, keep fullname.
      - Else if we have desig + suffix, format "desig (Suffix)".
      - Else if we only have fullname, return it.
      - Else return desig.
    This preserves slash forms like "65P/Gunn" or "141P-B/Machholz" when fullname already starts with the designation.
    """
    desig = _clean(desig)
    suffix = _clean(suffix)
    fullname = _clean(fullname)

    if desig and fullname and fullname.startswith(desig):
        return fullname

    if desig and suffix:
        # Avoid double parentheses if suffix is already parenthesized
        if suffix.startswith("(") and suffix.endswith(")"):
            return f"{desig} {suffix}"
        return f"{desig} ({suffix})"

    if fullname:
        return fullname

    return desig or ""


def _horizons_targetname(desig: str) -> Optional[str]:
    """
    Ask JPL Horizons for a `targetname` for this designation.
    Returns a string like "C/2025 A6 (Lemmon)" if available.
    """
    if not _HAS_HORIZONS:
        return None
    try:
        # One epoch is enough — we only want metadata/targetname
        # 500@10: geocentric, but the site doesn’t matter for targetname
        now_jd = _now_jd()
        obj = Horizons(id=desig, id_type='designation', epochs=now_jd, location='500@10')
        eph = obj.ephemerides()
        # astroquery returns an astropy Table. 'targetname' is a column on rows.
        if eph and "targetname" in eph.colnames and len(eph) > 0:
            tn = str(eph[0]["targetname"])
            return _clean(tn)
    except Exception:
        # Horizons might not know the very newest objects (or rate-limited) — just skip
        return None
    return None


def _now_jd() -> float:
    # astropy not strictly required here; we can compute a good-enough JD
    # but since astropy is present, do it properly if available.
    try:
        from astropy.time import Time  # type: ignore
        return Time(datetime.now(timezone.utc)).jd
    except Exception:
        # Fallback: rough JD (not used for precise ephemerides here)
        #  Unix epoch JD ~ 2440587.5
        return 2440587.5 + (datetime.now(timezone.utc).timestamp() / 86400.0)


# ----------------------------- data model ----------------------------- #

@dataclass
class CometItem:
    # Core identifiers
    desig: str
    name_suffix: Optional[str] = None       # e.g. "Lemmon"
    name_full: Optional[str] = None         # e.g. "C/2025 A6 (Lemmon)"
    display_name: Optional[str] = None      # always designation-first

    # Basic observability (from COBS)
    magnitude: Optional[float] = None
    best_time: Optional[str] = None
    best_ra: Optional[str] = None
    best_dec: Optional[str] = None
    best_alt: Optional[float] = None
    trend: Optional[str] = None
    constellation: Optional[str] = None

    # Raw COBS refs for debugging / provenance
    cobs_type: Optional[str] = None
    cobs_mpc_name: Optional[str] = None
    cobs_fullname: Optional[str] = None


# ----------------------------- main logic ----------------------------- #

def load_cobs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_items_from_cobs(cobs: Dict[str, Any]) -> List[CometItem]:
    out: List[CometItem] = []
    lst = cobs.get("comet_list", []) or []

    for row in lst:
        try:
            mag = row.get("magnitude", None)
            # Filter by bright limit
            if mag is None or float(mag) > BRIGHT_LIMIT:
                continue

            comet_name = row.get("comet_name")  # often "C/2025 A6"
            comet_fullname = row.get("comet_fullname")  # often "C/2025 A6 (Lemmon)" or "65P/Gunn"
            mpc_name = row.get("mpc_name")

            desig = _designation_from_fields(comet_name, mpc_name, comet_fullname)

            # Preferred suffix: Horizons (if returns), else from COBS fullname parsing
            horizons_fullname = _horizons_targetname(desig) or None
            suffix = _suffix_from_fullname(horizons_fullname) or _suffix_from_fullname(comet_fullname)

            display_name = _display_name_from_parts(desig, suffix, horizons_fullname or comet_fullname)

            item = CometItem(
                desig=desig,
                name_suffix=suffix,
                name_full=display_name,            # keep full aligned with display
                display_name=display_name,
                magnitude=float(mag) if mag is not None else None,
                best_time=row.get("best_time"),
                best_ra=row.get("best_ra"),
                best_dec=row.get("best_dec"),
                best_alt=row.get("best_alt"),
                trend=row.get("trend"),
                constellation=row.get("constelation") or row.get("constellation"),
                cobs_type=row.get("comet_type"),
                cobs_mpc_name=mpc_name,
                cobs_fullname=comet_fullname,
            )
            out.append(item)
        except Exception:
            # Never let one bad row kill the run
            continue

    # Sort by magnitude ascending (brightest first), then by designation
    out.sort(key=lambda it: (it.magnitude if it.magnitude is not None else 99.9, it.desig))
    return out


def main() -> None:
    try:
        cobs = load_cobs(COBS_PATH)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: COBS file not found at {COBS_PATH}. Make sure the workflow fetched it first.")

    items = build_items_from_cobs(cobs)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "cobs": True,
            "jpl_horizons": _HAS_HORIZONS,
        },
        "filters": {
            "bright_limit_le": BRIGHT_LIMIT,
        },
        "items": [asdict(it) for it in items],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUT_PATH} with {len(items)} items ≤ mag {BRIGHT_LIMIT}")


if __name__ == "__main__":
    main()
