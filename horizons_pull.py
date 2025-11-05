#!/usr/bin/env python3
"""
horizons_pull.py — build a JSON list of comets visible from COBS + names from JPL Horizons

- Reads COBS planner JSON from: data/cobs_list.json
- Filters by observed magnitude (<= BRIGHT_LIMIT, default 15)
- Enriches each comet's human-facing name, enforcing:
    * "Designation (Suffix)" — e.g. "C/2025 A6 (Lemmon)"
    * Keeps slash forms already correct: "65P/Gunn", "141P-B/Machholz"
- Writes: data/comets_ephem.json

Dependencies:
    pip install numpy astropy astroquery
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# astroquery is optional at runtime (network hiccups etc.)
try:
    from astroquery.jplhorizons import Horizons  # type: ignore
    _HAS_HORIZONS = True
except Exception:
    _HAS_HORIZONS = False

BRIGHT_LIMIT = float(os.environ.get("BRIGHT_LIMIT", "15") or 15)

COBS_PATH = os.path.join("data", "cobs_list.json")
OUT_PATH = os.path.join("data", "comets_ephem.json")

# ---------- helpers: cleaning & patterns ---------- #

_DESIG_RE = re.compile(
    r"""^\s*(?:
            (?P<cat>[CPDXAI])/             # C/ P/ D/ A/ X/ I/
            (?P<year>\d{4})\s*
            (?P<half>[A-Za-z]{1,3}\d*)     # e.g. A6, R2, J3, QE78
         |
            (?P<num>\d{1,3})(?P<ptype>[CPDXAI])  # 65P, 141P, etc.
            (?:-(?P<frag>[A-Za-z]))?             # optional fragment letter
        )\s*$""",
    re.VERBOSE,
)

def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _looks_like_designation(s: Optional[str]) -> bool:
    """Return True if s itself is a comet designation pattern (we don't want that as suffix)."""
    if not s:
        return False
    return bool(_DESIG_RE.match(_clean(s)))

def _designation_from_fields(comet_name: str | None, mpc_name: str | None, comet_fullname: str | None) -> str:
    """
    Prefer a clean designation from COBS `comet_name` (often like 'C/2025 A6').
    Fallbacks: parse from fullname or unpack MPC short code ('0065P' -> '65P').
    """
    name = _clean(comet_name)
    if name:
        return name

    full = _clean(comet_fullname)
    if full:
        m = _DESIG_RE.match(full)
        if m:
            if m.group("cat"):  # C/YYYY Half
                return f"{m.group('cat')}/{m.group('year')} {m.group('half')}"
            desig = f"{m.group('num')}{m.group('ptype')}"
            if m.group("frag"):
                desig = f"{desig}-{m.group('frag')}"
            return desig

        # Try extracting before slash: "65P/Gunn" -> "65P"
        if "/" in full:
            before = full.split("/", 1)[0]
            if _DESIG_RE.match(before):
                return _clean(before)

    mpc = _clean(mpc_name)
    if mpc and len(mpc) >= 5 and mpc[-1].isalpha():
        try:
            return f"{int(mpc[:-1])}{mpc[-1]}"  # 0065P -> 65P
        except Exception:
            pass

    return name or full or mpc or "Unknown"

def _suffix_from_fullname(fullname: str | None) -> Optional[str]:
    """
    Extract discoverer/short name from a fullname.
      - "C/2025 A6 (Lemmon)"           -> "Lemmon"
      - "65P/Gunn"                      -> "Gunn"
      - "141P-B/Machholz"               -> "Machholz"
      - "C/2023 A3 (Tsuchinshan-ATLAS)" -> "Tsuchinshan-ATLAS"
    """
    full = _clean(fullname)
    if not full:
        return None

    # Parenthetical form
    m = re.search(r"\(([^)]+)\)", full)
    if m:
        candidate = _clean(m.group(1))
        return candidate or None

    # Slash form
    if "/" in full:
        parts = full.split("/", 1)
        if len(parts) == 2:
            candidate = _clean(parts[1])
            return candidate or None

    return None

def _choose_suffix(desig: str, horizons_fullname: Optional[str], cobs_fullname: Optional[str]) -> Optional[str]:
    """
    Prefer a real discoverer suffix (not another designation).
    Order: Horizons fullname -> COBS fullname -> (as-is slash part if fullname starts with desig)
    Reject any candidate that equals the designation or matches a designation pattern.
    """
    candidates: List[Optional[str]] = [
        _suffix_from_fullname(horizons_fullname),
        _suffix_from_fullname(cobs_fullname),
    ]

    # As an extra fallback: if either fullname starts with the desig and has a slash,
    # keep the right-hand part (e.g., "65P/Gunn")
    for full in (horizons_fullname, cobs_fullname):
        f = _clean(full)
        if f and f.startswith(desig) and "/" in f:
            candidates.append(_clean(f.split("/", 1)[1]))

    for cand in candidates:
        cand = _clean(cand)
        if not cand:
            continue
        # Reject degenerate/alternate designations like "C/2025 A6" or "C/2025 N1"
        if cand == desig or _looks_like_designation(cand):
            continue
        return cand

    return None

def _display_name_from_parts(desig: str, suffix: Optional[str], fullname: Optional[str]) -> str:
    """
    Build a human-facing name with designation first, then discoverer in parentheses.
    Preserve slash-style names when fullname already starts with the designation.
    """
    desig = _clean(desig)
    suffix = _clean(suffix)
    fullname = _clean(fullname)

    # Keep already-correct slash/paren fullname if it starts with the desig
    if fullname and fullname.startswith(desig):
        return fullname

    if suffix:
        if suffix.startswith("(") and suffix.endswith(")"):
            return f"{desig} {suffix}"
        return f"{desig} ({suffix})"

    # Nothing better? fall back to fullname or just the desig
    return fullname or desig

def _now_jd() -> float:
    try:
        from astropy.time import Time  # type: ignore
        return Time(datetime.now(timezone.utc)).jd
    except Exception:
        return 2440587.5 + (datetime.now(timezone.utc).timestamp() / 86400.0)

def _horizons_targetname(desig: str) -> Optional[str]:
    """Return Horizons 'targetname' like 'C/2025 A6 (Lemmon)' when available."""
    if not _HAS_HORIZONS:
        return None
    try:
        obj = Horizons(id=desig, id_type='designation', epochs=_now_jd(), location='500@10')
        eph = obj.ephemerides()
        if eph and "targetname" in eph.colnames and len(eph) > 0:
            return _clean(str(eph[0]["targetname"]))
    except Exception:
        return None
    return None

# ---------- data model ---------- #

@dataclass
class CometItem:
    desig: str
    name_suffix: Optional[str] = None
    name_full: Optional[str] = None
    display_name: Optional[str] = None

    magnitude: Optional[float] = None
    best_time: Optional[str] = None
    best_ra: Optional[str] = None
    best_dec: Optional[str] = None
    best_alt: Optional[float] = None
    trend: Optional[str] = None
    constellation: Optional[str] = None

    cobs_type: Optional[str] = None
    cobs_mpc_name: Optional[str] = None
    cobs_fullname: Optional[str] = None

# ---------- main building ---------- #

def load_cobs(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_items_from_cobs(cobs: Dict[str, Any]) -> List[CometItem]:
    out: List[CometItem] = []
    lst = cobs.get("comet_list", []) or []

    for row in lst:
        try:
            mag = row.get("magnitude", None)
            if mag is None or float(mag) > BRIGHT_LIMIT:
                continue

            comet_name = row.get("comet_name")            # e.g., "C/2025 A6"
            comet_fullname = row.get("comet_fullname")    # e.g., "C/2025 A6 (Lemmon)" or "65P/Gunn"
            mpc_name = row.get("mpc_name")

            desig = _designation_from_fields(comet_name, mpc_name, comet_fullname)

            # Ask Horizons for a fullname (may be None or contain an alternate desig in parens)
            horizons_fullname = _horizons_targetname(desig)

            # Choose a good discoverer suffix, rejecting designation-like candidates
            suffix = _choose_suffix(desig, horizons_fullname, comet_fullname)

            # Final display name
            display_name = _display_name_from_parts(desig, suffix, horizons_fullname or comet_fullname)

            item = CometItem(
                desig=desig,
                name_suffix=suffix,
                name_full=display_name,               # keep 'name_full' aligned with display
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
            # Skip any problematic row without crashing the batch
            continue

    # Sort by brightness then by designation
    out.sort(key=lambda it: (it.magnitude if it.magnitude is not None else 99.9, it.desig))
    return out

def main() -> None:
    try:
        cobs = load_cobs(COBS_PATH)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: COBS file not found at {COBS_PATH}. Did the workflow fetch it?")

    items = build_items_from_cobs(cobs)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"cobs": True, "jpl_horizons": _HAS_HORIZONS},
        "filters": {"bright_limit_le": BRIGHT_LIMIT},
        "items": [asdict(it) for it in items],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUT_PATH} with {len(items)} items ≤ mag {BRIGHT_LIMIT}")

if __name__ == "__main__":
    main()
