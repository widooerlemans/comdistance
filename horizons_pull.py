#!/usr/bin/env python3
"""
horizons_pull.py — build a JSON list of comets visible from COBS + names from JPL Horizons

- Reads COBS planner JSON from: data/cobs_list.json
- Filters by observed magnitude (<= BRIGHT_LIMIT, default 15)
- Enriches each comet's human-facing name with designation-first formatting:
    * Prefer slash forms from COBS when present: "3I/ATLAS", "65P/Gunn", "141P-B/Machholz"
    * Otherwise: "Designation (Suffix)" e.g. "C/2025 A6 (Lemmon)"
- Writes: data/comets_ephem.json
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
    if not s:
        return False
    return bool(_DESIG_RE.match(_clean(s)))

def _designation_from_fields(comet_name: str | None, mpc_name: str | None, comet_fullname: str | None) -> str:
    """Prefer a clean designation from COBS `comet_name`. Fallbacks parse fullname or unpack MPC short code."""
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
    """Extract discoverer/short name from fullname."""
    full = _clean(fullname)
    if not full:
        return None

    m = re.search(r"\(([^)]+)\)", full)  # parenthetical
    if m:
        cand = _clean(m.group(1))
        return cand or None

    if "/" in full:  # slash
        parts = full.split("/", 1)
        if len(parts) == 2:
            cand = _clean(parts[1])
            return cand or None

    return None

def _choose_suffix(desig: str, horizons_fullname: Optional[str], cobs_fullname: Optional[str]) -> Optional[str]:
    """
    Prefer a real discoverer suffix (not another designation).
    Order: Horizons fullname -> COBS fullname -> RHS of COBS slash if startswith desig.
    """
    candidates: List[Optional[str]] = [
        _suffix_from_fullname(horizons_fullname),
        _suffix_from_fullname(cobs_fullname),
    ]

    # If COBS has "Desig/Name", use that name as a candidate explicitly
    cf = _clean(cobs_fullname)
    if cf and cf.startswith(desig) and "/" in cf:
        candidates.append(_clean(cf.split("/", 1)[1]))

    for cand in candidates:
        cand = _clean(cand)
        if not cand:
            continue
        if cand == desig or _looks_like_designation(cand):
            continue
        return cand

    return None

def _display_name_from_parts(desig: str,
                             suffix: Optional[str],
                             horizons_fullname: Optional[str],
                             cobs_fullname: Optional[str]) -> str:
    """
    Preference order for final human-facing name:
      1) If COBS fullname starts with desig and contains '/', KEEP IT (e.g., '3I/ATLAS', '65P/Gunn').
      2) Else if Horizons fullname starts with desig, keep it (e.g., 'C/2025 A6 (Lemmon)').
      3) Else if COBS fullname starts with desig, keep it (covers forms like '141P-B/Machholz').
      4) Else if we have a suffix, format 'desig (Suffix)'.
      5) Else fallback to whichever fullname is present, otherwise 'desig'.
    """
    desig = _clean(desig)
    suffix = _clean(suffix)
    hf = _clean(horizons_fullname)
    cf = _clean(cobs_fullname)

    if cf and cf.startswith(desig) and "/" in cf:
        return cf
    if hf and hf.startswith(desig):
        return hf
    if cf and cf.startswith(desig):
        return cf
    if suffix:
        if suffix.startswith("(") and suffix.endswith(")"):
            return f"{desig} {suffix}"
        return f"{desig} ({suffix})"
    return hf or cf or desig

def _now_jd() -> float:
    try:
        from astropy.time import Time  # type: ignore
        return Time(datetime.now(timezone.utc)).jd
    except Exception:
        return 2440587.5 + (datetime.now(timezone.utc).timestamp() / 86400.0)

def _horizons_targetname(desig: str) -> Optional[str]:
    """Return Horizons 'targetname' (e.g., 'C/2025 A6 (Lemmon)') when available."""
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
            comet_fullname = row.get("comet_fullname")    # e.g., "C/2025 A6 (Lemmon)" or "3I/ATLAS"
            mpc_name = row.get("mpc_name")

            desig = _designation_from_fields(comet_name, mpc_name, comet_fullname)

            horizons_fullname = _horizons_targetname(desig)

            suffix = _choose_suffix(desig, horizons_fullname, comet_fullname)

            display_name = _display_name_from_parts(desig, suffix, horizons_fullname, comet_fullname)

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
            continue

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
