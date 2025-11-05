#!/usr/bin/env python3
"""
Post-process data/comets_ephem.json to add:
- name_suffix   (e.g., 'Lemmon', 'ATLAS', 'Schwassmann-Wachmann')
- name_full     (e.g., 'C/2025 A6 (Lemmon)' or '24P/Schaumasse' or '3I/ATLAS')
- display_name  (same as name_full)

This is NON-DESTRUCTIVE: all existing keys/values produced by your original
horizons_pull.py are kept exactly the same. We only add the three name fields,
and we only overwrite them if they are missing.

How names are found:
1) Prefer any existing 'cobs_fullname' already on the item.
2) Otherwise, read data/cobs_list.json (COBS planner), build a map:
     map[comet_name] = comet_fullname
   and use that to fill cobs_fullname for the item by matching 'id'/'desig'.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any

INPATH  = Path("data/comets_ephem.json")
OUTPATH = Path("data/comets_ephem.json")
COBS_PATH = Path("data/cobs_list.json")

# Regex: trailing "(...)" to extract suffix like "(Lemmon)"
_PAREN_SUFFIX = re.compile(r"\(([^)]+)\)\s*$")

def _extract_suffix_from_fullname(cobs_fullname: str | None) -> str | None:
    """
    From 'C/2025 A6 (Lemmon)' -> 'Lemmon'
    From '24P/Schaumasse'      -> 'Schaumasse'
    From '3I/ATLAS'            -> 'ATLAS'
    """
    if not cobs_fullname:
        return None
    s = str(cobs_fullname).strip()
    m = _PAREN_SUFFIX.search(s)
    if m:
        return m.group(1).strip()
    if "/" in s:
        parts = s.split("/", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return None

def _choose_desig(it: Dict[str, Any]) -> str | None:
    """
    Find the designation/id field in your old/new outputs.
    """
    for k in ("desig","designation","id","desig_str","comet_name"):
        v = it.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    # Derive from cobs_fullname if present
    v = it.get("cobs_fullname")
    if isinstance(v,str) and v.strip():
        s = v.strip()
        if "/" in s and "(" not in s:
            # '24P/Schaumasse' -> '24P'
            return s.split("/",1)[0].strip()
        m = _PAREN_SUFFIX.search(s)
        if m:
            # 'C/2025 A6 (Lemmon)' -> 'C/2025 A6'
            return s[:m.start()].strip()
    return None

def _compose_name_fields(desig: str | None, cobs_fullname: str | None) -> tuple[str | None, str | None]:
    """
    Returns (name_suffix, name_full) with preferred formats:
      - C/2025 A6 (Lemmon)
      - 24P/Schaumasse
      - 3I/ATLAS
    """
    desig = (desig or "").strip()
    suffix = _extract_suffix_from_fullname(cobs_fullname)

    if suffix:
        if "/" in desig and "(" not in desig:
            # Periodic/interstellar with slash (e.g. '24P' -> '24P/Schaumasse', '3I' -> '3I/ATLAS')
            full = f"{desig}/{suffix}"
        else:
            # Long-period: 'C/2025 A6 (Lemmon)'
            full = f"{desig} ({suffix})"
        return suffix, full

    # If COBS fullname exists but no suffix parsed, reuse as full
    if cobs_fullname and str(cobs_fullname).strip():
        s = str(cobs_fullname).strip()
        if "/" in s or "(" in s:
            return None, s

    # Fallback: just designation
    return None, desig or None

def _build_cobs_name_map() -> Dict[str, str]:
    """
    Builds a mapping from COBS 'comet_name' -> 'comet_fullname'
    using data/cobs_list.json if present.
    """
    m: Dict[str, str] = {}
    if not COBS_PATH.exists():
        return m
    try:
        cobs = json.loads(COBS_PATH.read_text(encoding="utf-8"))
        for obj in cobs.get("comet_list", []):
            cn = obj.get("comet_name")       # e.g., "C/2025 A6", "24P", "3I"
            cf = obj.get("comet_fullname")   # e.g., "C/2025 A6 (Lemmon)", "24P/Schaumasse", "3I/ATLAS"
            if isinstance(cn, str) and cn.strip() and isinstance(cf, str) and cf.strip():
                m[cn.strip()] = cf.strip()
    except Exception:
        pass
    return m

def main():
    if not INPATH.exists():
        raise SystemExit(f"Input not found: {INPATH}")

    data = json.loads(INPATH.read_text(encoding="utf-8"))
    items = data.get("items") or []

    # Build COBS fullname map once
    cobs_map = _build_cobs_name_map()

    updated = 0
    for it in items:
        desig = _choose_desig(it) or ""

        # If there is no cobs_fullname on the item, try to inject it from cobs_map
        if not it.get("cobs_fullname") and desig in cobs_map:
            it["cobs_fullname"] = cobs_map[desig]

        # Skip if already has proper name fields
        if it.get("display_name") and it.get("name_full") and it.get("name_suffix"):
            continue

        cobs_full = it.get("cobs_fullname")
        name_suffix, name_full = _compose_name_fields(desig, cobs_full)
        display_name = name_full

        # Only set fields if missing (NON-DESTRUCTIVE)
        if it.get("name_suffix") is None and name_suffix is not None:
            it["name_suffix"] = name_suffix
            updated += 1
        if it.get("name_full") is None and name_full is not None:
            it["name_full"] = name_full
            updated += 1
        if it.get("display_name") is None and display_name is not None:
            it["display_name"] = display_name
            updated += 1

        # As a fallback, mirror designation
        if it.get("display_name") is None and desig:
            it["display_name"] = desig
            updated += 1

    if updated:
        OUTPATH.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":")), encoding="utf-8")
        print(f"Augmented {updated} name fields in {OUTPATH}")
    else:
        print("No items needed augmentation; leaving JSON unchanged.")

if __name__ == "__main__":
    main()
