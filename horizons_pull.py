#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (downloaded by the workflow into data/cobs_list.json)
- Geometry/brightness from JPL Horizons

Key points
- Queries "now" using epochs=[JD] to avoid TLIST/WLDINI errors.
- Resolves ambiguous periodic designations (e.g., 2P/12P/13P) to the most
  recent apparition (preferring the last N years) via Horizons record-id.
- Returns RA/DEC/Δ from ephemerides; computes r (heliocentric) and phase angle
  from state vectors so values exist even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude:
      v_pred = M1 + 5*log10(Δ) + k1*log10(r)
- If data/cobs_list.json exists, uses it to drive which comets to query and
  merges `cobs_mag` + (v_pred - cobs_mag). Otherwise falls back to COMETS list.
"""

import json
import time
import re
import math
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 10  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter; swap to a site dict for topocentric later
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V (even if some are missing)
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # written by the workflow step

# Optional: enable a brightness filter later (keeps “likely visible” only)
BRIGHT_LIMIT = None  # e.g. 17.5; keep as None for now

# Fallback list if COBS is missing/empty
COMETS: List[str] = [
    "2P",
    "12P",
    "13P",
    "C/2023 A3",
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------- Normalize COBS names to MPC-style designations --------
# Accepts: "2P", "2P/Encke", "C/2023 A3", "C/2023 A3 (Tsuchinshan-ATLAS)", etc.
_DESIG = re.compile(r"""
    ^\s*(
        \d+\s*P(?:/\w+)?                            # 2P or 2P/Encke
        |[PCADX]/\d{4}\s+[A-Z]\d+(?:\s*\([^)]+\))?  # C/2023 A3 (Name)
    )
""", re.IGNORECASE | re.VERBOSE)

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = _DESIG.match(s)
    if m:
        d = m.group(1)
        d = re.sub(r"\s*\([^)]+\)\s*$", "", d)      # drop trailing (Name)
        return re.sub(r"\s+", " ", d).upper()
    m2 = re.search(r"\(([^)]+)\)", s)               # sometimes designation inside ()
    if m2:
        return to_designation(m2.group(1))
    return None

def load_cobs_designations(path: Path) -> Dict[str, float]:
    """
    Read data/cobs_list.json and return {designation: observed_mag}.
    Accepts shapes like:
      { "comets": [ {"designation":"C/2023 A3","mag":14.2}, ... ] }
      [ {"mpc_name":"C/2023 A3","mag":14.2}, ... ]
      { "data":[...]} / {"objects":[...]} / {"items":[...]} variants
      or a mapping { "C/2023 A3": 14.2, ... }
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # normalize container -> list[dict] or mapping
    if isinstance(raw, dict):
        for key in ("comets", "objects", "data", "items", "list"):
            if key in raw and isinstance(raw[key], list):
                raw_list = raw[key]
                break
        else:
            # mapping {name: mag}
            if all(isinstance(k, str) for k in raw.keys()):
                out = {}
                for k, v in raw.items():
                    try:
                        desig = to_designation(k)
                        if desig:
                            out[desig] = float(v)
                    except Exception:
                        pass
                return {k: v for k, v in out.items() if k}
            raw_list = []
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    result: Dict[str, float] = {}
    for o in raw_list:
        if not isinstance(o, dict):
            continue
        # common magnitude keys
        mag = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag"):
            if k in o:
                try:
                    mag = float(o[k]); break
                except Exception:
                    pass
        name = o.get("designation") or o.get("mpc_name") or o.get("fullname") or o.get("name")
        desig = to_designation(str(name)) if name else None
        if desig and (mag is not None):
            # keep brightest duplicate
            if desig not in result or mag < result[desig]:
                result[desig] = mag
    return result
# ---------------------------------------------------------------


# -------------------- Horizons helpers -------------------------
_ROW = re.compile(r"^\s*(?P<rec>9\d{7})\s+(?P<epoch>\d{4})\s+")

def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rec = m.group("rec")
        epoch = int(m.group("epoch"))
        if epoch >= now_year - years_window and epoch > best_epoch:
            best, best_epoch = rec, epoch
    if best:
        return best
    # fallback to latest epoch overall
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        rec = m.group("rec")
        epoch = int(m.group("epoch"))
        if epoch > best_epoch:
            best, best_epoch = rec, epoch
    return best

def resolve_ambiguous_to_record_id(designation: str) -> Optional[str]:
    try:
        jd_now = Time.now().jd
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now])\
            .ephemerides(quantities="1")
        return None  # not ambiguous
    except Exception as e:
        msg = str(e)
        if "Ambiguous target name" not in msg:
            return None
        return _pick_recent_record(msg, YEARS_WINDOW)

def _query_ephem(id_value: str, id_type: str, observer, jd_now: float, try_again: bool = True):
    try:
        obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
        return obj.ephemerides(quantities=QUANTITIES)
    except Exception as e:
        msg = str(e)
        if try_again and ("no TLIST" in msg or "WLDINI" in msg):
            time.sleep(0.8)
            obj = Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now])
            return obj.ephemerides(quantities=QUANTITIES)
        raise

def _query_vectors(location: str, id_value: str, id_type: str, jd_now: float):
    obj = Horizons(id=id_value, id_type=id_type, location=location, epochs=[jd_now])
    return obj.vectors()

def _vec_norm(x, y, z):
    return (x*x + y*y + z*z) ** 0.5

def _phase_from_vectors(v_sun_row, v_earth_row):
    sx, sy, sz = float(v_sun_row["x"]), float(v_sun_row["y"]), float(v_sun_row["z"])
    ex, ey, ez = float(v_earth_row["x"]), float(v_earth_row["y"]), float(v_earth_row["z"])
    rn = _vec_norm(sx, sy, sz)      # r (AU)
    dn = _vec_norm(ex, ey, ez)      # Δ (AU)
    dot = sx*ex + sy*ey + sz*ez
    c = max(-1.0, min(1.0, dot / (rn * dn)))
    return degrees(acos(c)), rn, dn

def _colmap(cols) -> Dict[str, str]:
    return {c.lower(): c for c in cols}

def _get_optional_float(row, cmap: Dict[str, str], key_lower: str) -> Optional[float]:
    k = cmap.get(key_lower)
    if not k:
        return None
    try:
        return float(row[k])
    except Exception:
        return None

def _row_to_payload_with_photometry(row, r_au: Optional[float], delta_vec_au: Optional[float]) -> Dict[str, Any]:
    cols = getattr(row, "colnames", None) or row.table.colnames
    cmap = _colmap(cols)

    ra   = _get_optional_float(row, cmap, "ra")
    dec  = _get_optional_float(row, cmap, "dec")
    delt = _get_optional_float(row, cmap, "delta")
    alpha= _get_optional_float(row, cmap, "alpha")
    vmag = _get_optional_float(row, cmap, "v")
    M1   = _get_optional_float(row, cmap, "m1")
    k1   = _get_optional_float(row, cmap, "k1")

    delta_au = delt if delt is not None else delta_vec_au

    out = {
        "r_au": r_au,
        "delta_au": delta_au,
        "phase_deg": alpha,  # may be None; fill from vectors if needed
        "ra_deg": ra,
        "dec_deg": dec,
        "vmag": vmag,        # may be None
    }

    # If V is missing but M1/k1 are present, compute predicted magnitude
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    # Debug aid if critical bits missing
    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out
# ---------------------------------------------------------------


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/Δ from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # 1) Try by designation
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
        # vectors (designation)
        try:
            v_sun   = _query_vectors("@10",  comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "epoch_utc": now_iso(), **core}
        except Exception:
            # fall back: resolve to a unique record id and use vectors with that id
            rec_id = resolve_ambiguous_to_record_id(comet_id)
            if rec_id is None:
                raise
            v_sun   = _query_vectors("@10",  rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}

    except Exception as e1:
        # 2) Full retry by numeric record id
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now)
            row = eph[0]
            v_sun   = _query_vectors("@10",  rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}
        except Exception as e2:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": f"{e1} | retry:{e2}"}


def main():
    # 0) Load COBS designations (observed list) if present
    cobs_map = load_cobs_designations(COBS_PATH)  # {designation: observed_mag}
    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else COMETS

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)
        # Merge observed mag if available
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]
            vpred = item.get("v_pred") or item.get("vmag")
            if vpred is not None:
                try:
                    item["mag_diff_pred_minus_obs"] = round(float(vpred) - float(cobs_map[cid]), 2)
                except Exception:
                    pass
        results.append(item)
        time.sleep(PAUSE_S)

    # Optional visibility cut
    if BRIGHT_LIMIT is not None:
        filtered = []
        for it in results:
            vm = it.get("vmag")
            vp = it.get("v_pred")
            keep = (vm is not None and vm <= BRIGHT_LIMIT) or (vp is not None and vp <= BRIGHT_LIMIT)
            if keep:
                filtered.append(it)
        results = filtered

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": BRIGHT_LIMIT,
        "source": {
            "observations": "COBS (downloaded in workflow) if present",
            "theory": "JPL Horizons",
        },
        "count": len(results),
        "items": results,
    }
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
