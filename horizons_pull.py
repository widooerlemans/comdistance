#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (prefer data/cobs_list.json if present; supports 'comet_list')
- Geometry/brightness from JPL Horizons

Key points
- Queries "now" using epochs=[JD] to avoid TLIST/WLDINI errors.
- Resolves ambiguous periodic designations (e.g., 2P/12P/13P) to the most
  recent apparition (preferring the last N years) via Horizons record-id.
- Returns RA/DEC/Δ from ephemerides; computes r (heliocentric) and phase angle
  from state vectors so values exist even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude:
      v_pred = M1 + 5*log10(Δ) + k1*log10(r)

Outputs a compact JSON to data/comets_ephem.json.
"""

import json, time, re, math
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 15  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.30                   # small polite pause between calls
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow should write this

# Optional: filter by observed magnitude (COBS). If env var empty => no filter.
BRIGHT_LIMIT_ENV = os.getenv("BRIGHT_LIMIT", "").strip()
BRIGHT_LIMIT = float(BRIGHT_LIMIT_ENV) if BRIGHT_LIMIT_ENV else None

# Fallback hand list (only used if no COBS file)
COMETS_FALLBACK: List[str] = ["24P", "29P", "240P", "141P", "C/2025 A6", "C/2025 R2", "C/2024 E1"]

# ----------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -------- Robust designation extraction --------
# Finds MPC-style chunks ANYWHERE in the string, tolerant of extra text/spacing.
# Normalizes to:
#   - "C/YYYY LNN"  e.g., "C/2025 A6"
#   - "NNP"         e.g., "24P"
#   - "NNI"         e.g., "3I"
_C_STYLE = re.compile(
    r'(?i)(?P<class>[PCDXAI])\s*/\s*(?P<year>\d{4})\s*(?P<letter>[A-Z])\s*(?P<index>\d+)'  # C/2025 A6
)
_P_STYLE = re.compile(r'(?i)\b(?P<num>\d+)\s*P\b')  # 24P / 0024P
_I_STYLE = re.compile(r'(?i)\b(?P<num>\d+)\s*I\b')  # 3I

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = str(s)

    m = _C_STYLE.search(s)
    if m:
        cls = m.group('class').upper()
        year = m.group('year')
        let = m.group('letter').upper()
        idx = int(m.group('index'))
        return f"{cls}/{year} {let}{idx}"

    m = _P_STYLE.search(s)
    if m:
        n = int(m.group('num'))
        return f"{n}P"

    m = _I_STYLE.search(s)
    if m:
        n = int(m.group('num'))
        return f"{n}I"

    m_paren = re.search(r'\(([^)]+)\)', s)
    if m_paren:
        return to_designation(m_paren.group(1))

    return None

def load_cobs_designations(path: Path) -> Dict[str, float]:
    """
    Read data/cobs_list.json and return {designation: observed_mag}.
    Accepts shapes like:
      { "comet_list": [ {...}, ... ] }  # COBS planner
      { "comets": [ ... ] }  or {"objects": [...]}, {"data":[...]}, {"items":[...]}, {"list":[...]}
      [ {...}, ... ]
      Or even a mapping { "C/2025 A6": 11.4, ... }
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # normalize to list of dicts
    raw_list = []
    if isinstance(raw, dict):
        if isinstance(raw.get("comet_list"), list):
            raw_list = raw["comet_list"]
        else:
            for key in ("comets", "objects", "data", "items", "list"):
                if isinstance(raw.get(key), list):
                    raw_list = raw[key]; break
            else:
                # mapping case { designation: mag }
                if all(isinstance(k, str) for k in raw.keys()):
                    out = {}
                    for k, v in raw.items():
                        d = to_designation(k)
                        if not d: continue
                        try:
                            out[d] = float(v)
                        except Exception:
                            pass
                    return out
    elif isinstance(raw, list):
        raw_list = raw

    result: Dict[str, float] = {}
    for o in raw_list:
        if not isinstance(o, dict):
            continue
        # try many name fields commonly seen
        name = (
            o.get("designation") or o.get("mpc_name") or o.get("comet_name") or
            o.get("comet_fullname") or o.get("name") or o.get("title") or o.get("comet")
        )
        d = to_designation(name) if name else None

        # magnitude fields commonly seen
        mag_val = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag", "v"):
            if k in o:
                try:
                    mag_val = float(o[k]); break
                except Exception:
                    pass

        if d and (mag_val is not None):
            # keep brightest if duplicates
            if d not in result or mag_val < result[d]:
                result[d] = mag_val

    # optional filter by observed magnitude
    if BRIGHT_LIMIT is not None:
        result = {k: v for k, v in result.items() if v <= BRIGHT_LIMIT}

    return result
# ----------------------------------------

# -------- Horizons helpers --------
_ROW = re.compile(r"^\s*(?P<rec>9\d{7})\s+(?P<epoch>\d{4})\s+")

def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    now_year = datetime.utcnow().year
    best = None
    best_epoch = -1
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m: continue
        rec = m.group("rec"); epoch = int(m.group("epoch"))
        if epoch >= now_year - years_window and epoch > best_epoch:
            best, best_epoch = rec, epoch
    if best: return best
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m: continue
        rec = m.group("rec"); epoch = int(m.group("epoch"))
        if epoch > best_epoch:
            best, best_epoch = rec, epoch
    return best

def resolve_ambiguous_to_record_id(designation: str) -> Optional[str]:
    try:
        jd_now = Time.now().jd
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now])\
            .ephemerides(quantities="1")
        return None
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
    if not k: return None
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
        "vmag": vmag,        # may be None (Horizons sometimes omits)
    }

    # If V is missing but M1/k1 are present, compute predicted magnitude
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0 * math.log10(delta_au) + k1 * math.log10(r_au)
        except ValueError:
            pass

    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out
# -----------------------------------

def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/Δ from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # 1) Try by designation directly
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
        try:
            v_sun   = _query_vectors("@10",  comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "epoch_utc": now_iso(), **core}
        except Exception:
            # fall back to record id if vectors-by-designation fail
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
        # 2) Resolve and retry entirely by record id
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
    # Prefer COBS file if present
    cobs_map = load_cobs_designations(COBS_PATH)  # {designation: observed_mag}
    cobs_used = bool(cobs_map)
    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else COMETS_FALLBACK

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)
        # attach observed mag if available
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

    # Prepare simple diagnostics about COBS source
    cobs_bytes = None
    cobs_ct = None
    cobs_err = None
    try:
        if COBS_PATH.exists():
            cobs_bytes = COBS_PATH.stat().st_size
            cobs_ct = "application/json"
    except Exception as e:
        cobs_err = str(e)

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": BRIGHT_LIMIT,
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
            "cobs_source": "file" if cobs_used else "fallback",
            "cobs_bytes": cobs_bytes,
            "cobs_content_type": cobs_ct,
            "cobs_error": cobs_err,
        },
        "cobs_designations": len(cobs_map),
        "cobs_used": cobs_used,
        "count": len(results),
        "items": results,
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
