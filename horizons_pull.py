#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed comets from COBS (prefer data/cobs_list.json if present; supports 'comet_list')
- Geometry/brightness from JPL Horizons.

Key points
- Queries "now" using epochs=[JD] to avoid TLIST/WLDINI errors.
- Resolves ambiguous periodic designations (e.g., 2P/12P/13P) to the most
  recent apparition (preferring the last N years) via Horizons record-id.
- Returns RA/DEC/Δ from ephemerides; computes r (heliocentric) and phase angle
  from state vectors so values exist even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude: v_pred = M1 + 5*log10(Δ) + k1*log10(r)
- Filters by observed magnitude if BRIGHT_LIMIT env var is set (e.g., 15).
"""

import os, json, time, re, math
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 14  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.30
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this

# Optional observed-mag limit (e.g., 15) from env
BRIGHT_LIMIT = None
try:
    _bl = os.environ.get("BRIGHT_LIMIT", "").strip()
    if _bl:
        BRIGHT_LIMIT = float(_bl)
except Exception:
    BRIGHT_LIMIT = None
# -------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------- COBS name normalization (improved) --------
# Accepts:
#   - Periodic: "29P", "0029P", "29P/Schwassmann-Wachmann" -> "29P"
#   - Year designations: "C/2025 A6", "P/2025 T1", "D/2024 E1", "A/2023 G2", "X/2022 N2", "I/2017 U1"
#   - Interstellar number form: "1I/ʻOumuamua", "2I/Borisov", "3I/ATLAS" -> "1I", "2I", "3I"
#   - Trailing "(Name)" is ignored for Horizons id.

_RE_PERIODIC = re.compile(
    r'^\s*0*(?P<num>\d+)\s*P(?:\s*/\s*[-A-Za-z0-9 .\'’]+)?\s*$', re.IGNORECASE
)
_RE_YEAR_CLASS = re.compile(
    r'^\s*(?P<class>[PCDAXI])\s*/\s*(?P<year>\d{4})\s+(?P<code>[A-Z]\d+)(?:\s*\([^)]+\))?\s*$',
    re.IGNORECASE
)
_RE_INTERSTELLAR_NUM = re.compile(
    r'^\s*(?P<num>\d+)\s*I(?:\s*/\s*[-A-Za-z0-9 .\'’]+)?\s*$',
    re.IGNORECASE
)

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()

    # 1) Periodic: "0029P", "29P/Name" -> "29P"
    m = _RE_PERIODIC.match(s)
    if m:
        return f"{int(m.group('num'))}P"

    # 2) Year+class: "C/2025 A6 (Name)" -> "C/2025 A6"
    m = _RE_YEAR_CLASS.match(s)
    if m:
        klass = m.group('class').upper()
        year  = m.group('year')
        code  = m.group('code').upper().replace('  ', ' ')
        return f"{klass}/{year} {code}"

    # 3) Interstellar numbered: "3I/ATLAS" -> "3I"
    m = _RE_INTERSTELLAR_NUM.match(s)
    if m:
        return f"{int(m.group('num'))}I"

    # 4) Sometimes designation is inside parentheses
    m = re.search(r'\(([^)]+)\)', s)
    if m:
        return to_designation(m.group(1))

    return None


def load_cobs_designations(path: Path) -> Dict[str, float]:
    """
    Return {designation: observed_mag} from a variety of possible COBS shapes.
    Honors BRIGHT_LIMIT if set.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # find the list of comet dicts
    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        items = raw["comet_list"]
    elif isinstance(raw, dict):
        items = None
        for key in ("comets", "objects", "data", "items", "list"):
            if isinstance(raw.get(key), list):
                items = raw[key]; break
        if items is None:
            # mapping {name: mag}
            out = {}
            for k, v in raw.items():
                d = to_designation(k)
                if d is not None:
                    try:
                        mval = float(v)
                        if (BRIGHT_LIMIT is None) or (mval <= BRIGHT_LIMIT):
                            out[d] = mval
                    except Exception:
                        pass
            return out
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    result: Dict[str, float] = {}
    for o in items:
        if not isinstance(o, dict):
            continue
        # Try several name fields
        name = (
            o.get("mpc_name")
            or o.get("comet_name")
            or o.get("comet_fullname")
            or o.get("designation")
            or o.get("name")
            or o.get("fullname")
        )
        d = to_designation(name) if name else None

        # magnitude field candidates
        mag = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag", "obs_mag"):
            if k in o:
                try:
                    mag = float(o[k])
                except Exception:
                    mag = None
                break

        if d is not None and mag is not None:
            if (BRIGHT_LIMIT is None) or (mag <= BRIGHT_LIMIT):
                # keep the brightest (smallest) if duplicates appear
                if (d not in result) or (mag < result[d]):
                    result[d] = mag

    return result
# ----------------------------------------------------


# -------- Horizons plumbing --------
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
    # otherwise, newest overall
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

    # If V missing but M1/k1 present, compute predicted magnitude
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out
# -----------------------------------


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/delta from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # 1) Try by designation
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]

        # vectors for r & phase (try designation, fall back to record id if needed)
        try:
            v_sun = _query_vectors("@10", comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "epoch_utc": now_iso(), **core}
        except Exception:
            rec_id = resolve_ambiguous_to_record_id(comet_id)
            if rec_id is None:
                raise
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}

    except Exception as e1:
        # 2) Resolve and retry by record id
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now)
            row = eph[0]
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None:
                core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}
        except Exception as e2:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": f"{e1} | retry:{e2}"}


def main():
    # 0) Load COBS designations -> observed mag
    meta = {
        "observations": "COBS (file or direct fetch)",
        "theory": "JPL Horizons",
        "cobs_source": "file" if COBS_PATH.exists() else "none",
        "cobs_bytes": COBS_PATH.stat().st_size if COBS_PATH.exists() else 0,
        "cobs_content_type": "application/json" if COBS_PATH.exists() else None,
        "cobs_error": None,
    }
    cobs_map = load_cobs_designations(COBS_PATH)

    if cobs_map:
        comet_ids: List[str] = sorted(cobs_map.keys())
    else:
        # empty → nothing to do
        comet_ids = []

    print("COBS → Horizons designations:", comet_ids)

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

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": BRIGHT_LIMIT,
        "source": meta,
        "cobs_designations": len(comet_ids),
        "cobs_used": bool(comet_ids),
        "count": len(results),
        "items": results,
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
