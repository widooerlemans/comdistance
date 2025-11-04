#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (prefer data/cobs_list.json; supports 'comet_list')
- Geometry/brightness from JPL Horizons

Filters to observed (COBS) magnitude <= BRIGHT_LIMIT (env), if present.

Notes
- Queries "now" using epochs=[JD] to avoid TLIST/WLDINI errors.
- Resolves ambiguous periodic designations (e.g., 2P/12P/13P) to the most
  recent apparition (preferring last N years) via Horizons record-id.
- Returns RA/DEC/Δ from ephemerides; computes r (heliocentric) and phase angle
  from state vectors so values exist even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude:
      v_pred = M1 + 5*log10(Δ) + k1*log10(r)
"""

import json, time, re, math, os
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
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this
# Fallback if COBS missing:
COMETS: List[str] = ["24P", "29P", "141P", "C/2023 A3"]
# ---------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -------- COBS normalization --------
_DESIG = re.compile(r"""
    ^\s*(
        \d+\s*P(?:/[A-Za-z0-9-]+)?           # e.g. 24P or 24P/Schaumasse
        |[PCADX]/\d{4}\s+[A-Z]\d+            # e.g. C/2023 A3
    )
""", re.IGNORECASE | re.VERBOSE)

def to_designation(s: str) -> Optional[str]:
    if not s: return None
    s = str(s).strip()
    m = _DESIG.match(s)
    if m:
        d = m.group(1)
        d = re.sub(r"\s*\([^)]+\)\s*$", "", d)      # strip trailing (Name)
        return re.sub(r"\s+", " ", d).upper()
    m2 = re.search(r"\(([^)]+)\)", s)
    return to_designation(m2.group(1)) if m2 else None

def load_cobs_designations(path: Path) -> Dict[str, float]:
    """Return {designation: observed_mag} from data/cobs_list.json (flexible shapes)."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # pull list
    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        raw_list = raw["comet_list"]
    elif isinstance(raw, dict):
        for k in ("comets","objects","data","items","list"):
            if isinstance(raw.get(k), list):
                raw_list = raw[k]; break
        else:
            raw_list = []
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    out: Dict[str, float] = {}
    for o in raw_list:
        if not isinstance(o, dict): continue
        # name fields
        name = o.get("mpc_name") or o.get("comet_name") or o.get("comet_fullname") or o.get("designation") or o.get("name")
        d = to_designation(name) if name else None
        # mag fields (take the brightest if multiple)
        mag = None
        for k in ("mag","magnitude","current_mag","peak_mag","estimated_mag"):
            if k in o:
                try:
                    mag = float(o[k]); break
                except Exception:
                    pass
        if d and (mag is not None):
            if d not in out or mag < out[d]:
                out[d] = mag
    # fix periodic leading zeros like "0024P" → "24P"
    fix = {}
    for k, v in out.items():
        m = re.match(r"^0+(\d+P)(?:/.*)?$", k, re.I)
        fix[m.group(1).upper() if m else k] = v
    return fix

# -------- Horizons helpers --------
_ROW = re.compile(r"^\s*(?P<rec>9\d{7})\s+(?P<epoch>\d{4})\s+")

def _pick_recent_record(ambig_text: str, years_window: int) -> Optional[str]:
    now_year = datetime.utcnow().year
    best = None; best_epoch = -1
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m: continue
        rec, epoch = m.group("rec"), int(m.group("epoch"))
        if epoch >= now_year - years_window and epoch > best_epoch:
            best, best_epoch = rec, epoch
    if best: return best
    for line in ambig_text.splitlines():
        m = _ROW.match(line)
        if not m: continue
        rec, epoch = m.group("rec"), int(m.group("epoch"))
        if epoch > best_epoch:
            best, best_epoch = rec, epoch
    return best

def resolve_ambiguous_to_record_id(designation: str) -> Optional[str]:
    try:
        jd_now = Time.now().jd
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now]).ephemerides(quantities="1")
        return None
    except Exception as e:
        msg = str(e)
        if "Ambiguous target name" not in msg:
            return None
        return _pick_recent_record(msg, YEARS_WINDOW)

def _query_ephem(id_value: str, id_type: str, observer, jd_now: float, try_again: bool = True):
    try:
        return Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now]).ephemerides(quantities=QUANTITIES)
    except Exception as e:
        msg = str(e)
        if try_again and ("no TLIST" in msg or "WLDINI" in msg):
            time.sleep(0.8)
            return Horizons(id=id_value, id_type=id_type, location=observer, epochs=[jd_now]).ephemerides(quantities=QUANTITIES)
        raise

def _query_vectors(location: str, id_value: str, id_type: str, jd_now: float):
    return Horizons(id=id_value, id_type=id_type, location=location, epochs=[jd_now]).vectors()

def _vec_norm(x, y, z): return (x*x + y*y + z*z) ** 0.5

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
    try: return float(row[k])
    except Exception: return None

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
        "phase_deg": alpha,     # fill from vectors if None
        "ra_deg": ra,
        "dec_deg": dec,
        "vmag": vmag,           # may be None
    }
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)

    if any(out.get(k) is None for k in ("delta_au","ra_deg","dec_deg")):
        out["_cols"] = list(cmap.values())

    return out

def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    jd_now = Time.now().jd
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
        try:
            v_sun = _query_vectors("@10", comet_id, "designation", jd_now)[0]
            v_earth = _query_vectors("@399", comet_id, "designation", jd_now)[0]
        except Exception:
            rec_id = resolve_ambiguous_to_record_id(comet_id)
            if rec_id is None: raise
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None: core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}

        alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
        core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
        if core.get("phase_deg") is None: core["phase_deg"] = alpha_deg
        return {"id": comet_id, "epoch_utc": now_iso(), **core}

    except Exception as e1:
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": str(e1)}
        try:
            eph = _query_ephem(rec_id, "smallbody", observer, jd_now); row = eph[0]
            v_sun = _query_vectors("@10", rec_id, "smallbody", jd_now)[0]
            v_earth = _query_vectors("@399", rec_id, "smallbody", jd_now)[0]
            alpha_deg, r_au, delta_vec_au = _phase_from_vectors(v_sun, v_earth)
            core = _row_to_payload_with_photometry(row, r_au, delta_vec_au)
            if core.get("phase_deg") is None: core["phase_deg"] = alpha_deg
            return {"id": comet_id, "horizons_id": rec_id, "epoch_utc": now_iso(), **core}
        except Exception as e2:
            return {"id": comet_id, "epoch_utc": now_iso(), "error": f"{e1} | retry:{e2}"}    

def main():
    bright_limit_env = os.environ.get("BRIGHT_LIMIT", "").strip()
    bright_limit = None
    if bright_limit_env:
        try: bright_limit = float(bright_limit_env)
        except Exception: bright_limit = None

    cobs_map = load_cobs_designations(COBS_PATH)  # {designation: observed_mag}
    source_meta = {
        "observations": "COBS (file or direct fetch)",
        "theory": "JPL Horizons",
        "cobs_source": "file" if COBS_PATH.exists() else "fallback",
        "cobs_bytes": COBS_PATH.stat().st_size if COBS_PATH.exists() else 0,
        "cobs_content_type": "application/json" if COBS_PATH.exists() else None,
        "cobs_error": None,
    }

    if cobs_map:
        # optional filter by observed magnitude
        if bright_limit is not None:
            cobs_map = {k:v for k,v in cobs_map.items() if v <= bright_limit}
        comet_ids: List[str] = sorted(cobs_map.keys())
        cobs_used = True
    else:
        comet_ids = COMETS
        cobs_used = False

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]
            item["visible"] = True  # by construction when bright_limit applied
            vpred = item.get("v_pred") or item.get("vmag")
            if vpred is not None:
                try:
                    item["mag_diff_pred_minus_obs"] = round(float(vpred) - float(cobs_map[cid]), 2)
                except Exception:
                    pass
        results.append(item)
        time.sleep(PAUSE_S)

    # sort: brightest observed first if we have COBS
    if cobs_used:
        results.sort(key=lambda it: (999.0 if it.get("cobs_mag") is None else it["cobs_mag"], it.get("id","")))
    else:
        # no COBS → sort by predicted if present
        results.sort(key=lambda it: (999.0 if it.get("v_pred") is None else it["v_pred"], it.get("id","")))

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": bright_limit,
        "source": source_meta,
        "cobs_designations": len(cobs_map),
        "cobs_used": cobs_used,
        "count": len(results),
        "items": results,
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    Path(OUTPATH).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
