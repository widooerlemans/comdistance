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
"""

import json, time, re, math
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 15  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this
BRIGHT_LIMIT_ENV = "BRIGHT_LIMIT"        # optional env var filter on v_pred or cobs_mag
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- Unicode/Name Normalization ----------
# Many COBS fields contain NARROW NO-BREAK SPACE (U+202F) or NO-BREAK SPACE (U+00A0)
# between year and letter (e.g., "2025\u202fA6"). Normalize them to a regular space.
_SPACE_PAT = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]+")

def _norm_spaces(s: str) -> str:
    # collapse exotic spaces to ASCII space, then collapse runs to one space
    s = _SPACE_PAT.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# Accepts things like:
#   "2P", "24P/Schaumasse", "240P/NEAT",
#   "3I/ATLAS",
#   "C/2025 A6 (Lemmon)", "C/2024 E1 (Wierzchos)", "C/2021 G2 (ATLAS)"
# Returns a Horizons-friendly designation such as "24P", "240P", "3I", "C/2025 A6", "C/2024 E1".
_PAT_PERIODIC = re.compile(r"^\s*(\d+)\s*P(?:/.*)?\s*$", re.IGNORECASE)
_PAT_INTERSTELLAR = re.compile(r"^\s*(\d+)\s*I(?:/.*)?\s*$", re.IGNORECASE)
# C/2025 A6 or C/2024 E1 etc. Allow any weird spaces and parentheses after.
_PAT_C_PROV = re.compile(
    r"""^\s*([PCADX])\s*/\s*(\d{4})\s+([A-Z]{1,2}\d{1,3})""",
    re.IGNORECASE | re.VERBOSE
)

def to_designation(name_like: str) -> Optional[str]:
    if not name_like:
        return None
    s = _norm_spaces(str(name_like))

    # Drop trailing "(Name)" part if present for simpler matching
    s_no_paren = re.sub(r"\s*\([^)]+\)\s*$", "", s)

    # 24P, 240P, 2P/Encke -> "24P", "240P", "2P"
    m = _PAT_PERIODIC.match(s_no_paren)
    if m:
        return f"{int(m.group(1))}P"

    # 3I/ATLAS -> "3I"
    m = _PAT_INTERSTELLAR.match(s_no_paren)
    if m:
        return f"{int(m.group(1))}I"

    # C/2025 A6 (…)
    m = _PAT_C_PROV.match(s)
    if m:
        fam = m.group(1).upper()
        year = m.group(2)
        code = m.group(3).upper()
        # normalize single or double letter + number, keep the space between year and code
        return f"{fam}/{year} {code}"

    # Sometimes the designation sits inside parentheses
    m2 = re.search(r"\(([PCADX]/\s*\d{4}\s+[A-Za-z]{1,2}\d{1,3})\)", s, re.IGNORECASE)
    if m2:
        inner = _norm_spaces(m2.group(1))
        mm = _PAT_C_PROV.match(inner)
        if mm:
            return f"{mm.group(1).upper()}/{mm.group(2)} {mm.group(3).upper()}"

    # Last chance: very compact like "C/2025A6"
    m3 = re.match(r"^\s*([PCADX])\s*/\s*(\d{4})([A-Za-z]{1,2}\d{1,3})\s*$", s, re.IGNORECASE)
    if m3:
        return f"{m3.group(1).upper()}/{m3.group(2)} {m3.group(3).upper()}"

    return None
# ------------------------------------------------

def load_cobs_designations(path: Path) -> Dict[str, float]:
    """
    Read data/cobs_list.json and return {designation: observed_mag}.
    Accepts these shapes:
      {"comet_list": [ { "mpc_name": "C/2025 A6 (Lemmon)", "mag": 12.3 }, ... ]}
      {"comets":[...]}, {"objects":[...]}, {"items":[...]}, {"data":[...]} or a top-level list
      Also tolerates different magnitude keys.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        raw_list = raw["comet_list"]
    elif isinstance(raw, dict):
        raw_list = []
        for k in ("comets", "objects", "items", "data", "list"):
            if isinstance(raw.get(k), list):
                raw_list = raw[k]; break
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    result: Dict[str, float] = {}
    # keep a few raw names for debugging visibility in JSON
    debug_seen = []

    for o in raw_list:
        if not isinstance(o, dict):
            continue
        # Most reliable naming field order
        name = (
            o.get("mpc_name") or
            o.get("comet_fullname") or
            o.get("comet_name") or
            o.get("designation") or
            o.get("name")
        )
        if name and len(debug_seen) < 8:
            debug_seen.append(str(name))

        # magnitude field candidates
        mag = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag", "cur_mag"):
            if k in o:
                try:
                    mag = float(o[k]); break
                except Exception:
                    pass

        desig = to_designation(name) if name else None
        if desig and (mag is not None):
            # Use brightest if multiple entries
            if desig not in result or mag < result[desig]:
                result[desig] = mag

    # attach some debug to help confirm parsing
    result["_debug_first_names"] = debug_seen
    return result

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

    # If V is missing but M1/k1 are present, compute predicted magnitude
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out

def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/delta from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # 1) Try by designation
    try:
        eph = _query_ephem(comet_id, "designation", observer, jd_now)
        row = eph[0]
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
        # 2) If ephemeris failed, resolve and retry entirely by record id
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

def try_float_env(name: str) -> Optional[float]:
    import os
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None

def main():
    bright_limit = try_float_env(BRIGHT_LIMIT_ENV)

    # Load COBS list → {designation: observed_mag}
    cobs_map = load_cobs_designations(COBS_PATH)  # has special key "_debug_first_names"
    debug_first_names = cobs_map.pop("_debug_first_names", [])
    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else []

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

        results.append({"id": cid, **item})
        time.sleep(PAUSE_S)

    # Optional filter on brightness after merge
    if bright_limit is not None:
        filtered = []
        for it in results:
            obs = it.get("cobs_mag")
            pred = it.get("v_pred") or it.get("vmag")
            keep = (obs is not None and obs <= bright_limit) or (pred is not None and pred <= bright_limit)
            if keep:
                filtered.append(it)
        results = filtered

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": bright_limit,
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
        },
        "cobs_designations": len(comet_ids),
        "cobs_used": bool(comet_ids),
        "debug_first_cobs_names": debug_first_names,  # to confirm parsing
        "count": len(results),
        "items": results,
    }
    Path(OUTPATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
