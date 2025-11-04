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

import json, time, re, math, os
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 13  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this

# Fallback (used only if COBS missing/empty)
COMETS: List[str] = [
    "2P",
    "12P",
    "13P",
    "C/2023 A3",
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------- COBS name normalization (handles zero-padded periodic IDs) --------
# Accepts:
#   "0024P", "0240P/LINEAR", "2P/Encke", "12P", "C/2023 A3 (Tsuchinshan-ATLAS)", etc.
_DESIG = re.compile(r"""
    ^\s*(
        (?P<num>\d+)\s*P(?:/[A-Za-z0-9\-]+)?        # e.g. 0024P or 2P/ENCKE
        |[PCADX]/\d{4}\s+[A-Z]\d+(?:\s*\([^)]+\))?  # e.g. C/2023 A3 (Name)
    )
""", re.IGNORECASE | re.VERBOSE)

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()

    m = _DESIG.match(s)
    if m:
        raw = m.group(1)
        raw = re.sub(r"\s+", " ", raw).upper()
        # Periodic: strip leading zeros and drop trailing '/NAME'
        if m.groupdict().get("num") is not None:
            n = int(m.group("num"))  # removes leading zeros
            return f"{n}P"
        # Non-periodic (C/…): drop trailing (Name)
        raw = re.sub(r"\s*\([^)]+\)\s*$", "", raw)
        return raw

    # Sometimes designation appears inside parentheses: "... (C/2023 A3) ..."
    m2 = re.search(r"\(([^)]+)\)", s)
    if m2:
        return to_designation(m2.group(1))

    return None
# ---------------------------------------------------------------------------


def load_cobs_designations(path: Path) -> Dict[str, float]:
    """
    Read data/cobs_list.json and return {designation: observed_mag}.
    Supports COBS 'planner.api' shape with 'comet_list', plus a few variants.

    Magnitude field candidates searched: mag, magnitude, current_mag, peak_mag, estimated_mag.
    Name field candidates: mpc_name, comet_name, comet_fullname, designation, name.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Pull list container
    items = []
    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        items = raw["comet_list"]
    elif isinstance(raw, dict):
        for k in ("comets", "objects", "data", "items", "list"):
            if isinstance(raw.get(k), list):
                items = raw[k]
                break
    elif isinstance(raw, list):
        items = raw

    result: Dict[str, float] = {}
    for o in items:
        if not isinstance(o, dict):
            continue
        # magnitude candidates
        mag = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag"):
            if k in o:
                try:
                    mag = float(o[k])
                    break
                except Exception:
                    pass
        # name/designation candidates
        name = (
            o.get("mpc_name")
            or o.get("comet_name")
            or o.get("comet_fullname")
            or o.get("designation")
            or o.get("name")
        )
        desig = to_designation(str(name)) if name else None
        if desig and (mag is not None):
            # keep brightest if duplicates
            if desig not in result or mag < result[desig]:
                result[desig] = mag
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
    # fallback to newest overall
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
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now]) \
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

    ra    = _get_optional_float(row, cmap, "ra")
    dec   = _get_optional_float(row, cmap, "dec")
    delt  = _get_optional_float(row, cmap, "delta")
    alpha = _get_optional_float(row, cmap, "alpha")
    vmag  = _get_optional_float(row, cmap, "v")
    M1    = _get_optional_float(row, cmap, "m1")
    k1    = _get_optional_float(row, cmap, "k1")

    delta_au = delt if delt is not None else delta_vec_au

    out = {
        "r_au": r_au,
        "delta_au": delta_au,
        "phase_deg": alpha,  # may be None; fill from vectors if needed
        "ra_deg": ra,
        "dec_deg": dec,
        "vmag": vmag,        # may be None
    }

    # Predicted magnitude if V missing and M1/k1 present
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())

    return out
# ---------------------------------------------------------------------------


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


def main():
    # Load COBS designations (observed list)
    cobs_map = load_cobs_designations(COBS_PATH)  # {designation: observed_mag}
    bright_limit = os.environ.get("BRIGHT_LIMIT", "").strip()
    bright_limit_f = None
    if bright_limit:
        try:
            bright_limit_f = float(bright_limit)
        except Exception:
            bright_limit_f = None

    comet_ids: List[str]
    if cobs_map:
        # Optional brightness filter
        if bright_limit_f is not None:
            ids = [d for d, m in cobs_map.items() if m <= bright_limit_f]
        else:
            ids = list(cobs_map.keys())
        comet_ids = sorted(set(ids))
    else:
        comet_ids = COMETS

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)
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

    # Build metadata
    cobs_bytes = COBS_PATH.stat().st_size if COBS_PATH.exists() else 0
    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": bright_limit_f,
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
            "cobs_source": "file" if COBS_PATH.exists() else "none",
            "cobs_bytes": cobs_bytes,
            "cobs_content_type": "application/json" if COBS_PATH.exists() else None,
            "cobs_error": None,
        },
        "cobs_designations": len(cobs_map) if cobs_map else 0,
        "cobs_used": bool(cobs_map),
        "count": len(results),
        "items": results,
    }

    Path(OUTPATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
