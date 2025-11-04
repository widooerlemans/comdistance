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

import os
import io
import csv
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

SCRIPT_VERSION = 15  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.25
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this

# Optional: filter observed comets at or brighter than this mag (env BRIGHT_LIMIT)
def _env_bright_limit() -> Optional[float]:
    v = os.getenv("BRIGHT_LIMIT", "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None

BRIGHT_LIMIT = _env_bright_limit()

# Fallback hand list if COBS file missing/empty
COMETS_FALLBACK: List[str] = [
    "24P", "29P", "141P", "240P",
    "2P", "12P", "13P",
    "C/2023 A3",
]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------- COBS name normalization --------
# Accepts:
#   "24P", "24P/Schaumasse", "0024P/Schaumasse"
#   "C/2025 A6 (Lemmon)", "C/2024 E1 (Wierzchos)"
#   "3I/ATLAS" (interstellar)
_DESIG_PATTERNS = [
    # Periodic with optional name after slash
    re.compile(r'^\s*(\d+)\s*P(?:/[^,;()]+)?\s*$', re.I),
    # Interstellar class with number, e.g. 1I/ʻOumuamua, 2I/Borisov, 3I/ATLAS
    re.compile(r'^\s*(\d+)\s*I(?:/[^,;()]+)?\s*$', re.I),
    # MPC-style class/year/letter+index, optional (Name)
    re.compile(r'^\s*([PCDAXI])\s*/\s*(\d{4})\s*([A-Z])\s*(\d+)\s*(?:\([^)]+\))?\s*$',
               re.I),
]

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()

    # If an MPC designation appears inside parentheses, try that first.
    m_in = re.search(r'\(([^)]+)\)', s)
    if m_in:
        d = to_designation(m_in.group(1))
        if d:
            return d

    for pat in _DESIG_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        if pat is _DESIG_PATTERNS[0]:     # N P[/Name]
            n = str(int(m.group(1)))      # drop zero padding: "0024" -> "24"
            return f"{n}P"
        if pat is _DESIG_PATTERNS[1]:     # N I[/Name]
            n = str(int(m.group(1)))
            return f"{n}I"
        if pat is _DESIG_PATTERNS[2]:     # Class/Year Letter Index
            cls, yr, let, idx = m.groups()
            return f"{cls.upper()}/{yr} {let.upper()}{int(idx)}"
    return None


def load_cobs_designations(path: Path,
                           bright_limit: Optional[float] = None) -> Dict[str, float]:
    """
    Read data/cobs_list.json and return {designation: observed_mag}, filtered by bright_limit if provided.
    Tries several field names for both the comet name and the magnitude.
    """
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        src_bytes = len(path.read_bytes())
        src_ct = "application/json"
        src_err = None
    except Exception as e:
        # not JSON? try naive CSV/TSV (best effort)
        txt = path.read_text(encoding="utf-8", errors="ignore")
        src_bytes = len(txt.encode("utf-8", errors="ignore"))
        src_ct = "text/plain"
        src_err = str(e)
        dialect = csv.Sniffer().sniff(txt.splitlines()[0] if txt else "", [",", "\t", ";"])
        rows = list(csv.DictReader(io.StringIO(txt), dialect=dialect))
        raw = {"items": rows}

    # Choose the item list
    items = []
    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        items = raw["comet_list"]
    elif isinstance(raw, dict):
        for k in ("comets", "objects", "data", "items", "list"):
            if isinstance(raw.get(k), list):
                items = raw[k]; break
    elif isinstance(raw, list):
        items = raw

    name_keys = ("mpc_name", "comet_name", "comet_fullname", "designation", "name")
    mag_keys  = (
        "mag", "magnitude", "vmag", "v_mag", "current_mag", "best_mag", "est_mag",
        "estimated_mag", "mag_est", "mag_best"
    )

    out: Dict[str, float] = {}

    for o in items:
        if not isinstance(o, dict):
            continue

        # pick name
        name_val = None
        for nk in name_keys:
            if nk in o:
                name_val = o[nk]
                if name_val:
                    break
        desig = to_designation(name_val) if name_val else None
        if not desig:
            continue

        # pick magnitude
        mag_val = None
        for mk in mag_keys:
            if mk in o and o[mk] not in (None, "", "-"):
                try:
                    mag_val = float(o[mk])
                    break
                except Exception:
                    pass

        # apply bright_limit if given; if mag unknown, skip (we only want ≤ limit)
        if bright_limit is not None:
            if mag_val is None or not (mag_val <= bright_limit):
                continue

        # store; keep the brightest (lowest) mag if duplicates
        if mag_val is not None:
            if (desig not in out) or (mag_val < out[desig]):
                out[desig] = mag_val
        else:
            # no mag and no limit -> still include (rare)
            if bright_limit is None and desig not in out:
                out[desig] = float("nan")

    # record a tiny breadcrumb about the COBS source in a module-global
    load_cobs_designations._last_meta = {
        "cobs_bytes": src_bytes,
        "cobs_content_type": src_ct,
        "cobs_error": src_err,
        "cobs_source": "file",
    }
    return out

# a place to hang metadata
load_cobs_designations._last_meta = {
    "cobs_bytes": None,
    "cobs_content_type": None,
    "cobs_error": None,
    "cobs_source": None,
}
# ----------------------------------------


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
    # else pick the latest epoch overall
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
        Horizons(id=designation, id_type="designation", location=OBSERVER, epochs=[jd_now]).ephemerides(quantities="1")
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
# -----------------------------------


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    """Fetch RA/DEC/delta from ephemeris; compute r & phase from vectors; compute v_pred if needed."""
    jd_now = Time.now().jd

    # Try by designation
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
        # Resolve & retry entirely by record id
        rec_id = resolve_ambiguous_to_record_id(comet_id)
        if rec_id is None:
            # could be unknown ID (e.g., misparsed); surface error
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
    # 0) Load COBS (observations)
    cobs_map = load_cobs_designations(COBS_PATH, bright_limit=BRIGHT_LIMIT)
    cobs_meta = getattr(load_cobs_designations, "_last_meta", {})

    if cobs_map:
        comet_ids: List[str] = sorted(cobs_map.keys())
        used_cobs = True
    else:
        comet_ids = COMETS_FALLBACK
        used_cobs = False

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)

        # Merge observed mag if available
        if cid in cobs_map:
            try:
                item["cobs_mag"] = float(cobs_map[cid])
            except Exception:
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
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
            **({k: v for k, v in cobs_meta.items() if v is not None}),
        },
        "cobs_designations": len(cobs_map),
        "cobs_used": used_cobs,
        "count": len(results),
        "items": results,
    }
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
