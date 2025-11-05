#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (prefer data/cobs_list.json if present; supports 'comet_list')
- Geometry/brightness from JPL Horizons

Highlights
- Converts MPC packed codes like CK25A060 -> C/2025 A6 (divide ddd by 10; strips fragment suffixes).
- Queries "now" with epochs=[JD] to avoid TLIST/WLDINI.
- Resolves ambiguous periodic designations to most recent apparition.
- Returns RA/DEC/Δ; computes r & phase from state vectors; predicts v if Horizons doesn’t supply it.
- Sorts output by observed brightness (cobs_mag asc), then by v_pred.

Optional filter: set env BRIGHT_LIMIT (e.g., 15.0) to keep only comets with cobs_mag<=limit OR v_pred<=limit.
"""

import os
import json, time, re, math
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 17  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
OUT_JSON_PATH = Path(OUTPATH)
COBS_PATH = Path("data/cobs_list.json")  # workflow writes this
BRIGHT_LIMIT_ENV = "BRIGHT_LIMIT"
# ----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- Unicode/Name Normalization ----------
_SPACE_PAT = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]+")
def _norm_spaces(s: str) -> str:
    s = _SPACE_PAT.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# MPC packed provisional comet code: e.g., CK25A060 → C/2025 A6
# Format: [C|D|P|A|X] K yy L ddd  (ddd = N*10; divide by 10 to get N)
_PACKED = re.compile(r"^\s*([CDPAX])K(\d{2})([A-Z])(\d{3})(?:-[A-Z])?\s*$", re.IGNORECASE)

def unpack_mpc_packed(name_like: str) -> Optional[str]:
    if not name_like:
        return None
    s = _norm_spaces(str(name_like))
    m = _PACKED.match(s)
    if not m:
        return None
    fam = m.group(1).upper()
    yy  = int(m.group(2))
    half = m.group(3).upper()
    ddd = int(m.group(4))
    n = ddd // 10            # e.g. 060 -> 6, 010 -> 1, 020 -> 2
    year = 1900 + yy if yy >= 50 else 2000 + yy
    return f"{fam}/{year} {half}{n}"

# Accept conventional designations:
_PAT_PERIODIC = re.compile(r"^\s*(\d+)\s*P(?:/.*)?\s*$", re.IGNORECASE)       # 24P, 240P/Name -> 24P
_PAT_INTERSTELLAR = re.compile(r"^\s*(\d+)\s*I(?:/.*)?\s*$", re.IGNORECASE)   # 3I/ATLAS -> 3I
_PAT_C_PROV = re.compile(r"^\s*([PCADX])\s*/\s*(\d{4})\s+([A-Z]{1,2}\d{1,3})", re.IGNORECASE)

def strip_fragment(desig: str) -> str:
    s = _norm_spaces(desig)
    s = re.sub(r"[-\s]?[A-Z]$", "", s) if re.match(r"^\d+\s*[PI]\s*[-\s]?[A-Z]$", s, re.I) else s
    s = re.split(r"[^\dPI/]", s, 1, flags=re.I)[0]
    return s

def to_designation(name_like: str) -> Optional[str]:
    if not name_like:
        return None
    s = _norm_spaces(str(name_like))

    unpacked = unpack_mpc_packed(s)
    if unpacked:
        return unpacked

    s_no_paren = re.sub(r"\s*\([^)]+\)\s*$", "", s)

    m = _PAT_PERIODIC.match(s_no_paren)
    if m:
        return strip_fragment(f"{int(m.group(1))}P")

    m = _PAT_INTERSTELLAR.match(s_no_paren)
    if m:
        return f"{int(m.group(1))}I"

    m = _PAT_C_PROV.match(s)
    if m:
        fam = m.group(1).upper()
        year = m.group(2)
        code = m.group(3).upper()
        return f"{fam}/{year} {code}"

    m2 = re.search(r"\(([PCADX]/\s*\d{4}\s+[A-Za-z]{1,2}\d{1,3})\)", s, re.IGNORECASE)
    if m2:
        inner = _norm_spaces(m2.group(1))
        mm = _PAT_C_PROV.match(inner)
        if mm:
            return f"{mm.group(1).upper()}/{mm.group(2)} {mm.group(3).upper()}"

    m3 = re.match(r"^\s*([PCADX])\s*/\s*(\d{4})([A-Za-z]{1,2}\d{1,3})\s*$", s, re.IGNORECASE)
    if m3:
        return f"{m3.group(1).upper()}/{m3.group(2)} {m3.group(3).upper()}"

    return None

def load_cobs_designations(path: Path) -> Dict[str, Any]:
    """
    Load COBS (or MPC-like) comet entries, normalize to a designation map, and
    also capture a map of designation -> human-friendly full name.

    Returns a dict like:
        {
          "C/2025 A6": 4.5,           # observed/estimated mag
          ...
          "_fullname_map": {"C/2025 A6": "C/2025 A6 (Lemmon)", "210P": "210P/Christensen"},
          "_debug_first_names": [...],
          "_debug_counts": {...}
        }
    """
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Accept multiple shapes:
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
    dbg_first: List[str] = []
    dbg_counts = {"packed_unpacked": 0, "fragments": 0, "plain": 0}
    fullname_map: Dict[str, str] = {}

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    for o in items:
        if not isinstance(o, dict):
            continue

        name = (
            o.get("mpc_name")
            or o.get("comet_fullname")
            or o.get("comet_name")
            or o.get("designation")
            or o.get("fullname")
            or o.get("name")
        )
        if name and len(dbg_first) < 12:
            dbg_first.append(str(name))

        # Prefer a nice full name for display, if present
        full = o.get("comet_fullname") or o.get("fullname") or o.get("comet_name") or o.get("name")
        if isinstance(full, str):
            full = _norm_spaces(full)
            # drop simple fragment suffixes like "(A)"
            full = re.sub(r"\s*\([A-Z]\)\s*$", "", full).strip()

        # Try to find a magnitude-ish field
        mag = (
            _to_float(o.get("mag"))
            or _to_float(o.get("magnitude"))
            or _to_float(o.get("current_mag"))
            or _to_float(o.get("peak_mag"))
            or _to_float(o.get("estimated_mag"))
            or _to_float(o.get("v"))
            or _to_float(o.get("vmag"))
            or _to_float(o.get("est_mag"))
            or _to_float(o.get("pred_mag"))
        )

        # Normalize designation
        desig = None
        if name and _PACKED.match(_norm_spaces(str(name))):
            unpacked = unpack_mpc_packed(name)
            if unpacked:
                desig = unpacked
                dbg_counts["packed_unpacked"] += 1
        else:
            d0 = to_designation(name) if name else None
            if d0:
                desig = d0
                dbg_counts["plain"] += 1

        # Strip fragments like "210P-A" -> "210P"
        if desig and re.match(r"^\d+\s*[PI]\s*[-\s]?[A-Z]$", desig, re.I):
            desig = strip_fragment(desig)
            dbg_counts["fragments"] += 1

        if desig:
            if full:
                fullname_map.setdefault(desig, full)
            if mag is not None:
                # keep the brightest (smallest) value
                if desig not in result or mag < result[desig]:
                    result[desig] = mag

    result["_debug_first_names"] = dbg_first
    result["_debug_counts"] = dbg_counts
    result["_fullname_map"] = fullname_map
    return result

# -------- Horizons utilities --------
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
    rn = _vec_norm(sx, sy, sz)
    dn = _vec_norm(ex, ey, ez)
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
    out = {"r_au": r_au, "delta_au": delta_au, "phase_deg": alpha, "ra_deg": ra, "dec_deg": dec, "vmag": vmag}

    # Mirror Horizons of-date RA/Dec as JNow
    out["ra_jnow_deg"] = ra
    out["dec_jnow_deg"] = dec

    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass

    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
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
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None

def main():
    # Load COBS designations & metadata
    cobs_map = load_cobs_designations(COBS_PATH)

    debug_first_names = cobs_map.pop("_debug_first_names", [])
    debug_counts      = cobs_map.pop("_debug_counts", {})
    fullname_map      = cobs_map.pop("_fullname_map", {})

    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else []
    print(f"Loaded {len(comet_ids)} COBS designations; sample: {comet_ids[:10]}")

    results: List[Dict[str, Any]] = []

    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)

        # attach COBS mag and Δmag if possible
        if cid in cobs_map:
            item["cobs_mag"] = cobs_map[cid]
            vpred = item.get("v_pred") or item.get("vmag")
            if vpred is not None and cobs_map[cid] is not None:
                try:
                    item["mag_diff_pred_minus_obs"] = round(float(vpred) - float(cobs_map[cid]), 2)
                except Exception:
                    pass

        # attach pretty full name if we have it
        if cid in fullname_map:
            item["name_full"] = fullname_map[cid]  # e.g. "C/2025 A6 (Lemmon)" or "210P/Christensen"

        results.append(item)
        time.sleep(PAUSE_S)

    # --- tolerant brightness filter (keeps item if either mag passes; keeps if both unknown) ---
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    bright_limit = try_float_env(BRIGHT_LIMIT_ENV)
    if bright_limit is not None:
        before = len(results)
        kept = []
        for r in results:
            vp = _f(r.get("v_pred"))
            cp = _f(r.get("cobs_mag"))
            if (
                (vp is None and cp is None) or
                (vp is not None and vp <= bright_limit) or
                (cp is not None and cp <= bright_limit)
            ):
                kept.append(r)
        results = kept
        print(f"Brightness filter {bright_limit}: kept {len(results)}/{before}")

    # sort: observed brightness first (cobs_mag asc), then predicted v
    def sort_key(r):
        cp = _f(r.get("cobs_mag"))
        vp = _f(r.get("v_pred"))
        return (
            (0, cp) if cp is not None else (1, float('inf')),
            (0, vp) if vp is not None else (1, float('inf')),
        )
    results.sort(key=sort_key)

    # write JSON
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "items": results,
        "_debug_first_names": debug_first_names,
        "_debug_counts": debug_counts,
    }
    OUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON_PATH} with {len(results)} items")

if __name__ == "__main__":
    main()
