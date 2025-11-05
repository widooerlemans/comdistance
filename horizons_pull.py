#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (data/cobs_list.json written by the workflow)
- Geometry/brightness from JPL Horizons

This keeps your original info (r_au, delta_au, phase, RA/Dec, v, v_pred, etc.)
and adds properly formatted names:

- C/...  -> "C/YYYY Xn (Suffix)", e.g. "C/2025 A6 (Lemmon)"
- P/...  -> "NNP/Suffix" or "NNP-Fragment/Suffix", e.g. "24P/Schaumasse", "141P-B/Machholz"
- I/...  -> "3I/ATLAS"

We also carry useful COBS fields when present (best_time, best_ra, best_dec, best_alt,
trend, constellation, comet_type, mpc_name, comet_fullname) and the observed mag.

Optional filter via env BRIGHT_LIMIT to keep comets with observed mag <= limit
(or Horizons v/v_pred <= limit if observed not available).
"""

import json, time, re, math, os
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 18  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")
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
    n = ddd // 10
    year = 1900 + yy if yy >= 50 else 2000 + yy
    return f"{fam}/{year} {half}{n}"

# Accept conventional designs:
_PAT_PERIODIC = re.compile(r"^\s*(\d+)\s*P(?:/.*)?\s*$", re.IGNORECASE)   # 24P, 240P/Name -> 24P
_PAT_INTERSTELLAR = re.compile(r"^\s*(\d+)\s*I(?:/.*)?\s*$", re.IGNORECASE)  # 3I/ATLAS -> 3I
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

# ---- Name formatting helpers -------------------------------------------------

def _suffix_from_fullname(cobs_fullname: str, desig: str) -> Tuple[Optional[str], str]:
    """
    Returns (suffix, name_full) from a COBS fullname and the canonical designation.
    - If COBS has 'C/2025 A6 (Lemmon)' -> ('Lemmon', 'C/2025 A6 (Lemmon)')
    - If '24P/Schaumasse' -> ('Schaumasse', '24P/Schaumasse')
    - If '3I/ATLAS' -> ('ATLAS', '3I/ATLAS')
    If COBS fullname missing/odd, fall back to desig-only with no suffix.
    """
    if not cobs_fullname:
        return None, desig
    s = _norm_spaces(str(cobs_fullname))

    # C/… (Suffix)
    m = re.match(r"^(C/[0-9]{4}\s+[A-Za-z]{1,2}\d{1,3})\s*\(([^)]+)\)\s*$", s, re.IGNORECASE)
    if m:
        d = m.group(1)
        suf = m.group(2).strip()
        # normalize designation casing/spaces to our computed desig
        name_full = f"{desig} ({suf})" if desig else f"{d} ({suf})"
        return suf, name_full

    # NNP[/fragment]/Suffix
    m = re.match(r"^(\d+\s*P(?:-[A-Z])?)\s*/\s*(.+)$", s, re.IGNORECASE)
    if m:
        idpart = re.sub(r"\s+", "", m.group(1)).upper().replace(" ", "")
        suf = _norm_spaces(m.group(2))
        name_full = f"{idpart}/{suf}"
        return suf, name_full

    # I/ style, e.g. 3I/ATLAS
    m = re.match(r"^(\d+\s*I)\s*/\s*(.+)$", s, re.IGNORECASE)
    if m:
        idpart = re.sub(r"\s+", "", m.group(1)).upper().replace(" ", "")
        suf = _norm_spaces(m.group(2))
        name_full = f"{idpart}/{suf}"
        return suf, name_full

    # If COBS fullname is just the designation, provide no suffix
    ds = to_designation(s)
    if ds:
        return None, ds

    # last resort
    return None, desig or s

# ---- Load COBS and build a rich index ---------------------------------------

def load_cobs_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns { desig: { 'mag': float, 'cobs': {...raw fields...}, 'suffix': str|None,
                       'name_full': str, 'display_name': str } }
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        raw_list = raw["comet_list"]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []
        for k in ("comets", "objects", "items", "data", "list"):
            if isinstance(raw.get(k), list):
                raw_list = raw[k]; break

    for o in raw_list:
        if not isinstance(o, dict):
            continue

        # choose something to convert to designation
        name_like = (
            o.get("mpc_name") or
            o.get("comet_fullname") or
            o.get("comet_name") or
            o.get("designation") or
            o.get("name")
        )
        desig = None
        if name_like and _PACKED.match(_norm_spaces(str(name_like))):
            desig = unpack_mpc_packed(name_like)
        if not desig and name_like:
            desig = to_designation(name_like)
        if not desig:
            continue

        # magnitude
        mag = None
        for k in ("magnitude", "mag", "current_mag", "peak_mag", "estimated_mag", "cur_mag"):
            if k in o:
                try:
                    mag = float(o[k]); break
                except Exception:
                    pass

        # name fields from comet_fullname when available
        cobs_fullname = o.get("comet_fullname") or ""
        suffix, name_full = _suffix_from_fullname(cobs_fullname, desig)
        display_name = name_full

        # stash
        out[desig] = {
            "mag": mag,
            "cobs": o,
            "suffix": suffix,
            "name_full": name_full,
            "display_name": display_name,
        }

    return out

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

    # also mirror of-date RA/Dec as JNow
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
    if not v: return None
    try:
        return float(v)
    except Exception:
        return None

def _sort_key(it: Dict[str, Any]):
    cm = it.get("cobs_mag")
    vp = it.get("v_pred")
    cm_key = cm if isinstance(cm, (int, float)) else 1e9
    vp_key = vp if isinstance(vp, (int, float)) else 1e9
    return (cm_key, vp_key)

def main():
    bright_limit = try_float_env(BRIGHT_LIMIT_ENV)

    # Build rich COBS index
    cobs_index = load_cobs_index(COBS_PATH)
    comet_ids: List[str] = sorted(cobs_index.keys())

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        # Horizons + geometry
        item = fetch_one(cid, OBSERVER)

        # Merge COBS metadata (observed magnitude + planner goodies when present)
        cobs = cobs_index.get(cid, {})
        if cobs:
            # names
            item["desig"] = cid
            if cobs.get("suffix") is not None:
                item["name_suffix"] = cobs["suffix"]
            if cobs.get("name_full"):
                item["name_full"] = cobs["name_full"]
                item["display_name"] = cobs["display_name"]

            # observed magnitude for sorting/filtering
            if cobs.get("mag") is not None:
                item["cobs_mag"] = cobs["mag"]

            # carry useful COBS planner fields when available
            raw = cobs.get("cobs") or {}
            for k_src, k_dst in [
                ("best_time","best_time"),
                ("best_ra","best_ra"),
                ("best_dec","best_dec"),
                ("best_alt","best_alt"),
                ("trend","trend"),
                ("constelation","constellation"),  # COBS typo → normalized
                ("comet_type","cobs_type"),
                ("mpc_name","cobs_mpc_name"),
                ("comet_fullname","cobs_fullname"),
            ]:
                if k_src in raw and raw[k_src] not in (None, ""):
                    item[k_dst] = raw[k_src]

            # quick diagnostic: predicted minus observed mag
            vpred = item.get("v_pred") or item.get("vmag")
            if (cobs.get("mag") is not None) and (vpred is not None):
                try:
                    item["mag_diff_pred_minus_obs"] = round(float(vpred) - float(cobs["mag"]), 2)
                except Exception:
                    pass

        results.append(item)
        time.sleep(PAUSE_S)

    # Filtering
    if bright_limit is not None:
        filtered = []
        for it in results:
            obs = it.get("cobs_mag")
            pred = it.get("v_pred") or it.get("vmag")
            keep = (obs is not None and obs <= bright_limit) or (pred is not None and pred <= bright_limit)
            if keep:
                filtered.append(it)
        results = filtered

    # Sort by observed brightness, then predicted
    results.sort(key=_sort_key)

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": bright_limit,
        "sorted_by": "cobs_mag asc, then v_pred asc",
        "source": {"cobs": True, "jpl_horizons": True},
        "count": len(results),
        "items": results,
    }
    Path(OUTPATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
