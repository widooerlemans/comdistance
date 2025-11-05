#!/usr/bin/env python3
# Plain UTF-8, no BOM.
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (data/cobs_list.json written by the workflow)
- Geometry/brightness from JPL Horizons

Keeps your original output structure and adds three fields per item:
- name_suffix   (e.g., 'Lemmon', 'ATLAS', 'Schaumasse')
- name_full     (e.g., 'C/2025 A6 (Lemmon)', '24P/Schaumasse', '3I/ATLAS')
- display_name  (same as name_full)

Rules:
- For 'C/...' => parenthetical: C/2025 A6 (Lemmon)
- For 'nP' and 'nI' => slash:    24P/Schaumasse, 3I/ATLAS

Optional filter via env BRIGHT_LIMIT (e.g., 15.0) keeps items with cobs_mag<=limit OR v_pred<=limit.
"""

import json, time, re, math, os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from math import acos, degrees

SCRIPT_VERSION = 18  # match your previous +1

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6
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

# MPC packed provisional code: CK25A060 -> C/2025 A6
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

_PAT_PERIODIC = re.compile(r"^\s*(\d+)\s*P(?:/.*)?\s*$", re.IGNORECASE)
_PAT_INTERSTELLAR = re.compile(r"^\s*(\d+)\s*I(?:/.*)?\s*$", re.IGNORECASE)
_PAT_C_PROV = re.compile(r"^\s*([PCADX])\s*/\s*(\d{4})\s+([A-Z]{1,2}\d{1,3})", re.IGNORECASE)

def strip_fragment(desig: str) -> str:
    s = _norm_spaces(desig)
    if re.match(r"^\d+\s*[PI]\s*[-\s]?[A-Z]$", s, re.I):
        s = re.sub(r"[-\s]?[A-Z]$", "", s)
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
        return f"{m.group(1).upper()}/{m.group(2)} {m.group(3).upper()}"
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

def _extract_suffix_from_fullname(fullname: str) -> Optional[str]:
    if not fullname:
        return None
    s = _norm_spaces(str(fullname))
    m = re.search(r"\(([^)]+)\)\s*$", s)
    if m:
        return m.group(1).strip()
    if "/" in s:
        parts = s.split("/", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return None

def load_cobs_designations_and_names(path: Path):
    mag_map: Dict[str, float] = {}
    suffix_map: Dict[str, str] = {}
    fullname_map: Dict[str, str] = {}
    dbg_first = []
    dbg_counts = {"packed_unpacked": 0, "fragments": 0, "plain": 0}

    if not path.exists():
        return mag_map, suffix_map, fullname_map, dbg_first, dbg_counts

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return mag_map, suffix_map, fullname_map, dbg_first, dbg_counts

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

    for o in raw_list:
        if not isinstance(o, dict):
            continue

        name_pref = (
            o.get("comet_fullname")
            or o.get("mpc_name")
            or o.get("comet_name")
            or o.get("designation")
            or o.get("name")
        )
        if name_pref and len(dbg_first) < 12:
            dbg_first.append(str(name_pref))

        # observed magnitude
        mag = None
        for k in ("mag", "magnitude", "current_mag", "peak_mag", "estimated_mag", "cur_mag"):
            if k in o:
                try:
                    mag = float(o[k]); break
                except Exception:
                    pass

        # canonical designation
        desig = None
        if name_pref and _PACKED.match(_norm_spaces(str(name_pref))):
            unpacked = unpack_mpc_packed(name_pref)
            if unpacked:
                desig = unpacked
                dbg_counts["packed_unpacked"] += 1
        else:
            d0 = to_designation(name_pref) if name_pref else None
            if d0:
                desig = d0
                dbg_counts["plain"] += 1

        if desig and re.match(r"^\d+\s*[PI]\s*[-\s]?[A-Z]$", desig, re.I):
            desig = strip_fragment(desig)
            dbg_counts["fragments"] += 1

        if not desig:
            continue

        if (mag is not None) and (desig not in mag_map or mag < mag_map[desig]):
            mag_map[desig] = mag

        full_pref = o.get("comet_fullname") or o.get("comet_name") or str(name_pref)
        suf = _extract_suffix_from_fullname(full_pref)
        if suf:
            suffix_map[desig] = suf
        if isinstance(o.get("comet_fullname"), str):
            fullname_map[desig] = _norm_spaces(o["comet_fullname"])

    return mag_map, suffix_map, fullname_map, dbg_first, dbg_counts

# -------- Horizons util --------
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

def _compose_full_name(desig: str, suffix: Optional[str]) -> (Optional[str], Optional[str]):
    if not desig or not suffix:
        return None, None
    if desig.startswith("C/"):
        full = f"{desig} ({suffix})"
    elif re.match(r"^\d+P$", desig, re.I) or re.match(r"^\d+I$", desig, re.I):
        full = f"{desig}/{suffix}"
    else:
        full = f"{desig} ({suffix})"
    return suffix, full

def main():
    bright_limit = try_float_env(BRIGHT_LIMIT_ENV)

    # COBS: observed magnitudes + names
    mag_map, suffix_map, fullname_map, debug_first_names, debug_counts = load_cobs_designations_and_names(COBS_PATH)
    comet_ids: List[str] = sorted(mag_map.keys()) if mag_map else []

    results: List[Dict[str, Any]] = []
    for cid in comet_ids:
        item = fetch_one(cid, OBSERVER)

        # attach observed mag and delta vs predicted
        if cid in mag_map:
            item["cobs_mag"] = mag_map[cid]
            vpred = item.get("v_pred") or item.get("vmag")
            if vpred is not None:
                try:
                    item["mag_diff_pred_minus_obs"] = round(float(vpred) - float(mag_map[cid]), 2)
                except Exception:
                    pass

        # names: prefer canonical fullname from COBS; else compose from suffix
        suffix = suffix_map.get(cid)
        cobs_full = fullname_map.get(cid)
        if cobs_full and to_designation(cobs_full) == cid:
            item["name_suffix"] = _extract_suffix_from_fullname(cobs_full) or suffix
            item["name_full"] = cobs_full
            item["display_name"] = cobs_full
        else:
            ns, nf = _compose_full_name(cid, suffix)
            if ns:
                item["name_suffix"] = ns
            if nf:
                item["name_full"] = nf
                item["display_name"] = nf

        results.append(item)
        time.sleep(PAUSE_S)

    # optional bright-limit filter
    if bright_limit is not None:
        filtered = []
        for it in results:
            obs = it.get("cobs_mag")
            pred = it.get("v_pred") or it.get("vmag")
            keep = (obs is not None and obs <= bright_limit) or (pred is not None and pred <= bright_limit)
            if keep:
                filtered.append(it)
        results = filtered

    # sort by observed brightness, then predicted
    results.sort(key=_sort_key)

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": bright_limit,
        "sorted_by": "cobs_mag asc, then v_pred asc",
        "source": {"observations": "COBS (file or direct fetch)", "theory": "JPL Horizons"},
        "cobs_designations": len(comet_ids),
        "cobs_used": bool(comet_ids),
        "debug_first_cobs_names": debug_first_names,
        "debug_counts": debug_counts,
        "count": len(results),
        "items": results,
    }
    Path(OUTPATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")

if __name__ == "__main__":
    main()
