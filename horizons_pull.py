#!/usr/bin/env python3
"""
Build data/comets_ephem.json by merging:
- Observed list from COBS (prefer data/cobs_list.json if present; supports 'comet_list')
- Geometry/brightness from JPL Horizons

If data/cobs_list.json is missing/empty/unparseable, this script will
fetch from COBS directly and try JSON first, then CSV/TSV fallback.

Key points
- Queries "now" using epochs=[JD] to avoid TLIST/WLDINI errors.
- Resolves ambiguous periodic designations (e.g., 2P/12P/13P) to the most
  recent apparition (preferring the last N years) via Horizons record-id.
- Returns RA/DEC/Δ from ephemerides; computes r (heliocentric) and phase angle
  from state vectors so values exist even if Horizons omits r/alpha columns.
- If V is missing, computes predicted magnitude:
      v_pred = M1 + 5*log10(Δ) + k1*log10(r)
"""

import json, time, re, math, io, csv
from math import acos, degrees
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from astropy.time import Time
from astroquery.jplhorizons import Horizons

SCRIPT_VERSION = 12  # bump when you update this file

# ---------- CONFIG ----------
OBSERVER = "500"                 # geocenter
YEARS_WINDOW = 6                 # prefer most recent apparition within N years
QUANTITIES = "1,3,4,20,21,31"    # r, delta, alpha, RA, DEC, V
PAUSE_S = 0.3
OUTPATH = "data/comets_ephem.json"
COBS_PATH = Path("data/cobs_list.json")  # workflow may write this

# Direct COBS endpoint (used as fallback by this script)
COBS_URL = ("https://cobs.si/api/planner.api"
            "?lat=52.09&long=5.12&alt=10&lim_mag=15&min_alt=0&min_sol=0&min_moon=0")

# Optional “likely visible” filter (None disables)
BRIGHT_LIMIT = None  # e.g. 17.5

# Fallback list if we get nothing from COBS
COMETS_FALLBACK: List[str] = ["2P", "12P", "13P", "C/2023 A3"]
# ---------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------- Normalize COBS names to MPC-style designations --------
_DESIG = re.compile(r"""
    ^\s*(
        \d+\s*P(?:/[A-Za-z0-9-]+)?            # 2P or 2P/Encke
        |[PCADX]/\d{4}\s+[A-Z]\d+             # C/2023 A3 etc.
        (?:\s*\([^)]+\))?                     # optional (Name)
    )
""", re.IGNORECASE | re.VERBOSE)

def to_designation(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = _DESIG.match(s)
    if m:
        d = m.group(1)
        d = re.sub(r"\s*\([^)]+\)\s*$", "", d)      # drop (Name)
        return re.sub(r"\s+", " ", d).upper()
    m2 = re.search(r"\(([^)]+)\)", s)               # sometimes designation inside ()
    if m2:
        return to_designation(m2.group(1))
    return None


# -------- Load/Fetch COBS list --------
def _http_get(url: str, timeout: int = 30) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        req = Request(url, headers={
            "User-Agent": "comdistance-bot/1.0",
            "Accept": "*/*",
        })
        with urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("content-type", "")
            return r.read(), ctype
    except (URLError, HTTPError) as e:
        return None, f"HTTP error: {e}"
    except Exception as e:
        return None, str(e)

def _parse_cobs_json_bytes(b: bytes) -> Dict[str, float]:
    raw = json.loads(b.decode("utf-8", errors="replace"))

    # 1) Planner API shape: top-level 'comet_list': [...]
    if isinstance(raw, dict) and isinstance(raw.get("comet_list"), list):
        out: Dict[str, float] = {}
        for o in raw["comet_list"]:
            if not isinstance(o, dict):
                continue
            name = o.get("mpc_name") or o.get("comet_name") or o.get("comet_fullname") or o.get("name")
            mag = o.get("magnitude") or o.get("mag")
            desig = to_designation(str(name)) if name else None
            try:
                mag_f = float(mag) if mag is not None else None
            except Exception:
                mag_f = None
            if desig and (mag_f is not None):
                if desig not in out or mag_f < out[desig]:
                    out[desig] = mag_f
        if out:
            return out
        # fall through if empty

    # 2) Older / custom shapes we used earlier
    if isinstance(raw, dict):
        for key in ("comets", "objects", "data", "items", "list"):
            if key in raw and isinstance(raw[key], list):
                raw_list = raw[key]
                break
        else:
            if all(isinstance(k, str) for k in raw.keys()):  # mapping {name: mag}
                out = {}
                for k, v in raw.items():
                    desig = to_designation(k)
                    if desig:
                        try:
                            out[desig] = float(v)
                        except Exception:
                            pass
                return out
            raw_list = []
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    result: Dict[str, float] = {}
    for o in raw_list:
        if not isinstance(o, dict):
            continue
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
            if desig not in result or mag < result[desig]:
                result[desig] = mag
    return result

def _parse_cobs_csv_bytes(b: bytes) -> Dict[str, float]:
    """
    Very tolerant CSV/TSV parser. Tries commas, tabs, semicolons.
    Looks for columns that smell like designation and magnitude.
    """
    text = b.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delim = "," if "," in first_line else ("\t" if "\t" in first_line else ";")

    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delim)
    name_keys = ("mpc_name","designation","comet_name","comet_fullname","fullname","name","object","id")
    mag_keys  = ("magnitude","mag","current_mag","peak_mag","est_mag","m1")
    out: Dict[str, float] = {}
    for row in reader:
        name = None
        for k in name_keys:
            if k in row and row[k]:
                name = row[k]; break
        if not name and reader.fieldnames:
            name = row.get(reader.fieldnames[0])
        desig = to_designation(str(name)) if name else None
        if not desig:
            continue
        val = None
        for k in mag_keys:
            if k in row and row[k]:
                try:
                    val = float(row[k]); break
                except Exception:
                    pass
        if val is not None:
            if desig not in out or val < out[desig]:
                out[desig] = val
    return out

def load_or_fetch_cobs_map(path: Path, fallback_url: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns ({designation: mag}, debug_info)
    debug_info includes: source ("file"/"http-json"/"http-csv"/"fallback"),
    bytes, content_type, error (if any)
    """
    dbg = {"source": None, "bytes": 0, "content_type": None, "error": None}

    # 1) Try file
    if path.exists():
        try:
            b = path.read_bytes()
            dbg.update({"source": "file", "bytes": len(b), "content_type": "application/json"})
            mp = _parse_cobs_json_bytes(b)
            if mp:
                return mp, dbg
        except Exception as e:
            dbg["error"] = f"file-parse: {e}"

    # 2) Try HTTP JSON
    b, ctype = _http_get(fallback_url)
    if b:
        dbg.update({"source": "http-json", "bytes": len(b), "content_type": ctype})
        try:
            mp = _parse_cobs_json_bytes(b)
            if mp:
                return mp, dbg
        except Exception as e:
            dbg["error"] = f"http-json-parse: {e}"

        # 3) Try CSV fallback
        try:
            mp = _parse_cobs_csv_bytes(b)
            if mp:
                dbg["source"] = "http-csv"
                return mp, dbg
        except Exception as e:
            dbg["error"] = f"http-csv-parse: {e}"

    # 4) Nothing worked → fallback list with no mags
    return {}, {"source": "fallback", "bytes": 0, "content_type": None, "error": dbg.get("error")}


# -------------------- Horizons helpers -------------------------
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
        "phase_deg": alpha,
        "ra_deg": ra,
        "dec_deg": dec,
        "vmag": vmag,
    }
    if (vmag is None) and (M1 is not None) and (k1 is not None) and (r_au is not None) and (delta_au is not None):
        try:
            out["v_pred"] = M1 + 5.0*math.log10(delta_au) + k1*math.log10(r_au)
        except ValueError:
            pass
    if any(out.get(k) is None for k in ("delta_au", "ra_deg", "dec_deg")):
        out["_cols"] = list(cmap.values())
    return out
# ---------------------------------------------------------------


def fetch_one(comet_id: str, observer) -> Dict[str, Any]:
    jd_now = Time.now().jd
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
    # 0) Prefer file; otherwise fetch live; otherwise fallback
    cobs_map_file = {}
    if COBS_PATH.exists():
        try:
            cobs_map_file = _parse_cobs_json_bytes(COBS_PATH.read_bytes())
        except Exception:
            cobs_map_file = {}

    if cobs_map_file:
        cobs_map = cobs_map_file
        cobs_dbg = {"source": "file", "bytes": COBS_PATH.stat().st_size, "content_type": "application/json", "error": None}
    else:
        cobs_map, cobs_dbg = load_or_fetch_cobs_map(COBS_PATH, COBS_URL)

    comet_ids: List[str] = sorted(cobs_map.keys()) if cobs_map else COMETS_FALLBACK

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

    # Optional brightness cut
    if BRIGHT_LIMIT is not None:
        results = [
            it for it in results
            if (it.get("vmag") is not None and it["vmag"] <= BRIGHT_LIMIT)
            or (it.get("v_pred") is not None and it["v_pred"] <= BRIGHT_LIMIT)
        ]

    payload = {
        "generated_utc": now_iso(),
        "observer": OBSERVER,
        "years_window": YEARS_WINDOW,
        "script_version": SCRIPT_VERSION,
        "bright_limit": BRIGHT_LIMIT,
        "source": {
            "observations": "COBS (file or direct fetch)",
            "theory": "JPL Horizons",
            "cobs_source": cobs_dbg.get("source"),
            "cobs_bytes": cobs_dbg.get("bytes"),
            "cobs_content_type": cobs_dbg.get("content_type"),
            "cobs_error": cobs_dbg.get("error"),
        },
        "cobs_designations": len(cobs_map),
        "cobs_used": bool(cobs_map),
        "count": len(results),
        "items": results,
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUTPATH} with {len(results)} comets.")


if __name__ == "__main__":
    main()
