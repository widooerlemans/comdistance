#!/usr/bin/env python3
# augment_names.py
# Post-process the JSON created by your existing horizons_pull.py.
# Adds:
#   - name_suffix   e.g., "ATLAS", "SWAN", "Schaumasse"
#   - name_full     e.g., "C/2025 K1 (ATLAS)" or "24P/Schaumasse"
#   - display_name  equal to name_full
# Keeps ALL existing fields from your original output.

import json, re, pathlib, sys

ephem_path = pathlib.Path("data/comets_ephem.json")
cobs_path  = pathlib.Path("data/cobs_list.json")

if not ephem_path.exists():
    print("augment_names.py: missing data/comets_ephem.json", file=sys.stderr)
    sys.exit(0)

try:
    ephem = json.loads(ephem_path.read_text(encoding="utf-8"))
except Exception as e:
    print("augment_names.py: could not read comets_ephem.json:", e, file=sys.stderr)
    sys.exit(0)

# Build designation -> (fullname, suffix) map from COBS
cobs_map = {}
if cobs_path.exists():
    try:
        cobs = json.loads(cobs_path.read_text(encoding="utf-8"))
        for it in (cobs.get("comet_list") or []):
            desig = (it.get("comet_name") or "").strip()
            fullname = (it.get("comet_fullname") or "").strip()
            if not desig or not fullname:
                continue
            # Prefer "(Name)" at the end; else the segment after "/"
            suf = None
            m = re.search(r"\(([^)]+)\)\s*$", fullname)
            if m:
                suf = m.group(1).strip()
            else:
                m2 = re.search(r"/\s*([A-Za-z0-9\-\s]+)$", fullname)
                if m2:
                    suf = m2.group(1).strip()
            cobs_map[desig] = (fullname, suf)
    except Exception as e:
        print("augment_names.py: could not read/parse cobs_list.json:", e, file=sys.stderr)

items = ephem.get("items") or []
aug_count = 0

def norm(x):
    return (x or "").strip()

for it in items:
    desig = norm(it.get("id") or it.get("desig"))
    if not desig:
        continue

    fullname = desig
    suffix = None
    if desig in cobs_map:
        fullname, suffix = cobs_map[desig]

    if suffix is not None:
        it["name_suffix"] = suffix
    if fullname is not None:
        it["name_full"] = fullname
        it["display_name"] = fullname
        aug_count += 1

ephem_path.write_text(
    json.dumps(ephem, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8"
)
print(f"augment_names.py: augmented display_name for {aug_count} items")
