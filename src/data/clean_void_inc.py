import os
import glob
import json
import math
import shutil
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Pool
from tqdm import tqdm


JSON_DIR = "../dataset/hMOF/hmof_data"
DST_DIR = "../dataset/hMOF/hmof_data_clean_void_inc"
REPORT_DIR = "../dataset/hMOF/check_report"
CHECK_ONLY = False
GEO_REQUIRED = True
INC_REQUIRED = True
COPY_CIF = True
NUM_WORKERS = 16
HEAT_MIN = 0.0
LOAD_MIN = 0.0
PLD_LCD_TOL = 1e-6
INC_MIN = 0.0

KEYS = ['co2_0.01', 'co2_0.05', 'co2_0.1', 'co2_0.5', 'co2_2.5',
        'ch4_0.05', 'ch4_0.5', 'ch4_0.9', 'ch4_2.5', 'ch4_4.5', 'ch4_35.0',
        'h2_2.0', 'h2_100.0']
SPECIES_PREFIX = {'CarbonDioxide': 'co2_', 'Methane': 'ch4_', 'Hydrogen': 'h2_'}
GEO_KEYS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd']


def fill_dict(entries, target):
    for iso in entries:
        now_species = iso['isotherm_data'][0]['species_data'][0]['name']
        if now_species not in SPECIES_PREFIX:
            continue
        for d in iso['isotherm_data']:
            if len(d['species_data']) != 1:
                raise AssertionError(f"species length: {len(d['species_data'])}")
            if d['species_data'][0]['composition'] != 1:
                raise AssertionError(f"composition: {d['species_data'][0]['composition']}")
            key = SPECIES_PREFIX[now_species] + str(d['pressure'])
            if key in target:
                target[key].append(d['species_data'][0]['adsorption'])


def is_bad_number(v):
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def check_geo(json_data):
    vals, problems = {}, []
    for k in GEO_KEYS:
        v = json_data.get(k)
        if is_bad_number(v):
            vals[k] = np.nan
            problems.append(f"{k}:missing")
        else:
            vals[k] = float(v)

    vf = vals['void_fraction']
    if not np.isnan(vf) and not (0.0 < vf <= 1.0):
        problems.append(f"void_fraction:out_of_range({vf:.4g})")
    for k in ['surface_area_m2g', 'surface_area_m2cm3']:
        if not np.isnan(vals[k]) and vals[k] < 0:
            problems.append(f"{k}:negative({vals[k]:.4g})")
    for k in ['pld', 'lcd']:
        if not np.isnan(vals[k]) and vals[k] <= 0:
            problems.append(f"{k}:non_positive({vals[k]:.4g})")
    if not np.isnan(vals['pld']) and not np.isnan(vals['lcd']) and vals['pld'] > vals['lcd'] + PLD_LCD_TOL:
        problems.append(f"pld_gt_lcd({vals['pld']:.4g}>{vals['lcd']:.4g})")

    if any(p.endswith(":missing") for p in problems):
        status = "missing"
    elif problems:
        status = "invalid"
    else:
        status = "ok"
    return vals, status, ";".join(problems), [p.split(":")[0].split("(")[0] for p in problems]


def check_increments(json_data):
    groups = defaultdict(dict)
    for iso in json_data.get('isotherms', []):
        sp = iso['isotherm_data'][0]['species_data']
        if len(sp) != 1 or sp[0]['name'] not in SPECIES_PREFIX:
            continue
        gas = SPECIES_PREFIX[sp[0]['name']].rstrip('_')
        unit = str(iso.get('adsorptionUnits', '')).strip().lower()
        for d in iso['isotherm_data']:
            if len(d['species_data']) != 1:
                continue
            p, v = d.get('pressure'), d['species_data'][0].get('adsorption')
            if is_bad_number(p) or is_bad_number(v):
                continue
            key = SPECIES_PREFIX[sp[0]['name']] + str(float(p))
            if key in KEYS:
                groups[(gas, unit)][float(p)] = float(v)

    problems, bad_intervals = [], []
    for (gas, unit), pv in groups.items():
        ps = sorted(pv)
        for a, b in zip(ps[:-1], ps[1:]):
            inc = pv[b] - pv[a]
            if inc <= INC_MIN:
                name = f"{gas}_{a}->{b}"
                problems.append(f"{name}:{inc:.4g}[{unit}]")
                bad_intervals.append(name)
    status = "ok" if not problems else "non_increasing"
    return status, ";".join(problems), bad_intervals


def check_file(json_file):
    res = {"file": json_file, "status": None, "detail": "", "bad_keys": "",
           "geo_status": "", "geo_detail": "", "geo_bad_keys": "",
           "inc_status": "", "inc_detail": "", "inc_bad_intervals": ""}
    for k in GEO_KEYS:
        res[k] = np.nan
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        load_d = {k: [] for k in KEYS}
        heat_d = {k: [] for k in KEYS}
        fill_dict(json_data['isotherms'], load_d)
        fill_dict(json_data['heats'], heat_d)
    except Exception as e:
        res.update(status="broken", detail=f"{type(e).__name__}: {e}")
        return res

    geo_vals, geo_status, geo_detail, geo_bad = check_geo(json_data)
    res.update(geo_vals)
    res.update(geo_status=geo_status, geo_detail=geo_detail, geo_bad_keys=",".join(geo_bad))

    inc_status, inc_detail, inc_bad = check_increments(json_data)
    res.update(inc_status=inc_status, inc_detail=inc_detail, inc_bad_intervals=",".join(inc_bad))

    miss_l = [k for k in KEYS if len(load_d[k]) != 1]
    miss_h = [k for k in KEYS if len(heat_d[k]) != 1]
    if miss_l or miss_h:
        res.update(status="incomplete", detail=f"load={','.join(miss_l)} heat={','.join(miss_h)}")
        return res

    bad = []
    for k in KEYS:
        h, l = heat_d[k][0], load_d[k][0]
        if is_bad_number(h):
            bad.append(f"heat_{k}:nonfinite")
        elif float(h) <= HEAT_MIN:
            bad.append(f"heat_{k}:{float(h):.4g}")
        if is_bad_number(l):
            bad.append(f"load_{k}:nonfinite")
        elif float(l) < LOAD_MIN:
            bad.append(f"load_{k}:{float(l):.4g}")
    if bad:
        res.update(status="invalid_value", detail=";".join(bad),
                   bad_keys=",".join(b.split(":")[0] for b in bad))
        return res

    if COPY_CIF and not os.path.exists(json_file[:-5] + ".cif"):
        res.update(status="cif_missing")
        return res

    if GEO_REQUIRED and geo_status != "ok":
        res.update(status="geo_invalid", detail=geo_detail)
        return res

    if INC_REQUIRED and inc_status != "ok":
        res.update(status="inc_invalid", detail=inc_detail)
        return res

    res["status"] = "complete"
    return res


def copy_one(src_json):
    rel = os.path.relpath(src_json, JSON_DIR)
    srcs = [src_json]
    if COPY_CIF:
        srcs.append(src_json[:-5] + ".cif")
    for src in srcs:
        dst = os.path.join(DST_DIR, os.path.relpath(src, JSON_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return rel


def geo_summary(df):
    rows = []
    for k in GEO_KEYS:
        v = df[k]
        valid = v.dropna()
        n_bad = df["geo_bad_keys"].fillna("").str.split(",").apply(lambda ks: k in ks).sum()
        rows.append({
            "key": k,
            "n": len(v),
            "n_missing": int(v.isna().sum()),
            "n_flagged": int(n_bad),
            "n_zero": int((valid == 0).sum()),
            "min": valid.min() if len(valid) else np.nan,
            "p01": valid.quantile(0.01) if len(valid) else np.nan,
            "median": valid.median() if len(valid) else np.nan,
            "p99": valid.quantile(0.99) if len(valid) else np.nan,
            "max": valid.max() if len(valid) else np.nan,
        })
    n_pld_gt_lcd = df["geo_detail"].fillna("").str.contains("pld_gt_lcd").sum()
    return pd.DataFrame(rows), int(n_pld_gt_lcd)


def inc_summary(df):
    cnt = Counter(k for s in df["inc_bad_intervals"].fillna("") for k in s.split(",") if k)
    out = pd.DataFrame(sorted(cnt.items(), key=lambda x: -x[1]), columns=["interval", "n_files"])
    return out


def main():
    out_dir = REPORT_DIR if CHECK_ONLY else DST_DIR
    if not CHECK_ONLY and os.path.isdir(DST_DIR) and any(f.endswith(".json") for _, _, fs in os.walk(DST_DIR) for f in fs):
        raise RuntimeError(f"{DST_DIR} is not empy.")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True))
    print(f"{len(files)} json files found")

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(check_file, files, chunksize=64),
                            total=len(files), desc="check"))
        df = pd.DataFrame(results).sort_values("file").reset_index(drop=True)

        complete = df.loc[df.status == "complete", "file"].tolist()
        if not CHECK_ONLY:
            list(tqdm(pool.imap_unordered(copy_one, complete, chunksize=64), total=len(complete), desc="copy"))

    df.to_csv(os.path.join(out_dir, "_status.csv"), index=False)

    key_count = Counter(k for s in df.loc[df.status == "invalid_value", "bad_keys"]
                        for k in s.split(",") if k)
    key_df = pd.DataFrame(sorted(key_count.items(), key=lambda x: -x[1]),
                          columns=["key", "n_files"])
    key_df.to_csv(os.path.join(out_dir, "_invalid_keys.csv"), index=False)

    base = df[df.status.isin(["complete", "geo_invalid", "inc_invalid"])]

    geo_df, n_pld_gt_lcd = geo_summary(base)
    geo_df.to_csv(os.path.join(out_dir, "_geo_summary.csv"), index=False)
    base[["file", "geo_status", "geo_detail"] + GEO_KEYS].to_csv(
        os.path.join(out_dir, "_geo_values.csv"), index=False)

    inc_df = inc_summary(base)
    inc_df.to_csv(os.path.join(out_dir, "_inc_summary.csv"), index=False)
    base.loc[base.inc_status != "ok", ["file", "inc_detail"]].to_csv(
        os.path.join(out_dir, "_inc_problems.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print("\n=== File status ===")
    print(df["status"].value_counts())
    print("\n=== Files with invalid_value, by key ===")
    print(key_df.to_string(index=False) if len(key_df) else "(none)")

    print(f"\n=== Pore descriptor status ({len(base)} files with complete adsorption data) ===")
    print(base["geo_status"].value_counts())
    print("\n=== Summary by pore descriptor ===")
    print(geo_df.to_string(index=False))
    print(f"\nFiles with PLD > LCD: {n_pld_gt_lcd}")
    problem_types = Counter(p.split("(")[0] for s in base["geo_detail"].fillna("") for p in s.split(";") if p)
    if problem_types:
        print("\n=== Pore descriptor issue types ===")
        for t, c in problem_types.most_common():
            print(f"  {t}: {c}")

    print(f"\n=== Loading increment status ({len(base)} files with complete adsorption data) ===")
    print(base["inc_status"].value_counts())
    print("\n=== Files with increment <= 0, by pressure interval ===")
    print(inc_df.to_string(index=False) if len(inc_df) else "(none)")

    both = ((base.geo_status != "ok") & (base.inc_status != "ok")).sum()
    print(f"\nFiles with both pore descriptor and increment issues: {both}")

    if CHECK_ONLY:
        print(f"\nCHECK_ONLY -> {out_dir}")
    else:
        print(f"\nSAVE: {len(complete)} / {len(files)} -> {DST_DIR}")
    print(f"CSV: {os.path.join(out_dir, '_status.csv')}")


if __name__ == "__main__":
    main()
