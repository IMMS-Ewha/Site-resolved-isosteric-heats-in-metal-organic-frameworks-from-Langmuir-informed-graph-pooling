"""
Mean +- std over seeds of the summary tables written by site_analysis.py and langmuir_analysis (one folder per seed).

    python aggregate_seeds.py                                   # CO2 and CH4 runs below, seeds 42 43 44
    python aggregate_seeds.py --runs <run dir> ... --prefixes site_analysis langmuir_analysis_porep --seeds 42 43 44

For every run and prefix the seed folders are <run>/<prefix>_<seed>. Every csv present in all of them with at most
--max_rows rows (summary tables; per-MOF tables such as sites.csv or per_mof.csv are skipped) is aligned on its key
columns (type, pressure, interval, trend, pair, sat_bin, index column, ...) and written to <run>/<prefix>_seeds/<name>
with <column>_mean, <column>_std (n-1 over seeds) and <column>_pm ("mean ± std") for every numeric column.
GCMC-based columns (dQ_true, heat_true, ...) must come out with std 0: a non-zero value means the seed folders do not
cover the same test set (checked for columns named dQ_true*, heat_true*, load_true*, gcmc*; not for
decomposition_by_trend.csv, whose groups follow each seed's predicted trend).
The tables used in the paper are printed.
"""
import argparse
import os
import re
import warnings
from glob import glob

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description='mean +- std of analysis summaries over seeds')
ap.add_argument('--runs', nargs='+', default=['../results_hmof/ViSNet/heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin',
                                              '../results_hmof/ViSNet/heat_ch4_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin'])
ap.add_argument('--prefixes', nargs='+', default=['site_analysis', 'langmuir_analysis_porep'])
ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
ap.add_argument('--max_rows', default=200, type=int, help='csv files with more rows are per-MOF tables and skipped')
args = ap.parse_args()

KEY_NAMES = {'type', 'pressure', 'pair', 'interval', 'trend', 'sat_bin', 'group', 'metric', 'Unnamed: 0', 'index',
             'quantity', 'target', 'model', 'term', 'bin', 'gas'}
PRINT = {'site_analysis': ['decomposition_summary.csv', 'decomposition_by_trend.csv', 'type_heat_paired.csv',
                           'type_heat.csv', 'type_share.csv', 'thermo_summary.csv'],
         'langmuir_analysis_porep': ['summary.csv', 'decomposition.csv', 'g_by_sat_bin.csv']}
GCMC_COL = re.compile(r'^(dQ_true|heat_true|load_true|gcmc)')
NO_GCMC_CHECK = {'decomposition_by_trend.csv'}
warnings.filterwarnings('ignore', category=RuntimeWarning)


def keys_of(df):
    return [c for c in df.columns if c in KEY_NAMES or (c == df.columns[0] and df[c].dtype == object)]


def fmt(m, s):
    if not np.isfinite(m):
        return ''
    if not np.isfinite(s):
        return f'{m:.4g}'
    return f'{m:.4g} ± ' + (f'{s:.0f}' if abs(s) >= 10 else f'{s:.2g}')


def aggregate(dfs):
    keys = keys_of(dfs[0])
    vals = [c for c in dfs[0].columns if c not in keys]
    base = dfs[0].set_index(keys) if keys else dfs[0]
    idx = base.index
    al = [(d.set_index(keys) if keys else d).reindex(idx) for d in dfs]
    cols = {}
    for c in vals:
        raw = [a[c] if c in a else pd.Series(np.nan, index=idx) for a in al]
        num = np.stack([pd.to_numeric(r, errors='coerce').values.astype(float) for r in raw])
        if not np.isfinite(num).any():
            cols[c] = raw[0].values
            continue
        m = np.nanmean(num, 0)
        s = np.nanstd(num, 0, ddof=1) if len(dfs) > 1 else np.full(len(idx), np.nan)
        first = raw[0].astype(str).values
        cols[f'{c}_mean'], cols[f'{c}_std'] = m, s
        cols[f'{c}_pm'] = [fmt(a, b) if np.isfinite(a) else first[i] for i, (a, b) in enumerate(zip(m, s))]
    out = pd.DataFrame(cols, index=idx)
    return out.reset_index() if keys else out


def show(name, t):
    pm = [c for c in t.columns if c.endswith('_pm')]
    keys = [c for c in t.columns if not (c.endswith('_mean') or c.endswith('_std') or c.endswith('_pm'))]
    v = t[keys + pm].rename(columns={c: c[:-3] for c in pm})
    if name == 'type_share.csv' and {'type', 'pressure', 'A_mean'} <= set(v.columns):
        v = v.pivot(index='type', columns='pressure', values='A_mean')
    print(f'\n--- {name} (mean ± std over seeds) ---')
    print(v.to_string(index=name == 'type_share.csv'))


def main():
    done = 0
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 40)
    for run in args.runs:
        for prefix in args.prefixes:
            dirs = [os.path.join(run, f'{prefix}_{s}') for s in args.seeds]
            have = [d for d in dirs if os.path.isdir(d)]
            if len(have) < len(dirs):
                print(f'[skip] {run}: {prefix}_<seed> missing for '
                      f'{[s for s, d in zip(args.seeds, dirs) if d not in have]}')
                continue
            files = set.intersection(*[{os.path.basename(f) for f in glob(os.path.join(d, '*.csv'))} for d in dirs])
            out_dir = os.path.join(run, f'{prefix}_seeds')
            os.makedirs(out_dir, exist_ok=True)
            print(f'\n==================== {os.path.basename(run)} / {prefix} ({len(dirs)} seeds) -> {out_dir}')
            tables, skipped = {}, []
            for f in sorted(files):
                if f.startswith('atoms_'):
                    continue
                dfs = [pd.read_csv(os.path.join(d, f)) for d in dirs]
                if len(dfs[0]) > args.max_rows:
                    skipped.append(f)
                    continue
                t = aggregate(dfs)
                t.to_csv(os.path.join(out_dir, f), index=False)
                tables[f] = t
                gc = [] if f in NO_GCMC_CHECK else [c for c in t.columns
                                                     if c.endswith('_std') and GCMC_COL.match(c[:-4])]
                bad = [c for c in gc if (t[c].abs() > 1e-6).any()]
                if bad:
                    print(f'  WARNING {f}: GCMC columns differ between seeds {bad} (not the same test set?)')
            print(f'  aggregated: {", ".join(tables) or "none"}' + (f' | skipped (per MOF): {", ".join(skipped)}' if skipped else ''))
            for f in PRINT.get(prefix, [f for f in tables if len(tables[f]) <= 60]):
                if f in tables:
                    show(f, tables[f])
            done += len(tables)
    if not done:
        raise SystemExit('nothing aggregated: check --runs, --prefixes and --seeds')


if __name__ == '__main__':
    main()
