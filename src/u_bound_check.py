#!/usr/bin/env python
"""Share of test MOFs whose mean-field slope u'(theta) reaches the bound u'_max = 0.9 * 4RT.

Reads <root>/heat_<gas>_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin/langmuir_analysis_porep_<seed>/per_mof.csv
for every gas and seed (written by analyze.py --steps langmuir).
u'(theta) = a1 + 2 a2 theta is linear, so its largest |value| on [0, 1] is at an end point:
    u'(0) = a1                      (column slope_0, or a1)
    u'(1) = a1 + 2 a2               (column slope_1, or 2 u(1) - u'(0) with u(1) = a1 + a2 from g_full)
A MOF is "near the bound" when max(|u'(0)|, |u'(1)|) >= frac * u'_max.

Outputs (in --out_dir):
    u_bound_per_seed.csv   one row per gas, seed and threshold
    u_bound_summary.csv    mean and standard deviation over seeds
    u_bound_lcd.csv        share near the bound per LCD range (if an LCD column exists)
    u_bound_hist.png       histograms of u'(0) and u'(1) with the bound marked
and prints the medians (to cross-check Table S5), the LaTeX row for Table S5 and the numbers for the ESI sentence.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 8.314462618e-3  # kJ mol-1 K-1

p = argparse.ArgumentParser()
p.add_argument('--root', default='../results_hmof/ViSNet')
p.add_argument('--gases', nargs='+', default=['co2', 'ch4'])
p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
p.add_argument('--run', default='heat_{gas}_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
p.add_argument('--sub', default='langmuir_analysis_porep_{seed}')
p.add_argument('--T', type=float, default=298.0)
p.add_argument('--bound_factor', type=float, default=0.9, help='u_max = factor * 4RT')
p.add_argument('--fracs', nargs='+', type=float, default=[0.95, 0.99])
p.add_argument('--lcd_edges', nargs='+', type=float, default=[0, 5, 6, 8, 10, 15, 100])
p.add_argument('--out_dir', default='../results_hmof/figures/u_bound')
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

U_MAX = args.bound_factor * 4 * R * args.T
print(f"u'_max = {args.bound_factor} x 4RT = {U_MAX:.3f} kJ/mol at {args.T:.0f} K")


def pick(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def slopes(df, path):
    """Return u'(0), u'(1) and the LCD column (or None) of one per_mof.csv."""
    c0 = pick(df, ['slope_0', 'u_prime_0', 'uprime_0', 'a1'])
    c1 = pick(df, ['slope_1', 'u_prime_1', 'uprime_1'])
    cg = pick(df, ['g_full', 'g_full(theta=1)', 'u_1', 'u_full', 'u1'])
    ca2 = pick(df, ['a2'])
    if c0 is None:
        raise SystemExit(f"[error] no u'(0) column (slope_0 / a1) in {path}\ncolumns: {list(df.columns)}")
    s0 = df[c0].astype(float).to_numpy()
    if c1 is not None:
        s1, how = df[c1].astype(float).to_numpy(), c1
    elif cg is not None:
        s1, how = 2 * df[cg].astype(float).to_numpy() - s0, f'2*{cg} - {c0}'
    elif ca2 is not None:
        s1, how = s0 + 2 * df[ca2].astype(float).to_numpy(), f'{c0} + 2*{ca2}'
    else:
        raise SystemExit(f"[error] no column for u'(1) (slope_1, g_full or a2) in {path}\ncolumns: {list(df.columns)}")
    lcd = pick(df, ['lcd', 'LCD', 'largest_cavity_diameter'])
    u1 = df[cg].astype(float).to_numpy() if cg is not None else (s0 + s1) / 2
    return s0, s1, u1, how, lcd


rows, lcd_rows, hist = [], [], {}
for gas in args.gases:
    for seed in args.seeds:
        path = os.path.join(args.root, args.run.format(gas=gas), args.sub.format(seed=seed), 'per_mof.csv')
        if not os.path.exists(path):
            print(f'[skip] missing {path}')
            continue
        df = pd.read_csv(path)
        s0, s1, u1, how, lcd_col = slopes(df, path)
        ok = np.isfinite(s0) & np.isfinite(s1)
        s0, s1, u1 = s0[ok], s1[ok], u1[ok]
        smax = np.maximum(np.abs(s0), np.abs(s1))
        if np.nanmax(smax) > U_MAX * 1.001:
            print(f'[warn] {gas} seed {seed}: max |u\'| = {np.nanmax(smax):.3f} exceeds u\'_max; check units or --T')
        print(f'[read] {gas} seed {seed}: {ok.sum()} MOFs, u\'(1) from {how}'
              f'{", LCD from " + lcd_col if lcd_col else ""}')
        hist.setdefault(gas, []).append((s0, s1))
        for f in args.fracs:
            thr = f * U_MAX
            rows.append(dict(gas=gas, seed=seed, frac=f, threshold=thr, n=int(ok.sum()),
                             near_any=100 * np.mean(smax >= thr),
                             near_u0=100 * np.mean(np.abs(s0) >= thr),
                             near_u1=100 * np.mean(np.abs(s1) >= thr),
                             near_positive=100 * np.mean(np.maximum(s0, s1) >= thr),
                             near_negative=100 * np.mean(np.minimum(s0, s1) <= -thr),
                             median_u0=np.median(s0), median_u1=np.median(u1),
                             median_uprime1=np.median(s1), max_abs=np.max(smax)))
        if lcd_col:
            lcd = df[lcd_col].astype(float).to_numpy()[ok]
            cut = pd.cut(lcd, args.lcd_edges, right=False)
            thr = min(args.fracs) * U_MAX
            for b, idx in pd.Series(np.arange(len(lcd))).groupby(cut, observed=True):
                ii = idx.to_numpy()
                lcd_rows.append(dict(gas=gas, seed=seed, lcd_lo=b.left, lcd_bin=str(b), n=len(ii),
                                     near_any=100 * np.mean(smax[ii] >= thr)))

if not rows:
    raise SystemExit('[error] no per_mof.csv found; check --root, --run and --sub')

per = pd.DataFrame(rows)
per.to_csv(os.path.join(args.out_dir, 'u_bound_per_seed.csv'), index=False)
cols = ['near_any', 'near_u0', 'near_u1', 'near_positive', 'near_negative',
        'median_u0', 'median_u1', 'median_uprime1', 'max_abs']
summ = per.groupby(['gas', 'frac'])[cols].agg(['mean', 'std'])
summ.columns = [f'{a}_{b}' for a, b in summ.columns]
summ = summ.reset_index()
summ.to_csv(os.path.join(args.out_dir, 'u_bound_summary.csv'), index=False)

if lcd_rows:
    lc = pd.DataFrame(lcd_rows).groupby(['gas', 'lcd_lo', 'lcd_bin'], observed=True).agg(
        n=('n', 'mean'), near_mean=('near_any', 'mean'), near_std=('near_any', 'std')).reset_index()
    lc = lc.sort_values(['gas', 'lcd_lo']).drop(columns='lcd_lo')
    lc.to_csv(os.path.join(args.out_dir, 'u_bound_lcd.csv'), index=False)

# ---------------------------------------------------------------- figure
gases = list(hist)
fig, axes = plt.subplots(1, len(gases), figsize=(4.2 * len(gases), 3.2), squeeze=False)
for ax, gas in zip(axes[0], gases):
    s0 = np.concatenate([h[0] for h in hist[gas]])
    s1 = np.concatenate([h[1] for h in hist[gas]])
    bins = np.linspace(-U_MAX * 1.02, U_MAX * 1.02, 61)
    ax.hist(s0, bins=bins, alpha=0.6, label="u'(0)", color='#2a6fb0')
    ax.hist(s1, bins=bins, alpha=0.6, label="u'(1)", color='#d0682c')
    for x in (-U_MAX, U_MAX):
        ax.axvline(x, color='k', lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel("u' (kJ mol$^{-1}$)")
    ax.set_ylabel('MOFs (all seeds)')
    ax.set_title(gas.upper().replace('2', '$_2$').replace('4', '$_4$'))
    ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, 'u_bound_hist.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- report
pd.set_option('display.width', 200)
print('\n=== share of MOFs near the bound (%), mean +- std over seeds ===')
show = summ[['gas', 'frac', 'near_any_mean', 'near_any_std', 'near_u0_mean', 'near_u1_mean',
             'near_positive_mean', 'near_negative_mean']].round(2)
print(show.to_string(index=False))
print("\n=== medians (cross-check with Table S5: CO2 u'(0) 5.11, u(1) 3.96; CH4 u'(0) 2.58, u(1) 2.30) ===")
print(summ[summ.frac == min(args.fracs)][['gas', 'median_u0_mean', 'median_u1_mean',
                                         'median_uprime1_mean', 'max_abs_mean']].round(3).to_string(index=False))
if lcd_rows:
    print(f'\n=== share near the bound (%) by LCD (A), frac {min(args.fracs)} ===')
    print(lc.round(2).to_string(index=False))

f0 = min(args.fracs)
cell = {}
for gas in args.gases:
    r = summ[(summ.gas == gas) & (summ.frac == f0)]
    if len(r):
        m, s = r.near_any_mean.iloc[0], r.near_any_std.iloc[0]
        cell[gas] = f'{m:.1f} $\\pm$ {0 if np.isnan(s) else s:.1f}'
print('\n=== LaTeX row for Table S5 (add under the mean-field rows) ===')
print(f"MOFs with $\\max(|u'(0)|, |u'(1)|) \\geq {f0:g}\\,u'_\\mathrm{{max}}$ (\\%) & "
      + ' & '.join(cell.get(g, '--') for g in ['co2', 'ch4']) + ' \\\\')
print('\n[saved]', ', '.join(['u_bound_per_seed.csv', 'u_bound_summary.csv']
                         + (['u_bound_lcd.csv'] if lcd_rows else []) + ['u_bound_hist.png']), '->', args.out_dir)
