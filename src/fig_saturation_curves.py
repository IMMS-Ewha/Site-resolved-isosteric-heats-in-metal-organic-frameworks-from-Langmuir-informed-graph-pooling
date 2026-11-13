"""
Mean Q_st(p) curves grouped by how far the MOF is filled at p_max, CO2 (a) and CH4 (b) in one figure.
GCMC = solid line + circles, model = dashed line + triangles (same colour per group).

theta_fit@p_max = fractional filling at the highest pressure from a single-site Langmuir fit of the GCMC loading
(written by langmuir_analysis to per_mof.csv). The groups and the GCMC curve come from the data and are the same
for every seed; the model curve = mean over seeds, band = +- std over seeds. Rows of the per_mof.csv files are
matched by order (same test set, shuffle=False); the GCMC columns are checked to be identical across seeds.

    python fig_saturation_curves.py                               # defaults below
    python fig_saturation_curves.py --seeds 42 43 44 --out_dir ./figures

Outputs (out_dir): fig_qst_by_saturation.pdf/.png, qst_by_saturation.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SAT_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0001]
SAT_LABELS = [r'$\theta_\mathrm{fit}<0.3$', '0.3–0.5', '0.5–0.7', '0.7–0.85', r'$\geq 0.85$']
PLAIN = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.85', '>=0.85']
COLORS = [matplotlib.colormaps['viridis'](v) for v in np.linspace(0.0, 0.9, 5)]
INK, MUTED = '#0b0b0b', '#898781'
GASES = [('co2', r'CO$_2$'), ('ch4', r'CH$_4$')]

ap = argparse.ArgumentParser(description='Q_st(p) by saturation group, GCMC vs model, CO2 and CH4')
ap.add_argument('--co2_run', default='../results_hmof/ViSNet/heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--ch4_run', default='../results_hmof/ViSNet/heat_ch4_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--lang_dir', default='langmuir_analysis_porep', help='langmuir analysis folder prefix inside each run')
ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
ap.add_argument('--out_dir', default='../results_hmof/figures', type=str)
ap.add_argument('--name', default='fig_qst_by_saturation', type=str)
ap.add_argument('--n_labels', default=True, type=lambda s: str(s).lower() in ('1', 'true', 'yes'),
                help='write the group size n next to the right end of each curve')
args = ap.parse_args()


def load(run):
    files = [os.path.join(run, f'{args.lang_dir}_{s}', 'per_mof.csv') for s in args.seeds]
    miss = [f for f in files if not os.path.isfile(f)]
    if miss:
        raise SystemExit(f'per_mof.csv not found: {miss}')
    runs = [pd.read_csv(f) for f in files]
    pl = sorted([c.split('@')[1] for c in runs[0].columns if c.startswith('heat_true@')], key=float)
    tcol = f'theta_fit@{pl[-1]}'
    if tcol not in runs[0].columns:
        raise SystemExit(f'{tcol} not in {files[0]} (langmuir_analysis needs the GCMC loading)')
    if any(len(r) != len(runs[0]) for r in runs):
        raise SystemExit(f'per_mof.csv files of {run} have different numbers of MOFs')
    tc = [f'heat_true@{x}' for x in pl]
    for f, r in zip(files[1:], runs[1:]):
        if not np.allclose(r[tc].values, runs[0][tc].values, equal_nan=True):
            raise SystemExit(f'GCMC Q_st differs between {files[0]} and {f}: rows are not aligned')
    return runs, pl, tcol, files


def table(gas, runs, pl, tcol):
    bins = pd.cut(runs[0][tcol], bins=SAT_BINS, labels=False, right=False).values
    true = runs[0][[f'heat_true@{x}' for x in pl]].values
    pred = np.stack([r[[f'heat_pred@{x}' for x in pl]].values for r in runs])
    rows = []
    for g, mk in [('all', np.ones(len(bins), bool))] + [(PLAIN[b], bins == b) for b in range(len(PLAIN))]:
        if not mk.any():
            continue
        seed_mean = pred[:, mk].mean(1)
        for j, x in enumerate(pl):
            rows.append({'gas': gas, 'group': g, 'n_mof': int(mk.sum()), 'pressure': x,
                         'gcmc_mean': true[mk, j].mean(), 'model_mean': seed_mean[:, j].mean(),
                         'model_std_seeds': seed_mean[:, j].std(ddof=1) if len(runs) > 1 else np.nan,
                         'mae': np.abs(pred[:, mk, j] - true[mk, j]).mean()})
    return pd.DataFrame(rows)


def n_labels(ax, p_end, items):
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = 0.075 * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    ys = [items[0][0]]
    for y, _ in items[1:]:
        ys.append(max(y, ys[-1] + gap))
    over = ys[-1] - (hi - 0.02 * (hi - lo))
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, n) in zip(ys, items):
        ax.text(p_end * 1.18, y, f'n = {n:,}', va='center', ha='left', fontsize=6, color=INK)


def panel(ax, tab, pl, lab, gas_tex):
    p = np.array([float(x) for x in pl])
    ends = []
    for b in range(len(PLAIN)):
        t = tab[tab['group'] == PLAIN[b]]
        if t.empty:
            continue
        c, z = COLORS[b], 2 * b + 2
        ax.plot(p, t['gcmc_mean'], '-o', color=c, lw=1.4, ms=4.5, zorder=z)
        ax.plot(p, t['model_mean'], '--^', color=c, lw=1.4, ms=4, zorder=z + 1)
        if t['model_std_seeds'].notna().all():
            ax.fill_between(p, t['model_mean'] - t['model_std_seeds'], t['model_mean'] + t['model_std_seeds'],
                            color=c, alpha=0.15, lw=0, zorder=z - 1)
        ends.append((0.5 * (t['gcmc_mean'].iloc[-1] + t['model_mean'].iloc[-1]), int(t['n_mof'].iloc[0])))
    ax.set_xscale('log')
    ax.set_xticks(p)
    ax.set_xticklabels(pl)
    ax.minorticks_off()
    ax.set_xlim(p[0] / 1.5, p[-1] * (4.0 if args.n_labels else 1.5))
    ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel(r'Mean $Q_\mathrm{st}$ (kJ mol$^{-1}$)')
    ax.set_title(rf'$\mathbf{{{lab}}}$   {gas_tex}', loc='left')
    if args.n_labels:
        ax.autoscale_view()
        n_labels(ax, p[-1], ends)


def main():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08)
    tabs, info = [], []
    for ax, lab, (gas, tex), run in zip(axes, 'ab', GASES, (args.co2_run, args.ch4_run)):
        runs, pl, tcol, files = load(run)
        tab = table(gas, runs, pl, tcol)
        panel(ax, tab, pl, lab, tex)
        tabs.append((tab, pl))
        info.append(f'{gas}: {len(files)} seed(s), {len(runs[0])} MOFs, groups by {tcol}')

    handles = [Line2D([], [], color=c, lw=1.4, label=l) for c, l in zip(COLORS, SAT_LABELS)]
    handles += [Line2D([], [], color=MUTED, lw=1.4, ls='-', marker='o', ms=4.5, label='GCMC'),
                Line2D([], [], color=MUTED, lw=1.4, ls='--', marker='^', ms=4, label='Model')]
    fig.legend(handles=handles, loc='outside upper center', ncol=7, handlelength=2.0, columnspacing=1.2)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 600})):
        fig.savefig(os.path.join(args.out_dir, f'{args.name}.{ext}'), **kw)
    plt.close(fig)

    out = pd.concat([t for t, _ in tabs], ignore_index=True)
    out.to_csv(os.path.join(args.out_dir, 'qst_by_saturation.csv'), index=False)

    pd.set_option('display.width', 200)
    print('[saturation curves] ' + ' | '.join(info) + f' -> {args.out_dir}')
    for tab, pl in tabs:
        order = [g for g in ['all'] + PLAIN if g in set(tab['group'])]
        n = tab.drop_duplicates('group').set_index('group')['n_mof']
        print(f"\n===== {tab['gas'].iloc[0].upper()} =====  n: " + ', '.join(f'{g} {n[g]}' for g in order))
        for v, title in (('gcmc_mean', 'GCMC mean Q_st'), ('model_mean', 'model mean Q_st'), ('mae', 'MAE')):
            print(f'\n--- {title} (kJ/mol) ---')
            print(tab.pivot(index='group', columns='pressure', values=v).loc[order, pl].round(2).to_string())


if __name__ == '__main__':
    main()
