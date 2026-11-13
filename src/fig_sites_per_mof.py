"""
ESI GDPool section, Figure S1: number of sites per MOF against the number of atoms (test MOFs, ViSNet). One run gives

  fig_sites_per_mof.pdf/.png   (a) CO2, (b) CH4: hexagonal-bin density of the number of framework sites M (pore token
                               not counted) against the number of atoms N in the unit cell for the map seed, with the
                               median M and the median number of active sites in equal-count bins of N; one colour
                               range for both panels
  table_sites_per_mof.csv      per gas and seed, then mean and std over seeds: MOFs, mean / median / IQR of N and M,
                               mean active sites, median atoms per site N/M, share of MOFs with a single site
  table_unassigned_sites_<gas>.csv  per seed, then mean and std over seeds: share of framework sites that own no atom
                               (argmax of the assignment, type 'empty' in site_analysis.py), share of MOFs with such a
                               site, their mean share A at every pressure, and the median e, n, dS of unassigned sites
                               and of assigned active sites
  sites_per_mof_all.csv        one row per gas, seed and MOF: n_atoms, n_sites, n_active, n_unassigned
  + printed: the numbers quoted in the main-text GDPool paragraph and in the caption of ESI Figure S1

Reads <root>/heat_<gas>_<run_tag>/site_analysis_<seed>/{mof_types.csv, sites.csv} written by site_analysis.py.

    python fig_sites_per_mof.py
    python fig_sites_per_mof.py --root ../results_hmof/ViSNet --seeds 42 43 44 --map_seed 42
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ap = argparse.ArgumentParser(description='sites per MOF against atoms per MOF (ESI Figure S1)')
ap.add_argument('--root', default='../results_hmof/ViSNet')
ap.add_argument('--run_tag', default='ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin', help='run folder = heat_<gas>_<run_tag>')
ap.add_argument('--folder', default='site_analysis_{seed}', help='site_analysis folder inside the run folder')
ap.add_argument('--gases', nargs='+', default=['co2', 'ch4'])
ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
ap.add_argument('--map_seed', default='42', help='seed drawn in the figure')
ap.add_argument('--bins', default=20, type=int, help='equal-count bins of N for the median lines')
ap.add_argument('--gridsize', default=40, type=int, help='hexagons across the x axis')
ap.add_argument('--out_dir', default='../results_hmof/figures')
args = ap.parse_args()

LABEL = {'co2': r'CO$_2$', 'ch4': r'CH$_4$'}
INK, MUTED, ACCENT = '#0b0b0b', '#898781', '#c4581a'


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4,
                         'ytick.minor.width': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def path(gas, seed, f):
    return os.path.join(args.root, f'heat_{gas}_{args.run_tag}', args.folder.format(seed=seed), f)


def load(gas, seed):
    f = path(gas, seed, 'mof_types.csv')
    if not os.path.isfile(f):
        raise SystemExit(f'{f} not found (run site_analysis.py for {gas}, seed {seed})')
    mt = pd.read_csv(f)
    miss = [c for c in ('name', 'n_atoms', 'n_sites', 'n_active') if c not in mt.columns]
    if miss:
        raise SystemExit(f'{f} lacks {miss}')
    key = 'idx' if 'idx' in mt.columns else 'name'
    sf = path(gas, seed, 'sites.csv')
    st = None
    if os.path.isfile(sf):
        st = pd.read_csv(sf)
        fw = st[st['site'] >= 0]
        mt['n_unassigned'] = mt[key].map(fw[fw['type'] == 'empty'].groupby(key).size()).fillna(0).astype(int)
        mt['n_unknown'] = mt[key].map(fw[fw['type'] == 'unknown'].groupby(key).size()).fillna(0).astype(int)
    else:
        print(f'[warn] {sf} not found: unassigned-site statistics skipped for {gas}, seed {seed}')
    return mt, st


def q(s, p):
    return float(np.nanpercentile(s, p)) if len(s) else np.nan


def size_row(gas, seed, mt):
    N, M = mt['n_atoms'].astype(float), mt['n_sites'].astype(float)
    return {'gas': gas, 'seed': seed, 'mofs': len(mt),
            'N_mean': N.mean(), 'N_median': N.median(), 'M_mean': M.mean(), 'M_median': M.median(),
            'M_q25': q(M, 25), 'M_q75': q(M, 75), 'M_min': M.min(), 'M_max': M.max(),
            'active_mean': mt['n_active'].mean(), 'active_median': mt['n_active'].median(),
            'active_share_of_sites': mt['n_active'].sum() / M.sum(),
            'atoms_per_site_median': float(np.median(N / M.clip(lower=1))),
            'share_single_site': float((M == 1).mean())}


def unassigned_row(gas, seed, mt, st):
    if st is None or 'n_unassigned' not in mt.columns:
        return None
    fw = st[st['site'] >= 0]
    un, asg = fw[fw['type'] == 'empty'], fw[(fw['type'] != 'empty') & fw['active']]
    r = {'gas': gas, 'seed': seed, 'sites': len(fw), 'unassigned_sites': len(un),
         'share_unassigned_sites': len(un) / max(len(fw), 1),
         'share_mofs_with_unassigned': float((mt['n_unassigned'] > 0).mean()),
         'share_unassigned_active': float(un['active'].mean()) if len(un) else np.nan,
         'unknown_sites': int(mt['n_unknown'].sum())}
    for c in [c for c in mt.columns if c.startswith('A_empty@')]:
        p = c.split('@')[1]
        a = mt[c] + (mt[f'A_unknown@{p}'] if f'A_unknown@{p}' in mt.columns else 0)
        r[f'A_unassigned_mean@{p}'] = float(a.mean())
    for col in ('e', 'n', 'dS'):
        if col in fw.columns:
            r[f'{col}_median_unassigned'] = float(un[col].median()) if len(un) else np.nan
            r[f'{col}_median_assigned_active'] = float(asg[col].median()) if len(asg) else np.nan
    return r


def with_seed_stats(df):
    out = []
    for gas, g in df.groupby('gas', sort=False):
        out.append(g)
        num = g.drop(columns=['gas', 'seed']).astype(float)
        out.append(pd.DataFrame([{'gas': gas, 'seed': 'mean', **num.mean().to_dict()},
                                 {'gas': gas, 'seed': 'std', **num.std(ddof=1).to_dict()}]))
    return pd.concat(out, ignore_index=True)


def tidy(g):
    """drop the pressures of the other gas and put the A_unassigned columns in pressure order"""
    g = g.dropna(axis=1, how='all')
    ac = sorted([c for c in g.columns if c.startswith('A_unassigned_mean@')], key=lambda c: float(c.split('@')[1]))
    rest = [c for c in g.columns if c not in ac]
    i = rest.index('unknown_sites') + 1 if 'unknown_sites' in rest else len(rest)
    return g[rest[:i] + ac + rest[i:]]


def median_line(N, y, bins):
    edges = np.unique(np.quantile(N, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, N, side='right') - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        k = idx == b
        if k.sum() >= 5:
            xs.append(np.median(N[k])); ys.append(np.median(y[k]))
    return np.array(xs), np.array(ys)


def figure(maps):
    cmap = LinearSegmentedColormap.from_list('blues_t', plt.get_cmap('Blues')(np.linspace(0.2, 1.0, 256)))
    fig, axes = plt.subplots(1, len(maps), figsize=(7.0, 2.7), sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)
    N_all = np.concatenate([m['n_atoms'].values for _, m in maps]).astype(float)
    M_all = np.concatenate([m['n_sites'].values for _, m in maps]).astype(float)
    ext = (np.log10(N_all.min() * 0.9), np.log10(N_all.max() * 1.1),
           np.log10(max(M_all.min(), 1) * 0.8), np.log10(M_all.max() * 1.25))
    hbs = []
    for ax, (gas, mt), tag in zip(axes, maps, 'abcdefgh'):
        N, M, A = (mt[c].values.astype(float) for c in ('n_atoms', 'n_sites', 'n_active'))
        hb = ax.hexbin(N, np.maximum(M, 1), xscale='log', yscale='log', gridsize=args.gridsize, extent=ext,
                       mincnt=1, cmap=cmap, linewidths=0.0, norm=LogNorm())
        hbs.append(hb)
        x, y = median_line(N, M, args.bins)
        ax.plot(x, y, '-o', color=INK, lw=1.0, ms=2.2, label='Median sites $M$')
        x, y = median_line(N, A, args.bins)
        ax.plot(x, y, '--', color=ACCENT, lw=1.0, label='Median active sites')
        ax.set_xlabel('Atoms in the unit cell, $N$')
        ax.text(0.03, 0.97, f'{tag}', transform=ax.transAxes, ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(0.09, 0.97, LABEL.get(gas, gas), transform=ax.transAxes, ha='left', va='top', fontsize=8)
        ax.legend(loc='lower right', handlelength=1.8)
    axes[0].set_ylabel('Sites per MOF, $M$')
    vmax = max(h.get_array().max() for h in hbs)
    for h in hbs:
        h.set_norm(LogNorm(1, vmax))
    cb = fig.colorbar(hbs[-1], ax=axes, shrink=0.85, pad=0.01)
    cb.set_label('MOFs per bin', fontsize=7)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    for ext_, kw in (('pdf', {}), ('png', {'dpi': 600})):
        fig.savefig(os.path.join(args.out_dir, f'fig_sites_per_mof.{ext_}'), **kw)
    plt.close(fig)


def main():
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    size, unas, rows, maps = [], [], [], []
    for gas in args.gases:
        for seed in args.seeds:
            mt, st = load(gas, seed)
            size.append(size_row(gas, seed, mt))
            r = unassigned_row(gas, seed, mt, st)
            if r is not None:
                unas.append(r)
            cols = ['name', 'n_atoms', 'n_sites', 'n_active'] + (['n_unassigned'] if 'n_unassigned' in mt else [])
            rows.append(mt[cols].assign(gas=gas, seed=seed))
            if seed == args.map_seed:
                maps.append((gas, mt))
            print(f'[read] {gas} seed {seed}: {len(mt)} MOFs')
    if not maps:
        raise SystemExit(f'--map_seed {args.map_seed} not among --seeds {args.seeds}')

    size = with_seed_stats(pd.DataFrame(size))
    size.to_csv(os.path.join(args.out_dir, 'table_sites_per_mof.csv'), index=False)
    if unas:
        unas = with_seed_stats(pd.DataFrame(unas))
        for gas, g in unas.groupby('gas', sort=False):
            tidy(g).to_csv(os.path.join(args.out_dir, f'table_unassigned_sites_{gas}.csv'), index=False)
    pd.concat(rows, ignore_index=True)[['gas', 'seed', 'name', 'n_atoms', 'n_sites', 'n_active'] +
                                       (['n_unassigned'] if 'n_unassigned' in rows[0] else [])] \
        .to_csv(os.path.join(args.out_dir, 'sites_per_mof_all.csv'), index=False)
    figure(maps)

    pd.set_option('display.width', 220)
    show = ['gas', 'seed', 'mofs', 'N_median', 'M_mean', 'M_median', 'M_q25', 'M_q75', 'active_mean',
            'atoms_per_site_median', 'share_single_site']
    print('\n=== sites per MOF (framework sites; pore token not counted) ===')
    print(size[show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    if len(unas):
        ucols = ['gas', 'seed', 'share_unassigned_sites', 'share_mofs_with_unassigned', 'share_unassigned_active'] + \
                [c for c in unas.columns if c.startswith('A_unassigned_mean@')] + \
                [c for c in unas.columns if c.endswith('_median_unassigned') or c.endswith('_median_assigned_active')]
        ucols = [c for c in ucols if c in unas.columns]
        print('\n=== sites that own no atom (argmax), type "empty" ===')
        for gas, g in unas[ucols].groupby('gas', sort=False):
            print(tidy(g).to_string(index=False, float_format=lambda v: f'{v:.3f}') + '\n')

    m = size[size['seed'] == 'mean'].set_index('gas')
    s = size[size['seed'] == 'std'].set_index('gas')
    txt = {g: f'{m.loc[g, "M_mean"]:.1f} $\\pm$ {s.loc[g, "M_mean"]:.1f}' for g in m.index}
    print('\n--- main-text GDPool paragraph ([ToDo]; mean over MOFs, mean +- std over seeds) ---')
    print('테스트 세트에서 MOF당 사이트 수 $M$은 평균 '
          + ', '.join(f'{txt[g]}개({LABEL.get(g, g)})' for g in m.index) + '였다.')
    print('\n--- Figure S1 caption ([ToDo]) ---')
    print('; '.join(f'{LABEL.get(g, g)}: mean $M$ = {m.loc[g, "M_mean"]:.1f}, median $N/M$ = '
                    f'{m.loc[g, "atoms_per_site_median"]:.1f}' for g in m.index)
          + f' (means over seeds {", ".join(args.seeds)}; panels show seed {args.map_seed})')
    print('\n[figure] fig_sites_per_mof.pdf/.png; [table] table_sites_per_mof.csv'
          + (''.join(f', table_unassigned_sites_{g}.csv' for g in args.gases) if len(unas) else '') + f', sites_per_mof_all.csv -> {args.out_dir}')


if __name__ == '__main__':
    main()
