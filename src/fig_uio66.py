"""
Case study: functionalized UiO-66, CH4 - model v7 vs GCMC of this work (run_uio66_gcmc.py). One run gives

  fig_uio66_qst.pdf/.png      Q_st at p_high per framework: GCMC (open circles, 95% confidence interval) vs model
                              (diamonds, mean +- std over seeds); text: Spearman rho over frameworks, MAE, MAE of the
                              shift from pristine UiO-66
  fig_uio66_pressure.pdf/.png Q_st (row 1) and loading (row 2) vs pressure, one column per framework: GCMC (open
                              circles, 95% confidence interval) vs model (diamonds, mean +- std over seeds)
  fig_uio66_density.pdf/.png  same projection for both at p_high: row 1 GCMC CH4 loading (RASPA density grid scaled to
                              the GCMC absolute loading), row 2 model framework loading q_i of the map seed; both in
                              10^-3 mmol g^-1 A^-2, summed along z over rep x rep cells; one colour range per row
  fig_uio66_gcmc_slices.pdf/.png  GCMC CH4 density (molecules nm^-3) in slabs of +- slab A around z = 0, 1/4, 1/2 of
                              the cell at p_high (one cell, one colour range); white dots = framework heavy atoms in the slab
  fig_uio66_share.pdf/.png    where the framework-near loading sits at p_high, per site type of the model
                              (node / linker / functional): G = GCMC density shared by the framework atoms within
                              attr_cut in proportion to their LJ attraction to CH4 (force field of the GCMC run) and
                              summed over the site type of each atom; M = model q_i summed per site type;
                              text = Spearman rho over sites between the GCMC and model site loadings
  fig_uio66_qmap.pdf/.png     model: projected framework loading q_i at p_high (map seed); one colour range
  fig_uio66_emap.pdf/.png     model: same projection for the site heat e_i (Gaussian-weighted mean along z), for the ESI
  fig_uio66_atoms.pdf/.png    model: atoms projected along z, row 1 loading q_i at p_high, row 2 site heat e_i
  table_uio66.csv/.tex        main table at p_high: Q_st and loading (GCMC, model), shift from UiO-66, functional and
                              pore-token weights, helium void fraction; footer with Spearman rho and MAE
  table_uio66_si.csv/.tex     ESI table: model Q_st at every pressure (mean +- std over seeds) with the GCMC values
                              (csv also holds the loadings)
  table_uio66_sites_si.tex    ESI table: same-MOF functional - linker and node - linker heats and the weights
  table_uio66_share.csv       per framework and pressure with a density grid: site-type shares of the framework-near
                              loading (GCMC, model), Spearman rho over sites, GCMC share farther than attr_cut from
                              every atom, model pore-token share
  table_uio66_sites_cmp.csv   site loadings (GCMC, model; mmol g^-1) per framework, pressure and site
  gcmc_input.csv              the GCMC values used

Model: <run>/<site_prefix>_<seed>/{mof_types.csv, sites.csv} for every seed and atoms_<name>.csv, <name>.extxyz for
the map seed (site_analysis.py --pt_path ... --out_dir ... --viz ...).
GCMC: <gcmc_dir>/gcmc_ch4_results.csv, helium_void_fraction.csv and
<gcmc_dir>/ch4/<name>/<p>bar/VTK/System_0/DensityProfile_methane.vtk (run_uio66_gcmc.py --density_pressures all).
RASPA2 writes the grid relative to its largest voxel, so it is scaled here to the absolute GCMC loading of that run.
RASPA2 uncertainties are 95% confidence intervals from five blocks.

    python fig_uio66.py                                   # defaults below
    python fig_uio66.py --gcmc_dir ../results_hmof/gcmc_uio66 --map_seed 42 --out_dir ../results_hmof/figures
"""
import argparse
import os
import re
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.ticker import NullLocator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(description='functionalized UiO-66 case study (CH4): model vs GCMC')
ap.add_argument('--run', default='../results_hmof/ViSNet/heat_ch4_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--site_prefix', default='site_analysis_uio66')
ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
ap.add_argument('--map_seed', default='42', help='seed of the maps and of the site-level comparison')
ap.add_argument('--p_low', default='0.05', help='not used (kept for old commands)')
ap.add_argument('--p_high', default='4.5')
ap.add_argument('--gcmc_dir', default='../results_hmof/gcmc_uio66', help='out_dir of gcmc/run_uio66_gcmc.py')
ap.add_argument('--load', default='abs', choices=['abs', 'exc'], help='GCMC loading compared with the model')
ap.add_argument('--attr', default='lj', choices=['lj', 'nearest'],
                help='how GCMC density is given to framework atoms: shared in proportion to the LJ attraction '
                     'eps_ij sigma_ij^6 / r^6 of the atoms within attr_cut (lj), or all to the nearest atom')
ap.add_argument('--attr_cut', default=5.0, type=float, help='A, atoms farther than this get no GCMC density')
ap.add_argument('--slices', nargs='+', type=float, default=[0.0, 0.25, 0.5], help='fractional z of the GCMC slices')
ap.add_argument('--slab', default=1.0, type=float, help='A, half thickness of the GCMC slices')
ap.add_argument('--slice_sigma', default=0.3, type=float, help='A, Gaussian smoothing of the GCMC slices (0 = none)')
ap.add_argument('--rep', default=2, type=int, help='cells along x and y in the projected maps (orthogonal cell)')
ap.add_argument('--box', nargs=3, type=float, default=None, help='A, map window (x y z) for a non-orthogonal cell')
ap.add_argument('--grid', default=0.2, type=float, help='A, pixel size of the maps')
ap.add_argument('--sigma', default=0.8, type=float, help='A, Gaussian width of the maps')
ap.add_argument('--pct', default=99.5, type=float, help='percentile of the shared upper colour limit')
ap.add_argument('--cmap', default='magma')
ap.add_argument('--pct_atoms', nargs=2, type=float, default=[1, 99], help='percentiles of the point-map colour ranges')
ap.add_argument('--out_dir', default='../results_hmof/figures')
args = ap.parse_args()

# framework order (pristine first) and labels; names as in index.csv and in the GCMC csv (CIF file names)
FRAMEWORKS = [('DB12-RUBTAK02_corrected', 'UiO-66'), ('DB12-RUBTAK02-NH2', r'–NH$_2$'),
              ('DB12-RUBTAK02-Br', '–Br'), ('DB12-RUBTAK02-NO2', r'–NO$_2$'),
              ('DB12-RUBTAK02-DiMe', r'–(CH$_3$)$_2$')]
NAMES = [n for n, _ in FRAMEWORKS]
PLAIN = {'DB12-RUBTAK02_corrected': 'UiO-66', 'DB12-RUBTAK02-NH2': 'UiO-66-NH2', 'DB12-RUBTAK02-Br': 'UiO-66-Br',
         'DB12-RUBTAK02-NO2': 'UiO-66-NO2', 'DB12-RUBTAK02-DiMe': 'UiO-66-(CH3)2'}
TEX = {'DB12-RUBTAK02_corrected': 'UiO-66', 'DB12-RUBTAK02-NH2': r'UiO-66-NH$_2$', 'DB12-RUBTAK02-Br': 'UiO-66-Br',
       'DB12-RUBTAK02-NO2': r'UiO-66-NO$_2$', 'DB12-RUBTAK02-DiMe': r'UiO-66-(CH$_3$)$_2$'}
SITE_COLS = ['e_func_minus_linker', 'e_node_minus_linker', 'A_functional', 'A_pore']
TYPES3 = ['node', 'linker', 'functional']
TYPE_COLOR = {'node': '#2a78d6', 'linker': '#eb6834', 'functional': '#1baf7a', 'other': '#bdbbb5'}   # site_analysis.py
SLICE_NAME = {0.0: r'(Zr$_6$ cluster plane)', 0.25: '(tetrahedral cages)', 0.5: '(octahedral cage)'}

INK, MUTED, GCMC_C = '#0b0b0b', '#898781', '#D62728'
COV_R = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'V': 1.53, 'Cu': 1.32,
         'Zn': 1.22, 'Zr': 1.75}
METALS = {'V', 'Cu', 'Zn', 'Zr'}


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(s))


def folder(seed):
    d = os.path.join(args.run, f'{args.site_prefix}_{seed}')
    if not os.path.isdir(d):
        raise SystemExit(f'{d} not found (run site_analysis.py --pt_path ... --out_dir {d})')
    return d


def sd(a):
    return a.std(0, ddof=1) if len(a) > 1 else np.full(a.shape[1:], np.nan)


def savefig(fig, name, **kw):
    for ext, k in (('pdf', {}), ('png', {'dpi': 600})):
        fig.savefig(os.path.join(args.out_dir, f'{name}.{ext}'), **k, **kw)
    plt.close(fig)


def pdir(name, x):
    return os.path.join(args.gcmc_dir, 'ch4', name, f'{float(x):g}bar')


# ---------------------------------------------------------------- model values (all seeds)
def model_tables():
    pl, qst, load, site = None, [], [], []
    for s in args.seeds:
        d = folder(s)
        mt = pd.read_csv(os.path.join(d, 'mof_types.csv')).set_index('name')
        st = pd.read_csv(os.path.join(d, 'sites.csv'))
        miss = [n for n in NAMES if n not in mt.index]
        if miss:
            raise SystemExit(f'{miss} not in {d}/mof_types.csv (names: {list(mt.index)})')
        mt = mt.loc[NAMES]
        pl = sorted([c.split('@')[1] for c in mt.columns if c.startswith('Qst_pred@')], key=float)
        if args.p_high not in pl:
            raise SystemExit(f'--p_high {args.p_high} not among the model pressures {pl}')
        qcols = [f'q@{x}' for x in pl]
        if any(c not in st.columns for c in qcols):
            raise SystemExit(f'{d}/sites.csv lacks some of {qcols}')
        qst.append(mt[[f'Qst_pred@{x}' for x in pl]].values)
        load.append(st.groupby('name')[qcols].sum().reindex(NAMES).values)        # framework + pore, [MOF, P]
        site.append(np.stack([mt['e_functional'] - mt['e_linker'], mt['e_node'] - mt['e_linker'],
                              mt[f'A_functional@{args.p_high}'], mt[f'A_pore@{args.p_high}']], 1))
    qst, load, site = map(np.array, (qst, load, site))                          # [seed, MOF, ...]
    out = pd.DataFrame(index=NAMES)
    for j, x in enumerate(pl):
        out[f'model_qst@{x}'], out[f'model_qst@{x}_std'] = qst[:, :, j].mean(0), sd(qst[:, :, j])
        out[f'model_load@{x}'], out[f'model_load@{x}_std'] = load[:, :, j].mean(0), sd(load[:, :, j])
    for j, c in enumerate(SITE_COLS):
        v = pd.DataFrame(site[:, :, j].T)                                       # [MOF, seed]; NaN = no such site type
        out[c], out[f'{c}_std'] = v.mean(axis=1).values, v.std(axis=1, ddof=1).values
    out['groups'] = pd.read_csv(os.path.join(folder(args.map_seed), 'mof_types.csv')).set_index('name') \
        .loc[NAMES, 'groups'].fillna('').values
    return out, pl


# ---------------------------------------------------------------- GCMC values (run_uio66_gcmc.py)
def gcmc_tables():
    f = os.path.join(args.gcmc_dir, 'gcmc_ch4_results.csv')
    if not os.path.isfile(f):
        raise SystemExit(f'{f} not found (run run_uio66_gcmc.py ... --out_dir {args.gcmc_dir})')
    g = pd.read_csv(f).rename(columns={'pressure_bar': 'pressure'})
    miss = [n for n in NAMES if n not in set(g['name'])]
    if miss:
        raise SystemExit(f'{miss} not in {f} (names there: {sorted(set(g["name"]))})')
    g = g[g['name'].isin(NAMES)].copy()
    keep = [c for c in ['name', 'pressure', 'qst', 'qst_err', 'load_abs', 'load_abs_err', 'load_exc', 'load_exc_err',
                        'n_abs_uc', 'n_exc_uc'] if c in g.columns]
    g[keep].to_csv(os.path.join(args.out_dir, 'gcmc_input.csv'), index=False)
    hv = os.path.join(args.gcmc_dir, 'helium_void_fraction.csv')
    void = pd.read_csv(hv).set_index('name') if os.path.isfile(hv) else pd.DataFrame()
    settings = {c: g[c].iloc[0] for c in ['cutoff', 'ch4_sigma', 'tail_corrections', 'unit_cells', 'init_cycles',
                                          'cycles'] if c in g.columns}
    return g, void, settings


def combine(t, g, void, pl):
    for x in pl:
        gx = g[np.isclose(g['pressure'], float(x))].set_index('name')
        t[f'gcmc_qst@{x}'] = gx['qst'].reindex(t.index)
        t[f'gcmc_qst@{x}_err'] = gx['qst_err'].reindex(t.index)
        t[f'gcmc_load@{x}'] = gx[f'load_{args.load}'].reindex(t.index)
        t[f'gcmc_load@{x}_err'] = gx[f'load_{args.load}_err'].reindex(t.index) \
            if f'load_{args.load}_err' in gx.columns else np.nan
    t['void_fraction'] = void['helium_void_fraction'].reindex(t.index).values \
        if 'helium_void_fraction' in void.columns else np.nan
    t['void_fraction_err95'] = void['err'].reindex(t.index).values if 'err' in void.columns else np.nan
    ph, ref = args.p_high, NAMES[0]
    for who in ('gcmc', 'model'):
        t[f'{who}_shift@{ph}'] = t[f'{who}_qst@{ph}'] - t.loc[ref, f'{who}_qst@{ph}']
    return t


def metrics(t):
    ph = args.p_high
    g, m = t[f'gcmc_qst@{ph}'], t[f'model_qst@{ph}']
    ok = g.notna() & m.notna()
    rest = ok & (t.index != NAMES[0])
    gl, ml = t[f'gcmc_load@{ph}'], t[f'model_load@{ph}']
    okl = gl.notna() & ml.notna()
    return {'n': int(ok.sum()), 'mae': float((m - g)[ok].abs().mean()),
            'rho': float(spearmanr(g[ok], m[ok])[0]) if ok.sum() >= 3 else np.nan,
            'mae_shift': float((t[f'model_shift@{ph}'] - t[f'gcmc_shift@{ph}'])[rest].abs().mean()),
            'mean_offset': float((m - g)[ok].mean()),
            'load_rho': float(spearmanr(gl[okl], ml[okl])[0]) if okl.sum() >= 3 else np.nan,
            'load_mae': float((ml - gl)[okl].abs().mean())}


# ---------------------------------------------------------------- Q_st per framework, and vs pressure
def fig_qst(t, mt):
    ph = args.p_high
    fig, ax = plt.subplots(figsize=(3.35, 2.5), layout='constrained')
    x = np.arange(len(t))
    ax.errorbar(x, t[f'gcmc_qst@{ph}'], yerr=t[f'gcmc_qst@{ph}_err'], fmt='o', mfc='none', mec=GCMC_C,
                ecolor=GCMC_C, ms=6.5, mew=1.2, elinewidth=0.8, capsize=2, label='GCMC', zorder=4)
    ax.errorbar(x, t[f'model_qst@{ph}'], yerr=t[f'model_qst@{ph}_std'], fmt='D', color=INK, ms=4,
                elinewidth=0.8, capsize=2, label='Model', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in FRAMEWORKS])
    ax.set_ylabel(rf'$Q_\mathrm{{st}}$ at {ph} bar (kJ mol$^{{-1}}$)')
    ax.legend(loc='upper left')
    ax.text(0.98, 0.04, rf'Spearman $\rho$ = {mt["rho"]:.2f}' + '\n'
            + f'MAE = {mt["mae"]:.2f}, shift MAE = {mt["mae_shift"]:.2f} kJ mol$^{{-1}}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5)
    savefig(fig, 'fig_uio66_qst')


def fig_pressure(t, pl):
    P = np.array([float(x) for x in pl])
    rows = [('qst', r'$Q_\mathrm{st}$ (kJ mol$^{-1}$)'),
            ('load', ('Absolute' if args.load == 'abs' else 'Excess') + ' loading\n(mmol g$^{-1}$)')]
    n = len(FRAMEWORKS)
    fig, axes = plt.subplots(2, n, figsize=(7.0, 3.3), sharex=True, sharey='row', layout='constrained')
    shown = [v for v in (P.min(), 0.5, P.max()) if np.isclose(P, v).any()]
    for c, (name, lab) in enumerate(FRAMEWORKS):
        r = t.loc[name]
        for i, (key, ylab) in enumerate(rows):
            ax = axes[i, c]
            gm = np.array([r[f'gcmc_{key}@{x}'] for x in pl], float)
            ge = np.array([r.get(f'gcmc_{key}@{x}_err', np.nan) for x in pl], float)
            mm = np.array([r[f'model_{key}@{x}'] for x in pl], float)
            ms = np.array([r[f'model_{key}@{x}_std'] for x in pl], float)
            ax.plot(P, gm, '-', color=GCMC_C, lw=0.6, alpha=0.6, zorder=2)
            ax.errorbar(P, gm, yerr=ge, fmt='o', mfc='none', mec=GCMC_C, ecolor=GCMC_C, ms=4.2, mew=0.9,
                        elinewidth=0.7, capsize=1.5, label='GCMC', zorder=4)
            ax.plot(P, mm, '-', color=INK, lw=0.6, alpha=0.6, zorder=3)
            ax.errorbar(P, mm, yerr=ms, fmt='D', color=INK, ms=2.8, elinewidth=0.7, capsize=1.5, label='Model',
                        zorder=5)
            ax.set_xscale('log')
            ax.set_xticks(P)
            ax.set_xticklabels([f'{v:g}' if any(np.isclose(v, s) for s in shown) else '' for v in P])
            ax.xaxis.set_minor_locator(NullLocator())
            if i == 0:
                ax.set_title(lab, pad=3)
            if c == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(loc='best', fontsize=6.5, handletextpad=0.3)
    fig.supxlabel('Pressure (bar)', fontsize=8)
    savefig(fig, 'fig_uio66_pressure')


# ---------------------------------------------------------------- projected maps (map seed)
def read_atoms(name):
    d = folder(args.map_seed)
    a = pd.read_csv(os.path.join(d, f'atoms_{safe_name(name)}.csv'))
    with open(os.path.join(d, f'{safe_name(name)}.extxyz')) as f:
        f.readline()
        m = re.search(r'Lattice="([^"]+)"', f.readline())
    if m is None:
        raise SystemExit(f'no lattice in {safe_name(name)}.extxyz')
    return a, np.array(m.group(1).split(), float).reshape(3, 3)


def merge_h(a, cell, v):
    """value of each H added to the nearest heavy atom (minimum image)"""
    el = a['element'].values
    pos = a[['x', 'y', 'z']].values
    inv = np.linalg.inv(cell)
    v = np.asarray(v, float).copy()
    heavy = np.where(el != 'H')[0]
    for i in np.where(el == 'H')[0]:
        df = (pos[heavy] - pos[i]) @ inv
        v[heavy[np.argmin(np.linalg.norm((df - np.round(df)) @ cell, axis=1))]] += v[i]
    return v


def window(a, cell):
    orth = np.allclose(cell - np.diag(np.diag(cell)), 0, atol=1e-3)
    if args.box is not None:
        return np.array(args.box), orth
    if orth:
        return np.diag(cell) * np.array([args.rep, args.rep, 1]), orth
    raise SystemExit('non-orthogonal cell: give --box Lx Ly Lz (A), e.g. the conventional cell edges')


def tile(a, cell, vals, L):
    """heavy atoms with values in the window [0, Lx) x [0, Ly) x [0, Lz) from periodic images"""
    keep = a['element'].values != 'H'
    el = a['element'].values[keep]
    frac = (a[['x', 'y', 'z']].values[keep] @ np.linalg.inv(cell)) % 1.0
    vals = {k: np.asarray(v)[keep] for k, v in vals.items()}
    h = np.abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    K = int(np.ceil(L.max() / h.min())) + 1
    sh = np.array(np.meshgrid(*[np.arange(-1, K + 1)] * 3, indexing='ij')).reshape(3, -1).T
    X = ((frac[None] + sh[:, None]) @ cell).reshape(-1, 3)
    idx = np.tile(np.arange(len(el)), len(sh))
    ok = np.all((X >= -1e-6) & (X < L - 1e-6), axis=1)
    return X[ok], el[idx[ok]], {k: v[idx[ok]] for k, v in vals.items()}


def bonds_of(X, el):
    r = np.array([COV_R.get(e, 1.5) for e in el])
    pairs = np.array(sorted(cKDTree(X).query_pairs(3.0)))
    if len(pairs) == 0:
        return pairs
    i, j = pairs.T
    d = np.linalg.norm(X[i] - X[j], axis=1)
    mm = np.isin(el, list(METALS))
    return pairs[(d <= r[i] + r[j] + 0.4) & ~(mm[i] & mm[j])]


def project(X, w, L, orth, cell_xy):
    """Gaussian-smoothed sum of the weights w over z on an xy grid (periodic for an orthogonal cell)"""
    if orth:                                                                     # one cell, periodic, then tiled
        nx, ny = (np.round(cell_xy / args.grid)).astype(int)
        H, _, _ = np.histogram2d(X[:, 0] % cell_xy[0], X[:, 1] % cell_xy[1], bins=[nx, ny],
                                 range=[[0, cell_xy[0]], [0, cell_xy[1]]], weights=w)
        H /= args.rep * args.rep                                                 # X holds rep x rep cells
        H = gaussian_filter(H, args.sigma / args.grid, mode='wrap')
        return np.tile(H, (int(round(L[0] / cell_xy[0])), int(round(L[1] / cell_xy[1]))))
    nx, ny = (np.round(L[:2] / args.grid)).astype(int)
    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[nx, ny], range=[[0, L[0]], [0, L[1]]], weights=w)
    return gaussian_filter(H, args.sigma / args.grid, mode='constant')


def maps():
    ph = args.p_high
    st = pd.read_csv(os.path.join(folder(args.map_seed), 'sites.csv'))
    out = []
    for name, lab in FRAMEWORKS:
        a, cell = read_atoms(name)
        L, orth = window(a, cell)
        q = merge_h(a, cell, a[f'q_i@{ph}'].values)
        e = a['e_i'].values.astype(float)
        X, el, v = tile(a, cell, {'q': q, 'e': np.nan_to_num(e), 'e_raw': e}, L)
        area = args.grid ** 2
        qmap = project(X, v['q'], L, orth, np.diag(cell)[:2]) / area * 1e3     # 10^-3 mmol g^-1 A^-2
        ok = np.isfinite(e[a['element'].values != 'H'])
        wsum = project(X, np.ones(len(X)), L, orth, np.diag(cell)[:2])
        esum = project(X, v['e'], L, orth, np.diag(cell)[:2])
        emap = np.where(wsum > 0.02 * wsum.max(), esum / np.maximum(wsum, 1e-12), np.nan) if ok.any() else None
        s = st[st['name'] == name]
        q_pore = float(s.loc[s['site'] == -1, f'q@{ph}'].sum())
        q_frame = float(s.loc[s['site'] != -1, f'q@{ph}'].sum())
        out.append({'name': name, 'lab': lab, 'a': a, 'cell': cell, 'orth': orth, 'X': X, 'el': el, 'L': L,
                    'bonds': bonds_of(X, el), 'q': qmap, 'e': emap, 'q_atom': v['q'], 'e_atom': v['e_raw'],
                    'q_frame': q_frame, 'q_pore': q_pore})
    return out


def map_panel(fig, rect, img, d, norm, cmap, title=None):
    ax = fig.add_axes(rect)
    ax.imshow(img.T, origin='lower', extent=(0, d['L'][0], 0, d['L'][1]), cmap=cmap, norm=norm,
              interpolation='bilinear')
    if len(d['bonds']):
        ax.add_collection(LineCollection(d['X'][d['bonds']][:, :, :2], colors='white', linewidths=0.15, alpha=0.25))
    ax.set_xlim(0, d['L'][0]); ax.set_ylim(0, d['L'][1]); ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, pad=2)
    return ax


def colorbar(fig, rect, norm, cmap, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.set_label(label, fontsize=6.5)


def shared_norm(imgs, lo_zero=True):
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in imgs if m is not None])
    lo = 0.0 if lo_zero else np.percentile(vals, 100 - args.pct)
    return Normalize(lo, np.percentile(vals, args.pct))


def fig_map(data, key, name, cbar_label, note):
    norm = shared_norm([d[key] for d in data], lo_zero=(key == 'q'))
    n = len(data)
    fig = plt.figure(figsize=(7.0, 1.95))
    w, gap, left = 0.172, 0.006, 0.005
    for c, d in enumerate(data):
        ax = map_panel(fig, [left + c * (w + gap), 0.17, w, 0.70], d[key], d, norm, args.cmap, d['lab'])
        if note:
            ax.text(0.5, -0.03, f'framework {d["q_frame"]:.2f}, pore {d["q_pore"]:.2f}\n(mmol g$^{{-1}}$)',
                    transform=ax.transAxes, ha='center', va='top', fontsize=5.8, color=INK, linespacing=1.1)
    colorbar(fig, [left + n * (w + gap) + 0.004, 0.22, 0.009, 0.60], norm, args.cmap, cbar_label)
    savefig(fig, name, bbox_inches='tight')


def seq_cmap(name):
    return LinearSegmentedColormap.from_list(name + '_t', plt.get_cmap(name)(np.linspace(0.12, 1.0, 256)))


def fig_atoms(data):
    """atoms projected along z, depth-sorted points coloured by q_i (row 1) and e_i (row 2)"""
    rows = [('q_atom', rf'Loading $q_i$ at {args.p_high} bar', 'Blues', r'$10^{-3}$ mmol g$^{-1}$', 1e3),
            ('e_atom', r'Site heat $e_i$', 'Oranges', r'kJ mol$^{-1}$', 1.0)]
    n = len(data)
    fig = plt.figure(figsize=(7.0, 3.25))
    w, gap, left = 0.172, 0.006, 0.005
    for r, (key, rlab, cmap_name, unit, scale) in enumerate(rows):
        allv = np.concatenate([d[key] for d in data]) * scale
        allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, args.pct_atoms)
        if key == 'q_atom':
            lo = 0.0
        norm, cmap = Normalize(lo, hi), seq_cmap(cmap_name)
        for c, d in enumerate(data):
            ax = fig.add_axes([left + c * (w + gap), 0.52 - r * 0.47, w, 0.43])
            X, el, L = d['X'], d['el'], d['L']
            pts = w * 7.0 * 72 / max(L[0], L[1])
            o = np.argsort(X[:, 2])
            if len(d['bonds']):
                ax.add_collection(LineCollection(X[d['bonds']][:, :, :2], colors=MUTED, linewidths=0.25, alpha=0.6,
                                                 zorder=1))
            rad = np.array([COV_R.get(e, 1.5) for e in el])
            v = d[key] * scale
            col = np.where(np.isfinite(v)[:, None], cmap(norm(np.nan_to_num(v))), to_rgba(MUTED))
            ax.scatter(X[o, 0], X[o, 1], s=((0.45 * rad[o] + 0.25) * pts) ** 2, c=col[o], edgecolors=INK,
                       linewidths=0.1, zorder=2)
            ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_aspect('equal'); ax.axis('off')
            if r == 0:
                ax.set_title(d['lab'], pad=2)
        colorbar(fig, [left + n * (w + gap) + 0.004, 0.57 - r * 0.47, 0.009, 0.33], norm, cmap, f'{rlab}\n({unit})')
    savefig(fig, 'fig_uio66_atoms', bbox_inches='tight')


# ---------------------------------------------------------------- GCMC density grid vs model
def vtk_path(name, x):
    f = sorted(glob(os.path.join(pdir(name, x), 'VTK', '**', 'DensityProfile_*.vtk'), recursive=True))
    return f[-1] if f else None


def read_vtk(path):
    """RASPA2 density grid -> [nx, ny, nz] on the fractional grid of the unit cell (relative values, x fastest)"""
    dims, n_head = None, 0
    with open(path) as fh:
        for line in fh:
            n_head += 1
            if line.startswith('DIMENSIONS'):
                dims = tuple(int(v) for v in line.split()[1:4])
            if line.startswith('LOOKUP_TABLE'):
                break
    v = pd.read_csv(path, skiprows=n_head, header=None, dtype=np.float64, engine='c').to_numpy().ravel()
    if dims is None or v.size != np.prod(dims):
        raise SystemExit(f'{path}: {v.size} values for DIMENSIONS {dims}')
    return v.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)


def gcmc_grid(name, x, g):
    """voxel loading in mmol g^-1 (sums to the absolute GCMC loading of that run), or None"""
    path = vtk_path(name, x)
    if path is None:
        return None
    rho = read_vtk(path)
    row = g[(g['name'] == name) & np.isclose(g['pressure'], float(x))]
    if row.empty or rho.sum() <= 0 or 'n_abs_uc' not in row.columns:
        return None
    n_uc, load = float(row['n_abs_uc'].iloc[0]), float(row['load_abs'].iloc[0])
    if not (np.isfinite(n_uc) and np.isfinite(load) and load > 0):
        return None
    mass = n_uc * 1000.0 / load                                                 # g mol^-1 per unit cell
    return {'w': rho / rho.sum() * load, 'mass': mass, 'path': path}


def read_p1_cif(path):
    """cell lengths and (element, fractional) of the P1 CIF written by run_uio66_gcmc.py"""
    lines = open(path).read().splitlines()
    txt = '\n'.join(lines)
    par = {k: float(re.search(rf'{k}\s+([-\d.]+)', txt).group(1))
           for k in ['_cell_length_a', '_cell_length_b', '_cell_length_c']}
    keys, rows = [], []
    for line in lines:
        s = line.strip()
        if s.startswith('_atom_site_'):
            keys.append(s)
        elif keys and s and not s.startswith(('_', 'loop_', '#')):
            rows.append(s.split())
        elif keys and rows:
            break
    k = {c: keys.index(c) for c in ['_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
                                    '_atom_site_fract_z']}
    el = np.array([r[k['_atom_site_type_symbol']] for r in rows])
    fr = np.array([[float(r[k[c]]) for c in ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')]
                   for r in rows])
    return el, fr % 1.0, par


def check_frame(name, a, cell):
    """the model atoms and the GCMC structure must share the cell and the origin for the spatial comparison"""
    f = os.path.join(pdir(name, args.p_high), f'{name}.cif')
    if not os.path.isfile(f):
        print(f'[check] {PLAIN[name]}: {f} not found, frame not checked')
        return True
    el_c, fr_c, par = read_p1_cif(f)
    fa = (a[['x', 'y', 'z']].values @ np.linalg.inv(cell)) % 1.0
    _, j = cKDTree(fr_c % 1.0, boxsize=1.0 + 1e-9).query(fa % 1.0)
    dfr = fa - fr_c[j]
    dist = np.linalg.norm((dfr - np.round(dfr)) @ cell, axis=1)
    bad_el = int((el_c[j] != a['element'].values).sum())
    dl = np.abs(np.linalg.norm(cell, axis=1) - [par['_cell_length_a'], par['_cell_length_b'], par['_cell_length_c']])
    ok = len(el_c) == len(a) and dist.max() < 0.3 and bad_el == 0 and dl.max() < 0.05
    print(f'[check] {PLAIN[name]}: {len(a)} model atoms vs {len(el_c)} GCMC atoms, max offset {dist.max():.3f} A, '
          f'element mismatches {bad_el}, cell length difference {dl.max():.3f} A'
          + ('' if ok else '  <-- MISMATCH: spatial comparison not valid'))
    return ok


def lj_weights(name, a):
    """C6-like weight eps_ij sigma_ij^6 of every atom with CH4 (Lorentz-Berthelot, force field of the GCMC run)"""
    f = os.path.join(pdir(name, args.p_high), 'force_field_mixing_rules.def')
    eps, sig = {}, {}
    if os.path.isfile(f):
        for line in open(f):
            t = line.split()
            if len(t) >= 4 and t[1].lower() == 'lennard-jones':
                eps[t[0]], sig[t[0]] = float(t[2]), float(t[3])
    el = a['element'].values
    if 'CH4_sp3' not in eps or any(f'Mof_{e}' not in eps for e in set(el)):
        print(f'[attr] {PLAIN[name]}: LJ parameters not found in {f} -> plain 1/r^6 weights')
        return np.ones(len(el))
    ec, sc = eps['CH4_sp3'], sig['CH4_sp3']
    return np.array([np.sqrt(eps[f'Mof_{e}'] * ec) * ((sig[f'Mof_{e}'] + sc) / 2) ** 6 for e in el])


def attribute(grid, a, cell, c6):
    """GCMC loading per atom (mmol g^-1). attr = lj: each voxel shared by the atoms within attr_cut in proportion to
    their LJ attraction eps_ij sigma_ij^6 / r^6; attr = nearest: each voxel to its nearest atom within attr_cut"""
    w = grid['w']
    nx, ny, nz = w.shape
    flat = w.ravel()
    nzi = np.flatnonzero(flat > 0)
    vals = flat[nzi]
    ix, iy, iz = np.unravel_index(nzi, w.shape)
    P = np.c_[(ix + 0.5) / nx, (iy + 0.5) / ny, (iz + 0.5) / nz] @ cell
    fa = (a[['x', 'y', 'z']].values @ np.linalg.inv(cell)) % 1.0
    sh = np.array(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing='ij')).reshape(3, -1).T
    img = ((fa[None] + sh[:, None]) @ cell).reshape(-1, 3)
    owner = np.tile(np.arange(len(fa)), len(sh))
    tree = cKDTree(img)
    g_i, beyond = np.zeros(len(fa)), 0.0
    if args.attr == 'nearest':
        d, j = tree.query(P, distance_upper_bound=args.attr_cut, workers=-1)
        ok = np.isfinite(d)
        g_i = np.bincount(owner[j[ok]], weights=vals[ok], minlength=len(fa))
        return g_i, float(vals[~ok].sum()), float(flat.sum())
    for s in range(0, len(P), 40000):
        v = vals[s:s + 40000]
        dm = cKDTree(P[s:s + 40000]).sparse_distance_matrix(tree, args.attr_cut, output_type='coo_matrix')
        wt = c6[owner[dm.col]] / np.maximum(dm.data, 0.5) ** 6
        tot = np.bincount(dm.row, weights=wt, minlength=len(v))
        g_i += np.bincount(owner[dm.col], weights=wt / tot[dm.row] * v[dm.row], minlength=len(fa))
        beyond += float(v[tot <= 0].sum())
    return g_i, beyond, float(flat.sum())


def type_shares(v, types):
    tot = v.sum()
    out = {c: float(v[types == c].sum() / tot) if tot > 0 else np.nan for c in TYPES3}
    out['other'] = float(1.0 - sum(out.values())) if tot > 0 else np.nan
    return out


def gcmc_projection(grid, d):
    """GCMC loading summed along z on the pixel grid of the model maps, 10^-3 mmol g^-1 A^-2 (orthogonal cell)"""
    w, cell = grid['w'], d['cell']
    nx, ny, _ = w.shape
    a, b = np.diag(cell)[:2]
    W = w.sum(2)
    XX, YY = np.meshgrid((np.arange(nx) + 0.5) / nx * a, (np.arange(ny) + 0.5) / ny * b, indexing='ij')
    px, py = np.round(np.array([a, b]) / args.grid).astype(int)
    H, _, _ = np.histogram2d(XX.ravel(), YY.ravel(), bins=[px, py], range=[[0, a], [0, b]], weights=W.ravel())
    H = gaussian_filter(H, args.sigma / args.grid, mode='wrap')
    H = np.tile(H, (int(round(d['L'][0] / a)), int(round(d['L'][1] / b))))
    return H / args.grid ** 2 * 1e3


def gcmc_slice(grid, d, z0):
    """CH4 density (molecules nm^-3) averaged over |z - z0| <= slab, one cell"""
    w, cell = grid['w'], d['cell']
    nx, ny, nz = w.shape
    vol = abs(np.linalg.det(cell)) / (nx * ny * nz)                             # A^3 per voxel
    dens = w * grid['mass'] / 1000.0 / vol * 1000.0                             # molecules nm^-3
    fz = (np.arange(nz) + 0.5) / nz
    dz = (fz - z0 + 0.5) % 1.0 - 0.5
    sel = np.abs(dz) * np.linalg.norm(cell[2]) <= args.slab
    if not sel.any():
        sel = np.abs(dz) == np.abs(dz).min()
    S = dens[:, :, sel].mean(2)
    if args.slice_sigma > 0:
        S = gaussian_filter(S, [args.slice_sigma / (np.linalg.norm(cell[0]) / nx),
                                args.slice_sigma / (np.linalg.norm(cell[1]) / ny)], mode='wrap')
    fa = (d['a'][['x', 'y', 'z']].values @ np.linalg.inv(cell)) % 1.0
    heavy = d['a']['element'].values != 'H'
    near = heavy & (np.abs((fa[:, 2] - z0 + 0.5) % 1.0 - 0.5) * np.linalg.norm(cell[2]) <= args.slab)
    return S, fa[near] @ cell


def gcmc_compare(data, g, pl):
    st = pd.read_csv(os.path.join(folder(args.map_seed), 'sites.csv'))
    share_rows, site_rows, maps_g, slices = [], [], {}, {}
    for d in data:
        name, a, cell = d['name'], d['a'], d['cell']
        aligned = check_frame(name, a, cell)
        c6 = lj_weights(name, a) if args.attr == 'lj' else None
        types = a['site_type'].values.astype(str) if 'site_type' in a.columns else np.full(len(a), 'other')
        site = a['site'].values.astype(int) if 'site' in a.columns else None
        found = []
        for x in pl:
            grid = gcmc_grid(name, x, g)
            if grid is None:
                continue
            found.append(x)
            g_i, beyond, tot = attribute(grid, a, cell, c6)
            q_i = a[f'q_i@{x}'].values.astype(float)
            sg, sm = type_shares(g_i, types), type_shares(q_i, types)
            s = st[st['name'] == name]
            q_pore, q_all = float(s.loc[s['site'] == -1, f'q@{x}'].sum()), float(s[f'q@{x}'].sum())
            rho, n_sites = np.nan, 0
            if site is not None:
                gs = pd.Series(g_i).groupby(site).sum()
                ms = s[s['site'] >= 0].set_index('site')[f'q@{x}']
                both = pd.DataFrame({'gcmc': gs, 'model': ms.reindex(gs.index)}).dropna()
                styp = pd.Series(types).groupby(site).agg(lambda v: v.value_counts().index[0])
                n_sites = len(both)
                if n_sites >= 3:
                    rho = float(spearmanr(both['gcmc'], both['model'])[0])
                for k, r in both.iterrows():
                    site_rows.append({'name': name, 'pressure': float(x), 'site': int(k), 'site_type': styp.get(k, ''),
                                      'gcmc_q': r['gcmc'], 'model_q': r['model']})
            share_rows.append({'name': name, 'pressure': float(x), 'frame_aligned': aligned,
                               **{f'gcmc_{c}': v for c, v in sg.items()}, **{f'model_{c}': v for c, v in sm.items()},
                               'rho_sites': rho, 'n_sites': n_sites,
                               'gcmc_beyond_cut': beyond / tot if tot > 0 else np.nan,
                               'model_pore_token': q_pore / q_all if q_all > 0 else np.nan,
                               'gcmc_load_abs': tot, 'vtk': grid['path']})
            if x == args.p_high:
                if d['orth']:
                    maps_g[name] = gcmc_projection(grid, d)
                slices[name] = [gcmc_slice(grid, d, z0) for z0 in args.slices]
            del grid
        print(f'[gcmc grid] {PLAIN[name]}: density grids at ' + (', '.join(found) + ' bar' if found else 'no pressure'))
    share = pd.DataFrame(share_rows)
    if len(share):
        share.to_csv(os.path.join(args.out_dir, 'table_uio66_share.csv'), index=False)
        pd.DataFrame(site_rows).to_csv(os.path.join(args.out_dir, 'table_uio66_sites_cmp.csv'), index=False)
    return share, maps_g, slices


def fig_density(data, maps_g):
    rows = [([maps_g.get(d['name']) for d in data], 'GCMC CH$_4$ loading'),
            ([d['q'] for d in data], 'Model framework loading')]
    n = len(data)
    fig = plt.figure(figsize=(7.0, 2.85))
    w, gap, left, h = 0.172, 0.006, 0.005, 0.44
    for r, (imgs, rlab) in enumerate(rows):
        norm = shared_norm(imgs)
        y0 = 0.50 - r * 0.47
        for c, d in enumerate(data):
            if imgs[c] is None:
                ax = fig.add_axes([left + c * (w + gap), y0, w, h]); ax.axis('off')
                ax.text(0.5, 0.5, 'no density grid', ha='center', va='center', fontsize=6.5, color=MUTED,
                        transform=ax.transAxes)
                continue
            map_panel(fig, [left + c * (w + gap), y0, w, h], imgs[c], d, norm, args.cmap,
                      d['lab'] if r == 0 else None)
        colorbar(fig, [left + n * (w + gap) + 0.004, y0 + 0.04, 0.009, h - 0.08], norm, args.cmap,
                 f'{rlab}\n($10^{{-3}}$ mmol g$^{{-1}}$ $\\mathrm{{\\AA}}^{{-2}}$)')
    savefig(fig, 'fig_uio66_density', bbox_inches='tight')


def fig_slices(data, slices):
    allv = np.concatenate([S.ravel() for d in data if d['name'] in slices for S, _ in slices[d['name']]])
    norm = Normalize(0.0, np.percentile(allv, args.pct))
    n, m = len(data), len(args.slices)
    fig = plt.figure(figsize=(7.0, 1.42 * m + 0.35))
    w, gap, left, h = 0.168, 0.006, 0.045, 0.86 / m
    frac = {0.0: '0', 0.25: '1/4', 0.5: '1/2', 0.75: '3/4'}
    for c, d in enumerate(data):
        for r, z0 in enumerate(args.slices):
            ax = fig.add_axes([left + c * (w + gap), 0.95 - (r + 1) * h, w, h * 0.96])
            ax.axis('off')
            if c == 0:
                lab = next((v for k, v in SLICE_NAME.items() if np.isclose(k, z0)), '')
                ax.text(-0.06, 0.5, f'$z$ = {frac.get(round(z0, 2), f"{z0:g}")}\n{lab}', transform=ax.transAxes,
                        rotation=90, ha='right', va='center', fontsize=6.5, linespacing=1.2)
            if r == 0:
                ax.set_title(d['lab'], pad=2)
            if d['name'] not in slices:
                ax.text(0.5, 0.5, 'no density grid', ha='center', va='center', fontsize=6.5, color=MUTED,
                        transform=ax.transAxes)
                continue
            S, pts = slices[d['name']][r]
            a, b = np.linalg.norm(d['cell'][0]), np.linalg.norm(d['cell'][1])
            ax.imshow(S.T, origin='lower', extent=(0, a, 0, b), cmap=args.cmap, norm=norm, interpolation='bilinear')
            if len(pts):
                ax.scatter(pts[:, 0] % a, pts[:, 1] % b, s=0.6, c='white', alpha=0.6, linewidths=0)
            ax.set_xlim(0, a); ax.set_ylim(0, b); ax.set_aspect('equal')
    colorbar(fig, [left + n * (w + gap) + 0.004, 0.25, 0.009, 0.55], norm, args.cmap,
             'CH$_4$ density (molecules nm$^{-3}$)')
    savefig(fig, 'fig_uio66_gcmc_slices', bbox_inches='tight')


def fig_share(share):
    s = share[np.isclose(share['pressure'], float(args.p_high))].set_index('name')
    have = [(n, lab) for n, lab in FRAMEWORKS if n in s.index]
    cats = TYPES3 + (['other'] if (s[['gcmc_other', 'model_other']].abs() > 0.005).any().any() else [])
    fig, ax = plt.subplots(figsize=(3.6, 2.6), layout='constrained')
    bw = 0.36
    for i, (name, lab) in enumerate(have):
        for k, who in enumerate(('gcmc', 'model')):
            x0, bottom = i + (k - 0.5) * (bw + 0.04), 0.0
            for c in cats:
                v = max(float(s.loc[name, f'{who}_{c}']), 0.0)
                ax.bar(x0, v, bw, bottom=bottom, color=TYPE_COLOR[c], edgecolor='white', linewidth=0.6,
                       label=c if (i == 0 and k == 0) else None)
                bottom += v
            ax.text(x0, 1.01, 'G' if who == 'gcmc' else 'M', ha='center', va='bottom', fontsize=6, color=INK)
        rho = s.loc[name, 'rho_sites']
        if np.isfinite(rho):
            ax.text(i, -0.16, rf'$\rho$ = {rho:.2f}', ha='center', va='top', fontsize=6, color=INK,
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([lab for _, lab in have])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f'Share of framework-near\nloading at {args.p_high} bar')
    ax.legend(ncol=len(cats), loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=6.5, handlelength=1.0,
              columnspacing=0.8)
    savefig(fig, 'fig_uio66_share')


# ---------------------------------------------------------------- tables
def f(m, s=np.nan, dec=2, sign=False):
    if not np.isfinite(m):
        return '--'
    v = f'{m:+.{dec}f}' if sign else f'{m:.{dec}f}'
    return f'${v}$' if not np.isfinite(s) else f'${v} \\pm {s:.{dec}f}$'


def write_tables(t, mt, pl):
    ph = args.p_high
    t.to_csv(os.path.join(args.out_dir, 'table_uio66.csv'))
    lines = [r'\begin{table*}[t]', r'\small', r'  \centering',
             rf'  \caption{{CH$_4$ in UiO-66 and its functionalized variants at {ph}~bar and 298~K: isosteric heat and '
             + ('absolute' if args.load == 'abs' else 'excess') + r' loading from GCMC of this work and from the model, '
             r'the shift of $Q_\mathrm{st}$ from UiO-66, and the weights $A_c$ of the functional sites and of the pore '
             r'token in the model, with the helium void fraction $\phi_\mathrm{He}$ of each framework. Model values are '
             r'mean $\pm$ standard deviation over three seeds; GCMC uncertainties are 95\% confidence intervals from '
             r'five blocks.}',
             r'  \label{tab:uio66}',
             r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc}', r'    \toprule',
             r'    & & \multicolumn{2}{c}{$Q_\mathrm{st}$ (kJ\,mol$^{-1}$)} & \multicolumn{2}{c}{Shift (kJ\,mol$^{-1}$)} '
             r'& \multicolumn{2}{c}{Loading (mmol\,g$^{-1}$)} & $A_\mathrm{func}$ & $A_\mathrm{pore}$ \\',
             r'    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}',
             r'    Framework & $\phi_\mathrm{He}$ & GCMC & Model & GCMC & Model & GCMC & Model & & \\', r'    \midrule']
    for name, r in t.iterrows():
        lines.append('    ' + ' & '.join([
            TEX[name], f(r['void_fraction'], dec=3),
            f(r[f'gcmc_qst@{ph}'], r[f'gcmc_qst@{ph}_err']), f(r[f'model_qst@{ph}'], r[f'model_qst@{ph}_std']),
            f(r[f'gcmc_shift@{ph}'], sign=True), f(r[f'model_shift@{ph}'], sign=True),
            f(r[f'gcmc_load@{ph}'], r[f'gcmc_load@{ph}_err']), f(r[f'model_load@{ph}'], r[f'model_load@{ph}_std']),
            f(r['A_functional'], r['A_functional_std']), f(r['A_pore'], r['A_pore_std'])]) + r' \\')
    lines += [r'    \midrule',
              rf'    \multicolumn{{10}}{{l}}{{$Q_\mathrm{{st}}$: Spearman $\rho$ {mt["rho"]:.2f}, MAE {mt["mae"]:.2f}~kJ\,mol$^{{-1}}$, '
              rf'MAE of the shift {mt["mae_shift"]:.2f}~kJ\,mol$^{{-1}}$; loading: Spearman $\rho$ {mt["load_rho"]:.2f}, '
              rf'MAE {mt["load_mae"]:.2f}~mmol\,g$^{{-1}}$}} \\',
              r'    \bottomrule', r'  \end{tabular*}', r'\end{table*}']
    with open(os.path.join(args.out_dir, 'table_uio66.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    lines = [r'\begin{table}[h]', r'  \centering',
             rf'  \caption{{Site-level quantities of the model for CH$_4$ in UiO-66 and its functionalized variants at '
             rf'{ph}~bar: same-MOF difference between the mean heat of the functional or node sites and that of the '
             r'linker sites, and the weights $A_c$ of the functional sites and of the pore token (mean $\pm$ standard '
             r'deviation over three seeds).}',
             r'  \label{tab:uio66_sites_si}', r'  \begin{tabular}{lcccc}', r'    \toprule',
             r'    Framework & $\bar e_\mathrm{func}-\bar e_\mathrm{linker}$ & $\bar e_\mathrm{node}-\bar e_\mathrm{linker}$ '
             r'& $A_\mathrm{func}$ & $A_\mathrm{pore}$ \\',
             r'    & (kJ\,mol$^{-1}$) & (kJ\,mol$^{-1}$) & & \\', r'    \midrule']
    for name, r in t.iterrows():
        lines.append('    ' + ' & '.join([
            TEX[name], f(r['e_func_minus_linker'], r['e_func_minus_linker_std'], sign=True),
            f(r['e_node_minus_linker'], r['e_node_minus_linker_std'], sign=True),
            f(r['A_functional'], r['A_functional_std']), f(r['A_pore'], r['A_pore_std'])]) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(os.path.join(args.out_dir, 'table_uio66_sites_si.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    cols = ['name'] + [c for x in pl for c in (f'model_qst@{x}', f'model_qst@{x}_std', f'gcmc_qst@{x}', f'gcmc_qst@{x}_err',
                                               f'model_load@{x}', f'model_load@{x}_std', f'gcmc_load@{x}',
                                               f'gcmc_load@{x}_err')]
    t.reset_index().rename(columns={'index': 'name'})[cols].to_csv(os.path.join(args.out_dir, 'table_uio66_si.csv'),
                                                                    index=False)
    lines = [r'\begin{table}[h]', r'  \centering',
             r'  \caption{Isosteric heat of CH$_4$ (kJ\,mol$^{-1}$) predicted for UiO-66 and its functionalized '
             r'variants at the five pressures of the training data (mean $\pm$ standard deviation over three seeds), '
             r'with the GCMC values of this work in parentheses.}',
             r'  \label{tab:uio66_si}', r'  \begin{tabular}{l' + 'c' * len(pl) + '}', r'    \toprule',
             '    Framework & ' + ' & '.join(f'{x}~bar' for x in pl) + r' \\', r'    \midrule']
    for name, r in t.iterrows():
        cells = []
        for x in pl:
            c = f(r[f'model_qst@{x}'], r[f'model_qst@{x}_std'])
            if np.isfinite(r[f'gcmc_qst@{x}']):
                c += f' ({r[f"gcmc_qst@{x}"]:.2f})'
            cells.append(c)
        lines.append(f'    {TEX[name]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(os.path.join(args.out_dir, 'table_uio66_si.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    t, pl = model_tables()
    g, void, settings = gcmc_tables()
    print('[gcmc] ' + (', '.join(f'{k} {v}' for k, v in settings.items()) or 'settings not recorded'))
    gp = sorted(set(np.round(g['pressure'], 6)))
    lack = [x for x in pl if not np.isclose(gp, float(x)).any()]
    if lack:
        print(f'[gcmc] no GCMC values at {lack} bar (GCMC pressures: {gp})')
    t = combine(t, g, void, pl)
    mt = metrics(t)
    fig_qst(t, mt)
    fig_pressure(t, pl)
    data = maps()
    fig_map(data, 'q', 'fig_uio66_qmap', r'Projected framework loading' + '\n'
            + rf'at {args.p_high} bar ($10^{{-3}}$ mmol g$^{{-1}}$ $\mathrm{{\AA}}^{{-2}}$)', note=True)
    figs = ['fig_uio66_qst', 'fig_uio66_pressure', 'fig_uio66_qmap']
    if all(d['e'] is not None for d in data):
        fig_map(data, 'e', 'fig_uio66_emap', r'Site heat $e_i$ (kJ mol$^{-1}$)', note=False)
        figs.append('fig_uio66_emap')
    fig_atoms(data)
    figs.append('fig_uio66_atoms')
    share, maps_g, slices = gcmc_compare(data, g, pl)
    if maps_g:
        fig_density(data, maps_g); figs.append('fig_uio66_density')
    if slices:
        fig_slices(data, slices); figs.append('fig_uio66_gcmc_slices')
    else:
        print(f'[gcmc grid] no density grid at {args.p_high} bar under {args.gcmc_dir}/ch4/<name>/<p>bar/VTK '
              '(run_uio66_gcmc.py --density_pressures all) -> GCMC maps skipped')
    if len(share) and np.isclose(share['pressure'], float(args.p_high)).any():
        fig_share(share); figs.append('fig_uio66_share')
    write_tables(t, mt, pl)

    ph = args.p_high
    pd.set_option('display.width', 240)
    show = t.copy()
    show.index = [PLAIN[n] for n in show.index]
    cols = ['void_fraction', f'gcmc_qst@{ph}', f'model_qst@{ph}', f'model_qst@{ph}_std', f'gcmc_shift@{ph}', f'model_shift@{ph}',
            f'gcmc_load@{ph}', f'model_load@{ph}', f'model_load@{ph}_std'] + \
           [c for k in SITE_COLS for c in (k, f'{k}_std')] + ['groups']
    print(f'\n=== UiO-66 frameworks, CH4 at {ph} bar (model and site columns: mean/std over seeds {", ".join(args.seeds)}) ===')
    print(show[cols].to_string(float_format=lambda v: f'{v:.3f}'))
    for key, unit in (('qst', 'kJ/mol'), ('load', 'mmol/g')):
        print(f'\n--- {key} vs pressure ({unit}): GCMC / model (mean over seeds) ---')
        print(pd.DataFrame({f'{x} bar': [f'{r[f"gcmc_{key}@{x}"]:.2f} / {r[f"model_{key}@{x}"]:.2f}'
                                         for _, r in show.iterrows()] for x in pl}, index=show.index).to_string())
    print(f'\nQ_st at {ph} bar: Spearman rho {mt["rho"]:.3f} (n = {mt["n"]}), MAE {mt["mae"]:.3f}, mean offset model - GCMC '
          f'{mt["mean_offset"]:+.3f}, MAE of the shift from UiO-66 {mt["mae_shift"]:.3f} kJ/mol')
    print(f'loading at {ph} bar: Spearman rho {mt["load_rho"]:.3f}, MAE {mt["load_mae"]:.3f} mmol/g')
    if len(share):
        s = share[np.isclose(share['pressure'], float(ph))].copy()
        s['name'] = s['name'].map(PLAIN)
        print(f'\n--- framework-near loading at {ph} bar, share per site type (attr {args.attr}, attr_cut '
              f'{args.attr_cut} A, seed {args.map_seed}) ---')
        print(s[['name'] + [f'{w}_{c}' for c in TYPES3 for w in ('gcmc', 'model')]
                + ['rho_sites', 'n_sites', 'gcmc_beyond_cut', 'model_pore_token']]
              .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nmaps (seed ' + args.map_seed + '): ' + ', '.join(
        f'{PLAIN[n]} framework {d["q_frame"]:.3f} / pore {d["q_pore"]:.3f}' for n, d in zip(NAMES, data)))
    print(f'\n[figure] {", ".join(figs)} (.pdf/.png); [table] table_uio66.csv/.tex, table_uio66_si.csv/.tex, '
          f'table_uio66_sites_si.tex' + (', table_uio66_share.csv, table_uio66_sites_cmp.csv' if len(share) else '')
          + f', gcmc_input.csv -> {args.out_dir}')


if __name__ == '__main__':
    main()
