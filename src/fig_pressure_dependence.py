"""
Pressure dependence of isosteric heats: main-text figure, ESI figure and ESI table in one run.

  main (1 x 2)  a, b  dQst split p_min -> p_max for CO2 and CH4: all MOFs + groups by predicted trend
                      (decreasing / flat / increasing); framework, pore share shift and interaction terms (bars),
                      model and GCMC dQst (markers); means over seeds
                      (site_analysis_seeds/decomposition_summary.csv, decomposition_by_trend.csv from aggregate_seeds.py)
  ESI  (1 x 2)  a     CO2: low-coverage interaction slope u'(0) (= a_1, column slope_0) against LCD
                b     CH4: interaction at full coverage u(1) = a_1 + a_2 against void fraction
                      one seed's per_mof.csv, hexagonal bins (log count) + median per x-quantile bin;
                      Spearman rho = mean +- std over seeds (langmuir_analysis_porep_seeds/summary.csv)
  ESI table           dQst decomposition (all MOFs), filling at p_max and mean-field interaction, CO2 vs CH4,
                      mean +- std over seeds

    python fig_pressure_dependence.py                       # defaults below
    python fig_pressure_dependence.py --seed 43 --out_dir ./figures

Outputs (out_dir):
    fig_pressure_dependence.pdf/.png,     fig_pressure_dependence_bars.csv      (main)
    fig_pressure_dependence_si.pdf/.png,  fig_pressure_dependence_si_bins.csv   (ESI figure)
    table_dqst_meanfield.csv/.tex                                               (ESI table)
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ap = argparse.ArgumentParser(description='pressure dependence figures and ESI table')
ap.add_argument('--co2_run', default='../results_hmof/ViSNet/heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--ch4_run', default='../results_hmof/ViSNet/heat_ch4_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--site_dir', default='site_analysis_seeds', help='aggregated site analysis folder inside each run')
ap.add_argument('--lang_dir', default='langmuir_analysis_porep', help='langmuir analysis folder prefix inside each run')
ap.add_argument('--seed', default='42', type=str, help='seed whose per_mof.csv is drawn in the ESI figure')
ap.add_argument('--co2_x', default='lcd'); ap.add_argument('--co2_y', default='slope_0', help="u'(0) (= a_1)")
ap.add_argument('--ch4_x', default='void_fraction'); ap.add_argument('--ch4_y', default='g_full(theta=1)')
ap.add_argument('--gridsize', default=45, type=int)
ap.add_argument('--n_bins', default=20, type=int, help='x-quantile bins for the median line')
ap.add_argument('--out_dir', default='../results_hmof/figures')
ap.add_argument('--name', default='fig_pressure_dependence')
ap.add_argument('--table', default='table_dqst_meanfield')
args = ap.parse_args()

INK, MUTED, GCMC = '#0b0b0b', '#898781', '#D62728'
TERMS = [('framework', 'framework', '#1F77B4'), ('pore_share_shift', 'pore share shift', '#FF7F0E'),
         ('interaction', 'interaction', '#2CA02C')]
TRENDS = ['decreasing', 'flat', 'increasing']
CMAP = LinearSegmentedColormap.from_list('blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
LABEL = {'lcd': r'LCD ($\mathrm{\AA}$)', 'pld': r'PLD ($\mathrm{\AA}$)', 'void_fraction': 'Void fraction',
         'a_1': r"$u'(0)$ (kJ mol$^{-1}$)", 'slope_0': r"$u'(0)$ (kJ mol$^{-1}$)",
         'g_full(theta=1)': r'$u(1)$ (kJ mol$^{-1}$)', 'slope_1': r"$u'(1)$ (kJ mol$^{-1}$)"}
RHO_KEY = {'a_1': "mf_rho_u'(0)_vs_{x}", 'slope_0': "mf_rho_u'(0)_vs_{x}", 'g_full(theta=1)': 'mf_rho_u(1)_vs_{x}'}
GEO = ('lcd', 'pld', 'void_fraction')
GASES = [('CO2', r'CO$_2$', r'0.01$\rightarrow$2.5 bar'), ('CH4', r'CH$_4$', r'0.05$\rightarrow$4.5 bar')]
TABLE = [('dQ', r'Model $\Delta Q_\mathrm{st}$', 'dec', 'dQ_pred', 2),
         ('dQ', r'GCMC $\Delta Q_\mathrm{st}$', 'dec', 'dQ_true', 2),
         ('dQ', 'Framework term', 'dec', 'framework', 2),
         ('dQ', 'Pore share shift term', 'dec', 'pore_share_shift', 2),
         ('dQ', 'Interaction term', 'dec', 'interaction', 2),
         ('fill', r'Model, mean $\theta_\mathrm{total}$', 'lang', 'theta_total@pmax_mean', 2),
         ('fill', r'GCMC Langmuir fit, mean $\theta_\mathrm{fit}$', 'lang', 'theta_fit@pmax_mean', 2),
         ('fill', r'Spearman $\rho(\theta_\mathrm{total}, \theta_\mathrm{fit})$', 'lang', 'rho_theta_total_vs_theta_fit', 2),
         ('mf', r"Median $u'(0)$ (kJ\,mol$^{-1}$)", 'lang', "mf_u'(0)_median", 2),
         ('mf', r'Median $u(1)$ (kJ\,mol$^{-1}$)', 'lang', 'g_full_median', 2),
         ('mf', r"Spearman $\rho(u'(0), \mathrm{LCD})$", 'lang', "mf_rho_u'(0)_vs_lcd", 2),
         ('mf', r'Spearman $\rho(u(1), \text{void fraction})$', 'lang', 'mf_rho_u(1)_vs_void_fraction', 2)]
SECTION = {'dQ': r'$\Delta Q_\mathrm{st}$ from $p_\mathrm{min}$ to $p_\mathrm{max}$, mean over MOFs (kJ\,mol$^{-1}$)',
           'fill': r'Filling at $p_\mathrm{max}$', 'mf': r'Mean-field interaction $u(\theta) = a_1\theta + a_2\theta^2$'}


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def need(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found (run analyze.py per seed and aggregate_seeds.py first)')
    return path


def read_indexed(path):
    t = pd.read_csv(path)
    return t.set_index(t.columns[0])


def title(ax, lab, text):
    ax.set_title(rf'$\mathbf{{{lab}}}$   {text}', loc='left')


def save(fig, name):
    fig.savefig(os.path.join(args.out_dir, name + '.pdf'))
    fig.savefig(os.path.join(args.out_dir, name + '.png'), dpi=600)
    plt.close(fig)


def full_range(run):
    t = read_indexed(need(os.path.join(run, args.site_dir, 'decomposition_summary.csv')))
    span = {iv: float(str(iv).split('->')[1]) / float(str(iv).split('->')[0]) for iv in t.index if '->' in str(iv)}
    return t.loc[max(span, key=span.get)]


def trend_table(run):
    return read_indexed(need(os.path.join(run, args.site_dir, 'decomposition_by_trend.csv')))


def lang_summary(run):
    f = os.path.join(run, f'{args.lang_dir}_seeds', 'summary.csv')
    if not os.path.isfile(f):
        print(f'[note] {f} not found -> no rho annotation / table rows from it')
        return None
    return read_indexed(f)


def bars(ax, run, lab, gas, gas_tex, rng, rows):
    t = trend_table(run)
    order = [g for g in TRENDS if g in t.index]
    a = full_range(run)
    n_all = float(t.loc[order, 'n_mof_mean'].sum())
    cols = [f'{k}_mean' for k, _, _ in TERMS] + ['dQ_pred_mean', 'dQ_true_mean']
    data = pd.DataFrame([a[cols]] + [t.loc[g, cols] for g in order], index=['all'] + order)
    n = [n_all] + [t.loc[g, 'n_mof_mean'] for g in order]
    x = np.r_[0.0, 1.5 + 1.15 * np.arange(len(order))]
    w = 0.24
    for i, (k, _, col) in enumerate(TERMS):
        ax.bar(x + (i - 1) * w, data[f'{k}_mean'].values, w, color=col, edgecolor='white', lw=0.6)
    ax.plot(x, data['dQ_pred_mean'], 'D', color=INK, ms=4, zorder=5)
    ax.plot(x, data['dQ_true_mean'], 'o', mfc='none', mec=GCMC, ms=6.5, mew=1.2, zorder=6)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0.75, color=MUTED, lw=0.6, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\n(n = {v:,.0f})' for g, v in zip(data.index, n)])
    ax.set_ylabel(r'Mean $\Delta Q_\mathrm{st}$ (kJ mol$^{-1}$)')
    title(ax, lab, f'{gas_tex}, {rng}')
    stds = {**{c.replace('_mean', '_std'): a.get(c.replace('_mean', '_std'), np.nan) for c in cols}}
    rows.append({'panel': lab, 'gas': gas, 'group': 'all', 'n_mof_mean': n_all, **a[cols].to_dict(), **stds})
    for g in order:
        rows.append({'panel': lab, 'gas': gas, 'group': g,
                     **{c: t.loc[g, c] for c in t.columns if c.endswith(('_mean', '_std'))}})


def main_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08)
    rows = []
    for ax, lab, run, (gas, tex, rng) in zip(axes, 'ab', (args.co2_run, args.ch4_run), GASES):
        bars(ax, run, lab, gas, tex, rng, rows)
    handles = ([Patch(fc=c, ec='white', label=l) for _, l, c in TERMS]
               + [Line2D([], [], ls='', marker='D', color=INK, ms=4, label=r'$\Delta Q_\mathrm{st}$, model'),
                  Line2D([], [], ls='', marker='o', mfc='none', mec=GCMC, mew=1.2, ms=6.5,
                         label=r'$\Delta Q_\mathrm{st}$, GCMC')])
    fig.legend(handles=handles, loc='outside upper center', ncol=5, handlelength=1.4, columnspacing=1.4)
    save(fig, args.name)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, args.name + '_bars.csv'), index=False)


def rho_text(s, ycol, xcol):
    if s is None or ycol not in RHO_KEY or xcol not in GEO:
        return None
    k = RHO_KEY[ycol].format(x=xcol)
    if k not in s.index:
        print(f'[note] {k} not in summary.csv -> no rho annotation')
        return None
    m, sd = float(s.loc[k, 'value_mean']), float(s.loc[k, 'value_std'])
    return rf'Spearman $\rho$ = {m:.2f} $\pm$ {sd:.2f}'.replace('= -', '= −')


def density(ax, run, xcol, ycol, lab, gas, gas_tex, rows):
    per = pd.read_csv(need(os.path.join(run, f'{args.lang_dir}_{args.seed}', 'per_mof.csv')))
    for c in (xcol, ycol):
        if c not in per.columns:
            raise KeyError(f'{c} not in per_mof.csv ({run}); columns: {", ".join(per.columns[:40])} ...')
    d = per[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    x, y = d[xcol].values, d[ycol].values
    xl, yl = np.percentile(x, [0.5, 99.5]), np.percentile(y, [0.5, 99.5])
    hb = ax.hexbin(x, y, gridsize=args.gridsize, extent=(*xl, *yl), cmap=CMAP, norm=LogNorm(), mincnt=1,
                   linewidths=0)
    med = d.groupby(pd.qcut(x, args.n_bins, duplicates='drop'), observed=True).agg(
        x=(xcol, 'median'), y=(ycol, 'median'), n=(ycol, 'size'))
    ax.plot(med['x'], med['y'], '-', color=INK, lw=1.4, zorder=4)
    ax.set_xlim(xl); ax.set_ylim(yl)
    xlab = LABEL.get(xcol, xcol)
    if xcol in ('lcd', 'pld') and x.min() < 0:
        xlab = xlab.split(' (')[0] + ' (standardized)'
        print(f'[note] {xcol} has negative values -> treated as standardized, axis label without unit')
    ax.set_xlabel(xlab); ax.set_ylabel(LABEL.get(ycol, ycol))
    title(ax, lab, gas_tex)
    txt = rho_text(lang_summary(run), ycol, xcol)
    if txt:
        ax.text(0.97, 0.96, txt, transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))
    for _, r in med.iterrows():
        rows.append({'panel': lab, 'gas': gas, 'x': xcol, 'y': ycol, 'x_median': r['x'], 'y_median': r['y'],
                     'n': int(r['n'])})
    return hb


def si_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08)
    rows = []
    spec = ((args.co2_run, args.co2_x, args.co2_y), (args.ch4_run, args.ch4_x, args.ch4_y))
    for ax, lab, (run, xc, yc), (gas, tex, _) in zip(axes, 'ab', spec, GASES):
        h = density(ax, run, xc, yc, lab, gas, tex, rows)
        cb = fig.colorbar(h, ax=ax, pad=0.01, fraction=0.05, aspect=25)
        cb.set_label('MOFs per bin'); cb.outline.set_linewidth(0.4)
    save(fig, args.name + '_si')
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, args.name + '_si_bins.csv'), index=False)


def fmt(m, s, dec):
    if not np.isfinite(m):
        return '--'
    if not np.isfinite(s) or round(s, dec) == 0:
        return f'${m:.{dec}f}$'
    return f'${m:.{dec}f} \\pm {s:.{dec}f}$'


def esi_table():
    src = {}
    for gas, run in (('CO2', args.co2_run), ('CH4', args.ch4_run)):
        src[gas] = {'dec': full_range(run), 'lang': lang_summary(run)}
    rows = []
    for sec, lab, s, key, dec in TABLE:
        r = {'section': sec, 'quantity': lab, 'key': key}
        for gas in ('CO2', 'CH4'):
            t = src[gas][s]
            if s == 'dec':
                m, sd = t.get(f'{key}_mean', np.nan), t.get(f'{key}_std', np.nan)
            elif t is not None and key in t.index:
                m, sd = t.loc[key, 'value_mean'], t.loc[key, 'value_std']
            else:
                m, sd = np.nan, np.nan
                print(f'[note] {gas}: {key} not found -> "--" in the table')
            r[f'{gas}_mean'], r[f'{gas}_std'], r[f'{gas}_tex'] = float(m), float(sd), fmt(float(m), float(sd), dec)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.drop(columns=['CO2_tex', 'CH4_tex']).to_csv(os.path.join(args.out_dir, args.table + '.csv'), index=False)

    lines = [r'\begin{table}[h]', r'  \centering',
             r'  \caption{Decomposition of the pressure-induced change in $Q_\mathrm{st}$ over all test MOFs, '
             r'filling at the highest pressure and mean-field interaction parameters '
             r'($p_\mathrm{min}\rightarrow p_\mathrm{max}$: 0.01$\rightarrow$2.5~bar for CO$_2$ and '
             r'0.05$\rightarrow$4.5~bar for CH$_4$). Values are mean $\pm$ standard deviation over three seeds; '
             r'quantities computed from GCMC data alone are given without a deviation. '
             r"$u(\theta) = a_1\theta + a_2\theta^2$, so $u'(0) = a_1$ and $u(1) = a_1 + a_2$.}",
             r'  \label{tab:dqst_meanfield}',
             r'  \begin{tabular}{lcc}', r'    \toprule', r'    & CO$_2$ & CH$_4$ \\', r'    \midrule']
    for i, sec in enumerate(dict.fromkeys(tab['section'])):
        if i:
            lines.append(r'    \midrule')
        lines.append(rf'    \multicolumn{{3}}{{l}}{{\textit{{{SECTION[sec]}}}}} \\')
        for _, r in tab[tab['section'] == sec].iterrows():
            lines.append(f"    {r['quantity']} & {r['CO2_tex']} & {r['CH4_tex']} \\\\")
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    tex = '\n'.join(lines)
    with open(os.path.join(args.out_dir, args.table + '.tex'), 'w') as f:
        f.write(tex + '\n')
    return tab


def main():
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    main_figure()
    si_figure()
    tab = esi_table()
    pd.set_option('display.width', 200)
    show = tab[['quantity', 'CO2_tex', 'CH4_tex']].copy()
    show['quantity'] = tab['key']
    print('\n=== ESI table (mean +- std over seeds) ===')
    print(show.rename(columns={'CO2_tex': 'CO2', 'CH4_tex': 'CH4'}).to_string(index=False))
    print(f'\n[figure] main: {args.name}.pdf/.png, {args.name}_bars.csv')
    print(f'[figure] ESI : {args.name}_si.pdf/.png, {args.name}_si_bins.csv (seed {args.seed})')
    print(f'[table]  ESI : {args.table}.csv/.tex')
    print(f'         -> {args.out_dir}')


if __name__ == '__main__':
    main()
