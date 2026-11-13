"""
Site types of the Langmuir-informed pooling model, CO2 vs CH4 (main-text figure + ESI table in one run).

  a, b  site heat per type (node / linker / functional / pore): per-MOF mean heat of the active sites of each type;
        box = interquartile range, line = median, whiskers = 5th-95th percentile, each averaged over seeds.
        The same-MOF node - linker difference is printed as a caption sentence (in-panel note only with --note).
  c, d  mean share A_c(p) of each type against pressure, mean over seeds (band = +- std over seeds);
        the remainder to 1 is carried by sites without assigned atoms.
        The legend sits inside one share panel (default: d, lower left), since panels a, b name the types on the x axis.
  ESI table  same-MOF difference of the type heat to the linker (median, IQR, share > 0), mean +- std over seeds

Reads <run>/<site_prefix>_<seed>/{mof_types.csv, type_share.csv, type_heat_paired.csv} written by site_analysis.py.

    python fig_site_types.py                          # defaults below
    python fig_site_types.py --seeds 42 43 44 --out_dir ./figures
    python fig_site_types.py --legend_panel c --legend_loc 'lower left'
    python fig_site_types.py --note                   # old layout: node - linker note inside panels a, b

Outputs (out_dir): fig_site_types.pdf/.png, fig_site_types_data.csv, fig_site_types_caption.txt,
                   table_type_heat_paired.csv/.tex
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser(description='site type figure (CO2 vs CH4) and ESI table')
ap.add_argument('--co2_run', default='../results_hmof/ViSNet/heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--ch4_run', default='../results_hmof/ViSNet/heat_ch4_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin')
ap.add_argument('--site_prefix', default='site_analysis', help='per-seed folders <site_prefix>_<seed> inside each run')
ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
ap.add_argument('--whis', nargs=2, type=float, default=[5, 95], help='whisker percentiles')
ap.add_argument('--note', action='store_true', help='also print the node - linker note inside panels a, b')
ap.add_argument('--legend_panel', default='d', choices=['c', 'd', 'top'],
                help="share panel that holds the legend; 'top' = outside above the figure (old layout)")
ap.add_argument('--legend_loc', default='lower left', help='matplotlib legend location inside the panel')
ap.add_argument('--legend_ncol', type=int, default=2)
ap.add_argument('--out_dir', default='../results_hmof/figures')
ap.add_argument('--name', default='fig_site_types')
ap.add_argument('--table', default='table_type_heat_paired')
args = ap.parse_args()

INK, MUTED = '#0b0b0b', '#898781'
TYPES = ['node', 'linker', 'functional', 'pore']
# matplotlib default colour cycle (tab10); pore in orange as the pore share shift term; markers as second encoding
COLOR = {'node': '#1F77B4', 'linker': '#2CA02C', 'functional': '#D62728', 'pore': '#FF7F0E'}
MARKER = {'node': 'o', 'linker': 's', 'functional': '^', 'pore': 'D'}
GASES = [('CO2', r'CO$_2$'), ('CH4', r'CH$_4$')]
PAIRS = ['node - linker', 'functional - linker', 'pore - linker']


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def seed_files(run, fname):
    files = [os.path.join(run, f'{args.site_prefix}_{s}', fname) for s in args.seeds]
    miss = [f for f in files if not os.path.isfile(f)]
    if miss:
        raise SystemExit(f'not found (run site_analysis.py per seed first): {miss}')
    return [pd.read_csv(f) for f in files]


def box_stats(run):
    """per type: percentiles of the per-MOF type heat, averaged over seeds"""
    q = [args.whis[0], 25, 50, 75, args.whis[1]]
    per_seed = []
    for t in seed_files(run, 'mof_types.csv'):
        per_seed.append({c: (np.nanpercentile(t[f'e_{c}'], q), int(t[f'e_{c}'].notna().sum())) for c in TYPES
                         if f'e_{c}' in t.columns and t[f'e_{c}'].notna().any()})
    out = {}
    for c in TYPES:
        vals = [s[c] for s in per_seed if c in s]
        if not vals:
            continue
        p = np.array([v[0] for v in vals])
        out[c] = {'whislo': p[:, 0].mean(), 'q1': p[:, 1].mean(), 'med': p[:, 2].mean(), 'q3': p[:, 3].mean(),
                  'whishi': p[:, 4].mean(), 'med_std': p[:, 2].std(ddof=1) if len(p) > 1 else np.nan,
                  'n_mof': float(np.mean([v[1] for v in vals]))}
    return out


def share_stats(run):
    """mean share A_c per type and pressure: mean and std over seeds"""
    tabs = [t.pivot(index='type', columns='pressure', values='A_mean') for t in seed_files(run, 'type_share.csv')]
    cols = sorted(tabs[0].columns, key=float)
    stack = np.stack([t.reindex(index=TYPES, columns=cols).values for t in tabs])          # [seed, type, P]
    sd = stack.std(0, ddof=1) if len(tabs) > 1 else np.full(stack.shape[1:], np.nan)
    return np.array([float(c) for c in cols]), pd.DataFrame(stack.mean(0), index=TYPES), pd.DataFrame(sd, index=TYPES)


def paired_stats(run):
    """same-MOF difference to the linker: mean and std over seeds"""
    tabs = [t.set_index('pair') for t in seed_files(run, 'type_heat_paired.csv')]
    cols = ['n_mof', 'median', 'q25', 'q75', 'frac_positive']
    st = np.stack([t.reindex(PAIRS)[cols].values for t in tabs])
    return (pd.DataFrame(st.mean(0), index=PAIRS, columns=cols),
            pd.DataFrame(st.std(0, ddof=1) if len(tabs) > 1 else np.full(st.shape[1:], np.nan), index=PAIRS, columns=cols))


def title(ax, lab, text):
    ax.set_title(rf'$\mathbf{{{lab}}}$   {text}', loc='left')


def heat_panel(ax, box, pm, lab, tex):
    order = [c for c in TYPES if c in box]
    stats = [{k: box[c][k] for k in ('whislo', 'q1', 'med', 'q3', 'whishi')} | {'fliers': []} for c in order]
    bp = ax.bxp(stats, positions=range(len(order)), widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(lw=0.6, color=INK),
                capprops=dict(lw=0.6, color=INK), boxprops=dict(lw=0.6, edgecolor=INK))
    for patch, c in zip(bp['boxes'], order):
        patch.set_facecolor(COLOR[c])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(r'Site heat $\bar{e}_c$ (kJ mol$^{-1}$)')
    title(ax, lab, tex)
    if args.note:                                                                  # old layout
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        m, f = pm.loc['node - linker', 'median'], pm.loc['node - linker', 'frac_positive']
        ax.text(0.98, 0.97, f'node $-$ linker, same MOF:\nmedian {m:+.2f} kJ mol$^{{-1}}$, > 0 in {100 * f:.0f}% of MOFs',
                transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color=INK)


def share_panel(ax, p, mean, sd, lab, tex):
    for c in TYPES:
        if mean.loc[c].isna().all():
            continue
        ax.plot(p, mean.loc[c], '-', marker=MARKER[c], color=COLOR[c], ms=4, lw=1.4)
        if sd.loc[c].notna().all():
            ax.fill_between(p, mean.loc[c] - sd.loc[c], mean.loc[c] + sd.loc[c], color=COLOR[c], alpha=0.15, lw=0)
    ax.set_xscale('log')
    ax.set_xticks(p)
    ax.set_xticklabels([f'{x:g}' for x in p])
    ax.minorticks_off()
    ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel(r'Mean share $A_c$')
    ax.set_ylim(0, 1.15 * np.nanmax(mean.values + np.nan_to_num(sd.values)))
    title(ax, lab, tex)


def legend_overlap(fig, ax, leg):
    """number of plotted points of ax that fall inside the legend box (0 = no overlap)"""
    fig.canvas.draw()
    box = leg.get_window_extent()
    n = 0
    for line in ax.get_lines():
        xy = ax.transData.transform(np.column_stack(line.get_data()))
        n += int(np.sum((xy[:, 0] >= box.x0) & (xy[:, 0] <= box.x1) & (xy[:, 1] >= box.y0) & (xy[:, 1] <= box.y1)))
    return n


def fmt(m, s, dec=2, sign=True):
    f = f'{m:+.{dec}f}' if sign else f'{m:.{dec}f}'
    return f'${f}$' if not np.isfinite(s) else f'${f} \\pm {s:.{dec}f}$'


def esi_table(paired):
    rows = []
    for pair in PAIRS:
        r = {'pair': pair}
        for gas, _ in GASES:
            pm, ps = paired[gas]
            for k in ('n_mof', 'median', 'q25', 'q75', 'frac_positive'):
                r[f'{gas}_{k}_mean'], r[f'{gas}_{k}_std'] = pm.loc[pair, k], ps.loc[pair, k]
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out_dir, args.table + '.csv'), index=False)
    lines = [r'\begin{table}[h]', r'  \centering',
             r'  \caption{Same-MOF difference of the mean site heat of each type to that of the linker sites '
             r'(kJ\,mol$^{-1}$): median, interquartile range (IQR) and share of MOFs with a positive difference. '
             r'Values are mean $\pm$ standard deviation over three seeds; $n$ is the number of MOFs that contain both types.}',
             r'  \label{tab:type_heat_paired}',
             r'  \begin{tabular}{lcccccc}', r'    \toprule',
             r'    & \multicolumn{3}{c}{CO$_2$} & \multicolumn{3}{c}{CH$_4$} \\',
             r'    \cmidrule(lr){2-4} \cmidrule(lr){5-7}',
             r'    Pair & Median & IQR & $>0$ (\%) & Median & IQR & $>0$ (\%) \\', r'    \midrule']
    for _, r in tab.iterrows():
        cells = []
        for gas, _ in GASES:
            cells += [fmt(r[f'{gas}_median_mean'], r[f'{gas}_median_std']),
                      f"[${r[f'{gas}_q25_mean']:+.2f}$, ${r[f'{gas}_q75_mean']:+.2f}$]",
                      fmt(100 * r[f'{gas}_frac_positive_mean'], 100 * r[f'{gas}_frac_positive_std'], 0, sign=False)]
        name = r['pair'].replace(' - ', r' $-$ ')
        lines.append(f'    {name} & ' + ' & '.join(cells) + r' \\')
    nrow = '    $n$ (node $-$ linker) & ' + ' & '.join(
        rf"\multicolumn{{3}}{{c}}{{{r0:,.0f}}}" for r0 in
        [tab.loc[0, f'{g}_n_mof_mean'] for g, _ in GASES]) + r' \\'
    lines += [r'    \midrule', nrow, r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(os.path.join(args.out_dir, args.table + '.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return tab


def caption_sentence(paired):
    """sentence for the figure caption that replaces the in-panel note"""
    m = {g: paired[g][0].loc['node - linker'] for g, _ in GASES}
    return (r'The median same-MOF difference between node and linker sites is '
            rf"${m['CO2']['median']:+.2f}$~kJ\,mol$^{{-1}}$ for CO$_2$ and "
            rf"${m['CH4']['median']:+.2f}$~kJ\,mol$^{{-1}}$ for CH$_4$, positive in "
            rf"{100 * m['CO2']['frac_positive']:.0f}\% and {100 * m['CH4']['frac_positive']:.0f}\% of MOFs.")


def main():
    set_style()
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.08, hspace=0.08)
    rows, paired, shares = [], {}, {}
    for j, ((gas, tex), run) in enumerate(zip(GASES, (args.co2_run, args.ch4_run))):
        box = box_stats(run)
        p, sm, ss = share_stats(run)
        paired[gas] = paired_stats(run)
        shares[gas] = (p, sm, ss)
        heat_panel(axes[0, j], box, paired[gas][0], 'ab'[j], tex)
        share_panel(axes[1, j], p, sm, ss, 'cd'[j], tex)
        for c, b in box.items():
            rows.append({'gas': gas, 'quantity': 'site_heat', 'type': c, **b})
        for c in TYPES:
            for k, x in enumerate(p):
                rows.append({'gas': gas, 'quantity': 'share', 'type': c, 'pressure': x,
                             'mean': sm.loc[c, k], 'std': ss.loc[c, k]})
    handles = [Line2D([], [], color=COLOR[c], marker=MARKER[c], ms=4.5, lw=1.4, label=c) for c in TYPES]
    if args.legend_panel == 'top':
        fig.legend(handles=handles, loc='outside upper center', ncol=4, handlelength=2.0, columnspacing=1.6)
    else:
        ax_leg = axes[1, 'cd'.index(args.legend_panel)]
        leg = ax_leg.legend(handles=handles, loc=args.legend_loc, ncol=args.legend_ncol,
                            handlelength=2.0, columnspacing=1.2, borderaxespad=0.3)
        n_hit = legend_overlap(fig, ax_leg, leg)
        if n_hit:
            print(f'[warn] the legend in panel {args.legend_panel} ({args.legend_loc}) covers {n_hit} data points; '
                  f'try --legend_loc "lower right" / "center left" or --legend_panel c')
    fig.savefig(os.path.join(args.out_dir, args.name + '.pdf'))
    fig.savefig(os.path.join(args.out_dir, args.name + '.png'), dpi=600)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, args.name + '_data.csv'), index=False)
    esi_table(paired)
    cap = caption_sentence(paired)
    with open(os.path.join(args.out_dir, args.name + '_caption.txt'), 'w') as f:
        f.write(cap + '\n')

    pd.set_option('display.width', 200)
    for gas, _ in GASES:
        p, sm, ss = shares[gas]
        pm, ps = paired[gas]
        print(f'\n===== {gas} (mean over seeds {", ".join(args.seeds)}) =====')
        b = pd.DataFrame(box_stats(args.co2_run if gas == 'CO2' else args.ch4_run)).T
        print('--- site heat per type (kJ/mol): median, IQR, n MOFs ---')
        print(b[['med', 'med_std', 'q1', 'q3', 'n_mof']].round(2).to_string())
        print('--- mean share A_c (rows: type, columns: pressure) ---')
        t = sm.copy(); t.columns = [f'{x:g}' for x in p]
        t.loc['sum'] = t.sum()
        print(t.round(3).to_string())
        print('--- same-MOF difference to the linker ---')
        print((pm.round(3).astype(str) + ' ± ' + ps.round(3).astype(str)).to_string())
    print('\n--- caption sentence (replaces the in-panel note) ---')
    print(cap)
    print(f'\n[figure] {args.name}.pdf/.png, {args.name}_data.csv, {args.name}_caption.txt; '
          f'[table] {args.table}.csv/.tex -> {args.out_dir}')


if __name__ == '__main__':
    main()
