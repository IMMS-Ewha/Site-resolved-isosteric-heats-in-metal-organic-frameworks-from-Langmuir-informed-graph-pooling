"""
Reproduction of the pressure dependence of Qst: Global vs Langmuir readout, mean +- std over seeds.
Reads the per-MOF csv files written by parity.py (run it once per seed with --compare_global), no model needed.

    python parity.py --compare_global --seed 42 --global_model_path <CO2 global>/best_{seed}.pt <CH4 global>/best_{seed}.pt
    (repeat for --seed 43, 44)
    python pressure_dependence.py                                    # parity_seed* folders under the default root
    python pressure_dependence.py --parity_dirs DIR_SEED42 DIR_SEED43 DIR_SEED44

Metrics per gas, model and seed (test set):
  r[a->b]      Pearson correlation between predicted and GCMC changes of Qst from pressure a to b,
               for each adjacent pair of pressures and for the full range (lowest -> highest)
  drop recall  among MOFs whose GCMC Qst falls by more than --drop_tol kJ/mol over the last (highest-pressure)
               interval, the share that is also predicted to fall (predicted change < 0)
  drop precision (csv only) among MOFs predicted to fall over that interval, the share whose GCMC Qst also falls
               by more than --drop_tol
Outputs (out_dir): pressure_dependence_seeds.csv, pressure_dependence_summary.csv, table_pressure_dependence.tex
"""
import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description='pressure dependence of Qst, Global vs Langmuir')
ap.add_argument('--parity_dirs', nargs='*', default=['../results_hmof/ViSNet/parity_seed*'],
                help='parity.py output folders (one per seed) or glob patterns')
ap.add_argument('--targets', nargs='+', default=['heat_co2', 'heat_ch4'])
ap.add_argument('--models', nargs='+', default=['global', 'langmuir'])
ap.add_argument('--drop_tol', default=1.0, type=float, help='kJ/mol, GCMC fall that counts as a drop')
ap.add_argument('--out_dir', default='../results_hmof/ViSNet/pressure_dependence', type=str)
args = ap.parse_args()

GAS_TEX = {'heat_co2': 'CO$_2$', 'heat_ch4': 'CH$_4$'}
MODEL_TEX = {'global': 'Global', 'langmuir': 'Langmuir'}


def folders(patterns):
    out = []
    for pat in patterns:
        for d in sorted(glob(pat)):
            if os.path.isdir(d) and d not in out:
                out.append(d)
    if not out:
        raise SystemExit(f'no parity folders found for {patterns}')
    return out


def load(folder, target, model):
    f = os.path.join(folder, f'parity_{target}_{model}.csv')
    if not os.path.isfile(f):
        raise FileNotFoundError(f'{f} not found (run parity.py with --compare_global for this seed)')
    df = pd.read_csv(f)
    pl = sorted([c.split('@')[1] for c in df.columns if c.startswith('heat_true@')], key=float)
    return df[[f'heat_true@{x}' for x in pl]].values, df[[f'heat_pred@{x}' for x in pl]].values, pl


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def metrics(y, p, pl):
    out = {}
    pairs = [(j, j + 1) for j in range(len(pl) - 1)] + [(0, len(pl) - 1)]
    for a, b in pairs:
        out[f'r {pl[a]}->{pl[b]}'] = pearson(p[:, b] - p[:, a], y[:, b] - y[:, a])
    dt, dp = y[:, -1] - y[:, -2], p[:, -1] - p[:, -2]
    drop, pred_drop = dt < -args.drop_tol, dp < 0
    out['drop recall'] = float(pred_drop[drop].mean()) if drop.any() else np.nan
    out['drop precision'] = float(drop[pred_drop].mean()) if pred_drop.any() else np.nan
    out['n drop (GCMC)'] = int(drop.sum())
    out['n MOF'] = len(y)
    return out


def tex_rows(summ, target, pl):
    lines = []
    keys = [f'r {pl[j]}->{pl[j + 1]}' for j in range(len(pl) - 1)] + [f'r {pl[0]}->{pl[-1]}', 'drop recall']
    for k in keys:
        cells = {m: summ.loc[(target, m, k)] for m in args.models}
        best = max(cells, key=lambda m: cells[m]['mean'])
        txt = []
        for m in args.models:
            mu, sd = cells[m]['mean'], cells[m]['std']
            v = f'{mu:.3f}'
            txt.append((f'\\textbf{{{v}}}' if m == best and len(args.models) > 1 else v) + f'${{\\scriptstyle \\pm {sd:.3f}}}$')
        label = (f'$r$, {k[2:].split("->")[0]}$\\rightarrow${k[2:].split("->")[1]} bar' if k.startswith('r ')
                 else 'Drop recall')
        lines.append(f'    {label:<32s} & ' + ' & '.join(txt) + r' \\')
    return lines


def main():
    dirs = folders(args.parity_dirs)
    os.makedirs(args.out_dir, exist_ok=True)
    rows, plist, ref = [], {}, {}
    for d in dirs:
        for t in args.targets:
            for m in args.models:
                y, p, pl = load(d, t, m)
                plist[t] = pl
                if t in ref and (ref[t].shape != y.shape or not np.allclose(ref[t], y, atol=1e-4)):
                    raise SystemExit(f'GCMC values differ between files for {t} ({d}, {m}): not the same test set/order')
                ref.setdefault(t, y)
                for k, v in metrics(y, p, pl).items():
                    rows.append({'target': t, 'model': m, 'seed_dir': os.path.basename(os.path.normpath(d)),
                                 'metric': k, 'value': v})
    seeds = pd.DataFrame(rows)
    seeds.to_csv(os.path.join(args.out_dir, 'pressure_dependence_seeds.csv'), index=False)
    summ = (seeds.groupby(['target', 'model', 'metric'])['value']
            .agg(mean='mean', std=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n_seeds='count'))
    summ.to_csv(os.path.join(args.out_dir, 'pressure_dependence_summary.csv'))

    body = []
    for i, t in enumerate(args.targets):
        if i:
            body.append(r'    \midrule')
        body.append(rf'    \multicolumn{{{1 + len(args.models)}}}{{l}}{{\textit{{{GAS_TEX.get(t, t)}}}}} \\')
        body += tex_rows(summ, t, plist[t])
    n_drop = {t: int(summ.loc[(t, args.models[0], 'n drop (GCMC)'), 'mean']) for t in args.targets}
    caption = (r'\ Reproduction of the pressure dependence of $Q_\mathrm{st}$ with the ViSNet encoder. $r$ is the Pearson '
               r'correlation between the predicted and GCMC changes of $Q_\mathrm{st}$ over each pressure interval; the last '
               r'interval spans the full pressure range. Drop recall is the fraction of MOFs whose GCMC $Q_\mathrm{st}$ '
               rf'decreases by more than {args.drop_tol:g}~kJ\,mol$^{{-1}}$ in the highest-pressure interval '
               + '(' + ', '.join(f'{n_drop[t]} MOFs for {GAS_TEX.get(t, t)}' for t in args.targets) + ') '
               r'that are also predicted to decrease. Global and Langmuir as in Table~\ref{tab:mae_qst}. Values are mean '
               r'$\pm$ standard deviation over three seeds; the better value is shown in bold.')
    tex = '\n'.join([r'\begin{table}[ht]', r'\small', f'  \\caption{{{caption}}}', r'  \label{tab:pressure_dependence}',
                     r'  \begin{tabular*}{0.48\textwidth}{@{\extracolsep{\fill}}l' + 'c' * len(args.models) + '}',
                     r'    \toprule', '    Metric & ' + ' & '.join(MODEL_TEX.get(m, m) for m in args.models) + r' \\',
                     r'    \midrule', *body, r'    \bottomrule', r'  \end{tabular*}', r'\end{table}'])
    with open(os.path.join(args.out_dir, 'table_pressure_dependence.tex'), 'w') as f:
        f.write(tex + '\n')

    pd.set_option('display.width', 200)
    print(f'[pressure dependence] {len(dirs)} seed folder(s): {", ".join(os.path.basename(os.path.normpath(d)) for d in dirs)}')
    show = summ.reset_index()
    show['value'] = show.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if np.isfinite(r['std']) else f"{r['mean']:.3f}", axis=1)
    for t in args.targets:
        print(f'\n--- {t} ---')
        print(show[show['target'] == t].pivot(index='metric', columns='model', values='value').to_string())
    print(f'\n[pressure dependence] saved to {args.out_dir}: pressure_dependence_seeds.csv, '
          f'pressure_dependence_summary.csv, table_pressure_dependence.tex')


if __name__ == '__main__':
    main()
