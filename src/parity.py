import os
import argparse

parser = argparse.ArgumentParser(description='Digital_Discovery parity plots')
parser.add_argument('--data_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym', type=str)
parser.add_argument('--output_path', default='../results_hmof', type=str)
parser.add_argument('--targets', nargs='+', default=['heat_co2', 'heat_ch4'], choices=['heat_co2', 'heat_ch4'])
parser.add_argument('--best_model_path', nargs='*',
                    default=[],
                    help='Langmuir checkpoint per target, same order as --targets; {seed} is filled with --seed '
                         '(empty: train.py naming)')
parser.add_argument('--global_model_path', nargs='*', default=[],
                    help='Global checkpoint per target for --compare_global, same order as --targets; {seed} is filled '
                         'with --seed (empty: train.py naming with --pool --attn)')
parser.add_argument('--out_dir', default='', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--encoder', default='ViSNet', type=str, choices=['SchNet', 'PaiNN', 'ViSNet'])
parser.add_argument('--pool', action='store_false')
parser.add_argument('--pool_num', default=1, type=int)
parser.add_argument('--attn', action='store_false')
parser.add_argument('--pore', action='store_false')
parser.add_argument('--occ', action='store_false')
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--mult', default=1, type=int)
parser.add_argument('--act', default='silu', type=str, choices=['elu', 'silu', 'gelu', 'relu', 'leaky_relu'])
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--mode', default='base+gp+ctx+thermo+mf', type=str, choices=['base+gp', 'base+gp+ctx', 'base+gp+ctx+thermo', 'base+gp+ctx+thermo+mf'])
parser.add_argument('--lambda_occ', default=1.0, type=float)
parser.add_argument('--lambda_aux', default=2.0, type=float)
parser.add_argument('--lambda_shape', default=1.0, type=float)
parser.add_argument('--n_head', default=4, type=int)
parser.add_argument('--num_layers', default=6, type=int)
parser.add_argument('--lmax', default=2, type=int, choices=[1, 2])
parser.add_argument('--er', default=8, type=int)
parser.add_argument('--readout', default='mean', type=str, choices=['mean', 'add'])
parser.add_argument('--heat_p', nargs='+', default=[0, -1], type=int, help='pressure indices of the Qst panels')
parser.add_argument('--load_p', nargs='*', default=[-1], type=int, help='pressure indices of the loading panels')
parser.add_argument('--compare_global', action='store_true', help='also plot the Global baseline (pool off)')
parser.add_argument('--gridsize', default=55, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from tqdm import tqdm
from utils.utils import seed_everything, PygDataset
from torch_geometric.loader import DataLoader
from model.hc_model import Base_Model, Hierarchical_Clustering_Model, Hierarchical_Clustering_Model_With_LA

TARGETS = {'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
           'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5']}
PRESSURES = {'heat_co2': [0.01, 0.05, 0.1, 0.5, 2.5],
             'heat_ch4': [0.05, 0.5, 0.9, 2.5, 4.5]}
HEAT_STD = {'heat_co2': [6.518, 6.084, 6.014, 6.282, 6.678],
            'heat_ch4': [4.917, 4.804, 4.780, 4.733, 4.684]}
LOAD_STD = {'heat_co2': [0.2098, 0.4950, 0.6960, 1.3995, 2.5644],
            'heat_ch4': [0.1070, 0.4802, 0.6532, 1.0429, 1.3295]}
REG_KEYS = ('rmse', 'r2', 'dev_mae', 'dev_r2', 'mean_mae', 'dev_std_true')
GAS_LABEL = {'co2': 'CO$_2$', 'ch4': 'CH$_4$'}
UNIT = {'heat': r'kJ mol$^{-1}$', 'load': r'mmol g$^{-1}$'}
QTY_LABEL = {'heat': r'$Q_\mathrm{st}$', 'load': 'Loading'}
INK, MUTED = '#0b0b0b', '#898781'
W2 = 7.0
CMAP = LinearSegmentedColormap.from_list('blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def target_args(target, pool):
    a = argparse.Namespace(**vars(args))
    a.target, a.pool, a.attn = target, pool, (args.attn if pool else False)
    a.pressures = PRESSURES[target]
    a.heat_std = HEAT_STD[target]
    a.load_std = LOAD_STD[target]
    a.n_out = len(TARGETS[target])
    a.gas = target.split('_')[-1]
    a.reg_keys = REG_KEYS
    return a


def checkpoint(a, i):
    given = args.best_model_path if a.pool else args.global_model_path
    if given:
        return given[i].format(seed=a.seed)
    ver = f'{a.target}_{a.encoder}_{a.lambda_aux}_{a.pool}_{a.attn}_{a.mode}_fin'
    return f'{a.output_path}/{a.encoder}/{ver}/best_{a.seed}.pt'


def build(a):
    if a.pool:
        return Hierarchical_Clustering_Model_With_LA(a) if a.attn else Hierarchical_Clustering_Model(a)
    return Base_Model(a)


def load(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    state = torch.load(path, map_location=device)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise RuntimeError(f'{path}: missing keys {res.missing_keys[:5]} ... (checkpoint and model code differ)')
    if res.unexpected_keys:
        print(f'  [load] ignored unused keys in the checkpoint: {res.unexpected_keys}')


@torch.no_grad()
def predict(model, loader, device, pool):
    model.eval()
    yq, pq, yn, pn, names = [], [], [], [], []
    for data in tqdm(loader, desc='predict'):
        data = data.to(device)
        out = model(data)
        if pool:
            heat, load_ = out[0], out[1][5]
        elif isinstance(out, (tuple, list)):
            heat, load_ = out[0], out[1]
        else:
            heat, load_ = out, None
        yq.append(data.y.cpu()); pq.append(heat.cpu())
        if load_ is not None:
            yn.append(data.y_load.cpu()); pn.append(load_.cpu())
        nm = getattr(data, 'name', None)
        names += list(nm) if isinstance(nm, (list, tuple)) else [None] * data.num_graphs
    cat = lambda xs: torch.cat(xs).numpy().astype(np.float64) if xs else None
    return {'heat': (cat(yq), cat(pq)), 'load': (cat(yn), cat(pn)) if pn else None, 'names': names}


def metrics(y, p):
    e = p - y
    return {'mae': np.abs(e).mean(), 'rmse': np.sqrt((e ** 2).mean()),
            'r2': 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()}


def per_mof_csv(res, pressures, path):
    df = pd.DataFrame({'idx': np.arange(len(res['names'])), 'name': res['names']})
    for q in ('heat', 'load'):
        if res[q] is None:
            continue
        y, p = res[q]
        for j, pr in enumerate(pressures):
            df[f'{q}_true@{pr:g}'] = y[:, j]
            df[f'{q}_pred@{pr:g}'] = p[:, j]
    df.to_csv(path, index=False)


def panels_of(res, pressures, with_load):
    out = [('heat', j) for j in args.heat_p]
    if with_load and res['load'] is not None:
        out += [('load', j) for j in args.load_p]
    return [(q, j % len(pressures)) for q, j in out]


def parity_figure(results, name, with_load):
    cols = max(len(panels_of(res, PRESSURES[t], with_load)) for t, res in results)
    fig, axes = plt.subplots(len(results), cols, figsize=(W2, 2.35 * len(results) + 0.2), squeeze=False)
    hbs = []
    for r, (target, res) in enumerate(results):
        gas, ps = target.split('_')[-1], PRESSURES[target]
        pans = panels_of(res, ps, with_load)
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(pans):
                ax.axis('off')
                continue
            q, j = pans[c]
            y, p = res[q][0][:, j], res[q][1][:, j]
            lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
            pad = 0.03 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            hb = ax.hexbin(y, p, gridsize=args.gridsize, extent=(lo, hi, lo, hi), cmap=CMAP, norm=LogNorm(), mincnt=1, linewidths=0)
            hbs.append(hb)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls='--', zorder=3)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
            m = metrics(y, p)
            ax.text(0.04, 0.96, f"MAE {m['mae']:.2f}\n$R^2$ {m['r2']:.3f}", transform=ax.transAxes, va='top', fontsize=7)
            ax.set_title(f'{QTY_LABEL[q]}, {GAS_LABEL[gas]} {ps[j]:g} bar', loc='left')
            ax.set_xlabel(f'GCMC ({UNIT[q]})')
            ax.set_ylabel(f'Predicted ({UNIT[q]})')
    vmax = max(h.get_array().max() for h in hbs)
    for h in hbs:
        h.set_norm(LogNorm(vmin=1, vmax=vmax))
    for k, ax in enumerate([a for a in axes.ravel() if a.axison]):
        ax.text(-0.3, 1.04, 'abcdefghij'[k], transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    fig.tight_layout()
    cb = fig.colorbar(hbs[-1], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cb.set_label('MOFs per bin')
    fig.savefig(os.path.join(out_dir, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, name + '.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    global out_dir
    seed_everything(args.seed)
    set_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or f'{args.output_path}/{args.encoder}/parity_seed{args.seed}'
    os.makedirs(out_dir, exist_ok=True)
    for flag, paths in (('--best_model_path', args.best_model_path), ('--global_model_path', args.global_model_path)):
        if paths and len(paths) != len(args.targets):
            raise ValueError(f'{flag} needs one checkpoint per target ({len(args.targets)}), got {len(paths)}')

    runs = [('langmuir', True)] + ([('global', False)] if args.compare_global else [])
    results = {k: [] for k, _ in runs}
    rows = []
    for i, target in enumerate(args.targets):
        dataset = PygDataset(root=f'{args.data_path}/test', target_name=TARGETS[target], gas=target.split('_')[-1])
        if len(dataset) == 0:
            raise FileNotFoundError(f'empty test set: {os.path.abspath(args.data_path)}/test')
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
        for kind, pool in runs:
            a = target_args(target, pool)
            path = checkpoint(a, i)
            print(f'[parity] {target} {kind}: {path} ({len(dataset)} test MOFs)')
            model = build(a).to(device)
            load(model, path, device)
            res = predict(model, loader, device, pool)
            results[kind].append((target, res))
            per_mof_csv(res, a.pressures, os.path.join(out_dir, f'parity_{target}_{kind}.csv'))
            for q in ('heat', 'load'):
                if res[q] is None:
                    continue
                y, p = res[q]
                for j, pr in enumerate(a.pressures):
                    rows.append({'target': target, 'model': kind, 'quantity': q, 'pressure_bar': pr, **metrics(y[:, j], p[:, j])})
                rows.append({'target': target, 'model': kind, 'quantity': q, 'pressure_bar': 'mean',
                             **{k: np.mean([metrics(y[:, j], p[:, j])[k] for j in range(y.shape[1])]) for k in ('mae', 'rmse', 'r2')}})
            del model
            torch.cuda.empty_cache()

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(out_dir, 'parity_metrics.csv'), index=False)
    parity_figure(results['langmuir'], 'fig_parity', with_load=True)
    if args.compare_global:
        parity_figure(results['global'], 'fig_parity_global', with_load=False)

    print('\n[parity] MAE / RMSE / R2 (mean over pressures; MAE should match test_heat / test_load of the test csv)')
    print(met[met['pressure_bar'] == 'mean'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\n[parity] saved to {out_dir}: fig_parity' + (', fig_parity_global' if args.compare_global else '')
          + ', parity_metrics.csv, parity_<target>_<model>.csv')


if __name__ == '__main__':
    main()
