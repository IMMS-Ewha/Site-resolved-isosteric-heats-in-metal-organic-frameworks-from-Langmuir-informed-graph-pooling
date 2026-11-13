"""
Symmetry check of the Langmuir-informed pooling model: do crystallographically equivalent atoms get the same
embedding, the same kind of site and the same site values, and is Qst unchanged under rotation and origin shift?

  equivalent atoms : spglib equivalent_atoms of each test structure (symprec, default 0.01 A)
  levels           : r = within-class sum of squares / total sum of squares over atoms in classes with >= 2 members
                     (0 = equivalent atoms identical, ~1 = no better than random)
                       r_h     encoder embedding h_i
                       r_site  heat of the site the atom is assigned to (argmax_k S_ik)
                       r_e     atom-level site heat e_i (active sites, as in site_analysis)
                       r_q_lo  atom-level loading q_i at the lowest pressure
                       r_q_hi  atom-level loading q_i at the highest pressure
                     type_agree = share of equivalence classes whose atoms all sit on sites of the same type
  global test      : random rotations and origin shifts of the whole structure (graph kept, vectors rotated)
                       dQst_max  max |change of Qst| over pressures and trials (kJ/mol)
                       dh_rel    max |change of h| / mean |h|

From analyze.py (step 'symmetry'):
    from utils.symmetry_check import check_symmetry
    check_symmetry(model, args.best_model_path, test_loader, device, args)
Standalone, from src/ (same arguments as train.py; the checkpoint path is rebuilt as in train.py):
    python -m utils.symmetry_check --target heat_co2 --seed 42
    python -m utils.symmetry_check --target heat_co2 --seed 42 --encoder PaiNN
Compare runs (after the runs above):
    python -m utils.symmetry_check --compare ViSNet=<dir>/symmetry_check PaiNN=<dir>/symmetry_check --out_dir ./symmetry_compare
Redraw the figures of a finished run from its csv (no model, no GPU):
    python -m utils.symmetry_check --replot <dir>/symmetry_check

Outputs (out_dir = dirname(best_model_path)/symmetry_check, or --out_dir)
  csv : symmetry_mof, symmetry_class, symmetry_summary   (+ symmetry_compare with --compare)
  fig : fig_symmetry_levels, fig_symmetry_rotation       (+ fig_symmetry_compare with --compare)
        dotted line r = 1e-6 (R_THR) with the share of MOFs below it above each column; dashed line r = 1
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spglib
import torch
from tqdm import tqdm

from site_analysis import (COLOR, INDEX_TO_ELEMENT, INK, MUTED, S_MAX, W1, _load_weights, edge_lengths,
                           functional_groups, neighbours, per_graph_assignment, save, set_style, site_types)

LEVELS = {'r_h': r'embedding $h_i$', 'r_site': 'assigned site', 'r_e': r'site heat $e_i$',
          'r_q_lo': r'$q_i$, lowest $p$', 'r_q_hi': r'$q_i$, highest $p$'}
R_FLOOR = 1e-8
R_THR = 1e-6
R_TOP = 20.0
SYMPRECS = (1e-3, 1e-2, 1e-1)


def equivalent_atoms(cell, pos, numbers, symprec):
    if cell is None:
        return np.arange(len(pos)), 'none'
    frac = pos @ np.linalg.inv(cell)
    frac = frac - np.floor(frac)
    ds = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=symprec)
    if ds is None:
        return np.arange(len(pos)), 'none'
    eq = ds.equivalent_atoms if hasattr(ds, 'equivalent_atoms') else ds['equivalent_atoms']
    sg = ds.international if hasattr(ds, 'international') else ds['international']
    return np.asarray(eq), str(sg)


def classes(eq):
    return [np.where(eq == c)[0] for c in np.unique(eq) if (eq == c).sum() >= 2]


def within_ratio(x, groups):
    if not groups:
        return np.nan
    x = np.asarray(x, float).reshape(len(x), -1)
    idx = np.concatenate(groups)
    ok = np.isfinite(x[idx]).all(1)
    if ok.sum() < 2:
        return np.nan
    xs = x[idx][ok]
    tot = ((xs - xs.mean(0)) ** 2).sum()
    wit = 0.0
    for g in groups:
        xg = x[g][np.isfinite(x[g]).all(1)]
        if len(xg) >= 2:
            wit += ((xg - xg.mean(0)) ** 2).sum()
    if tot <= 1e-20:
        return 0.0
    return float(wit / tot)


def max_within_std(x, groups):
    s = [np.nanstd(x[g]) for g in groups if np.isfinite(x[g]).sum() >= 2]
    return float(np.max(s)) if s else np.nan


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def transform(data, R, t):
    d = data.clone()
    R = torch.as_tensor(R, dtype=d.pos.dtype, device=d.pos.device)
    t = torch.as_tensor(t, dtype=d.pos.dtype, device=d.pos.device)
    d.pos = d.pos @ R.T + t
    if getattr(d, 'cell', None) is not None:
        shape = d.cell.shape
        d.cell = (d.cell.reshape(-1, 3, 3) @ R.T).reshape(shape)
    if getattr(d, 'edge_vec', None) is not None:
        d.edge_vec = d.edge_vec @ R.T
    return d


@torch.no_grad()
def collect(model, loader, device, args, names=None):
    P = len(args.pressures)
    s_max = S_MAX.get(args.gas, np.inf)
    dead_tol = getattr(args, 'dead_tol', 0.05)
    symprec = getattr(args, 'symprec', 1e-2)
    n_rot = getattr(args, 'n_rot', 3)
    max_mofs = getattr(args, 'max_mofs', 0) or np.inf
    rng = np.random.default_rng(getattr(args, 'seed', 42))
    to_np = lambda t: t.detach().cpu().numpy()

    model.eval()
    cache = {}
    h1 = model.encoder.register_forward_hook(
        lambda mod, inp, out: cache.__setitem__('h', out[0] if isinstance(out, (tuple, list)) else out))
    h2 = model.sites.register_forward_hook(lambda mod, inp, out: cache.__setitem__('sites', out))
    mof_rows, class_rows = [], []
    k = 0
    try:
        for data in tqdm(loader, desc='symmetry check'):
            if k >= max_mofs:
                break
            data = data.to(device)
            out = model(data)
            h_all = cache['h'].detach().clone()
            log_n, theta = cache['sites'][0], cache['sites'][2]
            pred, e, m, alpha = out[0], out[1][0], out[2], out[3]
            B = pred.shape[0]
            mf = m.to(pred.dtype)
            n = torch.exp(log_n) * mf
            q = n.unsqueeze(-1) * theta
            log_K = cache['sites'][1]
            dS = 8.314462618 * (log_K - e / (8.314462618e-3 * 298.0))
            assign = per_graph_assignment(out[5] if len(out) > 5 else None, data.batch, B)
            batch = to_np(data.batch)

            dq_max, dh_max = np.zeros(B), np.zeros(B)
            for _ in range(n_rot):
                R = random_rotation(rng)
                t = rng.normal(scale=5.0, size=3)
                out2 = model(transform(data, R, t))
                h_rot = cache['h']
                dq = to_np((out2[0] - pred).abs().amax(1))
                dh = to_np((h_rot - h_all).abs().amax(1))
                hn = to_np(h_all.abs().mean(1))
                for b in range(B):
                    sel = batch == b
                    dq_max[b] = max(dq_max[b], dq[b])
                    dh_max[b] = max(dh_max[b], dh[sel].max() / max(hn[sel].mean(), 1e-12))

            ee, aa, qq, dd = (to_np(v).astype(np.float64) for v in (e, alpha, q, dS))
            mm = to_np(m.bool())
            hh = to_np(h_all).astype(np.float64)
            pos_all = to_np(data.pos).astype(np.float64)
            cells = to_np(data.cell.reshape(B, 3, 3)).astype(np.float64) if getattr(data, 'cell', None) is not None else [None] * B
            elen = to_np(edge_lengths(data))
            g_of_edge = to_np(data.batch[data.edge_index[0]])
            ptr = to_np(data.ptr)
            dnames = getattr(data, 'name', None)

            for b in range(B):
                if k >= max_mofs:
                    break
                if names is not None:
                    name = str(names[k])
                elif isinstance(dnames, (list, tuple)):
                    name = str(dnames[b])
                else:
                    name = str(k)
                sel = batch == b
                xi = to_np(data.x[data.batch == b].view(-1)).astype(int)
                atom = np.array([INDEX_TO_ELEMENT.get(int(x), 'X') for x in xi])
                pos, cell, h = pos_all[sel], cells[b], hh[sel]

                eq, sg = equivalent_atoms(cell, pos, xi, symprec)
                groups = classes(eq)
                n_cls = {f'n_classes@{s:g}': len(classes(equivalent_atoms(cell, pos, xi, s)[0])) for s in SYMPRECS}

                fr = mm[b, :-1]
                e_s, a_s, q_s, dS_s = (v[b, :-1][fr] for v in (ee, aa, qq, dd))
                active = (dS_s > -(1.0 - dead_tol) * s_max) & (a_s.max(1) >= 1e-3)
                S = assign[b]
                row = {'idx': k, 'name': name, 'spacegroup': sg, 'n_atoms': len(atom), 'n_classes': len(groups),
                       'n_atoms_in_classes': int(sum(len(g) for g in groups)), **n_cls,
                       'dQst_max': dq_max[b], 'dh_rel': dh_max[b]}
                if S is not None and S.shape[1] in (len(fr), len(fr) + 1):
                    S = S[:, :len(fr)][:, fr]
                    ei_b = to_np(data.edge_index[:, torch.as_tensor(g_of_edge == b, device=data.edge_index.device)]) - ptr[b]
                    fg = functional_groups(atom, neighbours(atom, ei_b, elen[g_of_edge == b]))
                    stype, own = site_types(atom, fg, S)
                    Sl = S[:, active] / np.clip(S[:, active].sum(1, keepdims=True), 1e-12, None)
                    e_i = Sl @ e_s[active] if active.any() else np.full(len(atom), np.nan)
                    q_i = (S / np.clip(S.sum(0, keepdims=True), 1e-12, None)) @ q_s
                    site_e = e_s[own]
                    row.update({'r_h': within_ratio(h, groups), 'r_site': within_ratio(site_e, groups),
                                'r_e': within_ratio(e_i, groups), 'r_q_lo': within_ratio(q_i[:, 0], groups),
                                'r_q_hi': within_ratio(q_i[:, P - 1], groups),
                                'type_agree': float(np.mean([len(set(stype[own[g]])) == 1 for g in groups])) if groups else np.nan,
                                'std_e_max': max_within_std(e_i, groups)})
                    for c, g in enumerate(groups):
                        hg = h[g]
                        class_rows.append({'idx': k, 'name': name, 'class': c, 'element': atom[g[0]], 'size': len(g),
                                           'types': '+'.join(sorted(set(stype[own[g]]))),
                                           'n_sites': len(set(own[g])),
                                           'h_spread_rel': float(np.abs(hg - hg.mean(0)).max() / max(np.abs(h).mean(), 1e-12)),
                                           'std_site_e': float(np.std(site_e[g])), 'std_e_i': float(np.nanstd(e_i[g])),
                                           'std_q_lo': float(np.std(q_i[g, 0])), 'std_q_hi': float(np.std(q_i[g, P - 1]))})
                else:
                    row.update({'r_h': within_ratio(h, groups)})
                mof_rows.append(row)
                k += 1
    finally:
        h1.remove(); h2.remove()
    return pd.DataFrame(mof_rows), pd.DataFrame(class_rows)


def summarize(mofs):
    sym = mofs[mofs['n_classes'] > 0]
    rows = {'n_mof': len(mofs), 'n_mof_with_equivalent_atoms': len(sym)}
    for c in list(LEVELS) + ['type_agree', 'std_e_max']:
        if c in sym:
            v = sym[c].dropna()
            rows.update({f'{c}_median': v.median(), f'{c}_q90': v.quantile(0.9), f'{c}_max': v.max()})
    for c in LEVELS:
        if c in sym:
            rows[f'frac_{c}_below_{R_THR:g}'] = sym[c].dropna().lt(R_THR).mean()
    rows.update({'dQst_max_median': mofs['dQst_max'].median(), 'dQst_max_max': mofs['dQst_max'].max(),
                 'dh_rel_median': mofs['dh_rel'].median(), 'dh_rel_max': mofs['dh_rel'].max()})
    return pd.Series(rows)


def _share_below(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.mean(v < R_THR)) if len(v) else np.nan


def _threshold(ax):
    ax.set_yscale('log')
    ax.set_ylim(R_FLOOR / 3, R_TOP)
    ax.axhline(1, color=MUTED, lw=0.6, ls='--')
    ax.axhline(R_THR, color=MUTED, lw=0.6, ls=':')
    ax.text(1.0, R_THR, r' $10^{-6}$', transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=6,
            color=MUTED, clip_on=False)
    ax.set_ylabel('Within-class / total variance')


def _box(ax, data, labels, colors, shares):
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor(INK); patch.set_linewidth(0.6)
    rng = np.random.default_rng(0)
    for i, (d, c, sh) in enumerate(zip(data, colors, shares)):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=4, color=c, edgecolors='none', alpha=0.6, zorder=3)
        ax.text(i + 1, 2.5, f'{100 * sh:.1f}%', ha='center', va='bottom', fontsize=6, color=INK)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def fig_levels(mofs, out_dir):
    cols = [c for c in LEVELS if c in mofs]
    sym = mofs[mofs['n_classes'] > 0]
    raw = [sym[c].dropna().values for c in cols]
    data = [np.clip(v, R_FLOOR, None) for v in raw]
    fig, ax = plt.subplots(figsize=(W1 + 0.6, 2.6))
    _box(ax, data, [LEVELS[c] for c in cols], [COLOR['node']] * len(cols), [_share_below(v) for v in raw])
    _threshold(ax)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    save(fig, out_dir, 'fig_symmetry_levels')
    plt.close(fig)
    print(f'[fig_symmetry_levels] n = {len(sym)} MOFs with equivalent atoms; values below {R_FLOOR:g} drawn at '
          f'{R_FLOOR:g}; share below {R_THR:g}: '
          + ', '.join(f'{c} {100 * _share_below(v):.1f}%' for c, v in zip(cols, raw)))


def fig_rotation(mofs, out_dir):
    v = np.clip(mofs['dQst_max'].values, 1e-8, None)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.hist(np.log10(v), bins=40, color=COLOR['node'], edgecolor='white', lw=0.4)
    ax.set_xlabel(r'$\log_{10}$ max $|\Delta Q_\mathrm{st}|$ under rotation and origin shift (kJ mol$^{-1}$)')
    ax.set_ylabel('MOFs')
    save(fig, out_dir, 'fig_symmetry_rotation')
    plt.close(fig)


def compare(specs, out_dir):
    set_style()
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for s in specs:
        label, path = s.split('=', 1) if '=' in s else (os.path.basename(os.path.normpath(s)), s)
        runs.append((label, pd.read_csv(os.path.join(path, 'symmetry_mof.csv'))))
    rows = []
    for label, df in runs:
        r = summarize(df)
        r['run'] = label
        rows.append(r)
    table = pd.DataFrame(rows).set_index('run')
    table.to_csv(os.path.join(out_dir, 'symmetry_compare.csv'))
    cols = [c for c in LEVELS if all(c in df for _, df in runs)]
    palette = [COLOR['node'], COLOR['linker'], COLOR['functional'], COLOR['pore'], COLOR['framework']]
    fig, ax = plt.subplots(figsize=(max(W1, 1.1 * len(cols) * len(runs) * 0.45 + 1), 2.6))
    w = 0.8 / len(runs)
    for j, (label, df) in enumerate(runs):
        sym = df[df['n_classes'] > 0]
        raw = [sym[c].dropna().values for c in cols]
        data = [np.clip(v, R_FLOOR, None) for v in raw]
        pos = np.arange(len(cols)) + (j - (len(runs) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for patch in bp['boxes']:
            patch.set_facecolor(palette[j % len(palette)]); patch.set_alpha(0.75); patch.set_edgecolor(INK); patch.set_linewidth(0.6)
        rng = np.random.default_rng(j)
        for x0, d, v in zip(pos, data, raw):
            ax.scatter(x0 + rng.uniform(-w * 0.3, w * 0.3, len(d)), d, s=4, color=palette[j % len(palette)],
                       edgecolors='none', alpha=0.6, zorder=3)
            ax.text(x0, 2.5, f'{100 * _share_below(v):.0f}%', ha='center', va='bottom', fontsize=5.5, color=INK)
        ax.plot([], [], 's', color=palette[j % len(palette)], label=label)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LEVELS[c] for c in cols], rotation=20, ha='right')
    _threshold(ax)
    ax.legend(ncol=len(runs), loc='lower center', bbox_to_anchor=(0.5, 1.08))
    save(fig, out_dir, 'fig_symmetry_compare')
    plt.close(fig)
    pd.set_option('display.width', 200)
    print(table.T.to_string(float_format=lambda v: f'{v:.3g}'))
    return table


def report(mofs, cls, out_dir):
    set_style()
    os.makedirs(out_dir, exist_ok=True)
    mofs.to_csv(os.path.join(out_dir, 'symmetry_mof.csv'), index=False)
    cls.to_csv(os.path.join(out_dir, 'symmetry_class.csv'), index=False)
    summary = summarize(mofs)
    summary.to_frame('value').to_csv(os.path.join(out_dir, 'symmetry_summary.csv'))
    fig_levels(mofs, out_dir)
    fig_rotation(mofs, out_dir)
    print(f"\n[symmetry check] {len(mofs)} MOFs -> {out_dir}")
    print(summary.to_string(float_format=lambda v: f'{v:.3g}'))
    if len(cls):
        worst = cls.sort_values('std_e_i', ascending=False).head(10)
        print('\n--- classes with the largest spread of e_i (kJ/mol) ---')
        print(worst[['name', 'element', 'size', 'types', 'n_sites', 'h_spread_rel', 'std_e_i', 'std_site_e']]
              .to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    return summary


def replot(run_dir, out_dir=None):
    set_style()
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    mofs = pd.read_csv(os.path.join(run_dir, 'symmetry_mof.csv'))
    fig_levels(mofs, out_dir)
    fig_rotation(mofs, out_dir)
    print(f'[symmetry check] figures redrawn from {run_dir} -> {out_dir}')


def check_symmetry(model, best_model_path, loader, device, args, load_weights=True, names=None, out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(best_model_path), 'symmetry_check')
    if load_weights:
        _load_weights(model, best_model_path, device)
    mofs, cls = collect(model, loader, device, args, names)
    return report(mofs, cls, out_dir)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='symmetry check')
    ap.add_argument('--desc', default="base", type=str)
    ap.add_argument('--data_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym', type=str)
    ap.add_argument('--output_path', default='../results_hmof', type=str)
    ap.add_argument('--target', default='heat_co2', type=str, choices=['heat_co2', 'heat_ch4'])
    ap.add_argument('--gpu', default='0', type=str)
    ap.add_argument('--seed', default=42, type=int)
    ap.add_argument('--bs', default=64, type=int)
    ap.add_argument('--encoder', default='ViSNet', type=str, choices=['SchNet', 'PaiNN', 'ViSNet'])
    ap.add_argument('--pool', action='store_false')
    ap.add_argument('--pool_num', default=1, type=int)
    ap.add_argument('--attn', action='store_false')
    ap.add_argument('--pore', action='store_false')
    ap.add_argument('--occ', action='store_false')
    ap.add_argument('--d_model', default=64, type=int)
    ap.add_argument('--mult', default=1, type=int)
    ap.add_argument('--act', default='silu', type=str, choices=['elu', 'silu', 'gelu', 'relu', 'leaky_relu'])
    ap.add_argument('--drop_rate', default=0.0, type=float)
    ap.add_argument('--eps', default=1e-8, type=float)
    ap.add_argument('--mode', default='base+gp+ctx+thermo+mf', type=str, choices=['base+gp', 'base+gp+ctx', 'base+gp+ctx+thermo', 'base+gp+ctx+thermo+mf'])
    ap.add_argument('--lambda_occ', default=1.0, type=float)
    ap.add_argument('--lambda_aux', default=2.0, type=float)
    ap.add_argument('--lambda_shape', default=1.0, type=float)
    ap.add_argument('--n_head', default=4, type=int)
    ap.add_argument('--num_layers', default=6, type=int)
    ap.add_argument('--lmax', default=2, type=int, choices=[1, 2])
    ap.add_argument('--er', default=8, type=int)
    ap.add_argument('--readout', default='mean', type=str, choices=['mean', 'add'])
    ap.add_argument('--best_model_path', default='', type=str,
                    help='empty: <output_path>/<encoder>/<target>_<encoder>_<lambda_aux>_<pool>_<attn>_<mode>_fin/best_<seed>.pt as in train.py')
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--symprec', default=1e-2, type=float, help='A, spglib tolerance for equivalent atoms')
    ap.add_argument('--n_rot', default=3, type=int, help='random rotations + origin shifts per MOF')
    ap.add_argument('--max_mofs', default=0, type=int, help='0: all test MOFs')
    ap.add_argument('--dead_tol', default=0.05, type=float)
    ap.add_argument('--compare', nargs='*', default=[], help='LABEL=symmetry_check_dir ...: compare finished runs only')
    ap.add_argument('--replot', default='', type=str, help='symmetry_check_dir: redraw its figures from the csv only')
    args = ap.parse_args()
    if args.compare:
        compare(args.compare, args.out_dir or './symmetry_compare')
        raise SystemExit
    if args.replot:
        replot(args.replot, args.out_dir or None)
        raise SystemExit
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    TARGETS = {'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
               'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5']}
    args.pressures = [float(t.split('_')[-1]) for t in TARGETS[args.target]]
    args.n_out = len(args.pressures)
    args.gas = args.target.split('_')[-1]
    if not args.best_model_path:
        ver = f'{args.target}_{args.encoder}_{args.lambda_aux}_{args.pool}_{args.attn}_{args.mode}_fin'
        args.best_model_path = os.path.join(args.output_path, args.encoder, ver, f'best_{args.seed}.pt')
    print(f'[symmetry check] checkpoint: {args.best_model_path}')

    from torch_geometric.loader import DataLoader
    from utils.utils import seed_everything, PygDataset
    from model.hc_model import Hierarchical_Clustering_Model_With_LA
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DataLoader(PygDataset(root=f'{args.data_path}/test', target_name=TARGETS[args.target], gas=args.gas),
                        batch_size=args.bs, shuffle=False)
    model = Hierarchical_Clustering_Model_With_LA(args).to(device)
    check_symmetry(model, args.best_model_path, loader, device, args, out_dir=args.out_dir or None)
