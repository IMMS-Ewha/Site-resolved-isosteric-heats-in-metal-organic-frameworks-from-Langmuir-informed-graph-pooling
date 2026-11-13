"""
Site-level interpretation of the Langmuir-informed pooling model (Methods: Site-level interpretation;
Computational details: Site identification and analysis).

  (i)   site types      : A_c(p) = sum_{k in c} alpha_k(p), C_c(p) = sum_{k in c} alpha_k(p) e_k,
                          mean e of the active sites of each type (node / linker / functional / pore)
  (ii)  atom projection : e_i = sum_k S_ik e_k / sum_k S_ik (active sites), q_i(p) = sum_k S_ik / sum_j S_jk q_k(p)
                          for selected MOFs -> figure + .extxyz / .pdb (B-factor) for VESTA / OVITO / PyMOL
  (iii) dQst split      : dQ = (1 - mean a_pore) dF + d a_pore (e_pore - mean F) + du   (exact; adjacent pressures
                          and p_min -> p_max), grouped by the predicted trend p_min -> p_max
  (iv)  thermodynamics  : within-MOF Spearman(e, dS) and OLS slope dS/de of the active framework sites
                          (constant K <=> slope -1/T; compensation temperature T_c = -1/slope), dS distribution,
                          Henry constant H_k = n_k K_k vs e_k (rank correlation, top-site agreement, alpha(p_min) check)

From analyze.py (step 'site'):
    from site_analysis import analyze_sites
    analyze_sites(model, args.best_model_path, test_loader, device, args)        # optional: args.viz = ['name' or index]
Standalone (from src/, same arguments as train.py; the checkpoint path is rebuilt as in train.py):
    python site_analysis.py --target heat_co2 --seed 42
    python site_analysis.py ... --best_model_path .../best_42.pt
    python site_analysis.py ... --pt_path /path/to/pt_folder --viz "hMOF-1003160-(id:55390)"
    python site_analysis.py --target heat_co2 --from_csv .../site_analysis      # redo tables/figures from saved csv

Outputs (out_dir = dirname(best_model_path)/site_analysis_<seed>, .../site_analysis_<pt folder>_<seed> with --pt_path, or --out_dir)
  csv : sites, mof_types, mof_summary, type_share, type_heat, type_heat_paired, decomposition_mof, decomposition_summary, decomposition_by_trend,
        thermo_mof, thermo_summary, atoms_<name>  (+ <name>.extxyz, <name>_e.pdb, <name>_q<p>.pdb)
  fig : fig_type_share, fig_type_heat, fig_type_contribution, fig_dqst_intervals, fig_dqst_trend,
        fig_compensation, fig_rho_e_dS, fig_dS, fig_henry_heat, fig_atoms_<name>   (.pdf + .png 600 dpi, one per file)
"""
import itertools
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from tqdm import tqdm

R_J = 8.314462618                                   # J/mol/K
RT = R_J * 1e-3 * 298.0                             # kJ/mol
S_MAX = {'co2': 210.8, 'ch4': 143.5}                # J/mol/K
INDEX_TO_ELEMENT = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F', 6: 'Cl', 7: 'V', 8: 'Cu', 9: 'Zn', 10: 'Br', 11: 'Zr'}
COV_R = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20,
         'V': 1.53, 'Cu': 1.32, 'Zn': 1.22, 'Zr': 1.75}
METALS = {'V', 'Cu', 'Zn', 'Zr'}
TYPES = ['node', 'linker', 'functional', 'pore']
COLOR = {'node': '#2a78d6', 'linker': '#eb6834', 'functional': '#1baf7a', 'pore': '#eda100',
         'framework': '#6250d6', 'share': '#eda100', 'interaction': '#52514e', 'gcmc': '#e34948'}
INK, MUTED = '#0b0b0b', '#898781'
W1, W2 = 3.35, 7.0
TERMS = {'framework': 'framework', 'pore_share_shift': 'pore share shift', 'interaction': 'interaction'}
TERM_COLOR = {'framework': COLOR['framework'], 'pore_share_shift': COLOR['share'], 'interaction': COLOR['interaction']}


def set_style():
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, name + '.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def _load_weights(model, path, device):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ['model_state_dict', 'state_dict', 'model']:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    model.load_state_dict(state)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok])[0])


def per_graph_assignment(sa, batch, B):
    if isinstance(sa, (list, tuple)) and len(sa) > 0:
        sa = sa[0]
    if not torch.is_tensor(sa):
        return [None] * B
    if sa.dim() == 2 and sa.shape[0] == batch.shape[0]:
        return [sa[batch == b].detach().cpu().numpy() for b in range(B)]
    if sa.dim() == 3 and sa.shape[0] == B:
        cnt = torch.bincount(batch, minlength=B).tolist()
        return [sa[b, :cnt[b]].detach().cpu().numpy() for b in range(B)]
    return [None] * B


def edge_lengths(data):
    if getattr(data, 'edge_vec', None) is not None:
        return data.edge_vec.norm(dim=-1)
    src, dst = data.edge_index
    vec = data.pos[src] - data.pos[dst]
    off, cell = getattr(data, 'offsets', None), getattr(data, 'cell', None)
    if off is not None and cell is not None:
        cell = cell.view(-1, 3, 3)[data.batch[dst]]
        vec = vec + torch.bmm(off.view(-1, 1, 3).to(vec.dtype), cell).squeeze(1)
    return vec.norm(dim=-1)


def neighbours(atom, ei, dist, tol=0.4):
    r = np.array([COV_R.get(a, 1.5) for a in atom])
    i, j = ei
    ok = (i != j) & (dist <= r[i] + r[j] + tol) & ~((atom[i] == 'H') & (atom[j] == 'H'))
    nb = [set() for _ in atom]
    for a, b in zip(i[ok], j[ok]):
        nb[a].add(b); nb[b].add(a)
    return nb


def functional_groups(atom, nb):
    fg = np.full(len(atom), '', dtype=object)
    for i, a in enumerate(atom):
        if a in METALS or a == 'H':
            continue
        nbr = [atom[j] for j in nb[i]]
        own = lambda e: [j for j in nb[i] if atom[j] == e]
        if a == 'N' and len(own('O')) >= 2:
            lab, mem = 'NO2', own('O')
        elif a == 'N' and own('H') and not own('O'):
            lab, mem = 'NH2', own('H')
        elif a == 'C' and len(own('H')) >= 3:
            lab, mem = 'CH3', own('H')
        elif a == 'O' and own('H') and own('C') and not any(e in METALS for e in nbr):
            lab, mem = 'OH', own('H')
        elif a in ('F', 'Cl', 'Br') and 'C' in nbr:
            lab, mem = str(a), []
        else:
            continue
        fg[i] = lab
        for j in mem:
            fg[j] = lab
    return fg


def site_types(atom, fg, S):
    own = S.argmax(1)
    out = []
    for c in range(S.shape[1]):
        idx = np.where(own == c)[0]
        if len(idx) == 0:
            out.append('empty')
        elif any(atom[i] in METALS for i in idx):
            out.append('node')
        elif any(fg[i] != '' and atom[i] != 'H' for i in idx):
            out.append('functional')
        else:
            out.append('linker')
    return np.array(out), own


def unwrap_mean(pos, idx, cell):
    ref = pos[idx[0]]
    d = pos[idx] - ref
    if cell is not None:
        f = d @ np.linalg.inv(cell)
        d = (f - np.round(f)) @ cell
    return ref + d.mean(0)


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(s))


@torch.no_grad()
def collect(model, loader, device, args, names=None):
    P = len(args.pressures)
    pl = [f'{x:g}' for x in args.pressures]
    s_max = S_MAX.get(args.gas, np.inf)
    dead_tol = getattr(args, 'dead_tol', 0.05)
    viz = [str(v) for v in (getattr(args, 'viz', None) or [])]
    auto_left = {True: not viz, False: not viz}
    max_atoms = getattr(args, 'viz_max_atoms', 400)
    e_tol = getattr(args, 'e_tol', 0.5)                                       # kJ/mol, top-site agreement tolerance
    intervals = [(j, j + 1) for j in range(P - 1)] + [(0, P - 1)]
    to_np = lambda t: t.detach().cpu().numpy()

    model.eval()
    cache = {}
    h1 = model.sites.register_forward_hook(lambda mod, inp, out: cache.__setitem__('sites', out))
    h2 = model.pore_inc.register_forward_hook(lambda mod, inp, out: cache.__setitem__('inc', out))
    site_rows, mof_rows, dec_rows, thermo_rows, atoms = [], [], [], [], {}
    n_noS, k = 0, 0
    try:
        for data in tqdm(loader, desc='site analysis'):
            data = data.to(device)
            out = model(data)
            pred, extras, m, alpha = out[0], out[1], out[2], out[3]
            B = pred.shape[0]
            e = extras[0]
            log_n, log_K, theta = cache['sites'][0], cache['sites'][1], cache['sites'][2]
            u = cache['inc'][0].reshape(B, P)
            mf = m.to(pred.dtype)
            n = torch.exp(log_n) * mf
            q = n.unsqueeze(-1) * theta
            dS = R_J * (log_K - e / RT)                                         # J/mol/K
            H = torch.exp(log_n + log_K) * mf
            y = to_np(data.y.view(B, -1)[:, :P]).astype(np.float64) if getattr(data, 'y', None) is not None else np.full((B, P), np.nan)
            pr, ee, aa, qq, dd, HH, nn, uu, tt = (to_np(v).astype(np.float64) for v in (pred, e, alpha, q, dS, H, n, u, theta))
            mm = to_np(m.bool())
            assign = per_graph_assignment(out[5] if len(out) > 5 else None, data.batch, B)
            elen = to_np(edge_lengths(data))
            g_of_edge = to_np(data.batch[data.edge_index[0]])
            ptr = to_np(data.ptr)
            cells = to_np(data.cell.view(B, 3, 3)) if getattr(data, 'cell', None) is not None else [None] * B
            dnames = getattr(data, 'name', None)

            for b in range(B):
                if names is not None:
                    name = str(names[k])
                elif isinstance(dnames, (list, tuple)):
                    name = str(dnames[b])
                else:
                    name = str(k)
                fr = mm[b, :-1]
                e_s, a_s, q_s, dS_s, H_s, n_s, th_s = (v[b, :-1][fr] for v in (ee, aa, qq, dd, HH, nn, tt))
                e_p, a_p, q_p, dS_p, H_p = ee[b, -1], aa[b, -1], qq[b, -1], dd[b, -1], HH[b, -1]
                active = (dS_s > -(1.0 - dead_tol) * s_max) & (a_s.max(1) >= 1e-3)

                atom = np.array([INDEX_TO_ELEMENT.get(int(x), 'X') for x in to_np(data.x[data.batch == b].view(-1))])
                sel = g_of_edge == b
                ei_b = to_np(data.edge_index[:, torch.as_tensor(sel, device=data.edge_index.device)]) - ptr[b]
                nb = neighbours(atom, ei_b, elen[sel])
                fg = functional_groups(atom, nb)
                S = assign[b]
                if S is not None and S.shape[1] in (len(fr), len(fr) + 1):
                    S = S[:, :len(fr)][:, fr]
                    stype, own = site_types(atom, fg, S)
                else:
                    S, stype, own = None, np.array(['unknown'] * len(e_s)), None
                    n_noS += 1

                for c in range(len(e_s)):
                    r = {'idx': k, 'name': name, 'site': c, 'type': stype[c], 'active': bool(active[c]), 'e': e_s[c],
                         'n': n_s[c], 'dS': dS_s[c], 'H': H_s[c]}
                    r.update({f'alpha@{x}': a_s[c, j] for j, x in enumerate(pl)})
                    r.update({f'q@{x}': q_s[c, j] for j, x in enumerate(pl)})
                    site_rows.append(r)
                r = {'idx': k, 'name': name, 'site': -1, 'type': 'pore', 'active': True, 'e': e_p, 'n': nn[b, -1],
                     'dS': dS_p, 'H': H_p}
                r.update({f'alpha@{x}': a_p[j] for j, x in enumerate(pl)})
                r.update({f'q@{x}': q_p[j] for j, x in enumerate(pl)})
                site_rows.append(r)

                mrow = {'idx': k, 'name': name, 'n_atoms': len(atom), 'n_sites': len(e_s), 'n_active': int(active.sum()),
                        'groups': '+'.join(sorted(set(fg) - {''}))}
                closure = pr[b] - uu[b] - (a_s * e_s[:, None]).sum(0) - a_p * e_p
                mrow['closure_max'] = float(np.abs(closure).max())
                for c in ['node', 'linker', 'functional', 'empty', 'unknown']:
                    mk = stype == c
                    A, C = a_s[mk].sum(0), (a_s[mk] * e_s[mk, None]).sum(0)
                    mrow.update({f'A_{c}@{x}': A[j] for j, x in enumerate(pl)})
                    mrow.update({f'C_{c}@{x}': C[j] for j, x in enumerate(pl)})
                    mrow[f'e_{c}'] = e_s[mk & active].mean() if (mk & active).any() else np.nan
                mrow.update({f'A_pore@{x}': a_p[j] for j, x in enumerate(pl)})
                mrow.update({f'C_pore@{x}': a_p[j] * e_p for j, x in enumerate(pl)})
                mrow['e_pore'] = e_p
                mrow.update({f'u@{x}': uu[b, j] for j, x in enumerate(pl)})
                mrow.update({f'Qst_pred@{x}': pr[b, j] for j, x in enumerate(pl)})
                mrow.update({f'Qst_true@{x}': y[b, j] for j, x in enumerate(pl)})
                mof_rows.append(mrow)

                F = (a_s * e_s[:, None]).sum(0) / np.clip(1.0 - a_p, 1e-12, None)
                for j1, j2 in intervals:
                    am, Fm = 0.5 * (a_p[j1] + a_p[j2]), 0.5 * (F[j1] + F[j2])
                    t_f, t_s, t_u = (1 - am) * (F[j2] - F[j1]), (a_p[j2] - a_p[j1]) * (e_p - Fm), uu[b, j2] - uu[b, j1]
                    dq = pr[b, j2] - pr[b, j1]
                    dec_rows.append({'idx': k, 'name': name, 'interval': f'{pl[j1]}->{pl[j2]}', 'dQ_pred': dq,
                                     'dQ_true': y[b, j2] - y[b, j1], 'framework': t_f, 'pore_share_shift': t_s,
                                     'interaction': t_u, 'residual': dq - t_f - t_s - t_u})

                ia = np.where(active)[0]
                Hall = np.append(H_s, H_p)
                gap = e_s[ia][np.argmax(H_s[ia])] - e_s[ia].max() if len(ia) >= 2 else np.nan
                thermo_rows.append({
                    'idx': k, 'name': name, 'n_active_frame': len(ia),
                    'rho_e_dS': spearman(e_s[ia], dS_s[ia]) if len(ia) >= 3 else np.nan,
                    'rho_H_e': spearman(H_s[ia], e_s[ia]) if len(ia) >= 3 else np.nan,
                    'e_topH_minus_max_e': gap,
                    'top_site_agree': float(gap >= -e_tol) if np.isfinite(gap) else np.nan,
                    'theta_max_pmin': th_s[ia, 0].max() if len(ia) else np.nan,
                    'rho_alpha_pmin_vs_H_share': spearman(np.append(a_s[:, 0], a_p[0]), Hall / Hall.sum()),
                    'dS_pore': dS_p, 'e_pore': e_p})

                pick = name in viz or str(k) in viz
                has_fg = bool((stype == 'functional').any())
                if (not pick and auto_left[has_fg] and S is not None and len(atom) <= max_atoms
                        and (stype == 'node').any() and (stype == 'linker').any()):
                    pick, auto_left[has_fg] = True, False
                if pick and S is not None:
                    Sl = S[:, active] / np.clip(S[:, active].sum(1, keepdims=True), 1e-12, None)
                    e_i = Sl @ e_s[active] if active.any() else np.full(len(atom), np.nan)
                    q_i = (S / np.clip(S.sum(0, keepdims=True), 1e-12, None)) @ q_s
                    pos = to_np(data.pos[data.batch == b])
                    node_sites = np.where(stype == 'node')[0]
                    if len(node_sites):
                        c0 = node_sites[np.argmax(a_s[node_sites, 0])]
                        center = unwrap_mean(pos, np.where(own == c0)[0], cells[b])
                    else:
                        center = pos[np.nanargmax(e_i)]
                    parent = np.array([next((j for j in nb[i] if atom[j] != 'H'), -1) if a == 'H' else -1
                                       for i, a in enumerate(atom)])
                    atoms[name] = {'atom': atom, 'pos': pos, 'cell': cells[b], 'type': stype[own], 'site': own,
                                   'e_i': e_i, 'q_i': q_i, 'h_parent': parent, 'center': center,
                                   'q_pore': q_p, 'q_frame': q_s.sum(0)}
                k += 1
    finally:
        h1.remove(); h2.remove()
    if n_noS:
        print(f"[warn] {n_noS} MOFs without a usable assignment matrix -> site type 'unknown'")
    return pd.DataFrame(site_rows), pd.DataFrame(mof_rows), pd.DataFrame(dec_rows), pd.DataFrame(thermo_rows), atoms


def compensation(sites):
    fr = sites[(sites['type'] != 'pore') & sites['active']][['idx', 'e', 'dS']].copy()
    g = fr.groupby('idx')
    fr['de'] = fr['e'] - g['e'].transform('mean')
    fr['dds'] = fr['dS'] - g['dS'].transform('mean')
    fr = fr[g['e'].transform('size') >= 3]
    sxy = (fr['de'] * fr['dds']).groupby(fr['idx']).sum()
    sxx = (fr['de'] ** 2).groupby(fr['idx']).sum()
    slope = (sxy / sxx.where(sxx > 1e-12)).rename('slope_dS_e')
    pooled = (fr['de'] * fr['dds']).sum() / (fr['de'] ** 2).sum()
    return slope, pooled, fr[['de', 'dds']]


def summarize(sites, mofs, dec, thermo, pl, s_max, tol):
    share = pd.DataFrame([{'type': c, 'pressure': x, 'A_mean': mofs[f'A_{c}@{x}'].mean(),
                           'A_median': mofs[f'A_{c}@{x}'].median(), 'A_q25': mofs[f'A_{c}@{x}'].quantile(.25),
                           'A_q75': mofs[f'A_{c}@{x}'].quantile(.75), 'C_mean': mofs[f'C_{c}@{x}'].mean()}
                          for c in TYPES for x in pl])
    heat = pd.DataFrame([{'type': c, 'n_mof': int(mofs[f'e_{c}'].notna().sum()), 'e_median': mofs[f'e_{c}'].median(),
                          'e_q25': mofs[f'e_{c}'].quantile(.25), 'e_q75': mofs[f'e_{c}'].quantile(.75),
                          'n_active_sites': int(((sites['type'] == c) & sites['active']).sum())} for c in TYPES])
    paired = []
    for c in ['node', 'functional', 'pore']:
        d = (mofs[f'e_{c}'] - mofs['e_linker']).dropna()
        paired.append({'pair': f'{c} - linker', 'n_mof': len(d), 'median': d.median(), 'q25': d.quantile(.25),
                       'q75': d.quantile(.75), 'frac_positive': (d > 0).mean()})
    paired = pd.DataFrame(paired)

    d_all = dec[dec['interval'] == f'{pl[0]}->{pl[-1]}'].copy()
    d_all['trend'] = np.select([d_all['dQ_pred'] < -tol, d_all['dQ_pred'] > tol], ['decreasing', 'increasing'], 'flat')
    cols = ['dQ_pred', 'dQ_true'] + list(TERMS)
    by_iv = dec.groupby('interval', sort=False)[cols].mean()
    by_iv['residual_max'] = dec.groupby('interval', sort=False)['residual'].apply(lambda s: s.abs().max())
    by_trend = d_all.groupby('trend')[cols].mean()
    by_trend.insert(0, 'n_mof', d_all['trend'].value_counts())

    slope, pooled, cen = compensation(sites)
    thermo = thermo.drop(columns='slope_dS_e', errors='ignore').merge(slope, left_on='idx', right_index=True, how='left')
    b, iso = thermo['slope_dS_e'].dropna(), -1000.0 / 298.0
    fr = sites[(sites['type'] != 'pore') & sites['active']]
    po = sites[sites['type'] == 'pore']
    th = {'n_mof': len(thermo),
          'n_mof_rho_e_dS': int(thermo['rho_e_dS'].notna().sum()),
          'rho_e_dS_median': thermo['rho_e_dS'].median(),
          'frac_rho_e_dS_negative': thermo['rho_e_dS'].dropna().lt(0).mean(),
          'slope_dS_e_median': b.median(), 'slope_dS_e_pooled': pooled, 'slope_constant_K': iso,
          'T_c_pooled_K': -1000.0 / pooled if pooled < 0 else np.nan,
          'frac_partial_compensation': ((b < 0) & (b > iso)).mean(),
          'frac_full_or_over_compensation': (b <= iso).mean(),
          'frac_no_compensation': (b >= 0).mean(),
          'rho_H_e_median': thermo['rho_H_e'].median(),
          'frac_top_site_agree': thermo['top_site_agree'].mean(),
          'e_topH_minus_max_e_median': thermo['e_topH_minus_max_e'].median(),
          'frac_henry_regime_pmin': thermo['theta_max_pmin'].lt(0.1).mean(),
          'rho_alpha_pmin_vs_H_share_median': thermo['rho_alpha_pmin_vs_H_share'].median(),
          'rho_alpha_pmin_vs_H_share_median_henry': thermo.loc[thermo['theta_max_pmin'] < 0.1,
                                                               'rho_alpha_pmin_vs_H_share'].median(),
          'dS_frame_median': fr['dS'].median(), 'dS_pore_median': po['dS'].median(),
          'frac_dS_frame_near_lower_bound': (fr['dS'] < -0.9 * s_max).mean(),
          'frac_dS_frame_near_zero': (fr['dS'] > -0.1 * s_max).mean(),
          'frac_inactive_sites': 1 - sites.loc[sites['type'] != 'pore', 'active'].mean(),
          'closure_max': mofs['closure_max'].max(), 'decomposition_residual_max': dec['residual'].abs().max()}
    for x in (pl[0], pl[-1]):
        th[f'A_unassigned_mean@{x}'] = (mofs[f'A_empty@{x}'] + mofs[f'A_unknown@{x}']).mean()
    return share, heat, paired, by_iv, by_trend, d_all, pd.Series(th), thermo, (cen, pooled)


def p_axis(ax, p):
    ax.set_xscale('log')
    ax.set_xticks(p)
    ax.set_xticklabels([f'{x:g}' for x in p])
    ax.minorticks_off()
    ax.set_xlabel('Pressure (bar)')


def fig_type_share(share, p, pl, out_dir):
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for c in TYPES:
        d = share[share['type'] == c].set_index('pressure').loc[pl]
        ax.plot(p, d['A_mean'], '-o', color=COLOR[c], ms=4, label=c)
    p_axis(ax, p)
    ax.set_ylabel(r'Mean share $A_c$')
    ax.set_ylim(0, 1.2 * share['A_mean'].max())
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.0)
    save(fig, out_dir, 'fig_type_share')


def fig_type_heat(mofs, out_dir):
    data = [mofs[f'e_{c}'].dropna().values for c in TYPES]
    keep = [i for i, d in enumerate(data) if len(d)]
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    bp = ax.boxplot([data[i] for i in keep], widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, i in zip(bp['boxes'], keep):
        patch.set_facecolor(COLOR[TYPES[i]]); patch.set_alpha(0.7); patch.set_edgecolor(INK); patch.set_linewidth(0.6)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([f'{TYPES[i]}\n(n={len(data[i])})' for i in keep])
    ax.set_ylabel(r'Mean site heat $\bar{e}_c$ (kJ mol$^{-1}$)')
    save(fig, out_dir, 'fig_type_heat')


def fig_type_contribution(mofs, pl, out_dir):
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    x = np.arange(len(pl))
    bottom = np.zeros(len(pl))
    for c in TYPES:
        v = np.array([mofs[f'C_{c}@{s}'].mean() for s in pl])
        ax.bar(x, v, 0.6, bottom=bottom, color=COLOR[c], edgecolor='white', lw=0.8, label=c)
        bottom += v
    uv = np.array([mofs[f'u@{s}'].mean() for s in pl])
    ax.bar(x, uv, 0.6, bottom=bottom, color=COLOR['interaction'], edgecolor='white', lw=0.8, label=r'$u$')
    ax.plot(x, [mofs[f'Qst_pred@{s}'].mean() for s in pl], 'D', color=INK, ms=4, zorder=4, label=r'$Q_\mathrm{st}$ model')
    yt = np.array([mofs[f'Qst_true@{s}'].mean() for s in pl])
    if np.isfinite(yt).all():
        ax.plot(x, yt, 'o', mfc='none', mec=COLOR['gcmc'], ms=7, mew=1.3, zorder=5, label=r'$Q_\mathrm{st}$ GCMC')
    ax.set_xticks(x); ax.set_xticklabels(pl); ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel(r'Mean contribution $C_c$ (kJ mol$^{-1}$)')
    _legend(ax, ncol=4)
    save(fig, out_dir, 'fig_type_contribution')


def _legend(ax, ncol):
    h, l = ax.get_legend_handles_labels()
    o = sorted(range(len(h)), key=lambda i: 1 if isinstance(h[i], Line2D) else
               (0 if isinstance(h[i], matplotlib.container.BarContainer) else 2))
    ax.legend([h[i] for i in o], [l[i] for i in o], ncol=ncol, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=0.8)


def _term_bars(ax, means, labels):
    w = 0.24
    x = np.arange(len(labels))
    for i, t in enumerate(TERMS):
        ax.bar(x + (i - 1) * w, means[t].values, w, color=TERM_COLOR[t], edgecolor='white', lw=0.6, label=TERMS[t])
    ax.plot(x, means['dQ_pred'].values, 'D', color=INK, ms=4, zorder=4, label=r'$\Delta Q_\mathrm{st}$ model')
    if np.isfinite(means['dQ_true'].values).all():
        ax.plot(x, means['dQ_true'].values, 'o', mfc='none', mec=COLOR['gcmc'], ms=7, mew=1.3, zorder=5,
                label=r'$\Delta Q_\mathrm{st}$ GCMC')
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r'Mean change (kJ mol$^{-1}$)')


def fig_dqst_intervals(by_iv, out_dir):
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    _term_bars(ax, by_iv, [s.replace('->', '→') for s in by_iv.index])
    ax.set_xlabel('Pressure interval (bar)')
    _legend(ax, ncol=5)
    save(fig, out_dir, 'fig_dqst_intervals')


def fig_dqst_trend(by_trend, pl, out_dir):
    order = [t for t in ['decreasing', 'flat', 'increasing'] if t in by_trend.index]
    g = by_trend.loc[order]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    _term_bars(ax, g, [f'{t}\n(n={int(g.loc[t, "n_mof"])})' for t in order])
    ax.set_xlabel(f'Predicted trend, {pl[0]}→{pl[-1]} bar')
    _legend(ax, ncol=3)
    save(fig, out_dir, 'fig_dqst_trend')


def fig_compensation(comp, out_dir, T=298.0):
    cen, pooled = comp
    x, y = cen['de'].values, cen['dds'].values
    xr, yr = np.percentile(x, [0.5, 99.5]), np.percentile(y, [0.5, 99.5])
    Hc, xe, ye = np.histogram2d(x, y, bins=60, range=[xr, yr])
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    pc = ax.pcolormesh(xe, ye, np.ma.masked_equal(Hc.T, 0), cmap=_seq_cmap('Blues'), norm=LogNorm(), rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label('Sites (count)'); cb.outline.set_linewidth(0.4)
    ax.axhline(0, color=MUTED, lw=0.4); ax.axvline(0, color=MUTED, lw=0.4)
    ax.plot(xr, -1000.0 / T * xr, '--', color=INK, lw=1, label=r'constant $K$ (slope $-1/T$)')
    tc = f', $T_c$ = {-1000.0 / pooled:.0f} K' if pooled < 0 else ''
    ax.plot(xr, pooled * xr, '-', color=COLOR['linker'], lw=1.5, label=f'fit (slope {pooled:.2f}{tc})')
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_xlabel(r'$e_k - \langle e \rangle_\mathrm{MOF}$ (kJ mol$^{-1}$)')
    ax.set_ylabel(r'$\Delta S_k - \langle \Delta S \rangle_\mathrm{MOF}$ (J mol$^{-1}$ K$^{-1}$)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, out_dir, 'fig_compensation')


def _rho_hist(ax, r, xlabel, note):
    cnt = ax.hist(r, bins=np.linspace(-1, 1, 41), color=COLOR['node'], edgecolor='white', lw=0.4)[0]
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.axvline(r.median(), color=INK, lw=1, ls='--')
    ax.set_ylim(0, 1.45 * cnt.max())
    left = r.median() >= 0
    ax.text(0.03 if left else 0.97, 0.97, note, transform=ax.transAxes, va='top', ha='left' if left else 'right',
            fontsize=7, bbox=dict(fc='white', ec='none', pad=1))
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('MOFs')


def fig_rho_e_dS(thermo, out_dir):
    r = thermo['rho_e_dS'].dropna()
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    _rho_hist(ax, r, r'Within-MOF Spearman $\rho(e_k,\Delta S_k)$',
              f'median {r.median():.2f}\n' + r'$\rho<0$' + f' in {(r < 0).mean() * 100:.0f}% of {len(r)} MOFs')
    save(fig, out_dir, 'fig_rho_e_dS')


def fig_dS(sites, s_max, out_dir):
    fr = sites[(sites['type'] != 'pore') & sites['active']]['dS']
    po = sites[sites['type'] == 'pore']['dS']
    bins = np.linspace(-s_max, 0, 43)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.hist(fr, bins=bins, density=True, histtype='step', lw=1.5, color=COLOR['framework'], label='framework sites')
    ax.hist(po, bins=bins, density=True, histtype='step', lw=1.5, color=COLOR['pore'], label='pore')
    for v in (-s_max, 0):
        ax.axvline(v, color=MUTED, lw=0.6, ls='--')
    ax.set_xlabel(r'$\Delta S_k$ (J mol$^{-1}$ K$^{-1}$)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left')
    save(fig, out_dir, 'fig_dS')


def fig_henry_heat(thermo, e_tol, out_dir):
    r = thermo['rho_H_e'].dropna()
    agree = thermo['top_site_agree'].dropna()
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    _rho_hist(ax, r, r'Within-MOF Spearman $\rho(H_k,e_k)$',
              f'median {r.median():.2f}\n' + r'top-$H_k$ site is the top-$e_k$ site' + f'\n(within {e_tol:g} kJ/mol) in {agree.mean() * 100:.0f}% of MOFs')
    save(fig, out_dir, 'fig_henry_heat')


def _seq_cmap(name):
    return LinearSegmentedColormap.from_list(name + '_t', plt.get_cmap(name)(np.linspace(0.18, 1.0, 256)))


def slab_view(a, radius, slab, hide_h):
    pos, cell, atom = a['pos'], a['cell'], a['atom']
    N = len(pos)
    if cell is not None:
        inv = np.linalg.inv(cell)
        c = a['center'] @ inv
        center = (c - np.floor(c)) @ cell
        frac = pos @ inv
        frac = frac - np.floor(frac)
        sh = np.array(list(itertools.product([-1, 0, 1], repeat=3)), float)
        X = ((frac[None] + sh[:, None]) @ cell).reshape(-1, 3)
        idx = np.tile(np.arange(N), len(sh))
        rec = inv.T
        cand = [np.array(h) @ rec for h in itertools.product([-1, 0, 1], repeat=3) if any(h)]
    else:
        center, X, idx = a['center'], pos.copy(), np.arange(N)
        cand = list(np.eye(3))
    keep = np.linalg.norm(X - center, axis=1) <= radius
    X, idx = X[keep] - center, idx[keep]
    el = atom[idx]
    r = np.array([COV_R.get(x, 1.5) for x in el])
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    hh = (el[:, None] == 'H') & (el[None] == 'H')
    mm = np.isin(el, list(METALS))
    bond = (D <= r[:, None] + r[None] + 0.4) & (D > 0.1) & ~hh & ~(mm[:, None] & mm[None])
    heavy = el != 'H'
    cand = [v / np.linalg.norm(v) for v in cand]
    nrm = max(cand, key=lambda v: (np.abs(X[heavy] @ v) <= slab).sum())
    depth = X @ nrm
    keep = np.abs(depth) <= slab
    grow = np.isin(a['type'][idx], ['linker', 'functional'])
    for _ in range(2):
        keep |= bond[keep & grow].any(0) & (np.abs(depth) <= 2 * slab)
    if hide_h:
        keep &= heavy
    X, idx, depth, r, Bk = X[keep], idx[keep], depth[keep], r[keep], bond[np.ix_(keep, keep)]
    lab = connected_components(Bk, directed=False)[1]
    near = np.linalg.norm(X, axis=1) <= 4.0
    root = max(np.unique(lab[near] if near.any() else lab), key=lambda c: (lab == c).sum())
    s = lab == root
    X, idx, depth, r, Bk = X[s], idx[s], depth[s], r[s], Bk[np.ix_(s, s)]
    Pn = X - np.outer(X @ nrm, nrm)
    V = np.linalg.svd(Pn - Pn.mean(0), full_matrices=False)[2][:2]
    xy = Pn @ V.T
    bonds = np.argwhere(np.triu(Bk, 1))
    return idx, xy, depth, bonds, r


def fig_atoms(name, a, pl, out_dir, radius, slab, hide_h):
    q = a['q_i'].copy()
    if hide_h:
        for i in np.where(a['h_parent'] >= 0)[0]:
            q[a['h_parent'][i]] += q[i]
    idx, xy, depth, bonds, r = slab_view(a, radius, slab, hide_h)
    o = np.argsort(depth)
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) + 3.0
    mid = 0.5 * (xy.max(0) + xy.min(0))
    W, Hh, side = W2, 2.75, 0.215
    fig = plt.figure(figsize=(W, Hh))
    pts_per_A = side * W * 72 / span
    size = ((0.55 * r + 0.45) * pts_per_A) ** 2
    jp = [0, len(pl) - 1]
    titles = ['Site type', r'Site heat $e_i$', rf'Loading $q_i$, {pl[0]} bar', rf'Loading $q_i$, {pl[-1]} bar']
    for p_ in range(4):
        left = 0.01 + p_ * 0.25
        ax = fig.add_axes([left, 0.25, side, side * W / Hh])
        for i, j in bonds:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=MUTED, lw=0.7, zorder=1, solid_capstyle='round')
        if p_ == 0:
            col = np.array([to_rgba(COLOR.get(t, MUTED)) for t in a['type'][idx]])
            ax.scatter(xy[o, 0], xy[o, 1], s=size[o], c=col[o], edgecolors=INK, linewidths=0.3, zorder=2)
            hand = [Line2D([], [], ls='', marker='o', ms=5, mfc=COLOR[t], mec=INK, mew=0.3, label=t)
                    for t in ['node', 'linker', 'functional'] if (a['type'][idx] == t).any()]
            ax.legend(handles=hand, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=len(hand),
                      handletextpad=0.2, columnspacing=0.6, fontsize=6.5)
        else:
            v = a['e_i'][idx] if p_ == 1 else q[idx, jp[p_ - 2]]
            ok = np.isfinite(v)
            cmap = _seq_cmap('Oranges' if p_ == 1 else 'Blues')
            norm = Normalize(np.nanmin(v), np.nanmax(v)) if p_ == 1 else Normalize(0, np.nanmax(v))
            col = np.where(ok[:, None], cmap(norm(np.nan_to_num(v))), to_rgba(MUTED))
            ax.scatter(xy[o, 0], xy[o, 1], s=size[o], c=col[o], edgecolors=INK, linewidths=0.3, zorder=2)
            cax = fig.add_axes([left + 0.02, 0.16, side - 0.04, 0.03])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
            cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, width=0.4, length=2)
            if p_ == 1:
                cb.set_label(r'kJ mol$^{-1}$', fontsize=6.5)
            else:
                ex = int(np.floor(np.log10(max(norm.vmax, 1e-30))))
                ticks = np.linspace(0, norm.vmax, 3)
                cb.set_ticks(ticks)
                cb.set_ticklabels([f'{t / 10 ** ex:.2g}' for t in ticks])
                cb.set_label((rf'$10^{{{ex}}}$ ' if ex else '') + r'mmol g$^{-1}$', fontsize=6.5)
                j = jp[p_ - 2]
                fig.text(left + side / 2, 0.245, f'framework {a["q_frame"][j]:.3f}, pore {a["q_pore"][j]:.3f} mmol g$^{{-1}}$',
                         ha='center', va='top', fontsize=6, color=INK)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(titles[p_], pad=2)
    save(fig, out_dir, f'fig_atoms_{safe_name(name)}')


def export_atoms(name, a, pl, out_dir):
    s = safe_name(name)
    df = pd.DataFrame({'element': a['atom'], 'x': a['pos'][:, 0], 'y': a['pos'][:, 1], 'z': a['pos'][:, 2],
                       'site': a['site'], 'site_type': a['type'], 'e_i': a['e_i']})
    for j, x in enumerate(pl):
        df[f'q_i@{x}'] = a['q_i'][:, j]
    df.to_csv(os.path.join(out_dir, f'atoms_{s}.csv'), index=False)
    cell = a['cell']
    tag = [x.replace('.', 'p') for x in pl]
    with open(os.path.join(out_dir, f'{s}.extxyz'), 'w') as f:
        f.write(f'{len(df)}\n')
        lat = f'Lattice="{" ".join(f"{v:.6f}" for v in cell.ravel())}" pbc="T T T" ' if cell is not None else ''
        f.write(lat + 'Properties=species:S:1:pos:R:3:e_i:R:1' + ''.join(f':q_{t}:R:1' for t in tag) + '\n')
        for i in range(len(df)):
            f.write(f'{df.element[i]} {df.x[i]:.5f} {df.y[i]:.5f} {df.z[i]:.5f} {df.e_i[i]:.4f} '
                    + ' '.join(f'{a["q_i"][i, j]:.6e}' for j in range(len(pl))) + '\n')
    fields = [('e', df['e_i'].values, 'B-factor = e_i (kJ/mol)')]
    for j in (0, len(pl) - 1):
        qj = a['q_i'][:, j]
        fields.append((f'q{tag[j]}', 100 * qj / max(qj.sum(), 1e-30), f'B-factor = q_i, % of framework loading at {pl[j]} bar'))
    for t, val, remark in fields:
        with open(os.path.join(out_dir, f'{s}_{t}.pdb'), 'w') as f:
            f.write(f'REMARK   1 {remark}\n')
            if cell is not None:
                L = np.linalg.norm(cell, axis=1)
                ang = [np.degrees(np.arccos(cell[i] @ cell[j] / (L[i] * L[j]))) for i, j in ((1, 2), (0, 2), (0, 1))]
                f.write(f'CRYST1{L[0]:9.3f}{L[1]:9.3f}{L[2]:9.3f}{ang[0]:7.2f}{ang[1]:7.2f}{ang[2]:7.2f} P 1           1\n')
            for i in range(len(df)):
                v = val[i] if np.isfinite(val[i]) else 0.0
                el = df.element[i]
                f.write(f'HETATM{i + 1:5d} {el:<4s} MOF A   1    {df.x[i]:8.3f}{df.y[i]:8.3f}{df.z[i]:8.3f}'
                        f'{1.0:6.2f}{v:6.2f}          {el:>2s}\n')
            f.write('END\n')


def report(sites, mofs, dec, thermo, atoms, args, out_dir):
    set_style()
    os.makedirs(out_dir, exist_ok=True)
    p = np.asarray(args.pressures, float)
    pl = [f'{x:g}' for x in args.pressures]
    s_max = S_MAX.get(args.gas, np.inf)
    share, heat, paired, by_iv, by_trend, d_all, th, thermo, comp = summarize(sites, mofs, dec, thermo, pl, s_max,
                                                                              getattr(args, 'tol', 1.0))

    csv = lambda df, f, **kw: df.to_csv(os.path.join(out_dir, f), **kw)
    csv(sites, 'sites.csv', index=False)
    csv(mofs, 'mof_types.csv', index=False)
    csv(share, 'type_share.csv', index=False)
    csv(heat, 'type_heat.csv', index=False)
    csv(paired, 'type_heat_paired.csv', index=False)
    csv(dec.merge(d_all[['idx', 'trend']], on='idx', how='left'), 'decomposition_mof.csv', index=False)
    csv(by_iv, 'decomposition_summary.csv')
    csv(by_trend, 'decomposition_by_trend.csv')
    csv(thermo, 'thermo_mof.csv', index=False)
    csv(th.to_frame('value'), 'thermo_summary.csv')
    case = mofs[['idx', 'name', 'groups', 'n_atoms', 'n_active']].copy()
    for c in TYPES:
        case[f'e_{c}'] = mofs[f'e_{c}']
    for x in (pl[0], pl[-1]):
        for c in TYPES:
            case[f'A_{c}@{x}'] = mofs[f'A_{c}@{x}']
        case[f'Qst_pred@{x}'] = mofs[f'Qst_pred@{x}']
    case = case.merge(d_all[['idx', 'dQ_pred', 'dQ_true'] + list(TERMS)], on='idx', how='left')
    case = case.merge(thermo[['idx', 'rho_e_dS', 'slope_dS_e', 'theta_max_pmin']], on='idx', how='left')
    csv(case, 'mof_summary.csv', index=False)

    fig_type_share(share, p, pl, out_dir)
    fig_type_heat(mofs, out_dir)
    fig_type_contribution(mofs, pl, out_dir)
    fig_dqst_intervals(by_iv, out_dir)
    fig_dqst_trend(by_trend, pl, out_dir)
    fig_compensation(comp, out_dir)
    fig_rho_e_dS(thermo, out_dir)
    fig_dS(sites, s_max, out_dir)
    fig_henry_heat(thermo, getattr(args, 'e_tol', 0.5), out_dir)
    for name, a in atoms.items():
        export_atoms(name, a, pl, out_dir)
        fig_atoms(name, a, pl, out_dir, getattr(args, 'crop_radius', 10.0), getattr(args, 'slab', 2.0),
                  getattr(args, 'hide_h', True))

    pd.set_option('display.width', 200)
    print(f"\n[site analysis] {len(mofs)} MOFs, {int((sites['type'] != 'pore').sum())} framework sites -> {out_dir}")
    print("\n=== (i) type share A_c (mean) ===")
    print(share.pivot(index='type', columns='pressure', values='A_mean')[pl].loc[TYPES].round(3).to_string())
    print("\n=== (i) type contribution C_c (mean, kJ/mol) ===")
    print(share.pivot(index='type', columns='pressure', values='C_mean')[pl].loc[TYPES].round(2).to_string())
    print("\n=== (i) mean site heat of the active sites per MOF (kJ/mol) ===")
    print(heat.round(2).to_string(index=False))
    print("\n--- same-MOF difference of the type mean heat (kJ/mol) ---")
    print(paired.round(3).to_string(index=False))
    print("\n=== (iii) dQst decomposition (mean over MOFs, kJ/mol) ===")
    print(by_iv.round(3).to_string())
    print(f"\n--- {pl[0]} -> {pl[-1]} bar by predicted trend (tol {getattr(args, 'tol', 1.0)} kJ/mol) ---")
    print(by_trend.round(3).to_string())
    print("\n=== (iv) thermodynamic indicators ===")
    print(th.round(4).to_string())
    if len(case) <= 30:
        print(f"\n=== per MOF (dQ terms: {pl[0]} -> {pl[-1]} bar) ===")
        print(case.drop(columns='idx').set_index('name').T.to_string(float_format=lambda v: f'{v:.3g}'))
    print(f"\n(ii) atom-level views: {list(atoms) or 'none'}"
          + ('' if atoms or getattr(args, 'viz', None) else
             f"  (auto pick needs <= {getattr(args, 'viz_max_atoms', 400)} atoms; use --viz NAME or --viz_max_atoms)"))
    return sites, mofs, dec, thermo, th


def analyze_sites(model, best_model_path, loader, device, args, load_weights=True, names=None, out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(best_model_path), f'site_analysis_{args.seed}')
    if load_weights:
        _load_weights(model, best_model_path, device)
    sites, mofs, dec, thermo, atoms = collect(model, loader, device, args, names)
    return report(sites, mofs, dec, thermo, atoms, args, out_dir)


def from_csv(csv_dir, args, out_dir=None):
    rd = lambda f: pd.read_csv(os.path.join(csv_dir, f))
    dec = rd('decomposition_mof.csv').drop(columns='trend', errors='ignore')
    return report(rd('sites.csv'), rd('mof_types.csv'), dec, rd('thermo_mof.csv'), {}, args, out_dir or csv_dir)


if __name__ == '__main__':
    import argparse
    from glob import glob
    ap = argparse.ArgumentParser(description='site-level interpretation')
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
    ap.add_argument('--pt_path', default='', type=str, help='folder of <idx>.pt graphs + index.csv; overrides --data_path')
    ap.add_argument('--out_dir', default='', type=str)
    ap.add_argument('--viz', nargs='*', default=[], help='MOF names or loader indices for atom views (empty: auto, 2 MOFs)')
    ap.add_argument('--viz_max_atoms', default=400, type=int)
    ap.add_argument('--crop_radius', default=10.0, type=float, help='A, fragment radius around the strongest node')
    ap.add_argument('--slab', default=2.0, type=float, help='A, half thickness of the displayed slab')
    ap.add_argument('--hide_h', default=True, type=lambda s: str(s).lower() in ('1', 'true', 'yes'))
    ap.add_argument('--dead_tol', default=0.05, type=float)
    ap.add_argument('--tol', default=1.0, type=float, help='kJ/mol, flat trend threshold p_min -> p_max')
    ap.add_argument('--e_tol', default=0.5, type=float, help='kJ/mol, top-H site counts as the top-e site within this')
    ap.add_argument('--from_csv', default='', type=str, help='site_analysis folder of a previous run: redo tables/figures only')
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    TARGETS = {'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
               'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5']}
    GEO_KEYS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd']
    args.pressures = [float(t.split('_')[-1]) for t in TARGETS[args.target]]
    args.n_out = len(args.pressures)
    args.gas = args.target.split('_')[-1]
    if args.from_csv:
        from_csv(args.from_csv, args, args.out_dir or None)
        raise SystemExit
    if not args.best_model_path:
        ver = f'{args.target}_{args.encoder}_{args.lambda_aux}_{args.pool}_{args.attn}_{args.mode}_fin'
        args.best_model_path = os.path.join(args.output_path, args.encoder, ver, f'best_{args.seed}.pt')
    print(f'[site analysis] checkpoint: {args.best_model_path}')

    from torch_geometric.loader import DataLoader
    from utils.utils import seed_everything, PygDataset
    from model.hc_model import Hierarchical_Clustering_Model_With_LA
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    names = None
    if args.pt_path:
        files = glob(os.path.join(args.pt_path, '*.pt'))
        stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
        order = sorted(range(len(files)), key=lambda i: (0, int(stems[i]), '') if stems[i].isdigit() else (1, 0, stems[i]))
        idx_csv = os.path.join(args.pt_path, 'index.csv')
        lut = {}
        if os.path.exists(idx_csv):
            t = pd.read_csv(idx_csv)
            lut = dict(zip(t['idx'].astype(int), t['name'].astype(str)))
        graphs, names = [], []
        for i in order:
            d = torch.load(files[i], weights_only=False)
            if getattr(d, 'geo', None) is None:
                d.geo = torch.stack([d[g].view(-1) for g in GEO_KEYS], dim=1).float()
            graphs.append(d)
            names.append(lut.get(int(stems[i]), stems[i]) if stems[i].isdigit() else stems[i])
        loader = DataLoader(graphs, batch_size=args.bs, shuffle=False)
    else:
        loader = DataLoader(PygDataset(root=f'{args.data_path}/test', target_name=TARGETS[args.target], gas=args.gas),
                            batch_size=args.bs, shuffle=False)
    model = Hierarchical_Clustering_Model_With_LA(args).to(device)
    out_dir = args.out_dir or (os.path.join(os.path.dirname(args.best_model_path),
                                            f'site_analysis_{os.path.basename(os.path.normpath(args.pt_path))}_{args.seed}')
                               if args.pt_path else None)
    analyze_sites(model, args.best_model_path, loader, device, args, names=names, out_dir=out_dir)
