"""
Do 12-nearest-neighbour ties explain the symmetry breaking found by symmetry_check?

The graph keeps the 12 nearest neighbours of each atom within the cutoff. When the 12th and 13th neighbours are at
the same distance, the cut keeps one of them by atom order, so crystallographically equivalent atoms can get
different neighbour sets and different embeddings.

For a random sample of test MOFs with equivalent atoms (spglib, same symprec as symmetry_check):
  tie         atoms whose 12th and 13th neighbours are within tie_tol (the cut splits a shell)
  asymmetric  equivalence classes whose atoms see different neighbourhoods in the graph: the multiset of
              (class of the other atom, distance) and the first and second moments of the edge vectors, over
              incoming and outgoing edges, compared with the class representative
                stored     the graph in the .pt files
                tie-aware  rebuilt with all neighbours within d_12 + tie_tol (the proposed preprocessing)
  geometry    largest difference of the sorted neighbour distances between equivalent atoms (A); above geom_tol
              the structure is only approximately symmetric at symprec
  model       the same checkpoint on three graphs of each MOF, r_h = within-class / total variance of the embedding
              (as in symmetry_check):
                stored     the .pt graph
                tie-aware  the proposed preprocessing
                exact      tie-aware graph of an exactly symmetric copy of the structure (positions averaged over the
                           space group found at symprec, metric averaged over its rotations): removes the breaking that
                           comes from the coordinate precision of the CIF
              stored -> tie-aware is what the new preprocessing fixes, tie-aware -> exact is what the coordinates
              cause, and r_h left on the exact graph is the encoder (numerics)
MOF groups (stored graph, first match):
  asymmetric graph      some equivalence class has an asymmetric neighbourhood
  approximate geometry  geometry deviation > geom_tol
  tie, symmetric        a tie somewhere (e.g. an atom on a special position) but symmetric classes
  no tie                clean graph and geometry

Run from src/ after analyze.py --steps symmetry. Same model arguments as train.py / analyze.py:
    python tie_check.py --target heat_co2 --seed 42
--data_path holds graphs built with the plain 12-nearest-neighbour rule:
    python data/preprocess_hmof_consistent.py --tie_tol -1 --save_path ../dataset/hMOF/full_dataset_clean_void_inc_consistent
    --n_sample 0        all MOFs with equivalent atoms (default 1000)
    --model_check 0     graph diagnostics only (no checkpoint needed)

Outputs (out_dir = dirname(best_model_path)/tie_check, or --out_dir)
  csv : tie_mof, tie_class, tie_summary
  fig : fig_tie_check
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spglib
import torch
from pymatgen.core import Lattice, Structure
from scipy.spatial import cKDTree
from torch_geometric.data import Batch
from tqdm import tqdm

from site_analysis import COLOR, INDEX_TO_ELEMENT, INK, MUTED, W2, _load_weights, save, set_style
from utils.symmetry_check import classes, equivalent_atoms, within_ratio

EDGE_KEYS = ('edge_index', 'edge_attr', 'edge_vec', 'offsets')
GEO_KEYS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd']
GROUPS = ['no tie', 'tie, symmetric', 'approximate geometry', 'asymmetric graph']
GROUP_COLOR = {'no tie': COLOR['node'], 'tie, symmetric': COLOR['framework'],
               'approximate geometry': COLOR['pore'], 'asymmetric graph': COLOR['gcmc']}
GRAPH_COLOR = {'stored': MUTED, 'tie-aware': COLOR['linker']}
STAGES = {'stored': 'r_h_stored', 'tie-aware': 'r_h_tie', 'exact': 'r_h_exact'}
R_FLOOR = 1e-8


def to_np(t):
    return t.detach().cpu().numpy()


def structure(d):
    numbers = to_np(d.x.view(-1)).astype(int)
    unknown = sorted(set(numbers) - set(INDEX_TO_ELEMENT))
    if unknown:
        raise KeyError(f'atom indices {unknown} are not in INDEX_TO_ELEMENT')
    cell = to_np(d.cell.reshape(3, 3)).astype(np.float64)
    pos = to_np(d.pos).astype(np.float64)
    st = Structure(Lattice(cell), [INDEX_TO_ELEMENT[int(z)] for z in numbers], pos, coords_are_cartesian=True)
    return st, numbers, cell, pos


def full_neighbours(st, cutoff):
    c, n, img, dist = st.get_neighbor_list(cutoff)
    o = np.lexsort((dist, c))
    return c[o].astype(int), n[o].astype(int), np.asarray(img, np.float64)[o], dist[o]


def shell(c, dist, n_atoms, k):
    counts = np.bincount(c, minlength=n_atoms)
    first = np.searchsorted(c, np.arange(n_atoms), side='left')
    d_k = np.full(n_atoms, np.inf)
    has = counts >= k
    d_k[has] = dist[first[has] + k - 1]
    gap = np.full(n_atoms, np.inf)
    has1 = counts > k
    gap[has1] = dist[first[has1] + k] - d_k[has1]
    rank = np.arange(len(c)) - first[c]
    return counts, d_k, gap, rank


def tie_aware(st, cutoff, k, tol):
    n = len(st)
    cell, pos = st.lattice.matrix, st.cart_coords
    c, nb, img, dist = full_neighbours(st, cutoff)
    counts, d_k, gap, rank = shell(c, dist, n, k)
    keep = dist <= d_k[c] + tol
    src, dst, im = nb[keep], c[keep], img[keep]
    vec = pos[src] + im @ cell - pos[dst]
    return {'c': c, 'dist': dist, 'rank': rank, 'gap': gap, 'src': src, 'dst': dst, 'img': im, 'vec': vec,
            'deg': np.bincount(dst, minlength=n)}


def symmetrize(cell, pos, numbers, symprec):
    frac = pos @ np.linalg.inv(cell)
    f = frac - np.floor(frac)
    f[f >= 1.0] = 0.0
    ds = spglib.get_symmetry_dataset((cell, f, numbers), symprec=symprec)
    if ds is None:
        return None
    rots = np.asarray(ds.rotations if hasattr(ds, 'rotations') else ds['rotations'])
    trans = np.asarray(ds.translations if hasattr(ds, 'translations') else ds['translations'])
    G = cell @ cell.T
    Gs = np.mean([R.T @ G @ R for R in rots], axis=0)
    tree = cKDTree(f, boxsize=1.0)
    acc, cnt = np.zeros_like(f), np.zeros(len(f))
    for R, t in zip(rots, trans):
        y = f @ R.T + t
        y = y - np.floor(y)
        y[y >= 1.0] = 0.0
        _, j = tree.query(y)
        diff = y - f[j]
        diff -= np.round(diff)
        if (len(np.unique(j)) != len(f) or (numbers[j] != numbers).any()
                or np.linalg.norm(diff @ cell, axis=1).max() > 10 * symprec):
            return None
        acc[j] += f[j] + diff
        cnt[j] += 1
    a, b, c = np.sqrt(np.diag(Gs))
    ang = np.degrees(np.arccos(np.clip([Gs[1, 2] / (b * c), Gs[0, 2] / (a * c), Gs[0, 1] / (a * b)], -1, 1)))
    lat = Lattice.from_parameters(a, b, c, *ang)
    return lat.matrix, (acc / cnt[:, None]) @ lat.matrix


def rows(a, v, n, fill):
    deg = np.bincount(a, minlength=n)
    width = max(int(deg.max()) if len(a) else 0, 1)
    M = np.full((n, width), fill, dtype=np.float64)
    if len(a):
        o = np.lexsort((v, a))
        a_s = a[o]
        rank = np.arange(len(a)) - np.searchsorted(a_s, a_s, side='left')
        M[a_s, rank] = v[o]
    return M, deg


def row_diff(A, B):
    w = max(A.shape[1], B.shape[1])
    A = np.pad(A, ((0, 0), (0, w - A.shape[1])), constant_values=np.inf)
    B = np.pad(B, ((0, 0), (0, w - B.shape[1])), constant_values=np.inf)
    both = np.isinf(A) & np.isinf(B)
    D = np.abs(A - B)
    D[both] = 0.0
    return D.max(1)


def asymmetric_atoms(src, dst, vec, rep, n, atol, cutoff):
    scale = cutoff + 10.0
    bad = np.zeros(n, bool)
    for a, b, v in ((dst, src, vec), (src, dst, -vec)):
        dist = np.linalg.norm(v, axis=1)
        M, deg = rows(a, rep[b] * scale + dist, n, np.inf)
        m1 = np.zeros((n, 3))
        np.add.at(m1, a, v)
        m2 = np.zeros((n, 3, 3))
        np.add.at(m2, a, v[:, :, None] * v[:, None, :])
        i1 = np.linalg.norm(m1, axis=1)
        i2 = np.linalg.norm(m2.reshape(n, 9), axis=1)
        bad |= ((deg != deg[rep]) | (row_diff(M, M[rep]) > atol)
                | (np.abs(i1 - i1[rep]) > atol * (deg + 1))
                | (np.abs(i2 - i2[rep]) > 2 * cutoff * atol * (deg + 1)))
    return bad


def stored_edges(d):
    src, dst = (to_np(t).astype(int) for t in d.edge_index)
    if getattr(d, 'edge_vec', None) is not None:
        vec = to_np(d.edge_vec).astype(np.float64)
    else:
        cell = to_np(d.cell.reshape(3, 3)).astype(np.float64)
        pos = to_np(d.pos).astype(np.float64)
        vec = pos[src] - pos[dst]
        if getattr(d, 'offsets', None) is not None:
            vec = vec + to_np(d.offsets).astype(np.float64) @ cell
    return src, dst, vec


def with_edges(d, src, dst, vec, img, cell=None, pos=None):
    d2 = d.clone()
    if pos is not None:
        d2.pos = torch.as_tensor(pos, dtype=d.pos.dtype)
        d2.cell = torch.as_tensor(cell, dtype=d.cell.dtype).view(d.cell.shape)
    d2.edge_index = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    if getattr(d, 'edge_attr', None) is not None:
        if d.edge_attr[0].numel() != 1:
            raise ValueError('edge_attr is not a scalar distance: add its construction to with_edges')
        d2.edge_attr = torch.as_tensor(np.linalg.norm(vec, axis=1), dtype=d.edge_attr.dtype).view(-1, *d.edge_attr.shape[1:])
    if getattr(d, 'edge_vec', None) is not None:
        d2.edge_vec = torch.as_tensor(vec, dtype=d.edge_vec.dtype)
    if getattr(d, 'offsets', None) is not None:
        d2.offsets = torch.as_tensor(img, dtype=d.offsets.dtype)
    return d2


def extra_edge_keys(d):
    E, N = d.edge_index.size(1), d.num_nodes
    return [k for k, v in d.to_dict().items()
            if k not in EDGE_KEYS and torch.is_tensor(v) and v.dim() > 0 and v.size(0) == E and E != N]


def with_geo(d):
    if getattr(d, 'geo', None) is None and all(getattr(d, g, None) is not None for g in GEO_KEYS):
        d = d.clone()
        d.geo = torch.stack([d[g].view(-1) for g in GEO_KEYS], dim=1).float()
    return d


@torch.no_grad()
def run_model(model, cache, d, device):
    out = model(Batch.from_data_list([with_geo(d)]).to(device))
    return to_np(cache['h']).astype(np.float64), to_np(out[0]).astype(np.float64).ravel()


def spread(h, g):
    return float(np.abs(h[g] - h[g].mean(0)).max() / max(np.abs(h).mean(), 1e-12))


def collect(dataset, candidates, args, model=None, device='cpu'):
    k, tol, cutoff = args.max_neighbors, args.tie_tol, args.cutoff
    n_sample = args.n_sample or len(candidates)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(np.asarray(candidates, int))

    cache, hook = {}, None
    if model is not None:
        model.eval()
        hook = model.encoder.register_forward_hook(
            lambda mod, inp, out: cache.__setitem__('h', out[0] if isinstance(out, (tuple, list)) else out))
    mof_rows, class_rows = [], []
    deg_stored, deg_tie = [], []
    n_ignored = 0
    bar = tqdm(total=n_sample, desc='tie check')
    try:
        for idx in order:
            if len(mof_rows) >= n_sample:
                break
            d = dataset[int(idx)]
            st, numbers, cell, pos = structure(d)
            n = len(numbers)
            eq, sg = equivalent_atoms(cell, pos, numbers, args.symprec)
            groups = classes(eq)
            if not groups:
                continue
            rep = np.asarray(eq, int)
            in_class = np.zeros(n, bool)
            in_class[np.concatenate(groups)] = True

            T = tie_aware(st, cutoff, k, tol)
            tie = T['gap'] <= tol
            D, _ = rows(T['c'], T['dist'], n, np.nan)
            D = D[:, :k]
            with np.errstate(invalid='ignore'):
                dev = np.nan_to_num(np.nanmax(np.abs(D - D[rep]), axis=1), nan=0.0)
            deg_t = T['deg']

            s_src, s_dst, s_vec = stored_edges(d)
            s_dist = np.linalg.norm(s_vec, axis=1)
            old = T['rank'] < k
            A_s, deg_s = rows(s_dst, s_dist, n, np.inf)
            A_o, deg_o = rows(T['c'][old], T['dist'][old], n, np.inf)
            recon = float(np.mean((deg_s == deg_o) & (row_diff(A_s, A_o) <= 1e-3)))

            bad_s = asymmetric_atoms(s_src, s_dst, s_vec, rep, n, args.atol, cutoff)
            bad_t = asymmetric_atoms(T['src'], T['dst'], T['vec'], rep, n, args.atol, cutoff)
            deg_stored.append(deg_s)
            deg_tie.append(deg_t)

            sym = symmetrize(cell, pos, numbers, args.symprec)
            X = None
            if sym is not None:
                st_x = Structure(Lattice(sym[0]), [INDEX_TO_ELEMENT[int(z)] for z in numbers], sym[1],
                                 coords_are_cartesian=True)
                X = tie_aware(st_x, cutoff, k, tol)
                X['cell'], X['pos'] = sym
                X['bad'] = asymmetric_atoms(X['src'], X['dst'], X['vec'], rep, n, args.atol, cutoff)

            cls_flags = [(bool(tie[g].any()), bool(bad_s[g].any()), bool(bad_t[g].any()), float(dev[g].max()))
                         for g in groups]
            n_asym_s = sum(f[1] for f in cls_flags)
            geom_dev = float(dev.max())
            if n_asym_s:
                group = 'asymmetric graph'
            elif geom_dev > args.geom_tol:
                group = 'approximate geometry'
            elif tie.any():
                group = 'tie, symmetric'
            else:
                group = 'no tie'
            name = getattr(d, 'name', None)
            name = name if isinstance(name, str) and name else str(int(idx))
            row = {'idx': int(idx), 'name': name, 'spacegroup': sg, 'n_atoms': n, 'n_classes': len(groups),
                   'n_atoms_in_classes': int(in_class.sum()), 'group': group,
                   'n_tie_atoms': int(tie.sum()), 'n_tie_atoms_in_classes': int((tie & in_class).sum()),
                   'n_tie_classes': sum(f[0] for f in cls_flags),
                   'n_asym_classes_stored': n_asym_s, 'n_asym_classes_tie': sum(f[2] for f in cls_flags),
                   'geom_dev': geom_dev, 'recon_match': recon,
                   'symmetrized': X is not None,
                   'n_asym_classes_exact': int(sum(X['bad'][g].any() for g in groups)) if X is not None else np.nan,
                   'deg_mean_stored': float(deg_s.mean()), 'deg_mean_tie': float(deg_t.mean()),
                   'deg_max_tie': int(deg_t.max()) if n else 0, 'min_gap': float(np.min(T['gap'])) if n else np.inf}

            h0 = h1 = h2 = None
            if model is not None:
                h0, q0 = run_model(model, cache, d, device)
                h1, q1 = run_model(model, cache, with_edges(d, T['src'], T['dst'], T['vec'], T['img']), device)
                changed = len(T['src']) != len(s_src) or not np.array_equal(np.sort(T['dst']), np.sort(s_dst))
                if changed and np.abs(h1 - h0).max() == 0:
                    n_ignored += 1
                row.update({'r_h_stored': within_ratio(h0, groups), 'r_h_tie': within_ratio(h1, groups),
                            'r_h_exact': np.nan, 'dQst_graph': float(np.abs(q1 - q0).max())})
                if X is not None:
                    h2, _ = run_model(model, cache, with_edges(d, X['src'], X['dst'], X['vec'], X['img'],
                                                               X['cell'], X['pos']), device)
                    row['r_h_exact'] = within_ratio(h2, groups)
            mof_rows.append(row)

            for j, (g, (t_, a_s, a_t, dv)) in enumerate(zip(groups, cls_flags)):
                cr = {'idx': int(idx), 'name': name, 'class': j, 'element': INDEX_TO_ELEMENT[int(numbers[g[0]])],
                      'size': len(g), 'tie': t_, 'asym_stored': a_s, 'asym_tie': a_t, 'geom_dev': dv,
                      'deg_stored': '/'.join(map(str, sorted(set(deg_s[g])))),
                      'deg_tie': '/'.join(map(str, sorted(set(deg_t[g]))))}
                if h0 is not None:
                    cr.update({'h_spread_stored': spread(h0, g), 'h_spread_tie': spread(h1, g),
                               'h_spread_exact': spread(h2, g) if h2 is not None else np.nan})
                class_rows.append(cr)
            bar.update(1)
    finally:
        bar.close()
        if hook is not None:
            hook.remove()
    deg = {'stored': np.concatenate(deg_stored) if deg_stored else np.zeros(0, int),
           'tie-aware': np.concatenate(deg_tie) if deg_tie else np.zeros(0, int)}
    return pd.DataFrame(mof_rows), pd.DataFrame(class_rows), deg, n_ignored


def summarize(mofs, cls, deg, n_ignored, args, has_model):
    thr = args.r_thr
    s = {'n_mof': len(mofs), 'n_class': len(cls),
         'recon_match_mean': mofs['recon_match'].mean(),
         'frac_mof_with_tie': mofs['n_tie_atoms'].gt(0).mean(),
         'frac_class_with_tie': cls['tie'].mean(),
         'frac_mof_asym_stored': mofs['n_asym_classes_stored'].gt(0).mean(),
         'frac_mof_asym_tie': mofs['n_asym_classes_tie'].gt(0).mean(),
         'frac_class_asym_stored': cls['asym_stored'].mean(),
         'frac_class_asym_tie': cls['asym_tie'].mean(),
         'frac_asym_stored_among_tie_classes': cls.loc[cls['tie'], 'asym_stored'].mean(),
         'frac_asym_stored_among_no_tie_classes': cls.loc[~cls['tie'], 'asym_stored'].mean(),
         'frac_mof_approx_geometry': mofs['geom_dev'].gt(args.geom_tol).mean(),
         'frac_mof_symmetrized': mofs['symmetrized'].mean(),
         'frac_mof_asym_exact': mofs['n_asym_classes_exact'].gt(0).mean(),
         'deg_mean_stored': deg['stored'].mean(), 'deg_mean_tie': deg['tie-aware'].mean(),
         'deg_max_tie': deg['tie-aware'].max(), 'frac_atoms_deg_above_k': np.mean(deg['tie-aware'] > args.max_neighbors)}
    for g in GROUPS:
        s[f'n_mof[{g}]'] = int(mofs['group'].eq(g).sum())

    r = mofs['r_h_ref']
    if r.notna().any():
        broken = r > thr
        s['frac_mof_r_h_above_thr'] = broken.mean()
        for g in GROUPS:
            sel = mofs['group'].eq(g) & r.notna()
            s[f'frac_r_h_above_thr[{g}]'] = broken[sel].mean() if sel.any() else np.nan
        if broken.any():
            for g in GROUPS:
                s[f'share_of_broken[{g}]'] = mofs.loc[broken, 'group'].eq(g).mean()
    if has_model:
        both = mofs[list(STAGES.values())].notna().all(1)
        for lab, col in STAGES.items():
            v = mofs[col].dropna()
            s.update({f'frac_r_h_above_thr[{lab}]': v.gt(thr).mean(), f'r_h_mean[{lab}]': mofs.loc[both, col].mean(),
                      f'r_h_median[{lab}]': v.median(), f'r_h_q90[{lab}]': v.quantile(0.9), f'r_h_max[{lab}]': v.max()})
        m0 = s['r_h_mean[stored]']
        s['r_h_removed_by_tie_aware'] = 1 - s['r_h_mean[tie-aware]'] / m0 if m0 > 0 else np.nan
        s['r_h_removed_by_tie_aware_and_exact'] = 1 - s['r_h_mean[exact]'] / m0 if m0 > 0 else np.nan
        b0 = mofs['r_h_stored'] > thr
        b1 = mofs['r_h_tie'] > thr
        b2 = mofs['r_h_exact'] > thr
        nx = mofs['r_h_exact'].isna()
        if b0.any():
            s.update({'broken_fixed_by_tie_aware': (~b1[b0]).mean(),
                      'broken_fixed_only_by_exact_symmetry': (b1 & ~b2 & ~nx)[b0].mean(),
                      'broken_left_on_exact_graph': (b1 & b2)[b0].mean(),
                      'broken_not_symmetrized': (b1 & nx)[b0].mean()})
        s.update({'dQst_graph_median': mofs['dQst_graph'].median(), 'dQst_graph_q90': mofs['dQst_graph'].quantile(0.9),
                  'dQst_graph_max': mofs['dQst_graph'].max(), 'n_encoder_ignored_edges': n_ignored})
    return pd.Series(s, dtype=object), verdict(s, has_model)


def verdict(s, has_model):
    if has_model and np.isfinite(s.get('broken_fixed_by_tie_aware', np.nan)):
        f = s['broken_fixed_by_tie_aware']
        g = s['broken_fixed_only_by_exact_symmetry']
        e = s['broken_left_on_exact_graph']
        dt = s['r_h_removed_by_tie_aware']
        dx = s['r_h_removed_by_tie_aware_and_exact']
        head = (f'mean r_h falls by {dt:.0%} on the tie-aware graph and by {dx:.0%} on the exact graph. '
                f'Of the MOFs with r_h > thr: {f:.0%} fixed by the tie-aware graph, {g:.0%} only by exact symmetry '
                f'(coordinate precision), {e:.0%} left on the exact graph (encoder). ')
        if dt >= 0.5:
            tail = '; most of the rest comes from approximate coordinates' if dx - dt >= 0.2 else ''
            return head + 'The ties carry most of the symmetry breaking: re-preprocess with the tie-aware graph' + tail
        if dx >= 0.5:
            return head + ('Much of it comes from approximate coordinates: the tie-aware graph alone is not enough '
                           '(symmetrize the structures or qualify the invariance claim)')
        return head + 'Mostly the encoder: check it before re-preprocessing'
    if np.isfinite(s.get('frac_mof_r_h_above_thr', np.nan)):
        g = s.get('share_of_broken[asymmetric graph]', 0.0) + s.get('share_of_broken[tie, symmetric]', 0.0)
        a = s.get('frac_mof_approx_geometry', 0.0)
        tail = (f'; {a:.0%} of the MOFs are only approximately symmetric, so confirm with --model_check 1'
                if a > 0.1 else '; confirm with --model_check 1')
        if g >= 0.5:
            return f'{g:.0%} of the MOFs with r_h > thr have a tie in the stored graph: consistent with the ties' + tail
        return f'only {g:.0%} of the MOFs with r_h > thr have a tie: the graph is not the main cause' + tail
    return 'no r_h available (run symmetry_check first or use --model_check 1): graph statistics only'


def _floor(v):
    return np.clip(np.asarray(v, float), R_FLOOR, None)


def _set_floor(args):
    global R_FLOOR
    R_FLOOR = min(1e-8, args.r_thr * 1e-2)


def panel_groups(ax, mofs, args):
    r = mofs['r_h_ref']
    present = [g for g in GROUPS if (mofs['group'].eq(g) & r.notna()).any()]
    data = [_floor(r[mofs['group'].eq(g) & r.notna()]) for g in present]
    rng = np.random.default_rng(0)
    for i, (g, v) in enumerate(zip(present, data)):
        bp = ax.boxplot([v], positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        bp['boxes'][0].set_facecolor(GROUP_COLOR[g]); bp['boxes'][0].set_alpha(0.35)
        bp['boxes'][0].set_edgecolor(INK); bp['boxes'][0].set_linewidth(0.6)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=4, color=GROUP_COLOR[g], edgecolors='none', alpha=0.7, zorder=3)
        ax.text(i, 2.0, f'{np.mean(v > args.r_thr):.0%}', ha='center', va='bottom', fontsize=6, color=INK)
    ax.axhline(args.r_thr, color=MUTED, lw=0.6, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(R_FLOOR / 3, 30)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f'{g}\n(n = {len(v)})' for g, v in zip(present, data)], fontsize=6)
    ax.set_ylabel(r'$r_h$ on the stored graph')
    ax.set_title(f'share above {args.r_thr:g} on top', fontsize=6, color=MUTED, loc='left')


def panel_classes(ax, cls):
    cats = [('tie', cls['tie']), ('no tie', ~cls['tie'])]
    w = 0.36
    for j, (lab, col) in enumerate((('stored', 'asym_stored'), ('tie-aware', 'asym_tie'))):
        vals = [cls.loc[m, col].mean() if m.any() else 0.0 for _, m in cats]
        x = np.arange(len(cats)) + (j - 0.5) * w
        ax.bar(x, vals, width=w * 0.92, color=GRAPH_COLOR[lab], label=f'{lab} graph')
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=6, color=INK)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f'{lab} at its own cut\n({int(m.sum())} classes)' for lab, m in cats], fontsize=6)
    ax.set_ylabel('Share of classes with an\nasymmetric neighbourhood')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')


def panel_model(ax, mofs, args):
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.22, 0.22, len(mofs))
    for i, (lab, col) in enumerate(STAGES.items()):
        v = mofs[col]
        ok = v.notna().values
        bp = ax.boxplot([_floor(v[ok])], positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        bp['boxes'][0].set_facecolor('none'); bp['boxes'][0].set_edgecolor(INK); bp['boxes'][0].set_linewidth(0.6)
        for g in GROUPS:
            sel = ok & mofs['group'].eq(g).values
            if sel.any():
                ax.scatter(i + jitter[sel], _floor(v[sel]), s=4, color=GROUP_COLOR[g], edgecolors='none', alpha=0.7,
                           zorder=3, label=g if i == 0 else None)
        ax.text(i, 2.0, f'{np.mean(v[ok] > args.r_thr):.0%}', ha='center', va='bottom', fontsize=6, color=INK)
    ax.axhline(args.r_thr, color=MUTED, lw=0.6, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(R_FLOOR / 3, 30)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(['stored', 'tie-aware', 'tie-aware,\nexact symmetry'], fontsize=6)
    ax.set_ylabel(r'$r_h$, same checkpoint')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=5.5, markerscale=1.5, ncol=2,
              handletextpad=0.2, columnspacing=0.6)


def panel_degree(ax, deg, args):
    hi = int(max(deg['stored'].max(), deg['tie-aware'].max()))
    lo = int(min(deg['stored'].min(), deg['tie-aware'].min(), args.max_neighbors))
    xs = np.arange(lo, hi + 1)
    w = 0.4
    for j, lab in enumerate(('stored', 'tie-aware')):
        frac = np.array([np.mean(deg[lab] == x) for x in xs])
        ax.bar(xs + (j - 0.5) * w, frac, width=w * 0.92, color=GRAPH_COLOR[lab],
               label=f'{lab} (mean {deg[lab].mean():.2f})')
    ax.set_yscale('log')
    ax.set_xticks(xs)
    ax.set_xlabel('Incoming edges per atom')
    ax.set_ylabel('Share of atoms')
    ax.legend(loc='upper right', fontsize=6)


def figure(mofs, cls, deg, args, has_model, out_dir):
    _set_floor(args)
    panels = []
    if mofs['r_h_ref'].notna().any():
        panels.append(lambda ax: panel_groups(ax, mofs, args))
    panels.append(lambda ax: panel_classes(ax, cls))
    if has_model:
        panels.append(lambda ax: panel_model(ax, mofs, args))
    panels.append(lambda ax: panel_degree(ax, deg, args))
    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(W2, 2.5))
        axes = np.atleast_1d(axes)
    for ax, f, lab in zip(axes, panels, 'abcd'):
        f(ax)
        ax.text(-0.18, 1.04, lab, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    fig.tight_layout()
    save(fig, out_dir, 'fig_tie_check')


def report(mofs, cls, deg, n_ignored, args, has_model, out_dir):
    set_style()
    os.makedirs(out_dir, exist_ok=True)
    summary, text = summarize(mofs, cls, deg, n_ignored, args, has_model)
    summary['verdict'] = text
    mofs.to_csv(os.path.join(out_dir, 'tie_mof.csv'), index=False)
    cls.to_csv(os.path.join(out_dir, 'tie_class.csv'), index=False)
    summary.to_frame('value').to_csv(os.path.join(out_dir, 'tie_summary.csv'))
    figure(mofs, cls, deg, args, has_model, out_dir)

    fmt = lambda v: f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v)
    print(f'\n[tie check] {len(mofs)} MOFs with equivalent atoms -> {out_dir}')
    for k_, v in summary.items():
        if k_ != 'verdict':
            print(f'  {k_:45s} {fmt(v)}')
    if summary['recon_match_mean'] < 0.99:
        print('  WARNING: the stored graph is not reproduced by the 12-nearest rule on the stored positions '
              '(check --cutoff / --max_neighbors)')
    if n_ignored:
        print(f'  WARNING: in {n_ignored} MOFs the embedding did not change with the edges: the encoder may build '
              'its own graph, so the model rows say nothing about the preprocessing')
    if has_model and len(cls):
        worst = cls.sort_values('h_spread_stored', ascending=False).head(10)
        print('\n--- classes with the largest embedding spread on the stored graph ---')
        print(worst[['name', 'element', 'size', 'tie', 'asym_stored', 'asym_tie', 'deg_stored', 'deg_tie', 'geom_dev',
                     'h_spread_stored', 'h_spread_tie', 'h_spread_exact']]
              .to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print(f'\n[verdict] {text}')
    return summary


def run_tie_check(dataset, args, model=None, device='cpu', best_model_path=None, sym_csv=None, out_dir=None,
                  load_weights=True):
    if model is not None and load_weights:
        _load_weights(model, best_model_path, device)
    if model is not None:
        extra = extra_edge_keys(dataset[0])
        if extra:
            raise ValueError(f'per-edge fields {extra} are not rebuilt for the tie-aware graph: add them to with_edges '
                             'or run with --model_check 0')
    sym = None
    if sym_csv and os.path.isfile(sym_csv):
        sym = pd.read_csv(sym_csv)
        candidates = sym.loc[sym['n_classes'] > 0, 'idx'].astype(int).values
        print(f'[tie check] {len(candidates)} MOFs with equivalent atoms in {sym_csv}')
    else:
        candidates = np.arange(len(dataset))
        print('[tie check] no symmetry_mof.csv: equivalent atoms are found on the fly' + (f' ({sym_csv} not found)' if sym_csv else ''))
    mofs, cls, deg, n_ignored = collect(dataset, candidates, args, model, device)
    if mofs.empty:
        raise RuntimeError('no MOF with equivalent atoms in the sample')
    if sym is not None:
        keep = [c for c in ('r_h', 'r_e', 'std_e_max', 'type_agree', 'dQst_max', 'dh_rel') if c in sym]
        mofs = mofs.merge(sym[['idx'] + keep].rename(columns={c: f'{c}_symcheck' for c in keep}), on='idx', how='left')
    if model is not None:
        mofs['r_h_ref'] = mofs['r_h_stored']
    elif 'r_h_symcheck' in mofs:
        mofs['r_h_ref'] = mofs['r_h_symcheck']
    else:
        mofs['r_h_ref'] = np.nan
    return report(mofs, cls, deg, n_ignored, args, model is not None, out_dir)


if __name__ == '__main__':
    import argparse
    flag = lambda s: str(s).lower() in ('1', 'true', 'yes')
    ap = argparse.ArgumentParser(description='12-nearest-neighbour tie check')
    ap.add_argument('--desc', default="base", type=str)
    ap.add_argument('--data_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent', type=str)
    ap.add_argument('--output_path', default='../results_hmof', type=str)
    ap.add_argument('--best_model_path', default='', type=str,
                    help='empty: <output_path>/<encoder>/<target>_<encoder>_<lambda_aux>_<pool>_<attn>_<mode>_fin/best_<seed>.pt as in train.py')
    ap.add_argument('--sym_csv', default='', type=str, help='empty: <checkpoint dir>/symmetry_check/symmetry_mof.csv')
    ap.add_argument('--out_dir', default='', type=str, help='empty: <checkpoint dir>/tie_check')
    ap.add_argument('--target', default='heat_co2', type=str, choices=['heat_co2', 'heat_ch4'])
    ap.add_argument('--gpu', default='0', type=str)
    ap.add_argument('--seed', default=42, type=int)
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
    ap.add_argument('--cutoff', default=6.0, type=float, help='A, graph cutoff of the preprocessing')
    ap.add_argument('--max_neighbors', default=12, type=int, help='neighbours kept per atom in the preprocessing')
    ap.add_argument('--tie_tol', default=1e-2, type=float, help='A, tie tolerance of the tie-aware rule')
    ap.add_argument('--symprec', default=1e-2, type=float, help='A, spglib tolerance (same as symmetry_check)')
    ap.add_argument('--atol', default=3e-2, type=float, help='A, distance tolerance when comparing neighbourhoods')
    ap.add_argument('--geom_tol', default=1e-4, type=float, help='A, above this equivalent atoms are only approximately equivalent')
    ap.add_argument('--r_thr', default=1e-6, type=float, help='r_h above this counts as broken symmetry')
    ap.add_argument('--n_sample', default=1000, type=int, help='MOFs with equivalent atoms to check (0: all)')
    ap.add_argument('--model_check', default=True, type=flag, help='1: rerun the checkpoint on both graphs')
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    TARGETS = {'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
               'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5']}
    PRESSURES = {'heat_co2': [0.01, 0.05, 0.1, 0.5, 2.5], 'heat_ch4': [0.05, 0.5, 0.9, 2.5, 4.5]}
    HEAT_STD = {'heat_co2': [6.518, 6.084, 6.014, 6.282, 6.678], 'heat_ch4': [4.917, 4.804, 4.780, 4.733, 4.684]}
    LOAD_STD = {'heat_co2': [0.2098, 0.4950, 0.6960, 1.3995, 2.5644], 'heat_ch4': [0.1070, 0.4802, 0.6532, 1.0429, 1.3295]}
    args.pressures = PRESSURES[args.target]
    args.heat_std = HEAT_STD[args.target]
    args.load_std = LOAD_STD[args.target]
    args.n_out = len(args.pressures)
    args.gas = args.target.split('_')[-1]
    if not args.best_model_path:
        ver = f'{args.target}_{args.encoder}_{args.lambda_aux}_{args.pool}_{args.attn}_{args.mode}_fin'
        args.best_model_path = os.path.join(args.output_path, args.encoder, ver, f'best_{args.seed}.pt')
    run_dir = os.path.dirname(args.best_model_path)
    sym_csv = args.sym_csv or os.path.join(run_dir, 'symmetry_check', 'symmetry_mof.csv')
    out_dir = args.out_dir or os.path.join(run_dir, 'tie_check')

    from utils.utils import seed_everything, PygDataset
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PygDataset(root=f'{args.data_path}/test', target_name=TARGETS[args.target], gas=args.gas)
    if len(dataset) == 0:
        raise FileNotFoundError(f'empty test set: {os.path.abspath(args.data_path)}/test')
    print(f'[tie check] data: {len(dataset)} test structures from {os.path.abspath(args.data_path)}/test')

    model = None
    if args.model_check:
        if not os.path.isfile(args.best_model_path):
            raise FileNotFoundError(f'checkpoint not found: {args.best_model_path} (or run with --model_check 0)')
        print(f'[tie check] checkpoint: {args.best_model_path}')
        from model.hc_model import Hierarchical_Clustering_Model_With_LA
        model = Hierarchical_Clustering_Model_With_LA(args).to(device)
    run_tie_check(dataset, args, model, device, args.best_model_path, sym_csv, out_dir)
