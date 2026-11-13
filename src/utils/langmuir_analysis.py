"""
One analysis for the Langmuir-site model (Hierarchical_Clustering_Model_With_LA; also the older _Pore_Inc models).

  heat_mode 'pore_g'  : Qst(p) = sum_c alpha_c e_c + alpha_pore (e_pore + g(p))      g on the pore site only
  heat_mode 'lateral' : Qst(p) = sum_t alpha_t (e_t + g(p)) = (1-a)F + a e_pore + g  g on every site
  (args.heat_mode = 'pore_g' | 'lateral' | 'auto'; 'auto' picks the form that reconstructs the prediction)
  mean-field models (mode ...+mf): g = u(theta_total) = a_1 theta + a_2 theta^2 (Fowler-Guggenheim), the lateral form;
  u is also inside the isotherm, so N(p) and theta_c already carry it. Labels then read 'u' instead of 'g', and the
  summary adds mf_* rows: slopes u'(0) = a_1, u'(1) = a_1 + 2 a_2, share at the slope bound, Var(u)/Var(Qst),
  |du| / |dQst| from p_min to p_max, attractive share, Spearman of u(1) and u'(0) with the pore descriptors
  alpha_c = n_c dtheta_c/dp / sum = dN_c/dN,   N(p) = sum_c n_c theta_c(p)  (compared with GCMC loading)

Works for CO2 and CH4: the gas is args.gas (or read from args.target), every '@p' label uses the real pressures
(p_max is the last pressure, e.g. 2.5 bar for CO2, 4.5 bar for CH4).

Options (read from args, all optional)
  args.gas      : 'co2' | 'ch4'            (default: from args.target)
  args.sat_by   : 'fit'  -> theta of a single-site Langmuir fit of the GCMC loading at p_max, fixed bins
                  'curv' -> N(p_min)/N(p_max) of the GCMC loading (no fit), quintiles
                  'lcd'  -> largest cavity diameter, small LCD = high saturation (no fit), quintiles
  args.heat_mode, args.g_mode ('poly'), args.pore_ps (output folder suffix), args.mode (label only)

Intermediate values (n_c, K_c, theta_c, g coefficients) are read with forward hooks on model.sites and
model.pore_inc, so the model code does not need to change.

Sections
  1. accuracy          : MAE per pressure, by saturation bin
  2. loading (N_model) : isotherm shape vs GCMC, scale, theta_total vs Langmuir fit of GCMC, outliers
  3. saturation        : saturation score, volumetric filling (gas-specific liquid density), heat change by bin
  4. pore vs framework : alpha_pore, F (framework heat given the destination), e_pore(p), raw contributions,
                         exact decomposition of dQst (framework / pore share shift / pore heat change / interaction g)
  5. framework sites   : strongest / weakest cluster alpha, within-MOF Spearman(log K, e),
                         site entropy dS_c = R (ln K_c - e_c/RT) with the gas-specific window (-S_max, 0)
  5b. thermodynamic link (thermo modes, detected from model.sites.lnK0_lo; also reported for free-K models):
                         ln K = ln K0 + e/RT makes Spearman(log K, e) structural -> not evidence;
                         checked instead: dS piling at the bounds (sigmoid saturation; pore at -S_max = the model
                         tries to inflate e_pore and cancel it with K0), enthalpy-entropy compensation
                         Spearman(e, dS) (expected < 0), log n (n-K trade-off in the Henry regime)
                         free-K models: dS = R ln K - e/T contains -e, so Spearman(e, dS) is biased negative by
                         construction there (read rho(log K, e) instead); in thermo modes dS = R ln K0 comes from its
                         own head, so the compensation sign is a real check
  6. g                 : g(p), coefficients a_k, g(theta = 1),
                         shape of g on the OBSERVED theta_total range of each MOF (not extrapolated to theta = 1)

Outputs (out_dir = dirname(best_model_path)/langmuir_analysis_{args.pore_ps}[_{sat_by}]_{args.seed})
  summary.csv, by_pressure.csv, by_sat_bin.csv, decomposition.csv, g_by_sat_bin.csv,
  loading_shape_by_bin.csv, loading_outliers.csv, per_mof.csv,
  fig_heat.png, fig_loading.png, fig_sites_g.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

R_J = 8.314462618            # J/mol/K
R_KJ = R_J * 1e-3            # kJ/mol/K
N_A = 6.02214076e23
T_K = 298.0
GAS = {                      # liquid density (g/cm3, upper-bound packing), molar mass (g/mol),
    'co2': dict(rho_liq=1.18, M=44.01, S_max=210.8),     # entropy the adsorbate can lose (J/mol/K)
    'ch4': dict(rho_liq=0.4224, M=16.04, S_max=143.5),
}
CELL_KEYS = ['cell', 'lattice', 'lattice_matrix']
SAT_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0001]
SAT_LABELS = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.85', '>=0.85']
N_QUANT = 5
GEO_NAMES = ['void_fraction', 'gsa', 'vsa', 'pld', 'lcd']
K_GRID = np.logspace(-4, 4, 801)


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


def _gas(args):
    g = getattr(args, 'gas', None)
    if g is None:
        t = str(getattr(args, 'target', '')).lower()
        g = 'ch4' if 'ch4' in t else ('co2' if 'co2' in t else None)
    g = None if g is None else str(g).lower()
    if g not in GAS:
        raise ValueError(f"unknown gas {g!r}: set args.gas to one of {list(GAS)}")
    return g


def _check_pressures(model, p):
    lp = getattr(getattr(model, 'sites', None), 'log_p', None)
    if isinstance(lp, torch.Tensor) and lp.numel() == len(p):
        pm = lp.detach().exp().cpu().numpy().ravel()
        if not np.allclose(pm, p, rtol=1e-3):
            raise ValueError(f"pressure mismatch: model {pm} vs analysis {p}")


def langmuir_fit(N, p, k_grid=K_GRID):
    th = (k_grid[:, None] * p[None, :]) / (1.0 + k_grid[:, None] * p[None, :])
    a = th[None, :, :] / N[:, None, :]
    nmax = a.sum(-1) / (a ** 2).sum(-1).clip(1e-30)
    res = ((1.0 - nmax[..., None] * a) ** 2).mean(-1)
    j = res.argmin(1)
    return k_grid[j], np.sqrt(res[np.arange(N.shape[0]), j])


def _rank_spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return spearmanr(x[ok], y[ok]).correlation


def _sites_out(o):
    if isinstance(o, dict):
        return o['log_n'], o['log_K'], o['theta']
    return o[0], o[1], o[2]


def _poly(coef, t):
    out = np.zeros_like(t)
    for k in range(coef.shape[1]):
        out = out + coef[:, k].reshape((-1,) + (1,) * (t.ndim - 1)) * t ** (k + 1)
    return out


def g_shape_observed(coef, th_lo, th_hi, n_grid=41, tol=1e-6):
    s = np.linspace(0.0, 1.0, n_grid)[None, :]
    t = th_lo[:, None] + (th_hi - th_lo)[:, None] * s
    gv = _poly(coef, t)
    gv = gv - gv[:, :1]
    dg = np.diff(gv, axis=1)
    scale = np.abs(gv).max(1, keepdims=True).clip(1e-12)
    up, dn = dg > tol * scale, dg < -tol * scale
    inc, dec = ~dn.any(1), ~up.any(1)
    first, last = np.sign(dg[:, 0]), np.sign(dg[:, -1])
    shape = np.select([inc & dec, inc, dec, (first > 0) & (last < 0), (first < 0) & (last > 0)],
                      ['flat', 'increasing', 'decreasing', 'rise_then_fall', 'fall_then_rise'], 'mixed')
    return shape, gv.min(1), gv[:, -1]


def saturation_bins(mode, theta_fit, Ng, lcd, pmn, pmx):
    if mode == 'fit':
        score = theta_fit[:, -1]
        return score, pd.cut(score, bins=SAT_BINS, labels=SAT_LABELS, right=False), SAT_LABELS, f'θ_fit@{pmx}'
    if mode == 'curv':
        score = Ng[:, 0] / np.where(Ng[:, -1] > 0, Ng[:, -1], np.nan)
        name = f'N({pmn})/N({pmx}) quintile'
    elif mode == 'lcd':
        score = -lcd
        name = 'LCD quintile (q5 = smallest)'
    else:
        raise ValueError(f"sat_by must be 'fit', 'curv' or 'lcd', got {mode!r}")
    labels = [f'q{i + 1}' for i in range(N_QUANT)]
    return score, pd.qcut(score, q=N_QUANT, labels=labels), labels, name


@torch.no_grad()
def collect(model, loader, device, P):
    model.eval()
    cache = {}
    h1 = model.sites.register_forward_hook(lambda mod, inp, out: cache.__setitem__('sites', out))
    h2 = model.pore_inc.register_forward_hook(lambda mod, inp, out: cache.__setitem__('inc', out))

    mof = {k: [] for k in ['y', 'pred', 'y_load', 'N_model', 'theta_total', 'alpha_pore', 'F', 'e_pore0',
                           'e_pore_p', 'g', 'coef', 'frame_contrib', 'pore_contrib', 'geo', 'vol', 'mass', 'C',
                           'n_pore_share', 'logK_pore', 'theta_pore', 'e_mean', 'e_max', 'e_min',
                           'a_strong', 'a_weak', 'logK_strong', 'theta_strong', 'rho_logK_e', 'dS_frame_mean', 'dS_pore',
                           'rho_e_dS', 'logn_pore']}
    tok_dS_frame, tok_e_frame, tok_logn_frame = [], [], []
    has_load = True
    RT = R_KJ * T_K
    try:
        for data in tqdm(loader, desc="langmuir analysis"):
            data = data.to(device)
            out = model(data)
            pred, extras, m, alpha = out[0], out[1], out[2], out[3]
            e, theta_total, e_pore_p = extras[0], extras[3], extras[4]
            B = pred.shape[0]
            log_n, log_K, theta = _sites_out(cache['sites'])
            g, coef = cache['inc'][0], cache['inc'][1]

            mf = m.to(pred.dtype)
            n = torch.exp(log_n) * mf
            N_model = (n.unsqueeze(-1) * theta).sum(1)
            a_pore = alpha[:, -1]
            frame_contrib = (alpha[:, :-1] * e[:, :-1].unsqueeze(-1)).sum(1)
            F = frame_contrib / (1 - a_pore).clamp_min(1e-8)

            mof['y'].append(data.y.view(B, P).cpu()); mof['pred'].append(pred.cpu())
            if has_load and getattr(data, 'y_load', None) is not None:
                mof['y_load'].append(data.y_load.view(B, P).cpu())
            else:
                has_load = False
            mof['N_model'].append(N_model.cpu()); mof['theta_total'].append(theta_total.reshape(B, P).cpu())
            mof['alpha_pore'].append(a_pore.cpu()); mof['F'].append(F.cpu())
            mof['e_pore0'].append(e[:, -1].cpu()); mof['e_pore_p'].append(e_pore_p.cpu()); mof['g'].append(g.cpu())
            mof['coef'].append(coef.reshape(B, -1).cpu())
            mof['frame_contrib'].append(frame_contrib.cpu()); mof['pore_contrib'].append((a_pore * e_pore_p).cpu())
            mof['geo'].append(data.geo.view(B, -1).cpu())
            cell = next((getattr(data, k) for k in CELL_KEYS if getattr(data, k, None) is not None), None)
            mass = getattr(data, 'mass_cell', None)
            if cell is not None and mass is not None and cell.reshape(-1, 3, 3).shape[0] == B:
                mof['vol'].append(torch.linalg.det(cell.reshape(-1, 3, 3).double()).abs().float().cpu())
                mof['mass'].append(mass.view(B).float().cpu())
            else:
                mof['vol'].append(torch.full((B,), float('nan'))); mof['mass'].append(torch.full((B,), float('nan')))
            mof['n_pore_share'].append((n[:, -1] / n.sum(1).clamp_min(1e-12)).cpu())
            mof['logK_pore'].append(log_K[:, -1].cpu()); mof['theta_pore'].append(theta[:, -1].cpu())

            e_np, lk_np, th_np, ln_np = e.cpu().numpy(), log_K.cpu().numpy(), theta.cpu().numpy(), log_n.cpu().numpy()
            al_np, mk_np = alpha.cpu().numpy(), m.bool().cpu().numpy()
            for b in range(B):
                fr = np.where(mk_np[b, :-1])[0]
                C = len(fr)
                mof['C'].append(C)
                ef, lkf = e_np[b, fr], lk_np[b, fr]
                dS = R_J * (lkf - ef / RT)                                               # J/mol/K
                tok_dS_frame.append(dS); tok_e_frame.append(ef); tok_logn_frame.append(ln_np[b, fr])
                mof['dS_frame_mean'].append(dS.mean() if C else np.nan)
                mof['dS_pore'].append(R_J * (lk_np[b, -1] - e_np[b, -1] / RT))
                mof['logn_pore'].append(ln_np[b, -1])
                mof['rho_e_dS'].append(_rank_spearman(ef, dS) if C >= 3 else np.nan)
                mof['e_mean'].append(ef.mean() if C else np.nan)
                if C >= 2:
                    i_hi, i_lo = fr[np.argmax(ef)], fr[np.argmin(ef)]
                    mof['e_max'].append(e_np[b, i_hi]); mof['e_min'].append(e_np[b, i_lo])
                    mof['a_strong'].append(al_np[b, i_hi]); mof['a_weak'].append(al_np[b, i_lo])
                    mof['logK_strong'].append(lk_np[b, i_hi]); mof['theta_strong'].append(th_np[b, i_hi])
                else:
                    nanP = np.full(P, np.nan)
                    mof['e_max'].append(np.nan); mof['e_min'].append(np.nan)
                    mof['a_strong'].append(nanP); mof['a_weak'].append(nanP)
                    mof['logK_strong'].append(np.nan); mof['theta_strong'].append(nanP)
                mof['rho_logK_e'].append(_rank_spearman(lkf, ef) if C >= 3 else np.nan)
    finally:
        h1.remove(); h2.remove()

    d = {}
    for k, v in mof.items():
        if not v:
            continue
        if isinstance(v[0], torch.Tensor):
            d[k] = torch.cat(v).numpy().astype(np.float64)
        else:
            d[k] = np.array(v, dtype=np.float64)
    if not has_load:
        d.pop('y_load', None)
    for k, v in [('tok_dS_frame', tok_dS_frame), ('tok_e_frame', tok_e_frame), ('tok_logn_frame', tok_logn_frame)]:
        d[k] = np.concatenate(v).astype(np.float64) if v else np.array([])
    return d


def analyze_langmuir(model, best_model_path, loader, device, args, load_weights=True):
    gas = _gas(args)
    gp = GAS[gas]
    sat_by = getattr(args, 'sat_by', 'fit')
    pressures = list(args.pressures)
    p = np.asarray(pressures, dtype=np.float64)
    P = len(p)
    i05 = int(np.argmin(np.abs(p - 0.5)))
    pl = [str(x) for x in pressures]
    pmn, pmx, p05 = pl[0], pl[-1], pl[i05]
    iv = f'{p05}to{pmx}'
    out_dir = os.path.join(os.path.dirname(best_model_path), f"langmuir_analysis_{getattr(args, 'pore_ps', 'model')}")
    if sat_by != 'fit':
        out_dir += f'_{sat_by}'
    if getattr(args, 'seed', None) is not None:
        out_dir += f'_{args.seed}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"[gas] {gas}  pressures {pressures}  (p_max = {pmx} bar)  saturation by '{sat_by}'")

    if load_weights:
        _load_weights(model, best_model_path, device)
    model.to(device)
    _check_pressures(model, p)
    lo = getattr(getattr(model, 'sites', None), 'lnK0_lo', None)
    thermo = lo is not None
    S_max = -R_J * float(lo) if thermo else gp['S_max']
    slope_max = getattr(getattr(model, 'pore_inc', None), 'slope_max', None)
    mf = slope_max is not None
    gn = 'u' if mf else 'g'
    print(f"[thermo] {'ln K = ln K0 + e/RT, dS in (' + f'{-S_max:.1f}' + ', 0) J/mol/K' if thermo else 'free K (dS implied)'}"
          f"  mode = {getattr(args, 'mode', '?')}")
    d = collect(model, loader, device, P)
    n = d['y'].shape[0]
    y, pred = d['y'], d['pred']
    err = np.abs(pred - y)
    a, F, E, G = d['alpha_pore'], d['F'], d['e_pore_p'], d['g']
    recon_pg = (1 - a) * F + a * E
    recon_lat = recon_pg + G
    mode = getattr(args, 'heat_mode', 'auto')
    if mode not in ('pore_g', 'lateral'):
        mode = 'lateral' if np.abs(recon_lat - pred).max() < np.abs(recon_pg - pred).max() else 'pore_g'
    lateral = mode == 'lateral'
    H = G if lateral else np.zeros_like(G)
    recon = recon_pg + H
    print(f"\n[mode] heat_mode = {mode}  (max err pore_g form {np.abs(recon_pg - pred).max():.2e}, "
          f"lateral form {np.abs(recon_lat - pred).max():.2e})")
    print(f"[check] max |(1-α)F + α·e_pore(p) + g_all - pred| = {np.abs(recon - pred).max():.2e}")
    has_load = 'y_load' in d
    g_poly = getattr(args, 'g_mode', 'poly') == 'poly'
    Nm, tt = d['N_model'], d['theta_total']
    coef = d['coef'].reshape(n, -1)

    g_var_ok = False
    if g_poly:
        g_c = _poly(coef, tt) - _poly(coef, tt[:, :1])
        g_u = _poly(coef, tt)
        e_c, e_u = np.abs(g_c - G).max(), np.abs(g_u - G).max()
        g_var_ok = min(e_c, e_u) < 1e-3 * max(1.0, np.abs(G).max())
        print(f"[g] max |poly(theta_total) - g|: centred at p_min {e_c:.2e}, uncentred {e_u:.2e}"
              + ("" if g_var_ok else "  -> g is not a polynomial of theta_total; observed-range shape uses theta_total anyway"))

    theta_fit = np.full((n, P), np.nan); fit_err = np.full(n, np.nan); fill_vol = np.full(n, np.nan)
    Ng = d['y_load'] if has_load else np.full((n, P), np.nan)
    if has_load:
        ok = np.all(Ng > 0, axis=1) & np.all(np.isfinite(Ng), axis=1)
        K_fit = np.full(n, np.nan)
        if ok.any():
            K_fit[ok], fit_err[ok] = langmuir_fit(Ng[ok], p)
        theta_fit = K_fit[:, None] * p[None, :] / (1 + K_fit[:, None] * p[None, :])
        Vp = d['geo'][:, 0] * d['vol'] * 1e-24 / (d['mass'] / N_A)                     # cm3/g
        fill_vol = Ng[:, -1] / (Vp * gp['rho_liq'] / gp['M'] * 1000.0)
    else:
        print("[warn] data.y_load not found -> loading / saturation sections skipped")
    lcd = d['geo'][:, 4] if d['geo'].shape[1] > 4 else np.full(n, np.nan)
    if not has_load and sat_by in ('fit', 'curv'):
        sat_by = 'lcd'
    sat_score, sat_bin, labels, sat_name = saturation_bins(sat_by, theta_fit, Ng, lcd, pmn, pmx)

    intervals = [(j, j + 1) for j in range(P - 1)] + [(0, P - 1)]

    def terms(j1, j2):
        ab = 0.5 * (a[:, j1] + a[:, j2])
        t_f = (1 - ab) * (F[:, j2] - F[:, j1])
        t_s = (a[:, j2] - a[:, j1]) * 0.5 * ((E[:, j2] - F[:, j2]) + (E[:, j1] - F[:, j1]))
        t_g = ab * (E[:, j2] - E[:, j1])
        t_h = H[:, j2] - H[:, j1]
        return t_f, t_s, t_g, t_h

    dec_rows = []
    for j1, j2 in intervals:
        t_f, t_s, t_g, t_h = terms(j1, j2)
        tot = np.abs(t_f).mean() + np.abs(t_s).mean() + np.abs(t_g).mean() + np.abs(t_h).mean()
        dt = y[:, j2] - y[:, j1]
        dec_rows.append({'interval': f'{pl[j1]}->{pl[j2]}', 'dQ_true': dt.mean(), 'dQ_pred': (pred[:, j2] - pred[:, j1]).mean(),
                         'framework': t_f.mean(), 'pore_share_shift': t_s.mean(), 'pore_heat_change': t_g.mean(),
                         'interaction': t_h.mean(),
                         'frac_framework': np.abs(t_f).mean() / tot, 'frac_pore_share_shift': np.abs(t_s).mean() / tot,
                         'frac_pore_heat_change': np.abs(t_g).mean() / tot, 'frac_interaction': np.abs(t_h).mean() / tot,
                         'rho_framework_vs_true': _spearman(t_f, dt),
                         'rho_share_shift_vs_true': _spearman(t_s, dt),
                         'rho_heat_change_vs_true': _spearman(t_g, dt),
                         'rho_interaction_vs_true': _spearman(t_h, dt)})
    dec = pd.DataFrame(dec_rows)
    t_f05, t_s05, t_g05, t_h05 = terms(i05, P - 1)

    if has_load:
        okl = np.all(Ng > 0, axis=1) & np.all(Nm > 0, axis=1)
        lr_m = np.log(Nm.clip(1e-12)) - np.log(Nm[:, -1:].clip(1e-12))
        lr_g = np.log(Ng.clip(1e-12)) - np.log(Ng[:, -1:].clip(1e-12))
        shape_err = np.where(okl, np.abs(lr_m - lr_g)[:, :-1].mean(1), np.nan)
        log_scale = np.where(okl, np.log10(Ng[:, -1] / Nm[:, -1].clip(1e-12)), np.nan)
        abs_err = np.where(okl, np.abs(np.log(Nm.clip(1e-12)) - np.log(Ng.clip(1e-12))).mean(1), np.nan)
    else:
        shape_err = log_scale = abs_err = np.full(n, np.nan)

    tol_b = 0.05 * S_max
    dSf, dSp, ef_tok = d['tok_dS_frame'], d['dS_pore'], d['tok_e_frame']
    near_lo = lambda v: v < -S_max + tol_b
    near_hi = lambda v: v > -tol_b
    th_rows = {'thermo_link': float(thermo), 'S_max_used': S_max,
               'dS_frame_min': dSf.min() if dSf.size else np.nan, 'dS_frame_max': dSf.max() if dSf.size else np.nan,
               'dS_pore_min': dSp.min(), 'dS_pore_max': dSp.max(),
               'frac_dS_frame_at_lo': near_lo(dSf).mean() if dSf.size else np.nan,
               'frac_dS_frame_at_hi': near_hi(dSf).mean() if dSf.size else np.nan,
               'frac_dS_pore_at_lo': near_lo(dSp).mean(), 'frac_dS_pore_at_hi': near_hi(dSp).mean(),
               'frac_dS_frame_pos': (dSf > 0).mean() if dSf.size else np.nan, 'frac_dS_pore_pos': (dSp > 0).mean(),
               'rho_e_dS_frame_pooled': _spearman(ef_tok, dSf),
               'rho_e_dS_frame_within_median(C>=3)': np.nanmedian(d['rho_e_dS']),
               'frac_rho_e_dS_neg': (d['rho_e_dS'][np.isfinite(d['rho_e_dS'])] < 0).mean(),
               'rho_e_dS_pore_across_mof': _spearman(d['e_pore0'], dSp),
               'log10_n_frame_median': np.median(d['tok_logn_frame']) / np.log(10) if dSf.size else np.nan,
               'log10_n_pore_median': np.median(d['logn_pore']) / np.log(10),
               'log10_n_pore_p99': np.percentile(d['logn_pore'], 99) / np.log(10)}
    if thermo:
        out_b = (dSf < -S_max - 1e-3).sum() + (dSf > 1e-3).sum() + (dSp < -S_max - 1e-3).sum() + (dSp > 1e-3).sum()
        print(f"[thermo] tokens outside (-S_max, 0): {int(out_b)} (should be 0)")
    tok_bin = np.repeat(np.arange(n), d['C'].astype(int))

    per = pd.DataFrame({'C': d['C'], 'sat_score': sat_score, 'sat_bin': sat_bin,
                        f'theta_fit@{pmx}': theta_fit[:, -1], f'fill_vol@{pmx}': fill_vol,
                        'langmuir_fit_err': fit_err, f'theta_total@{pmx}': tt[:, -1], f'theta_pore@{pmx}': d['theta_pore'][:, -1],
                        'n_pore_share': d['n_pore_share'], 'log10_halfp_pore': -d['logK_pore'] / np.log(10),
                        'log10_halfp_strongest': -d['logK_strong'] / np.log(10),
                        'load_shape_err': shape_err, 'load_log10_scale': log_scale, 'load_abs_log_err': abs_err,
                        'e_pore0': d['e_pore0'], 'e_frame_mean': d['e_mean'], 'e_frame_max': d['e_max'], 'e_frame_min': d['e_min'],
                        'rho_logK_e_frame': d['rho_logK_e'], 'dS_frame_mean': d['dS_frame_mean'], 'dS_pore': d['dS_pore'],
                        'rho_e_dS_frame': d['rho_e_dS'], 'log10_n_pore': d['logn_pore'] / np.log(10),
                        f'dQ_true_{iv}': y[:, -1] - y[:, i05], f'dQ_pred_{iv}': pred[:, -1] - pred[:, i05],
                        f'term_framework_{iv}': t_f05, f'term_share_shift_{iv}': t_s05, f'term_heat_change_{iv}': t_g05,
                        f'term_interaction_{iv}': t_h05,
                        'void_fraction': d['geo'][:, 0], 'lcd': lcd})
    if g_poly:
        for k in range(coef.shape[1]):
            per[f'a_{k + 1}'] = coef[:, k]
        per['g_full(theta=1)'] = coef.sum(1)
        if coef.shape[1] >= 2:
            s0, s1 = coef[:, 0], coef[:, 0] + 2 * coef[:, 1]
            per['g_shape_full'] = np.select([(s0 >= 0) & (s1 >= 0), (s0 > 0) & (s1 < 0), (s0 <= 0) & (s1 <= 0)],
                                            ['increasing', 'rise_then_fall', 'decreasing'], 'fall_then_rise')
            per['slope_0'], per['slope_1'] = s0, s1
        else:
            per['g_shape_full'] = np.where(coef[:, 0] >= 0, 'increasing', 'decreasing')
        shp, gmin, gend = g_shape_observed(coef, tt[:, 0], tt[:, -1])
        per['g_shape_obs'], per['g_obs_min'], per['g_obs_end'] = shp, gmin, gend
        per['theta_obs_lo'], per['theta_obs_hi'] = tt[:, 0], tt[:, -1]
    for j, x in enumerate(pl):
        per[f'heat_true@{x}'] = y[:, j]; per[f'heat_pred@{x}'] = pred[:, j]; per[f'abs_err@{x}'] = err[:, j]
        per[f'alpha_pore@{x}'] = a[:, j]; per[f'F@{x}'] = F[:, j]; per[f'e_pore@{x}'] = E[:, j]; per[f'g@{x}'] = G[:, j]
        per[f'N_model@{x}'] = Nm[:, j]; per[f'theta_total@{x}'] = tt[:, j]
        per[f'alpha_strong@{x}'] = d['a_strong'][:, j]; per[f'alpha_weak@{x}'] = d['a_weak'][:, j]
        per[f'theta_strong@{x}'] = d['theta_strong'][:, j]
        if has_load:
            per[f'N_gcmc@{x}'] = Ng[:, j]
    per.to_csv(os.path.join(out_dir, 'per_mof.csv'), index=False)

    multi = d['C'] >= 2
    one_m_a = (1 - a).clip(1e-8)
    a_s_t, a_w_t = d['a_strong'] / one_m_a, d['a_weak'] / one_m_a
    rows = []
    for j, x in enumerate(pressures):
        rows.append({'pressure': x, 'mae': err[:, j].mean(), 'heat_true': y[:, j].mean(), 'heat_pred': pred[:, j].mean(),
                     'alpha_pore': a[:, j].mean(), 'uniform_1/(C+1)': (1 / (d['C'] + 1)).mean(),
                     'frame_contrib': d['frame_contrib'][:, j].mean(), 'pore_contrib': d['pore_contrib'][:, j].mean(),
                     'interaction_contrib': H[:, j].mean(),
                     'F_given_framework': F[:, j].mean(), 'e_frame_mean': np.nanmean(d['e_mean']),
                     'e_pore_p': E[:, j].mean(), 'g_mean': G[:, j].mean(), 'g_median': np.median(G[:, j]),
                     'alpha_strong_raw': np.nanmean(d['a_strong'][multi, j]), 'alpha_strong_renorm': np.nanmean(a_s_t[multi, j]),
                     'alpha_weak_raw': np.nanmean(d['a_weak'][multi, j]), 'alpha_weak_renorm': np.nanmean(a_w_t[multi, j]),
                     'theta_strong': np.nanmean(d['theta_strong'][multi, j]), 'theta_pore': d['theta_pore'][:, j].mean(),
                     'theta_total': tt[:, j].mean(),
                     'load_ratio_model_median': np.median(Nm[:, j] / Nm[:, -1].clip(1e-12)),
                     'load_ratio_gcmc_median': np.median(Ng[:, j] / Ng[:, -1]) if has_load else np.nan})
    by_p = pd.DataFrame(rows)
    by_p.to_csv(os.path.join(out_dir, 'by_pressure.csv'), index=False)
    dec.to_csv(os.path.join(out_dir, 'decomposition.csv'), index=False)

    b_rows, s_rows, gb = [], [], None
    for lab in labels:
        mk = (per['sat_bin'] == lab).values
        if mk.sum() == 0:
            continue
        mm = mk & multi
        r = {'sat_bin': lab, 'n': int(mk.sum()), 'frac': mk.mean(), 'sat_score_mean': np.nanmean(sat_score[mk]),
             f'theta_fit@{pmx}': np.nanmean(theta_fit[mk, -1]), f'theta_total@{pmx}': tt[mk, -1].mean(),
             f'fill_vol@{pmx}_median': np.nanmedian(fill_vol[mk]), 'lcd_median': np.nanmedian(lcd[mk]),
             f'dQ_true_{iv}': (y[mk, -1] - y[mk, i05]).mean(), 'frac_dQ_true_neg': ((y[mk, -1] - y[mk, i05]) < 0).mean(),
             f'dQ_pred_{iv}': (pred[mk, -1] - pred[mk, i05]).mean(),
             'term_framework': t_f05[mk].mean(), 'term_share_shift': t_s05[mk].mean(), 'term_heat_change': t_g05[mk].mean(),
             'term_interaction': t_h05[mk].mean(),
             'load_shape_err_median': np.nanmedian(shape_err[mk]),
             'dS_pore_median': np.median(dSp[mk]), 'frac_dS_pore_at_lo': near_lo(dSp[mk]).mean(),
             'dS_frame_median': np.median(dSf[mk[tok_bin]]) if dSf.size else np.nan,
             'frac_dS_frame_at_lo': near_lo(dSf[mk[tok_bin]]).mean() if dSf.size else np.nan,
             'e_pore0_mean': d['e_pore0'][mk].mean()}
        for j, x in enumerate(pl):
            r[f'mae@{x}'] = err[mk, j].mean()
        for j, x in enumerate(pl):
            r[f'alpha_pore@{x}'] = a[mk, j].mean()
        for j, x in enumerate(pl):
            r[f'alpha_strong_renorm@{x}'] = np.nanmean(a_s_t[mm, j])
        for j, x in enumerate(pl):
            r[f'F@{x}'] = F[mk, j].mean()
        for j, x in enumerate(pl):
            r[f'e_pore@{x}'] = E[mk, j].mean()
        b_rows.append(r)
        for j, x in enumerate(pressures):
            s_rows.append({'sat_bin': lab, 'pressure': x, 'heat_true': y[mk, j].mean(), 'heat_pred': pred[mk, j].mean(),
                           'gcmc_ratio_median': np.nanmedian(Ng[mk, j] / Ng[mk, -1]) if has_load else np.nan,
                           'model_ratio_median': np.median(Nm[mk, j] / Nm[mk, -1].clip(1e-12))})
    if g_poly:
        ccols = [c for c in per.columns if c.startswith('a_')] + ['g_full(theta=1)', f'g@{pmx}', 'g_obs_min',
                                                                   'theta_obs_hi']
        gb = per.groupby('sat_bin', observed=True)[ccols].mean()
        gb['frac_g_obs_min<-0.1'] = per.groupby('sat_bin', observed=True)['g_obs_min'].apply(lambda s: (s < -0.1).mean())
        gb = gb.join(pd.crosstab(per['sat_bin'], per['g_shape_obs'], normalize='index').add_prefix('obs_'))
        gb = gb.join(pd.crosstab(per['sat_bin'], per['g_shape_full'], normalize='index').add_prefix('full_'))
        gb.to_csv(os.path.join(out_dir, 'g_by_sat_bin.csv'))
    by_bin = pd.DataFrame(b_rows); by_bin.to_csv(os.path.join(out_dir, 'by_sat_bin.csv'), index=False)
    shape_bin = pd.DataFrame(s_rows); shape_bin.to_csv(os.path.join(out_dir, 'loading_shape_by_bin.csv'), index=False)

    out_mk = np.nan_to_num(shape_err) > 2.0
    per[out_mk].sort_values('load_shape_err', ascending=False).to_csv(os.path.join(out_dir, 'loading_outliers.csv'), index=False)

    rho_ok = np.isfinite(d['rho_logK_e'])
    summ = {'gas': gas, 'p_max': pressures[-1], 'sat_by': sat_by, 'mode': str(getattr(args, 'mode', '')),
            'n_mof': n, 'lateral(1)_or_pore_g(0)': float(lateral), 'mae': err.mean(), 'max_recon_err': np.abs(recon - pred).max(),
            'alpha_pore@pmin': a[:, 0].mean(), 'alpha_pore@pmax': a[:, -1].mean(), 'uniform_1/(C+1)': (1 / (d['C'] + 1)).mean(),
            'C_mean': d['C'].mean(),
            'F@pmin': F[:, 0].mean(), 'F@pmax': F[:, -1].mean(), 'e_frame_mean': np.nanmean(d['e_mean']),
            'alpha_strong_renorm@pmin': np.nanmean(a_s_t[multi, 0]), 'alpha_strong_renorm@pmax': np.nanmean(a_s_t[multi, -1]),
            'alpha_weak_renorm@pmin': np.nanmean(a_w_t[multi, 0]), 'alpha_weak_renorm@pmax': np.nanmean(a_w_t[multi, -1]),
            'rho_logK_e_median(C>=3)': np.nanmedian(d['rho_logK_e']), 'frac_rho_logK_e_pos': (d['rho_logK_e'][rho_ok] > 0).mean(),
            'dS_frame_median_J/molK': np.nanmedian(d['tok_dS_frame']), 'dS_pore_median_J/molK': np.nanmedian(d['dS_pore']),
            'dS_lower_bound_J/molK': -S_max,
            'frac_dS_frame_below_bound': (dSf < -S_max).mean() if dSf.size else np.nan,
            **th_rows,
            'n_pore_share_median': np.median(d['n_pore_share']),
            'log10_halfp_pore_median': np.median(-d['logK_pore'] / np.log(10)),
            'log10_halfp_strongest_median': np.nanmedian(-d['logK_strong'] / np.log(10)),
            'frac_halfp_pore_above_pmax': (-d['logK_pore'] / np.log(10) > np.log10(p[-1])).mean(),
            'theta_total@pmax_mean': tt[:, -1].mean(), 'theta_pore@pmax_mean': d['theta_pore'][:, -1].mean(),
            'theta_total@pmin_mean': tt[:, 0].mean(),
            'g@pmin_mean': G[:, 0].mean(), 'g@pmax_mean': G[:, -1].mean(), 'g@pmax_max': G[:, -1].max(), 'g@pmax_min': G[:, -1].min()}
    if has_load:
        okf = np.isfinite(theta_fit[:, -1])
        summ.update({'theta_fit@pmax_mean': np.nanmean(theta_fit[:, -1]), 'frac_theta_fit_gt_0.8': np.nanmean(theta_fit[:, -1] > 0.8),
                     'rho_theta_total_vs_theta_fit': spearmanr(tt[okf, -1], theta_fit[okf, -1]).correlation,
                     'load_shape_err_median': np.nanmedian(shape_err), 'load_shape_err_mean': np.nanmean(shape_err),
                     'n_load_shape_err_gt_2': int(out_mk.sum()),
                     'rho_Nmodel_vs_Ngcmc@pmax': spearmanr(Nm[:, -1], Ng[:, -1]).correlation,
                     'load_log10_scale_median': np.nanmedian(log_scale), 'load_log10_scale_std': np.nanstd(log_scale),
                     'load_abs_log_err_median': np.nanmedian(abs_err)})
    if g_poly:
        for k in range(coef.shape[1]):
            summ[f'a_{k + 1}_median'] = np.median(coef[:, k])
        summ['g_full_median'] = np.median(coef.sum(1))
        summ['frac_g_full_neg'] = (coef.sum(1) < 0).mean()
        summ['g_is_poly_of_theta_total'] = float(g_var_ok)
        summ['frac_g_obs_increasing'] = (per['g_shape_obs'] == 'increasing').mean()
        summ['frac_g_obs_min<-0.1'] = (per['g_obs_min'] < -0.1).mean()
    if mf and g_poly and coef.shape[1] >= 2:
        s0, s1, u1 = coef[:, 0], coef[:, 0] + 2 * coef[:, 1], coef.sum(1)
        at_b = lambda v: np.abs(v) >= 0.95 * slope_max
        dq, du = pred[:, -1] - pred[:, 0], G[:, -1] - G[:, 0]
        summ.update({'mf_slope_max': float(slope_max),
                     "mf_u'(0)_median": np.median(s0), "mf_u'(1)_median": np.median(s1),
                     "mf_frac_u'(0)_at_bound": at_b(s0).mean(), "mf_frac_u'(1)_at_bound": at_b(s1).mean(),
                     'mf_frac_any_slope_at_bound': (at_b(s0) | at_b(s1)).mean(),
                     'mf_var_u/var_Qst': G.var() / max(pred.var(), 1e-12),
                     'mf_mean|du|/mean|dQst|_pmin_to_pmax': np.abs(du).mean() / max(np.abs(dq).mean(), 1e-12),
                     'mf_frac_u(1)_attractive': (u1 > 0).mean()})
        for j, nm in enumerate(GEO_NAMES[:d['geo'].shape[1]]):
            summ[f'mf_rho_u(1)_vs_{nm}'] = _spearman(u1, d['geo'][:, j])
            summ[f"mf_rho_u'(0)_vs_{nm}"] = _spearman(s0, d['geo'][:, j])
    summ = pd.Series(summ)
    summ.to_frame('value').to_csv(os.path.join(out_dir, 'summary.csv'))

    x = np.arange(P)
    cmap = plt.get_cmap('viridis')
    bcol = {lab: cmap(i / max(1, len(labels) - 1)) for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax[0, 0].bar(x, by_p['frame_contrib'], label='framework Σ α_c e_c')
    ax[0, 0].bar(x, by_p['pore_contrib'], bottom=by_p['frame_contrib'], label='pore α_pore e_pore(p)')
    if lateral:
        ax[0, 0].bar(x, by_p['interaction_contrib'], bottom=by_p['frame_contrib'] + by_p['pore_contrib'],
                     color='C3', label=f'interaction {gn} (all sites)')
    ax[0, 0].plot(x, by_p['heat_true'], 'ko-', label='true Qst')
    ax[0, 0].set_title(f'contribution to Qst (raw, {gas.upper()})'); ax[0, 0].set_ylabel('kJ/mol')
    ax[0, 1].plot(x, by_p['F_given_framework'], 'o-', label='F: framework, given destination')
    if lateral:
        ax[0, 1].plot(x, by_p['e_pore_p'], 'k^-', label='e_pore (site heat, constant)')
        ax[0, 1].plot(x, by_p['F_given_framework'] + by_p['interaction_contrib'], 'o:', color='C0', label=f'F + {gn}')
        ax[0, 1].plot(x, by_p['e_pore_p'] + by_p['interaction_contrib'], 'k^:', label=f'e_pore + {gn}')
    else:
        ax[0, 1].plot(x, by_p['e_pore_p'], 'k^-', label='e_pore(p) = e_pore + g(p)')
    ax[0, 1].axhline(by_p['e_frame_mean'].iloc[0], color='gray', ls='--', label='mean e_c (framework)')
    ax[0, 1].set_title('heat given the destination'); ax[0, 1].set_ylabel('kJ/mol')
    ax[0, 2].plot(x, by_p['alpha_strong_raw'], 'o-', color='C3', label='strongest cluster α')
    ax[0, 2].plot(x, by_p['alpha_strong_renorm'], 'o--', color='C3', label='strongest cluster α̃')
    ax[0, 2].plot(x, by_p['alpha_weak_raw'], 's-', color='C0', label='weakest cluster α')
    ax[0, 2].plot(x, by_p['alpha_weak_renorm'], 's--', color='C0', label='weakest cluster α̃')
    ax[0, 2].plot(x, by_p['alpha_pore'], 'k^-', label='pore α')
    ax[0, 2].plot(x, by_p['uniform_1/(C+1)'], 'k:', label='uniform 1/(C+1)')
    ax[0, 2].set_title('where the next molecule goes (C ≥ 2)'); ax[0, 2].set_ylabel('mean α')
    for k in range(3):
        ax[0, k].set_xticks(x, pl); ax[0, k].set_xlabel('pressure (bar)'); ax[0, k].legend(fontsize=8)
    xd = np.arange(len(dec)); w = 0.2
    ax[1, 0].bar(xd - 1.5 * w, dec['frac_framework'], w, label='framework')
    ax[1, 0].bar(xd - 0.5 * w, dec['frac_pore_share_shift'], w, label='pore share shift')
    ax[1, 0].bar(xd + 0.5 * w, dec['frac_pore_heat_change'], w, label='pore heat change (g in pore)')
    ax[1, 0].bar(xd + 1.5 * w, dec['frac_interaction'], w, label=f'interaction ({gn} on all sites)')
    ax[1, 0].set_xticks(xd, dec['interval'], rotation=25); ax[1, 0].set_ylabel('fraction of mean |term|')
    ax[1, 0].set_title('decomposition of ΔQst'); ax[1, 0].legend(fontsize=8)
    if len(by_bin):
        for lab in by_bin['sat_bin']:
            s = shape_bin[shape_bin.sat_bin == lab]
            ax[1, 1].plot(x, s['heat_true'], 'o-', color=bcol[lab], label=lab)
            ax[1, 1].plot(x, s['heat_pred'], 'x--', color=bcol[lab], alpha=0.7)
        ax[1, 1].set_xticks(x, pl); ax[1, 1].set_xlabel('pressure (bar)'); ax[1, 1].set_ylabel('Qst (kJ/mol)')
        ax[1, 1].set_title(f'heat by {sat_name} bin (solid true, dashed pred)'); ax[1, 1].legend(fontsize=8)
        xb = np.arange(len(by_bin)); w = 0.16
        ax[1, 2].bar(xb - 2 * w, by_bin['term_framework'], w, label='framework')
        ax[1, 2].bar(xb - 1 * w, by_bin['term_share_shift'], w, label='pore share shift')
        ax[1, 2].bar(xb, by_bin['term_heat_change'], w, label='pore heat change')
        ax[1, 2].bar(xb + 1 * w, by_bin['term_interaction'], w, label='interaction')
        ax[1, 2].bar(xb + 2 * w, by_bin[f'dQ_true_{iv}'], w, color='k', alpha=0.6, label='true ΔQst')
        ax[1, 2].axhline(0, color='k', lw=0.8); ax[1, 2].set_xticks(xb, by_bin['sat_bin'])
        ax[1, 2].set_xlabel(f'{sat_name} bin'); ax[1, 2].set_ylabel('kJ/mol')
        ax[1, 2].set_title(f'ΔQst {p05}→{pmx} bar by saturation'); ax[1, 2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'fig_heat.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    if has_load:
        ax[0, 0].plot(x, by_p['load_ratio_gcmc_median'], 'ko-', label='GCMC (all)')
        ax[0, 0].plot(x, by_p['load_ratio_model_median'], 'k^--', label='N_model (all)')
        for lab in by_bin['sat_bin']:
            s = shape_bin[shape_bin.sat_bin == lab]
            ax[0, 0].plot(x, s['gcmc_ratio_median'], 'o-', color=bcol[lab], alpha=0.8, label=f'{lab} GCMC')
            ax[0, 0].plot(x, s['model_ratio_median'], '^--', color=bcol[lab], alpha=0.8)
        ax[0, 0].set_yscale('log'); ax[0, 0].set_xticks(x, pl); ax[0, 0].set_xlabel('pressure (bar)')
        ax[0, 0].set_ylabel(f'N(p) / N({pmx}) (median)'); ax[0, 0].set_title('isotherm shape (solid GCMC, dashed model)')
        ax[0, 0].legend(fontsize=7)
        se = shape_err[np.isfinite(shape_err)]
        ax[0, 1].hist(np.clip(se, 0, 3), bins=80)
        ax[0, 1].set_title(f'shape error  median {np.median(se):.3f}, >2: {int(out_mk.sum())} MOFs (clipped at 3)')
        ax[0, 1].set_xlabel('mean |log ratio_model − log ratio_GCMC|')
        okf = np.isfinite(theta_fit[:, -1])
        hb = ax[0, 2].hexbin(theta_fit[okf, -1], tt[okf, -1], gridsize=50, bins='log', mincnt=1, extent=(0, 1, 0, 1))
        ax[0, 2].plot([0, 1], [0, 1], 'k--', lw=1); fig.colorbar(hb, ax=ax[0, 2])
        ax[0, 2].set_xlabel(f'θ_fit@{pmx} (Langmuir fit of GCMC)'); ax[0, 2].set_ylabel(f'model θ_total@{pmx}')
        ax[0, 2].set_title(f"filling at {pmx} bar (Spearman {summ['rho_theta_total_vs_theta_fit']:.3f})")
        okn = np.all(Nm > 0, 1) & np.all(Ng > 0, 1)
        hb = ax[1, 0].hexbin(np.log10(Ng[okn, -1]), np.log10(Nm[okn, -1]), gridsize=50, bins='log', mincnt=1)
        fig.colorbar(hb, ax=ax[1, 0])
        ax[1, 0].set_xlabel(f'log10 GCMC N({pmx}) (mmol/g)'); ax[1, 0].set_ylabel(f'log10 N_model({pmx}) (model units)')
        ax[1, 0].set_title(f"loading magnitude (Spearman {summ['rho_Nmodel_vs_Ngcmc@pmax']:.3f})")
        ax[1, 1].hist(log_scale[np.isfinite(log_scale)], bins=60)
        ax[1, 1].set_xlabel(f'log10 (N_GCMC / N_model) at {pmx} bar'); ax[1, 1].set_title('scale of N_model (narrow = consistent units)')
    ax[1, 2].hist(d['n_pore_share'], bins=60, range=(0, 1))
    ax[1, 2].set_xlabel('n_pore / Σ n'); ax[1, 2].set_title('pore share of site capacity')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'fig_loading.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    if thermo and dSf.size:
        hb = ax[0, 0].hexbin(ef_tok, dSf, gridsize=60, bins='log', mincnt=1)
        fig.colorbar(hb, ax=ax[0, 0])
        ax[0, 0].scatter(d['e_pore0'], dSp, s=2, c='C3', alpha=0.3, label='pore')
        ax[0, 0].axhline(-S_max, color='C3', ls='--', lw=1); ax[0, 0].axhline(0, color='k', ls='--', lw=1)
        ax[0, 0].set_xlabel('e (kJ/mol)'); ax[0, 0].set_ylabel('ΔS (J/mol/K)'); ax[0, 0].legend(fontsize=8)
        ax[0, 0].set_title(f"compensation: Spearman(e, ΔS) framework {th_rows['rho_e_dS_frame_pooled']:.2f}, "
                           f"pore {th_rows['rho_e_dS_pore_across_mof']:.2f}")
    else:
        ax[0, 0].hist(d['rho_logK_e'][rho_ok], bins=40, range=(-1, 1))
        ax[0, 0].axvline(0, color='k', lw=0.8)
        ax[0, 0].set_title(f"within-MOF Spearman(log K_c, e_c), C≥3: median {summ['rho_logK_e_median(C>=3)']:.2f}")
    ax[0, 1].hist(np.clip(d['tok_dS_frame'], -300, 100), bins=80, alpha=0.6, density=True, label='framework clusters')
    ax[0, 1].hist(np.clip(d['dS_pore'], -300, 100), bins=80, alpha=0.6, density=True, label='pore')
    ax[0, 1].axvspan(-120, -60, color='gray', alpha=0.15, label='typical physisorption')
    ax[0, 1].axvline(-S_max, color='C3', ls='--', lw=1, label=f'−S_max ({gas.upper()})')
    ax[0, 1].axvline(0, color='k', ls='--', lw=1)
    ax[0, 1].set_xlabel('ΔS = R(ln K − e/RT) (J/mol/K, clipped)'); ax[0, 1].set_title('site entropy'); ax[0, 1].legend(fontsize=8)
    ax[0, 2].hist(-d['logK_strong'][multi] / np.log(10), bins=80, alpha=0.6, label='strongest framework cluster')
    ax[0, 2].hist(-d['logK_pore'] / np.log(10), bins=80, alpha=0.6, label='pore')
    for pp in p:
        ax[0, 2].axvline(np.log10(pp), color='gray', ls='--', lw=0.8)
    ax[0, 2].set_xlabel('log10(1/K) (bar)'); ax[0, 2].set_title('half-filling pressure (dashed: data pressures)')
    ax[0, 2].legend(fontsize=8)
    ax[1, 0].boxplot([G[:, j] for j in range(P)], showfliers=False)
    ax[1, 0].set_xticks(np.arange(1, P + 1), pl); ax[1, 0].axhline(0, color='k', lw=0.8)
    ax[1, 0].set_xlabel('pressure (bar)'); ax[1, 0].set_ylabel(f'{gn}(p) (kJ/mol)')
    ax[1, 0].set_title((f'mean field {gn}(θ_total) (all sites)' if mf else f'interaction {gn}(p) (added to all sites)')
                       if lateral else f'pore heat change {gn}(p)')
    if g_poly and coef.shape[1] >= 2:
        sc = ax[1, 1].scatter(coef[:, 0], coef[:, 1], s=2, c=np.nan_to_num(tt[:, -1]), cmap='viridis', alpha=0.5)
        fig.colorbar(sc, ax=ax[1, 1], label=f'model θ_total@{pmx}')
        ax[1, 1].axhline(0, color='k', lw=0.8); ax[1, 1].axvline(0, color='k', lw=0.8)
        ax[1, 1].set_xlabel('a_1'); ax[1, 1].set_ylabel('a_2'); ax[1, 1].set_title(f'{gn} coefficients')
    elif g_poly:
        ax[1, 1].hist(coef[:, 0], bins=60); ax[1, 1].set_xlabel('a_1'); ax[1, 1].set_title('g coefficient')
    if gb is not None:
        fcols = [c for c in gb.columns if c.startswith('obs_')]
        bottom = np.zeros(len(gb))
        for c in fcols:
            ax[1, 2].bar(np.arange(len(gb)), gb[c].values, bottom=bottom, label=c.replace('obs_', ''))
            bottom += gb[c].values
        ax[1, 2].set_xticks(np.arange(len(gb)), gb.index.astype(str)); ax[1, 2].set_xlabel(f'{sat_name} bin')
        ax[1, 2].set_ylabel('fraction'); ax[1, 2].set_title(f'shape of {gn} on the observed θ_total range')
        ax[1, 2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'fig_sites_g.png'), dpi=150); plt.close(fig)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
    fmt = lambda v: f"{v:.4g}"
    print(f"\n=== summary ({gas}, heat_mode = {mode}, {'mean field u(θ_total)' if mf else 'g'}, saturation by {sat_name}) ===")
    if thermo:
        print("  (thermo: rho_logK_e is structural; read frac_dS_*_at_lo/hi and rho_e_dS_* instead)")
    else:
        print("  (free K: rho_e_dS_* is biased negative by construction (dS contains -e/T); read rho_logK_e instead)")
    print(summ.to_string(float_format=fmt))
    print("\n=== by pressure ===")
    print(by_p.to_string(index=False, float_format=fmt))
    print("\n=== ΔQst decomposition ===")
    print(dec.to_string(index=False, float_format=fmt))
    if len(by_bin):
        print(f"\n=== by saturation bin ({sat_name}, ΔQst {p05}→{pmx} bar) ===")
        print(by_bin.to_string(index=False, float_format=fmt))
    if gb is not None:
        print(f"\n=== {gn} coefficients and shape (by saturation bin; obs_ = observed θ range, full_ = extrapolated to θ∈[0,1]) ===")
        print(gb.to_string(float_format=fmt))
    print(f"\nSaved to {out_dir}")
    return per, by_p, by_bin, dec, summ
