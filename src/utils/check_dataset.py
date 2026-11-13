import os
import numpy as np
import pandas as pd
import torch

GEO_COLS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd', 'density']


def check_dataset(ds, name='train', save_dir=None, n_show=3):
    store = getattr(ds, '_data', None) or ds.data
    geo, y, yl = store.geo.float(), store.y.float(), store.y_load.float()
    n = len(ds)
    print(f'===== [{name}] {n} graphs')
    print(f'geo {tuple(geo.shape)} | y {tuple(y.shape)} | y_load {tuple(yl.shape)}')
    ok = geo.shape == (n, len(GEO_COLS)) and y.shape[0] == n and yl.shape[0] == n
    print('shape check:', 'OK' if ok else f'MISMATCH (expected geo ({n}, {len(GEO_COLS)}))')

    for k, t in (('geo', geo), ('y', y), ('y_load', yl)):
        bad = (~torch.isfinite(t)).sum().item()
        print(f'{k:6s} non-finite values: {bad}')

    g = geo.numpy()
    q = np.nanpercentile(g, [0, 1, 50, 99, 100], axis=0)
    stats = pd.DataFrame(q.T, index=GEO_COLS[:g.shape[1]], columns=['min', 'p1', 'median', 'p99', 'max'])
    stats['n_zero'] = (g == 0).sum(0)
    pd.set_option('display.width', 160)
    print(stats.to_string(float_format=lambda v: f'{v:.4g}'))

    if g.shape[1] == len(GEO_COLS):
        dens = g[:, 5]
        out = ((dens < 0.1) | (dens > 3.0)).sum()
        print(f'density outside 0.1-3 g/cm3: {out} of {n}')
        sa_g, sa_v = g[:, 1], g[:, 2]
        m = sa_g > 0
        rel = np.abs(sa_v[m] / sa_g[m] - dens[m]) / dens[m]
        print(f'density vs SA_m2cm3/SA_m2g ({m.sum()} MOFs with SA > 0): '
              f'median rel. diff {np.median(rel):.2e}, max {rel.max():.2e} '
              f"-> {'OK' if np.median(rel) < 1e-2 else 'CHECK the density formula / units'}")
        print(f'MOFs with SA = 0 (density only from the new column): {(~m).sum()}')

    print(f'first {n_show} samples:')
    fmt = lambda a, d: '[' + ', '.join(f'{v:.{d}f}' for v in a) + ']'
    for i in range(min(n_show, n)):
        print(f'  geo {fmt(g[i], 4)} | y {fmt(y[i].numpy(), 2)} | y_load {fmt(yl[i].numpy(), 3)}')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stats.to_csv(os.path.join(save_dir, f'check_{name}_geo.csv'))
    return stats
