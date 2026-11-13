import os
import re
import zlib
import shutil
import argparse
import tempfile
import subprocess
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from torch_geometric.data import Data


parser = argparse.ArgumentParser()
parser.add_argument('--cif_path', default='../dataset/UiO-66/cif', help='folder of .cif files')
parser.add_argument('--save_path', default='../dataset/UiO-66/uio66',
                    help='output folder; its name is the tag of the analysis folders (site_analysis_<tag>_<seed>)')
parser.add_argument('--zeopp', default='network', help='Zeo++ binary')
parser.add_argument('--insertions', default=200000, type=int, help='helium insertions per MOF (MOFX-DB: 200,000)')
parser.add_argument('--he_potential', default='truncated', choices=['truncated', 'shifted'],
                    help='LJ at the 12.8 A cutoff: truncated (matches MOFX-DB, checked on 100 hMOFs) or shifted to zero')
args = parser.parse_args()


CUTOFF, MAX_NEIGHBORS = 6.0, 12
ELEMENT_INDEX = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Cl': 6, 'V': 7, 'Cu': 8, 'Zn': 9, 'Br': 10, 'Zr': 11}
UFF_RAD = {'H': 1.285, 'C': 1.715, 'N': 1.63, 'O': 1.56, 'F': 1.5, 'Cl': 1.76,       # MOFX-DB radii/UFF.rad
           'V': 1.4, 'Cu': 1.555, 'Zn': 1.23, 'Br': 1.865, 'Zr': 1.39}
# UFF Lennard-Jones: sigma [A], epsilon/kB [K]  (sigma = x_i / 2^(1/6), eps = D_i)
UFF_LJ = {'H': (2.57, 22.14), 'C': (3.43, 52.83), 'N': (3.26, 34.72), 'O': (3.12, 30.19),
          'F': (2.997, 25.16), 'Cl': (3.517, 114.23), 'V': (2.80, 8.05), 'Cu': (3.114, 2.516),
          'Zn': (2.462, 62.40), 'Br': (3.73, 126.30), 'Zr': (2.783, 34.72)}
HE_SIGMA, HE_EPS = 2.64, 10.9
TEMPERATURE, HE_CUTOFF = 298.0, 12.8
ZEOPP = shutil.which(args.zeopp) or args.zeopp
TIE_TOL = 1e-2  # A: keep all neighbours within TIE_TOL of the MAX_NEIGHBORS-th distance


def _framework_images(crystal, rc):
    m = crystal.lattice.matrix
    vol = abs(np.linalg.det(m))
    n = [int(np.ceil(rc / (vol / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3]))))) for i in range(3)]
    shifts = np.array([[i, j, k] for i in range(-n[0], n[0] + 1)
                       for j in range(-n[1], n[1] + 1) for k in range(-n[2], n[2] + 1)], dtype=float)
    frac = crystal.frac_coords % 1.0
    pos = ((frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)) @ m
    sig = np.array([(UFF_LJ[s.specie.symbol][0] + HE_SIGMA) / 2 for s in crystal])
    eps = np.array([np.sqrt(UFF_LJ[s.specie.symbol][1] * HE_EPS) for s in crystal])
    return pos, np.tile(sig, len(shifts)), np.tile(eps, len(shifts))


def he_energies(crystal, points, rc=HE_CUTOFF, shifted=False, batch=4000, images=None):
    pos, sig, eps = images if images is not None else _framework_images(crystal, rc)
    tree = cKDTree(pos)
    U = np.empty(len(points))
    for s in range(0, len(points), batch):
        p = points[s:s + batch]
        pairs = cKDTree(p).sparse_distance_matrix(tree, rc, output_type='ndarray')
        r = np.maximum(pairs['v'], 1e-3)
        j = pairs['j']
        sr6 = (sig[j] / r) ** 6
        u = 4.0 * eps[j] * (sr6 * sr6 - sr6)
        if shifted:
            src6 = (sig[j] / rc) ** 6
            u -= 4.0 * eps[j] * (src6 * src6 - src6)
        U[s:s + batch] = np.bincount(pairs['i'], weights=u, minlength=len(p))
    return U


def helium_void_fraction(crystal, n_insert=200000, shifted=False, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((n_insert, 3)) @ crystal.lattice.matrix
    U = he_energies(crystal, points, shifted=shifted, images=_framework_images(crystal, HE_CUTOFF))
    return float(np.mean(np.exp(-np.clip(U / TEMPERATURE, -700, 700))))


def zeopp(crystal):
    with tempfile.TemporaryDirectory() as td:
        CifWriter(crystal).write_file(os.path.join(td, 'm.cif'))
        with open(os.path.join(td, 'UFF.rad'), 'w') as f:
            for el in sorted({s.specie.symbol for s in crystal}):
                f.write(f'{el}\t{UFF_RAD[el]}\n')
        for ha in (['-ha'], []):
            subprocess.run([ZEOPP, *ha, '-r', 'UFF.rad', '-res', 'm.res', '-sa', '1.86', '1.86', '1000', 'm.sa', 'm.cif'],
                           cwd=td, capture_output=True, text=True)
            if os.path.exists(os.path.join(td, 'm.res')) and os.path.exists(os.path.join(td, 'm.sa')):
                break
        else:
            raise RuntimeError('Zeo++ failed')
        res = open(os.path.join(td, 'm.res')).read().split()
        sa = {k: float(v) for k, v in re.findall(r'(\S+):\s+(-?[\d.eE+-]+)', open(os.path.join(td, 'm.sa')).read())}
    return {'pld': float(res[2]), 'lcd': float(res[1]),
            'surface_area_m2g': sa['ASA_m^2/g'] + sa['NASA_m^2/g'],
            'surface_area_m2cm3': sa['ASA_m^2/cm^3'] + sa['NASA_m^2/cm^3']}


def edges(crystal):
    """neighbours within CUTOFF: the MAX_NEIGHBORS nearest, plus every neighbour within TIE_TOL of the
    MAX_NEIGHBORS-th distance, so that equivalent atoms keep the same neighbour set (no cut inside a shell)"""
    center, nbr, images, dist = crystal.get_neighbor_list(CUTOFF)
    if len(center) == 0:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)),
                torch.empty((0, 3)), torch.empty((0, 3)))
    order = np.lexsort((dist, center))
    center, nbr, images, dist = center[order], nbr[order], images[order], dist[order]
    n_atoms = len(crystal)
    counts = np.bincount(center, minlength=n_atoms)
    first = np.searchsorted(center, np.arange(n_atoms), side='left')
    d_k = np.full(n_atoms, np.inf)
    has_k = counts >= MAX_NEIGHBORS
    d_k[has_k] = dist[first[has_k] + MAX_NEIGHBORS - 1]
    keep = dist <= d_k[center] + TIE_TOL
    center, nbr, images, dist = center[keep], nbr[keep], images[keep], dist[keep]
    vec = crystal.cart_coords[nbr] + images @ crystal.lattice.matrix - crystal.cart_coords[center]
    return (torch.tensor(np.vstack([nbr, center]), dtype=torch.long),
            torch.tensor(dist, dtype=torch.float).unsqueeze(1),
            torch.tensor(images, dtype=torch.float),
            torch.tensor(vec, dtype=torch.float))

def process(job):
    idx, cif = job
    name = os.path.splitext(os.path.basename(cif))[0]
    row = {'idx': idx, 'name': name, 'status': 'fail', 'error': ''}
    try:
        crystal = Structure.from_file(cif)
        unknown = sorted({s.specie.symbol for s in crystal} - set(ELEMENT_INDEX))
        if unknown:
            raise ValueError(f'elements not in training vocabulary: {unknown}')
        geo = {'void_fraction': helium_void_fraction(crystal, args.insertions, args.he_potential == 'shifted',
                                                     seed=zlib.crc32(name.encode())),
               **zeopp(crystal)}
        edge_index, edge_attr, offsets, edge_vec = edges(crystal)
        data = Data(x=torch.tensor([ELEMENT_INDEX[s.specie.symbol] for s in crystal], dtype=torch.long),
                    pos=torch.tensor(crystal.cart_coords, dtype=torch.float),
                    cell=torch.tensor(crystal.lattice.matrix, dtype=torch.float).unsqueeze(0),
                    edge_index=edge_index, edge_attr=edge_attr, edge_vec=edge_vec, offsets=offsets,
                    mass_cell=torch.tensor([crystal.composition.weight], dtype=torch.float),
                    num_nodes=len(crystal))
        for k, v in geo.items():
            data[k] = torch.tensor([v], dtype=torch.float)
        torch.save(data, os.path.join(args.save_path, f'{idx}.pt'))
        row.update(status='ok', **geo)
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def main():
    cifs = sorted(glob(os.path.join(args.cif_path, '*.cif')))
    if not cifs:
        raise SystemExit(f'no .cif files in {args.cif_path} (check the folder path)')
    if not (shutil.which(ZEOPP) or os.path.exists(ZEOPP)):
        raise SystemExit(f'Zeo++ binary not found: {ZEOPP}')
    os.makedirs(args.save_path, exist_ok=True)
    print(f'{len(cifs)} CIFs | He insertions {args.insertions} | potential {args.he_potential}')
    with Pool(max(1, len(os.sched_getaffinity(0)) - 1)) as pool:
        rows = list(tqdm(pool.imap_unordered(process, list(enumerate(cifs))), total=len(cifs)))
    df = pd.DataFrame(rows).sort_values('idx')
    df.to_csv(os.path.join(args.save_path, 'index.csv'), index=False)
    print(f"saved {(df.status == 'ok').sum()} / {len(df)} to {args.save_path}")
    if (df.status != 'ok').any():
        print(df[df.status != 'ok'][['name', 'error']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
