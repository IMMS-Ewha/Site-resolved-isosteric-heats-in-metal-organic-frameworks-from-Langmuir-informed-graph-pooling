import os
import argparse
import re
import random
import torch
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Structure
from collections import defaultdict
from multiprocessing import Pool
from torch_geometric.data import Data
from glob import glob


GEO_KEYS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd']

parser = argparse.ArgumentParser(description='preprocessing hmof (pore descriptors: MOFX-DB consistent set)')
parser.add_argument('--data_path', default='../dataset/hMOF/hmof_data_clean_void_inc', type=str)
parser.add_argument('--save_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym', type=str)
parser.add_argument('--consistent_csv', default='../dataset/hMOF/geo_mapping/consistent_hMOF.csv', type=str)
parser.add_argument('--cutoff', default=6.0, type=float)
parser.add_argument('--max_neighbors', default=12, type=int)
parser.add_argument('--tie_tol', default=1e-2, type=float,
                    help='A, also keep neighbours within d_12 + tie_tol (symmetric graphs of the paper); '
                         'negative: keep exactly the max_neighbors nearest (graphs read by tie_check.py)')
args = parser.parse_args()


# [consistent] pore descriptors from the MOFX-DB consistent-set CSV, keyed by hMOF number ('hMOF-12...' -> 12).
# Loaded at import so the worker processes inherit it. Surface area = total (accessible + inaccessible),
# the same quantity that preprocess_cif_consistent.py computes for new CIFs.
_c = pd.read_csv(args.consistent_csv)
CONSISTENT = {int(r.id): {'void_fraction': r.vf,
                          'surface_area_m2g': r.sa_tot_m2g,
                          'surface_area_m2cm3': r.sa_tot_m2cm3,
                          'pld': r.pld,
                          'lcd': r.lcd} for r in _c.itertuples()}


def hmof_id(path):
    m = re.search(r'hMOF-(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def coverage_split(
    filenames,
    file_to_elements,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    random.seed(seed)
    filenames = list(filenames)
    random.shuffle(filenames)
    n = len(filenames)
    target = {"train": int(round(train_ratio * n)), "valid": int(round(valid_ratio * n))}
    target["test"] = n - target["train"] - target["valid"]

    splits = {"train": [], "valid": [], "test": []}
    split_sets = {"train": set(), "valid": set(), "test": set()}
    assigned = set()
    element_to_files = defaultdict(list)

    for f in tqdm(filenames):
        for e in file_to_elements[f]:
            element_to_files[e].append(f)

    rare_elements = sorted(element_to_files, key=lambda x: len(element_to_files[x]))

    for element in tqdm(rare_elements):
        candidates = element_to_files[element].copy()
        random.shuffle(candidates)
        for split in ["train", "valid", "test"]:
            if element in split_sets[split]:
                continue
            chosen = None

            for f in candidates:
                if f not in assigned:
                    chosen = f
                    break

            if chosen is None:
                continue

            splits[split].append(chosen)
            assigned.add(chosen)
            split_sets[split].update(file_to_elements[chosen])

    remaining = [f for f in filenames if f not in assigned]
    random.shuffle(remaining)
    for f in tqdm(remaining):
        if len(splits["train"]) < target["train"]:
            splits["train"].append(f)
        elif len(splits["valid"]) < target["valid"]:
            splits["valid"].append(f)
        else:
            splits["test"].append(f)

    return splits


def build_edge_index_and_attr(crystal, cutoff, max_neighbors, tie_tol=1e-2):
    center, nbr, images, dist = crystal.get_neighbor_list(cutoff)

    if len(center) == 0:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
                torch.empty((0, 3), dtype=torch.float),
                torch.empty((0, 3), dtype=torch.float))

    order = np.lexsort((dist, center))
    center, nbr, images, dist = center[order], nbr[order], images[order], dist[order]

    n_atoms = len(crystal)
    counts = np.bincount(center, minlength=n_atoms)
    first = np.searchsorted(center, np.arange(n_atoms), side="left")
    d_k = np.full(n_atoms, np.inf)
    has_k = counts >= max_neighbors
    d_k[has_k] = dist[first[has_k] + max_neighbors - 1]
    if tie_tol < 0:
        keep = (np.arange(len(center)) - first[center]) < max_neighbors
    else:
        keep = dist <= d_k[center] + tie_tol
    center, nbr, images, dist = center[keep], nbr[keep], images[keep], dist[keep]

    cart = crystal.cart_coords
    lattice = crystal.lattice.matrix
    edge_vec = cart[nbr] + images @ lattice - cart[center]

    edge_index = torch.tensor(np.vstack([nbr, center]), dtype=torch.long)
    edge_attr = torch.tensor(dist, dtype=torch.float).unsqueeze(1)
    offsets = torch.tensor(images, dtype=torch.float)
    edge_vec = torch.tensor(edge_vec, dtype=torch.float)

    return edge_index, edge_attr, offsets, edge_vec


element_to_atomic_number = {
    'H': 1,
    'C': 2,
    'N': 3,
    'O': 4,
    'F': 5,
    'Cl': 6,
    'V': 7,
    'Cu': 8,
    'Zn': 9,
    'Br': 10,
    'Zr': 11
}

STP_CM3_PER_MMOL = 22.414                                        # cm3(STP) / mmol
MOLAR_MASS = {'co2': 44.0095, 'ch4': 16.0425, 'h2': 2.01588}     # g/mol = mg/mmol


def to_mmol_per_g(value, unit, gas, rho):
    """rho: framework density [g/cm3]"""
    u = str(unit).strip().lower().replace(" ", "")
    M = MOLAR_MASS[gas]
    if u in ("mmol/g", "mol/kg"):
        return value
    if u == "mol/g":
        return value * 1000.0
    if u in ("cm3(stp)/g", "cm^3(stp)/g", "cc(stp)/g", "cm3/g"):
        return value / STP_CM3_PER_MMOL
    if u in ("cm3(stp)/cm3", "cm^3(stp)/cm^3", "cc(stp)/cc", "cm3/cm3", "v/v"):
        return value / (STP_CM3_PER_MMOL * rho)
    if u in ("g/l", "mg/cm3", "mg/ml"):
        return value / M / rho
    if u == "mg/g":
        return value / M
    raise ValueError(f"unsupported load unit: {unit}")


def to_kj_per_mol(value, unit):
    u = str(unit).strip().lower().replace(" ", "")
    if u == "kj/mol":
        return value
    if u == "j/mol":
        return value / 1000.0
    if u == "kcal/mol":
        return value * 4.184
    raise ValueError(f"unsupported heat unit: {unit}")


def cif_to_data(cif_path, load_target, heat_target, geo_values):
    crystal = Structure.from_file(cif_path)

    x = torch.tensor([element_to_atomic_number[site.specie.symbol] for site in crystal], dtype=torch.long)
    pos = torch.tensor(crystal.cart_coords, dtype=torch.float)
    cell = torch.tensor(crystal.lattice.matrix, dtype=torch.float).unsqueeze(0)
    edge_index, edge_attr, offsets, edge_vec = build_edge_index_and_attr(crystal=crystal, cutoff=args.cutoff, max_neighbors=args.max_neighbors, tie_tol=args.tie_tol)
    data = Data(
        x=x,
        pos=pos,
        cell=cell,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vec=edge_vec,
        offsets=offsets,
        mass_cell=torch.tensor([crystal.composition.weight], dtype=torch.float),
        num_nodes=len(crystal)
    )

    for key, value in load_target.items():
        data["load_" + key] = torch.tensor([value], dtype=torch.float)   # mmol/g
    for key, value in heat_target.items():
        data["heat_" + key] = torch.tensor([value], dtype=torch.float)   # kJ/mol
    for key, value in geo_values.items():
        data[key] = torch.tensor([value], dtype=torch.float)

    return data


def process_file(conf):
    idx, row, save_dir_path = conf
    cif_file = row
    json_file = row.replace('.cif', '.json')

    rho = float(Structure.from_file(cif_file).density)

    with open(json_file, 'r') as f:
        load_target_dict = {'co2_0.01': [], 'co2_0.05': [], 'co2_0.1': [], 'co2_0.5': [], 'co2_2.5': [],
                    'ch4_0.05': [], 'ch4_0.5': [], 'ch4_0.9': [], 'ch4_2.5': [], 'ch4_4.5': [], 'ch4_35.0': [],
                    'h2_2.0': [], 'h2_100.0': []}
        heat_target_dict = {'co2_0.01': [], 'co2_0.05': [], 'co2_0.1': [], 'co2_0.5': [], 'co2_2.5': [],
                    'ch4_0.05': [], 'ch4_0.5': [], 'ch4_0.9': [], 'ch4_2.5': [], 'ch4_4.5': [], 'ch4_35.0': [],
                    'h2_2.0': [], 'h2_100.0': []}
        json_data = json.load(f)
        for iso in range(len(json_data['isotherms'])):
            now_species = json_data['isotherms'][iso]['isotherm_data'][0]['species_data'][0]['name']
            if now_species == 'CarbonDioxide' or now_species == 'Methane' or now_species == 'Hydrogen':
                unit = json_data['isotherms'][iso]['adsorptionUnits']
                for k in range(len(json_data['isotherms'][iso]['isotherm_data'])):
                    now_isotherm_data = json_data['isotherms'][iso]['isotherm_data'][k]
                    if len(now_isotherm_data['species_data']) != 1:
                        raise AssertionError('species length :', len(now_isotherm_data['species_data']))
                    if now_isotherm_data['species_data'][0]['composition'] != 1:
                        raise AssertionError('composition :', now_isotherm_data['species_data'][0]['composition'])
                    if now_species == 'CarbonDioxide':
                        key = 'co2_' + str(now_isotherm_data['pressure'])
                        gas = 'co2'
                    if now_species == 'Methane':
                        key = 'ch4_' + str(now_isotherm_data['pressure'])
                        gas = 'ch4'
                    if now_species == 'Hydrogen':
                        key = 'h2_' + str(now_isotherm_data['pressure'])
                        gas = 'h2'
                    value = to_mmol_per_g(float(now_isotherm_data['species_data'][0]['adsorption']),
                                          unit, gas, rho)
                    load_target_dict[key].append(value)
        for iso in range(len(json_data['heats'])):
            now_species = json_data['heats'][iso]['isotherm_data'][0]['species_data'][0]['name']
            if now_species == 'CarbonDioxide' or now_species == 'Methane' or now_species == 'Hydrogen':
                unit = json_data['heats'][iso]['adsorptionUnits']
                for k in range(len(json_data['heats'][iso]['isotherm_data'])):
                    now_isotherm_data = json_data['heats'][iso]['isotherm_data'][k]
                    if len(now_isotherm_data['species_data']) != 1:
                        raise AssertionError('species length :', len(now_isotherm_data['species_data']))
                    if now_isotherm_data['species_data'][0]['composition'] != 1:
                        raise AssertionError('composition :', now_isotherm_data['species_data'][0]['composition'])
                    if now_species == 'CarbonDioxide':
                        key = 'co2_' + str(now_isotherm_data['pressure'])
                    if now_species == 'Methane':
                        key = 'ch4_' + str(now_isotherm_data['pressure'])
                    if now_species == 'Hydrogen':
                        key = 'h2_' + str(now_isotherm_data['pressure'])
                    value = to_kj_per_mol(float(now_isotherm_data['species_data'][0]['adsorption']),
                                          unit)
                    heat_target_dict[key].append(value)

        cons = CONSISTENT.get(hmof_id(cif_file))
        if cons is None:
            raise KeyError(f'{os.path.basename(cif_file)} not found in the consistent CSV')
        vals = {}
        for k in GEO_KEYS:
            v = cons[k]
            vals[k] = float(v) if np.isfinite(v) else float(json_data.get(k))

    data = cif_to_data(cif_path=cif_file, load_target=load_target_dict, heat_target=heat_target_dict, geo_values=vals)

    save_path = os.path.join(save_dir_path, f"{idx}.pt")
    torch.save(data, save_path)

    first = lambda d: {k: float(v[0]) for k, v in d.items() if len(v) > 0}
    return {'heat': first(heat_target_dict), 'load': first(load_target_dict)}


def usable_cpus():
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, math.floor(int(quota) / int(period)))
    except (FileNotFoundError, ValueError):
        try:
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, math.floor(quota / period))
        except (FileNotFoundError, ValueError):
            pass
    return max(1, n)


def main():
    data_dir = Path(args.data_path)
    file_list = glob(args.data_path + '/*')
    cif_file_list = [f for f in file_list if f.endswith('.cif')]

    if not os.path.exists(data_dir.parent / "train_files.csv"):
        file_to_elements = {}
        for fname in tqdm(cif_file_list):
            structure = Structure.from_file(fname)
            file_to_elements[fname] = {str(sp) for sp in structure.species}

        splits = coverage_split(
                    cif_file_list,
                    file_to_elements,
                    train_ratio=0.80,
                    valid_ratio=0.10,
                    test_ratio=0.10,
                    seed=42
                )

        train_files = splits["train"]
        valid_files = splits["valid"]
        test_files = splits["test"]
        for ty, files in [("train", train_files), ("valid", valid_files), ("test", test_files)]:
            pd.DataFrame({"filename": [os.path.basename(f) for f in files]}).to_csv(
                data_dir.parent / f"{ty}_files.csv", index=False)

        all_elements = set()
        for elems in file_to_elements.values():
            all_elements |= elems
        print(all_elements)
        print(len(all_elements))

        for split_name, split_files in splits.items():
            elems = set()
            for f in split_files:
                elems |= file_to_elements[f]
            missing = all_elements - elems

            print("=" * 40)
            print(split_name)
            print("#structures :", len(split_files))
            print("#elements   :", len(elems))

            if len(missing) == 0:
                print("PASS")
            else:
                print("Missing:", sorted(missing))

    stats_rows = []
    n_workers = max(1, usable_cpus() - 1)
    print(f"using {n_workers} workers | pore descriptors: consistent set ({args.consistent_csv})")

    for ty in ['train', 'valid', 'test']:
        save_dir_path = os.path.join(args.save_path, f"{ty}")
        os.makedirs(save_dir_path, exist_ok=True)

        target_files = [os.path.join(args.data_path, os.path.basename(f))
                        for f in pd.read_csv(data_dir.parent / f"{ty}_files.csv")["filename"]]
        print(len(target_files))

        files_to_process = [(idx, row, save_dir_path) for idx, row in enumerate(target_files)]

        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_file, files_to_process), total=len(files_to_process)))

        print(f"Finished processing {len(results)} files out of {len(files_to_process)}.")
        for r in results:
            stats_rows.append({'split': ty, **{f'heat_{k}': v for k, v in r['heat'].items()},
                               **{f'load_{k}': v for k, v in r['load'].items()}})

    df = pd.DataFrame(stats_rows)
    out = []
    for split, g in [*df.groupby('split', sort=False), ('total', df)]:
        for gas in ['co2', 'ch4']:
            for p in sorted({float(c.split('_')[-1]) for c in df.columns if c.startswith(f'heat_{gas}_')}):
                h, l = g.get(f'heat_{gas}_{p}'), g.get(f'load_{gas}_{p}')
                out.append({'split': split, 'gas': gas.upper(), 'pressure_bar': p,
                            'heat_mean': h.mean(), 'heat_std': h.std(), 'load_mean': l.mean(), 'load_std': l.std(),
                            'n': int(h.notna().sum())})
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(args.save_path, 'dataset_stats.csv'), index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__=='__main__':
    main()
