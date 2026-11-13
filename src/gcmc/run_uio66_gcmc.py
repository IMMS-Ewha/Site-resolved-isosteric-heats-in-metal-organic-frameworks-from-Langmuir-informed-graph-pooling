"""
GCMC of CH4 in UiO-66 and its functionalized variants with the hMOF (MOFX-DB) labelling protocol,
plus an hMOF-0 control run that checks the setup against the MOFX-DB labels.

Protocol (MOFX-DB simin of the hMOF CH4 isotherms/heats, DOI 10.1039/C2EE23201D):
  UFF framework (CrystalGenerator, Mof_* pseudo atoms) + TraPPE-UA CH4, Lorentz-Berthelot, truncated, no tail correction,
  CutOff 12.0 A, no charges, 298 K, 0.05/0.5/0.9/2.5/4.5 bar, 1000 initialization + 1000 production cycles.
  Unit cells: hMOF uses 2x2x2; here the number is computed so that every perpendicular width exceeds 2 x cutoff
  (gives 2x2x2 for hMOF-0 and for cubic UiO-66).

One run does everything:
  1) writes P1 CIFs with RASPA labels (Mof_<element>), wrapped fractional coordinates
  2) helium Widom runs -> helium void fraction (needed for excess loading)
  3) CH4 GCMC runs for every structure x pressure, in parallel
  4) parses loading (absolute and excess) and Qst (= -enthalpy of adsorption, kJ/mol); loading is read in
     molecules/unit cell and converted to mmol/g with the framework mass from the CIF, so it does not depend on the
     masses RASPA reads from pseudo_atoms.def
  5) compares hMOF-0 with its MOFX-DB labels to tell whether the labels are absolute or excess loading
  6) optional: CH4 density on a 3D grid (VTK) at the pressures given by --density_pressures, accumulated over the
     production cycles with the supercell folded back into one unit cell (DensityAveragingTypeVTK UnitCell)

pseudo_atoms.def in the older 10-column layout (type print as scat mass charge ...) is rewritten into the RASPA2
14-column layout (type print as chem oxidation mass charge ... anisotropic anisotropic-type tinker-type); read in the
old layout, RASPA2 takes the integer part of the mass as the oxidation state and only the decimals as the mass.

    # run from src/; settings of the paper (cutoff 12.8 A, CH4 sigma 3.75 A, no tail corrections, 2x2x2 cells)
    python gcmc/run_uio66_gcmc.py --cifs ../dataset/UiO-66/cif/*.cif --ff_dir gcmc/ff \
        --out_dir ../results_hmof/gcmc_uio66 --cutoff 12.8 --ch4_sigma 3.75 --tail_corrections no \
        --unit_cells 2 2 2 --init_cycles 5000 --cycles 50000 --density_pressures all
    python gcmc/run_uio66_gcmc.py --hmof_cif gcmc/hMOF-0.cif --hmof_json gcmc/hMOF-0.json --ff_dir gcmc/ff \
        --out_dir ../results_hmof/hmof0_check --cutoff 12.8 --ch4_sigma 3.75 --tail_corrections no \
        --unit_cells 2 2 2 --init_cycles 5000 --cycles 50000
                                                     # hMOF-0 control only
    python run_uio66_gcmc.py ... --collect_only      # re-parse finished runs
    python run_uio66_gcmc.py ... --dry_run           # write inputs only, no RASPA
    python run_uio66_gcmc.py ... --tail_corrections yes   # test: LJ tail corrections on (file default: no)
    python run_uio66_gcmc.py ... --cutoff 12.8 --ch4_sigma 3.75 --unit_cells 2 2 2
                                                     # settings stated in the ESI of Wilmer et al. 2012
    python run_uio66_gcmc.py ... --density_pressures all          # VTK density at every pressure (or e.g. 4.5)

ff_dir must contain force_field_mixing_rules.def, pseudo_atoms.def and force_field.def.
methane.def (TraPPE-UA, bead CH4_sp3) and helium.def (bead He) are written by this script in the standard RASPA
single-bead format, so no molecule file and no RASPA share directory are needed.
Outputs (out_dir): helium_void_fraction.csv, gcmc_ch4_results.csv (with the path of the VTK density file when
computed), hmof0_check.csv
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

ap = argparse.ArgumentParser(description='CH4 GCMC with the hMOF protocol for UiO-66 variants + hMOF-0 control')
ap.add_argument('--cifs', nargs='*', default=[], help='UiO-66 and functionalized variant CIFs (file stem = name); '
                                                    'omit to run the hMOF-0 control only')
ap.add_argument('--hmof_cif', default=None, type=str, help='hMOF-0 CIF for the control run')
ap.add_argument('--hmof_json', default=None, type=str, help='hMOF-0 MOFX-DB JSON with the CH4 labels')
ap.add_argument('--ff_dir', required=True, type=str)
ap.add_argument('--out_dir', default='../results_hmof/gcmc_uio66', type=str)
ap.add_argument('--pressures', nargs='+', type=float, default=[0.05, 0.5, 0.9, 2.5, 4.5], help='bar')
ap.add_argument('--temperature', default=298.0, type=float)
ap.add_argument('--cutoff', default=12.0, type=float)
ap.add_argument('--init_cycles', default=1000, type=int, help='hMOF protocol: 1000')
ap.add_argument('--cycles', default=1000, type=int, help='hMOF protocol: 1000')
ap.add_argument('--widom_cycles', default=2000, type=int)
ap.add_argument('--tail_corrections', default='no', choices=['no', 'yes'],
                help='overrides the tail-correction rule of force_field_mixing_rules.def (hMOF files: no)')
ap.add_argument('--unit_cells', nargs=3, type=int, default=None,
                help='fixed replication for every structure (Wilmer 2012: 2 2 2); default: smallest with widths > 2 x cutoff')
ap.add_argument('--ch4_sigma', default=None, type=float,
                help='overrides sigma of CH4_sp3 in force_field_mixing_rules.def (files: 3.73 A; Wilmer 2012 ESI: 3.75 A)')
ap.add_argument('--density_pressures', nargs='*', default=[],
                help="pressures (bar) at which the CH4 density grid is accumulated, or 'all'; default: none")
ap.add_argument('--density_grid', default=150, type=int, help='grid points per unit-cell axis (default 150)')
ap.add_argument('--workers', default=0, type=int, help='0 = all usable CPUs')
ap.add_argument('--raspa', default=None, type=str, help='path to RASPA simulate (default: $RASPA_DIR/bin/simulate)')
ap.add_argument('--collect_only', action='store_true')
ap.add_argument('--dry_run', action='store_true')
args = ap.parse_args()

FF_FILES = ['force_field_mixing_rules.def', 'pseudo_atoms.def', 'force_field.def']
# single-bead molecules: (pseudo atom, Tc [K], Pc [Pa], acentric factor); critical constants enter only the fugacity
MOLECULES = {'methane': ('CH4_sp3', 190.564, 4599200.0, 0.01142),
             'helium': ('He', 5.2, 228000.0, -0.39)}
CH4_SIMIN_PRESSURES = {0.05, 0.5, 0.9, 2.5, 4.5}


def usable_cpus():
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if quota != 'max':
            n = min(n, math.floor(int(quota) / int(period)))
    except (FileNotFoundError, ValueError):
        try:
            quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
            period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
            if quota > 0:
                n = min(n, math.floor(quota / period))
        except (FileNotFoundError, ValueError):
            pass
    return max(1, n)


def pseudo_atom_names(ff_dir):
    lines = [l for l in open(os.path.join(ff_dir, 'pseudo_atoms.def')).read().splitlines() if l.strip()]
    return {l.split()[0] for l in lines[3:] if not l.startswith('#')}


def check_ff_dir(ff_dir):
    for f in FF_FILES:
        if not os.path.isfile(os.path.join(ff_dir, f)):
            raise SystemExit(f'{f} not found in {ff_dir}')


def write_pseudo_atoms(src, dst):
    """copy pseudo_atoms.def, converting the old 10-column layout into the RASPA2 14-column layout"""
    lines = open(src).read().splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.lstrip().startswith('#type')), None)
    if hdr is None or 'oxidation' in lines[hdr]:
        shutil.copy(src, dst)
        return False
    out = lines[:hdr] + ['#type      print   as    chem  oxidation   mass        charge   polarization B-factor radii  '
                         'connectivity anisotropic anisotropic-type   tinker-type']
    for l in lines[hdr + 1:]:
        t = l.split()
        if not t or l.lstrip().startswith('#'):
            out.append(l)
            continue
        if len(t) != 10:
            raise SystemExit(f'pseudo_atoms.def: unexpected line "{l}"')
        out.append(f'{t[0]:<10s} {t[1]:<6s} {t[2]:<5s} {t[3]:<5s} 0  {t[4]:>11s} {t[5]:>8s} {t[6]:>8s} {t[7]:>6s} '
                   f'{t[8]:>6s} {t[9]:>3s}  0  relative  0')
    with open(dst, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    return True


def write_mixing_rules(src, dst):
    """copy force_field_mixing_rules.def with the tail-correction rule set by --tail_corrections"""
    lines = open(src).read().splitlines()
    i = next((k for k, l in enumerate(lines) if 'tailcorrection' in l.lower().replace(' ', '')), None)
    if i is None or i + 1 >= len(lines):
        raise SystemExit('force_field_mixing_rules.def: tail-correction rule not found')
    lines[i + 1] = args.tail_corrections
    if args.ch4_sigma is not None:
        k = next((k for k, l in enumerate(lines) if l.split()[:1] == ['CH4_sp3']), None)
        if k is None:
            raise SystemExit('force_field_mixing_rules.def: CH4_sp3 not found')
        t = lines[k].split()
        lines[k] = f'{t[0]:<15s}{t[1]:<17s}{t[2]:>9s}{args.ch4_sigma:>10.3f}'
    with open(dst, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def molecule_def_text(molecule):
    """standard RASPA definition of a single-bead molecule (one flexible group with one atom)"""
    bead, tc, pc, omega = MOLECULES[molecule]
    return '\n'.join([
        '# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]',
        f'{tc}', f'{pc}', f'{omega}',
        '# Number Of Atoms', '1',
        '# Number Of Groups', '1',
        f'# {molecule}-group', 'flexible',
        '# number of atoms', '1',
        '# atomic positions', f'0 {bead}',
        '# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend '
        'Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb',
        '               0    0            0    0            0       0        0            0         0            0'
        '         0               0            0        0            0',
        '# Number of config moves', '0', ''])


def load_structure(cif):
    s = Structure.from_file(cif)
    if not s.is_ordered:
        raise SystemExit(f'{cif}: partial occupancies found; give an ordered P1 structure')
    return s


def write_raspa_cif(struct, path, name, known):
    """P1 CIF, labels = Mof_<element> (pseudo-atom names), fractional coordinates wrapped into [0, 1)"""
    lat = struct.lattice
    rows = []
    for site in struct:
        el = site.specie.symbol
        label = f'Mof_{el}'
        if label not in known:
            raise SystemExit(f'{name}: pseudo atom {label} is not defined in pseudo_atoms.def')
        f = np.mod(site.frac_coords, 1.0)
        rows.append(f'{label}\t{el}\t{f[0]:.6f}\t{f[1]:.6f}\t{f[2]:.6f}')
    txt = '\n'.join([
        f'data_{name}',
        "_symmetry_space_group_name_H-M\t'P1'",
        '_symmetry_Int_Tables_number\t1',
        'loop_', '_symmetry_equiv_pos_as_xyz', '  x,y,z',
        f'_cell_length_a\t{lat.a:.6f}', f'_cell_length_b\t{lat.b:.6f}', f'_cell_length_c\t{lat.c:.6f}',
        f'_cell_angle_alpha\t{lat.alpha:.6f}', f'_cell_angle_beta\t{lat.beta:.6f}', f'_cell_angle_gamma\t{lat.gamma:.6f}',
        'loop_', '_atom_site_label', '_atom_site_type_symbol',
        '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z', *rows, ''])
    with open(path, 'w') as fh:
        fh.write(txt)


def unit_cells(struct, cutoff):
    """smallest replication with every perpendicular width > 2 x cutoff"""
    m = struct.lattice.matrix
    vol = abs(np.linalg.det(m))
    widths = [vol / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3)]
    return [max(1, math.ceil(2 * cutoff / w + 1e-9)) for w in widths]


def common_header(name, cells, cycles, init_cycles, print_every):
    return '\n'.join([
        'SimulationType                MonteCarlo',
        f'NumberOfCycles                {cycles}',
        f'NumberOfInitializationCycles  {init_cycles}',
        f'PrintEvery                    {print_every}',
        'RestartFile                   no',
        '',
        'Forcefield                    Local',
        f'CutOff                        {args.cutoff}',
        'ChargeMethod                  None',
        'UseChargesFromCIFFile         no',
        'RemoveAtomNumberCodeFromLabel no',
        '',
        'Framework 0',
        f'FrameworkName                 {name}',
        f'UnitCells                     {cells[0]} {cells[1]} {cells[2]}'])


def helium_input(name, cells):
    return '\n'.join([
        common_header(name, cells, args.widom_cycles, 0, max(1, args.widom_cycles // 4)),
        f'ExternalTemperature           {args.temperature}',
        '',
        'Component 0 MoleculeName             helium',
        '            MoleculeDefinition       Local',
        '            WidomProbability         1.0',
        '            CreateNumberOfMolecules  0', ''])


def density_on(p_bar):
    dp = args.density_pressures
    return bool(dp) and ('all' in dp or any(abs(float(x) - p_bar) < 1e-9 for x in dp if x != 'all'))


def density_block():
    every = 5000 if args.cycles >= 5000 else max(1, args.cycles // 2)
    g = args.density_grid
    return ['ComputeDensityProfile3DVTKGrid    yes',
            f'WriteDensityProfile3DVTKGridEvery {every}',
            f'DensityProfile3DVTKGridPoints     {g} {g} {g}',
            'DensityAveragingTypeVTK           UnitCell']


def ch4_input(name, cells, p_bar, hvf):
    return '\n'.join([
        common_header(name, cells, args.cycles, args.init_cycles, 250),
        *(density_block() if density_on(p_bar) else []),
        f'HeliumVoidFraction            {hvf:.6f}',
        f'ExternalTemperature           {args.temperature}',
        f'ExternalPressure              {p_bar * 1e5:.1f}',
        '',
        'Component 0 MoleculeName             methane',
        '            StartingBead             0',
        '            MoleculeDefinition       Local',
        '            IdealGasRosenbluthWeight 1.0',
        '            TranslationProbability   1.0',
        '            RotationProbability      1.0',
        '            ReinsertionProbability   1.0',
        '            SwapProbability          1.0',
        '            CreateNumberOfMolecules  0', ''])


def prepare_dir(run_dir, name, cif_src, simin, molecule):
    """molecule: 'methane' or 'helium'; the molecule definition is written next to simulation.input"""
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(cif_src, os.path.join(run_dir, f'{name}.cif'))
    write_pseudo_atoms(os.path.join(args.ff_dir, 'pseudo_atoms.def'), os.path.join(run_dir, 'pseudo_atoms.def'))
    write_mixing_rules(os.path.join(args.ff_dir, 'force_field_mixing_rules.def'),
                       os.path.join(run_dir, 'force_field_mixing_rules.def'))
    shutil.copy(os.path.join(args.ff_dir, 'force_field.def'), os.path.join(run_dir, 'force_field.def'))
    with open(os.path.join(run_dir, f'{molecule}.def'), 'w') as fh:
        fh.write(molecule_def_text(molecule))
    with open(os.path.join(run_dir, 'simulation.input'), 'w') as fh:
        fh.write(simin)


def raspa_binary():
    if args.raspa:
        return args.raspa
    rd = os.environ.get('RASPA_DIR')
    cand = os.path.join(rd, 'bin', 'simulate') if rd else shutil.which('simulate')
    if not cand or not os.path.isfile(cand):
        raise SystemExit('RASPA simulate not found: set $RASPA_DIR or pass --raspa')
    return cand


def run_one(job):
    binary, run_dir = job
    with open(os.path.join(run_dir, 'raspa.log'), 'w') as log:
        rc = subprocess.run([binary], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT).returncode
    return run_dir, rc


def run_all(dirs, workers):
    binary = raspa_binary()
    with Pool(workers) as pool:
        for run_dir, rc in pool.imap_unordered(run_one, [(binary, d) for d in dirs]):
            print(f'  {"done " if rc == 0 else "FAIL "} {run_dir}' + ('' if rc == 0 else f' (exit {rc}, see raspa.log)'))


def output_text(run_dir):
    files = sorted(glob(os.path.join(run_dir, 'Output', 'System_0', '*.data')))
    if not files:
        files = sorted(f for f in glob(os.path.join(run_dir, 'Output', '**', '*'), recursive=True) if os.path.isfile(f))
    return open(files[-1], errors='replace').read() if files else None


def density_file(run_dir):
    files = sorted(glob(os.path.join(run_dir, 'VTK', '**', '*.vtk'), recursive=True))
    return os.path.relpath(files[-1]) if files else ''


def diagnose(run_dir, n=15):
    """print what RASPA left behind when a run could not be parsed"""
    out_files = [os.path.relpath(f, run_dir) for f in glob(os.path.join(run_dir, 'Output', '**', '*'), recursive=True)
                 if os.path.isfile(f)]
    print(f'    output files: {out_files if out_files else "none (RASPA stopped before writing output)"}')
    log = os.path.join(run_dir, 'raspa.log')
    if os.path.isfile(log):
        tail = open(log, errors='replace').read().strip().splitlines()[-n:]
        print('    raspa.log (last lines):' + ('' if tail else ' empty'))
        for line in tail:
            print('      | ' + line)


NUM = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'


def parse_widom(txt):
    m = re.findall(r'Average Widom Rosenbluth-weight:\s+' + NUM + r'\s+\+/-\s+' + NUM, txt)
    return (float(m[-1][0]), float(m[-1][1])) if m else (np.nan, np.nan)


def parse_ch4(txt, mass_uc):
    """loading from molecules/unit cell -> mmol/g with the CIF framework mass (g/mol per unit cell)"""
    out = {}
    for key, pat in [('n_abs_uc', r'Average loading absolute \[molecules/unit cell\]'),
                     ('n_exc_uc', r'Average loading excess \[molecules/unit cell\]'),
                     ('raspa_abs_molkg', r'Average loading absolute \[mol/kg framework\]')]:
        m = re.search(pat + r'\s+' + NUM + r'\s+\+/-\s+' + NUM, txt)
        out[key], out[key + '_err'] = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    for key in ['abs', 'exc']:
        out[f'load_{key}'] = out[f'n_{key}_uc'] / mass_uc * 1000.0
        out[f'load_{key}_err'] = out[f'n_{key}_uc_err'] / mass_uc * 1000.0
    i = txt.find('Enthalpy of adsorption:')
    m = re.search(NUM + r'\s+\+/-\s+' + NUM + r'\s+\[KJ/MOL\]', txt[i:]) if i >= 0 else None
    out['qst'], out['qst_err'] = (-float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    return out


def hmof_labels(path):
    """MOFX-DB CH4 labels at the hMOF pressures: {p: (loading mol/kg, Qst kJ/mol)}"""
    d = json.load(open(path))

    def pick(entries, unit):
        for e in entries:
            names = {s['name'] for row in e['isotherm_data'] for s in row['species_data']}
            ps = {round(r['pressure'], 4) for r in e['isotherm_data']}
            if names == {'Methane'} and CH4_SIMIN_PRESSURES <= ps and e['adsorptionUnits'].lower() == unit:
                return {round(r['pressure'], 4): r['species_data'][0]['adsorption'] for r in e['isotherm_data']}
        return {}
    load, heat = pick(d['isotherms'], 'mol/kg'), pick(d['heats'], 'kj/mol')
    return {p: (load.get(p, np.nan), heat.get(p, np.nan)) for p in sorted(CH4_SIMIN_PRESSURES)}, d.get('void_fraction')


def main():
    check_ff_dir(args.ff_dir)
    known = pseudo_atom_names(args.ff_dir)
    out = Path(args.out_dir)
    (out / 'cif').mkdir(parents=True, exist_ok=True)
    workers = args.workers or usable_cpus()

    systems = [(Path(c).stem, c) for c in args.cifs]
    if args.hmof_cif:
        systems.append(('hMOF-0_control', args.hmof_cif))
    if not systems:
        raise SystemExit('nothing to run: give --cifs and/or --hmof_cif')
    info = {}
    for name, cif in systems:
        s = load_structure(cif)
        dst = out / 'cif' / f'{name}.cif'
        write_raspa_cif(s, dst, name, known)
        cells = list(args.unit_cells) if args.unit_cells else unit_cells(s, args.cutoff)
        if args.unit_cells and cells != unit_cells(s, args.cutoff):
            print(f'[note] {name}: fixed unit cells {cells} are smaller than the minimum-image requirement '
                  f'{unit_cells(s, args.cutoff)} for a {args.cutoff} A cutoff')
        info[name] = {'cif': str(dst), 'cells': cells, 'n_atoms': len(s),
                      'mass': float(s.composition.weight)}
        print(f'[structure] {name}: {len(s)} atoms, {info[name]["mass"]:.2f} g/mol per unit cell, '
              f'unit cells {info[name]["cells"]}')
    if write_pseudo_atoms(os.path.join(args.ff_dir, 'pseudo_atoms.def'), os.devnull):
        print('[ff] pseudo_atoms.def is in the old 10-column layout -> rewritten in the RASPA2 layout for every run')
    bad = [x for x in args.density_pressures if x != 'all' and float(x) not in args.pressures]
    if bad:
        raise SystemExit(f'--density_pressures {bad} not among --pressures {args.pressures}')
    if args.density_pressures:
        print(f'[density] {args.density_grid}^3 grid per unit cell at '
              f'{"all pressures" if "all" in args.density_pressures else ", ".join(args.density_pressures) + " bar"}')
    print(f'[ff] cutoff {args.cutoff} A, tail corrections: {args.tail_corrections}, '
          f'CH4 sigma: {args.ch4_sigma if args.ch4_sigma is not None else "from file"}')

    # 1) helium void fraction
    he_dirs = {name: str(out / 'helium' / name) for name in info}
    if not args.collect_only:
        for name, d in he_dirs.items():
            prepare_dir(d, name, info[name]['cif'], helium_input(name, info[name]['cells']), molecule='helium')
        if args.dry_run:
            print('[dry run] helium inputs written')
        else:
            print(f'[helium] {len(he_dirs)} Widom runs on {min(workers, len(he_dirs))} workers')
            run_all(list(he_dirs.values()), workers)
    he_rows = []
    for name, d in he_dirs.items():
        txt = output_text(d)
        vf, err = parse_widom(txt) if txt else (np.nan, np.nan)
        if not args.dry_run and not np.isfinite(vf):
            print(f'[warn] could not parse helium void fraction in {d}')
            diagnose(d)
        info[name]['hvf'] = vf
        he_rows.append({'name': name, 'helium_void_fraction': vf, 'err': err})
    pd.DataFrame(he_rows).to_csv(out / 'helium_void_fraction.csv', index=False)

    # 2) CH4 GCMC
    ch4_dirs = []
    for name in info:
        hvf = info[name]['hvf'] if np.isfinite(info[name].get('hvf', np.nan)) else 0.0
        if hvf == 0.0 and not args.dry_run:
            print(f'[warn] {name}: helium void fraction not available -> excess loading will equal absolute loading')
        for p in args.pressures:
            d = str(out / 'ch4' / name / f'{p:g}bar')
            ch4_dirs.append((name, p, d))
            if not args.collect_only:
                prepare_dir(d, name, info[name]['cif'], ch4_input(name, info[name]['cells'], p, hvf), molecule='methane')
    if args.dry_run:
        print(f'[dry run] {len(ch4_dirs)} CH4 inputs written under {out / "ch4"}')
        return
    if not args.collect_only:
        print(f'[ch4] {len(ch4_dirs)} GCMC runs on {min(workers, len(ch4_dirs))} workers')
        run_all([d for _, _, d in ch4_dirs], workers)

    rows = []
    shown = 0
    for name, p, d in ch4_dirs:
        txt = output_text(d)
        r = parse_ch4(txt, info[name]['mass']) if txt else {}
        if not txt or not np.isfinite(r.get('qst', np.nan)):
            print(f'[warn] could not parse {d}')
            if shown < 2:                              # details for the first two failures only
                diagnose(d)
                shown += 1
        rows.append({'name': name, 'pressure_bar': p, **r, 'density_vtk': density_file(d) if density_on(p) else '',
                     'unit_cells': 'x'.join(map(str, info[name]['cells'])),
                     'helium_void_fraction': info[name]['hvf'], 'init_cycles': args.init_cycles, 'cycles': args.cycles})
    cols = ['name', 'pressure_bar', 'load_abs', 'load_abs_err', 'load_exc', 'load_exc_err', 'qst', 'qst_err',
            'n_abs_uc', 'n_exc_uc', 'raspa_abs_molkg', 'density_vtk', 'unit_cells', 'helium_void_fraction',
            'tail_corrections',
            'init_cycles', 'cycles']
    res = pd.DataFrame(rows).assign(tail_corrections=args.tail_corrections, cutoff=args.cutoff,
                                    ch4_sigma=args.ch4_sigma).reindex(columns=cols + ['cutoff', 'ch4_sigma'])
    res.to_csv(out / 'gcmc_ch4_results.csv', index=False)
    if res['qst'].isna().all():
        raise SystemExit('\nno CH4 run could be parsed; see the raspa.log lines above')
    ratio = (res['raspa_abs_molkg'] / res['load_abs']).median()
    if np.isfinite(ratio) and abs(ratio - 1) > 0.01:
        print(f'[note] RASPA mol/kg differs from the CIF-mass conversion by a factor {ratio:.2f} '
              f'(framework mass read by RASPA is wrong); mmol/g values below use the CIF mass')

    pd.set_option('display.width', 200)
    print('\n=== CH4 GCMC (loading mmol/g, Qst kJ/mol) ===')
    print(res[['name', 'pressure_bar', 'load_abs', 'load_exc', 'qst', 'qst_err']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    # 3) hMOF-0 control vs MOFX-DB labels
    if args.hmof_cif and args.hmof_json:
        labels, vf_db = hmof_labels(args.hmof_json)
        c = res[res['name'] == 'hMOF-0_control'].set_index('pressure_bar')
        chk = []
        for p, (lab_load, lab_heat) in labels.items():
            if p not in c.index:
                continue
            r = c.loc[p]
            chk.append({'pressure_bar': p, 'label_loading': lab_load, 'gcmc_abs': r['load_abs'], 'gcmc_exc': r['load_exc'],
                        'label_qst': lab_heat, 'gcmc_qst': r['qst'], 'gcmc_qst_err': r['qst_err']})
        chk = pd.DataFrame(chk)
        chk.to_csv(out / 'hmof0_check.csv', index=False)
        print('\n=== hMOF-0 control vs MOFX-DB labels ===')
        print(chk.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        rel = lambda a: float(np.nanmean(np.abs(chk[a] - chk['label_loading']) / chk['label_loading']))
        e_abs, e_exc = rel('gcmc_abs'), rel('gcmc_exc')
        print(f'mean relative deviation of loading: absolute {e_abs:.3f}, excess {e_exc:.3f} '
              f'-> labels look like {"ABSOLUTE" if e_abs < e_exc else "EXCESS"} loading')
        print(f'Qst MAE vs labels: {np.nanmean(np.abs(chk["gcmc_qst"] - chk["label_qst"])):.3f} kJ/mol')
        print(f'helium void fraction: GCMC {info["hMOF-0_control"]["hvf"]:.4f} vs MOFX-DB {vf_db}')

    if args.density_pressures:
        miss = res[(res['density_vtk'] == '') & res['pressure_bar'].apply(density_on)]
        print(f'\n[density] {int((res["density_vtk"] != "").sum())} VTK files found'
              + (f'; missing for {len(miss)} runs (check VTK/ in those folders)' if len(miss) else ''))
    print(f'\n[saved] {out}/helium_void_fraction.csv, gcmc_ch4_results.csv' + (', hmof0_check.csv' if args.hmof_json else ''))


if __name__ == '__main__':
    main()