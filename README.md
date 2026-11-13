# Site-resolved isosteric heats in metal–organic frameworks from Langmuir-informed graph pooling

Code and data for a Langmuir-informed graph pooling model that predicts CO2 and CH4 isosteric heats of adsorption
at five pressures from the crystal structure of a MOF (hMOF database), for the site-level analysis of the trained
models, and for the GCMC case study of UiO-66 and four functionalized variants.

## Repository layout

**Run every command from `src/`.** All default paths are relative to `src/` and point to `../dataset` and
`../results_hmof` inside this repository.

```
<this repository>/
├── src/                                   code (run commands here)
│   ├── data/                              cleaning of the MOFX-DB download, graph building for hMOF
│   ├── model/                             encoders (SchNet, PaiNN, ViSNet), GDPool, Langmuir-informed readout
│   ├── utils/                             training/testing loops, Langmuir analysis, symmetry check
│   └── gcmc/                              RASPA2 GCMC of UiO-66 (force-field files, hMOF-0 control input)
├── dataset/
│   ├── hMOF/
│   │   ├── train_files.csv, valid_files.csv, test_files.csv    split of the paper (CIF file names)
│   │   ├── geo_mapping/consistent_hMOF.csv                      MOFX-DB consistent textural properties
│   │   ├── hmof_data/                                           MOFX-DB download (not tracked)
│   │   ├── hmof_data_clean_void_inc/                            after data/clean_void_inc.py (not tracked)
│   │   └── full_dataset_clean_void_inc_consistent_sym/          graphs: train/ valid/ test/ (not tracked)
│   └── UiO-66/
│       ├── cif/                                                 UiO-66 and its –NH2, –Br, –NO2, –(CH3)2 variants
│       └── uio66/                                               graphs of these structures (not tracked)
└── results_hmof/                          checkpoints, analysis folders, GCMC runs, figures (not tracked)
```

External programs: [RASPA2](https://github.com/iRASPA/RASPA2) for the GCMC runs and
[Zeo++](https://www.zeoplusplus.org/) (binary `network` on `PATH`) for the pore descriptors of new structures.
Python packages are listed in `requirements.txt`.

## 1. Data

### hMOF structures and labels (download)

From MOFX-DB (<https://mof.tech.northwestern.edu/databases>), **Download Databases → hMOF**: CIF files and JSON
files with the simulated isotherms and isosteric heats. The cleaning step needs the CO2, CH4 and H2 data of each
structure. Put all `.cif` and `.json` files in `dataset/hMOF/hmof_data/`. The force fields of the hMOF simulations
are listed on the same page under **Download ForceFields**.

### Pore descriptors (included)

`dataset/hMOF/geo_mapping/consistent_hMOF.csv` is the hMOF table of the MOFX-DB **consistent set** of textural
properties (**Download Textural Properties → Consistent_set**, `Textural_Properties_CSV.zip`), included without
changes: 137,953 structures, columns `id` (hMOF number), `vf`, `sa_acc_m2g`, `sa_tot_m2g`, `sa_acc_m2cm3`,
`sa_tot_m2cm3`, `pld`, `lcd`, `lcd_free`. The model uses `vf`, `sa_tot_m2g`, `sa_tot_m2cm3`, `pld` and `lcd`.
If you use these data, cite MOFX-DB and the hMOF database:

- N. S. Bobbitt, K. Shi, B. J. Bucior, H. Chen, N. Tracy-Amoroso, Z. Li, Y. Sun, J. H. Merlin, J. I. Siepmann,
  D. W. Siderius and R. Q. Snurr, *J. Chem. Eng. Data*, 2023, **68**, 483–498.
- C. E. Wilmer, M. Leaf, C. Y. Lee, O. K. Farha, B. G. Hauser, J. T. Hupp and R. Q. Snurr, *Nat. Chem.*, 2012,
  **4**, 83–89.

### Clean

```bash
python data/clean_void_inc.py
```

Input and output folders are set at the top of the script (`JSON_DIR`, `DST_DIR`; the output folder must be empty).
A structure is kept when

1. loading and heat are available at all 13 gas/pressure points
   (CO2 0.01, 0.05, 0.1, 0.5, 2.5 bar; CH4 0.05, 0.5, 0.9, 2.5, 4.5, 35 bar; H2 2, 100 bar),
2. every heat is > 0 and every loading is ≥ 0,
3. the CIF file exists,
4. the pore descriptors in the JSON file are valid (0 < void fraction ≤ 1, surface areas ≥ 0, PLD > 0, LCD > 0,
   PLD ≤ LCD),
5. the loading increases with pressure for every gas.

126,944 structures are kept. The kept `.cif` / `.json` pairs are copied to `DST_DIR`, and a report goes to
`dataset/hMOF/check_report/`. Only the CO2 points up to 2.5 bar and the CH4 points up to 4.5 bar are used as labels;
the other points are used only in the checks above.

### Build graphs

```bash
python data/preprocess_hmof_consistent.py
```

Defaults: `--data_path ../dataset/hMOF/hmof_data_clean_void_inc`,
`--save_path ../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym`,
`--consistent_csv ../dataset/hMOF/geo_mapping/consistent_hMOF.csv`.

- Graphs: neighbours within 6 Å, the 12 nearest per atom plus every neighbour within 0.01 Å of the 12th distance
  (`--tie_tol`), so that symmetry-equivalent atoms keep the same neighbour set.
- Labels: loading in mmol/g and isosteric heat in kJ/mol at every pressure.
- Pore descriptors (void fraction, surface areas, PLD, LCD) come from the consistent set.
- Split: the split of the paper in `dataset/hMOF/{train,valid,test}_files.csv` (101,555 / 12,694 / 12,695
  structures) is used as it is. Without these files, a new 80/10/10 split (seed 42) that keeps every element in
  every split is made and written there.

Output: `<save_path>/{train,valid,test}/<idx>.pt` and `<save_path>/dataset_stats.csv`.

### Other structures (e.g. UiO-66)

```bash
python preprocess_cif_consistent.py            # defaults: --cif_path ../dataset/UiO-66/cif --save_path ../dataset/UiO-66/uio66
```

The pore descriptors are computed from the CIF with the protocol of the consistent set, so that hMOF structures
get the same values as in the table: PLD, LCD and surface area (accessible + non-accessible, 1.86 Å probe) with
Zeo++ and UFF radii, and the helium void fraction by Widom insertion (UFF + helium Lennard-Jones, 12.8 Å cutoff,
298 K). The Zeo++ binary `network` must be on `PATH` (or pass `--zeopp`). Graphs are built as for hMOF.
Supported elements: H, C, N, O, F, Cl, V, Cu, Zn, Br, Zr.

Output: `<save_path>/<idx>.pt` and `<save_path>/index.csv`. These structures have no labels (see section 5).

## 2. Training

```bash
python train.py --target heat_co2 --encoder ViSNet --seed 42 --gpu 0
```

| Argument | Values | Default |
|---|---|---|
| `--target` | `heat_co2` (0.01–2.5 bar), `heat_ch4` (0.05–4.5 bar) | `heat_co2` |
| `--encoder` | `SchNet`, `PaiNN`, `ViSNet` | `ViSNet` |
| `--mode` | `base+gp`, `base+gp+ctx`, `base+gp+ctx+thermo`, `base+gp+ctx+thermo+mf` | `base+gp+ctx+thermo+mf` (full model) |
| `--seed` | any integer (paper: 42, 43, 44) | `42` |
| `--pool --attn` | add both flags to train the Global baseline (no pooling) | pooling on |
| `--data_path` | graph folder | `../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym` |
| `--output_path` | result folder | `../results_hmof` |
| `--wandb` | log to Weights & Biases (API key with `--wandb_key`) | off |

```bash
# full model, CO2 and CH4, three seeds
for target in heat_co2 heat_ch4; do
  for seed in 42 43 44; do
    python train.py --target $target --seed $seed
  done
done

python train.py --target heat_co2 --mode base+gp+ctx       # ablation step
python train.py --target heat_co2 --encoder PaiNN          # encoder comparison
python train.py --target heat_co2 --pool --attn            # Global baseline (no pooling)
```

The checkpoint is saved as
`../results_hmof/<encoder>/<target>_<encoder>_<lambda_aux>_<pool>_<attn>_<mode>_fin/best_<seed>.pt`
(full model: `heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin`; Global baseline: `..._False_False_...`),
and the test metrics of each run are appended to `result.csv` in the same folder.

## 3. Testing a trained model

```bash
python test.py --model_dir ../results_hmof/ViSNet/heat_co2_ViSNet_2.0_True_True_base+gp+ctx+thermo+mf_fin \
               --target heat_co2 --seed 42
```

`test.py` loads `<model_dir>/best_<seed>.pt` and evaluates it on the test set.
Pass the same model arguments as in training (`--encoder`, `--mode`, `--pool`, `--attn`, ...).

## 4. Analysis of the trained models

`analyze.py` rebuilds the checkpoint path as in `train.py` (or takes `--best_model_path`) and runs the steps given
in `--steps` (default: all four). The figure scripts expect seeds 42, 43 and 44 of the full ViSNet model.

```bash
for target in heat_co2 heat_ch4; do
  for seed in 42 43 44; do
    python analyze.py --target $target --seed $seed --steps test langmuir site
  done
  python analyze.py --target $target --seed 42 --steps symmetry
done
python aggregate_seeds.py          # mean ± std over seeds -> <run>/site_analysis_seeds/, langmuir_analysis_porep_seeds/
```

| Step | Folder in the run folder | Content |
|---|---|---|
| `test` | `result.csv` | test metrics |
| `langmuir` | `langmuir_analysis_porep_<seed>/` | per-MOF filling and mean-field parameters (`per_mof.csv`), Q<sub>st</sub> decomposition |
| `site` | `site_analysis_<seed>/` | sites, site types, type shares, thermodynamic parameters, compensation figure |
| `symmetry` | `symmetry_check/` | equivalent-atom consistency, rotation and origin-shift test |

Further analyses:

```bash
# parity plots with the Global baseline -> ../results_hmof/ViSNet/parity_seed<seed>/ (once per seed)
for seed in 42 43 44; do python parity.py --compare_global --seed $seed; done
python pressure_dependence.py      # Global vs Langmuir pressure trends, reads parity_seed*/

# 12-nearest-neighbour ties: needs graphs built with the plain 12-nearest-neighbour rule
python data/preprocess_hmof_consistent.py --tie_tol -1 --save_path ../dataset/hMOF/full_dataset_clean_void_inc_consistent
python tie_check.py --target heat_co2 --seed 42

# share of MOFs whose mean-field slope reaches its bound
python u_bound_check.py
```

## 5. UiO-66 case study

```bash
# graphs and pore descriptors of the five structures -> ../dataset/UiO-66/uio66
python preprocess_cif_consistent.py
# site analysis with the CH4 models -> <CH4 run>/site_analysis_uio66_<seed>/
for seed in 42 43 44; do
  python analyze.py --target heat_ch4 --seed $seed --pt_path ../dataset/UiO-66/uio66 --steps site
done
# GCMC with the hMOF protocol (RASPA2), CH4 density at every pressure -> ../results_hmof/gcmc_uio66
python gcmc/run_uio66_gcmc.py --cifs ../dataset/UiO-66/cif/*.cif --ff_dir gcmc/ff \
    --out_dir ../results_hmof/gcmc_uio66 --cutoff 12.8 --ch4_sigma 3.75 --tail_corrections no \
    --unit_cells 2 2 2 --init_cycles 5000 --cycles 50000 --density_pressures all
# hMOF-0 control: same settings, compared with its MOFX-DB labels
python gcmc/run_uio66_gcmc.py --hmof_cif gcmc/hMOF-0.cif --hmof_json gcmc/hMOF-0.json --ff_dir gcmc/ff \
    --out_dir ../results_hmof/hmof0_check --cutoff 12.8 --ch4_sigma 3.75 --tail_corrections no \
    --unit_cells 2 2 2 --init_cycles 5000 --cycles 50000
# model vs GCMC: figures and tables
python fig_uio66.py
```

The flags above are the settings of the paper; the script defaults are shorter test settings.

## 6. Figures and tables of the paper

The figure scripts read the analysis folders above and write to `../results_hmof/figures` by default.

| Figure / table | Script | Input |
|---|---|---|
| Fig. 2 | `parity.py` | checkpoints |
| Fig. 3 | `fig_saturation_curves.py` | `langmuir_analysis_porep_<seed>/` |
| Fig. 4, ESI Fig. S3, ESI Table S5 | `fig_pressure_dependence.py` (+ `u_bound_check.py` for the bound row of Table S5) | `site_analysis_seeds/`, `langmuir_analysis_porep_*` |
| Fig. 5, ESI Table S6 | `fig_site_types.py` | `site_analysis_<seed>/` |
| Fig. 6, Fig. 7, Table 5, ESI Figs. S5–S6, ESI Tables S7–S9 | `fig_uio66.py` | `site_analysis_uio66_<seed>/`, `gcmc_uio66/` |
| Tables 2–4, ESI Table S2 | `train.py` / `analyze.py --steps test` | `result.csv` of each run |
| ESI Fig. S1 | `fig_sites_per_mof.py` | `site_analysis_<seed>/` |
| ESI Fig. S2, ESI Table S3 | `analyze.py --steps symmetry`, `tie_check.py` | `symmetry_check/` |
| ESI Fig. S4 | `analyze.py --steps site` | checkpoints |
| ESI Table S4 | `pressure_dependence.py` | `parity_seed*/` |

Fig. 1 is a schematic.

## License

Code: see [LICENSE](LICENSE). The MOFX-DB data in `dataset/hMOF/` (split file names, `geo_mapping/consistent_hMOF.csv`)
remain under the terms of MOFX-DB.
