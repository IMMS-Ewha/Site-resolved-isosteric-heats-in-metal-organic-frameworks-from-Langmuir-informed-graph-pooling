import os
import argparse
import traceback
from glob import glob


parser = argparse.ArgumentParser(description='Digital_Discovery analysis')
parser.add_argument('--desc', default="base", type=str)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--data_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym', type=str)
parser.add_argument('--output_path', default='../results_hmof', type=str)
parser.add_argument('--best_model_path', default='', type=str,
                    help='empty: <output_path>/<encoder>/<target>_<encoder>_<lambda_aux>_<pool>_<attn>_<mode>_fin/best_<seed>.pt as in train.py')
parser.add_argument('--pt_path', default='', type=str, help='folder of <idx>.pt graphs + index.csv (e.g. UiO-66 series): site and symmetry steps only')
parser.add_argument('--target', default='heat_co2', type=str, choices=['heat_co2', 'heat_ch4'])
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
parser.add_argument('--pore_ps', default='porep', type=str, choices=['poreps', 'pores', 'porep'])
parser.add_argument('--steps', nargs='*', default=['test', 'langmuir', 'site', 'symmetry'],
                    choices=['test', 'langmuir', 'site', 'symmetry'])
parser.add_argument('--viz', nargs='*', default=[], help='MOF names or loader indices for atom views (empty: auto, 2 MOFs)')
parser.add_argument('--viz_max_atoms', default=400, type=int)
parser.add_argument('--crop_radius', default=10.0, type=float)
parser.add_argument('--slab', default=2.0, type=float)
parser.add_argument('--hide_h', default=True, type=lambda s: str(s).lower() in ('1', 'true', 'yes'))
parser.add_argument('--dead_tol', default=0.05, type=float)
parser.add_argument('--tol', default=1.0, type=float)
parser.add_argument('--e_tol', default=0.5, type=float)
parser.add_argument('--symprec', default=1e-2, type=float)
parser.add_argument('--n_rot', default=3, type=int)
parser.add_argument('--max_mofs', default=0, type=int)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


import pandas as pd
import torch
from utils.utils import seed_everything, PygDataset
from torch_geometric.loader import DataLoader
from model.hc_model import Hierarchical_Clustering_Model_With_LA
from utils.langmuir_analysis import analyze_langmuir
from utils.test_utils import test_pool_attn_occ
from site_analysis import analyze_sites
from utils.symmetry_check import check_symmetry


TARGETS = {'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
           'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5']}
PRESSURES = {'heat_co2': [0.01, 0.05, 0.1, 0.5, 2.5],
             'heat_ch4': [0.05, 0.5, 0.9, 2.5, 4.5]}
HEAT_STD = {'heat_co2': [6.518, 6.084, 6.014, 6.282, 6.678],
            'heat_ch4': [4.917, 4.804, 4.780, 4.733, 4.684]}
LOAD_STD = {'heat_co2': [0.2098, 0.4950, 0.6960, 1.3995, 2.5644],
            'heat_ch4': [0.1070, 0.4802, 0.6532, 1.0429, 1.3295]}
REG_KEYS = ('rmse', 'r2', 'dev_mae', 'dev_r2', 'mean_mae', 'dev_std_true')
GEO_KEYS = ['void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd']


@torch.no_grad()
def diff_std(loader, device):
    y = torch.cat([d.y.view(d.num_graphs, -1) for d in loader])
    return (y[:, 1:] - y[:, :-1]).std(0).clamp_min(1e-3).to(device)


def pt_loader(pt_path, bs):
    files = glob(os.path.join(pt_path, '*.pt'))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    order = sorted(range(len(files)), key=lambda i: (0, int(stems[i]), '') if stems[i].isdigit() else (1, 0, stems[i]))
    idx_csv = os.path.join(pt_path, 'index.csv')
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
    return DataLoader(graphs, batch_size=bs, shuffle=False), names


def run_step(name, fn):
    print(f'\n{"=" * 20} {name} {"=" * 20}')
    try:
        fn()
        return 'done'
    except Exception:
        traceback.print_exc()
        return 'failed'


def main():
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_name = TARGETS[args.target]
    args.pressures = PRESSURES[args.target]
    args.heat_std = HEAT_STD[args.target]
    args.load_std = LOAD_STD[args.target]
    args.n_out = len(target_name)
    args.gas = args.target.split('_')[-1]
    args.reg_keys = REG_KEYS
    if not args.best_model_path:
        ver = f'{args.target}_{args.encoder}_{args.lambda_aux}_{args.pool}_{args.attn}_{args.mode}_fin'
        args.best_model_path = os.path.join(args.output_path, args.encoder, ver, f'best_{args.seed}.pt')
    if not os.path.isfile(args.best_model_path):
        raise FileNotFoundError(f'checkpoint not found: {args.best_model_path}')
    print(f'[analysis] checkpoint: {args.best_model_path}')

    model = Hierarchical_Clustering_Model_With_LA(args).to(device)
    print(model)
    print(f'[analysis] parameters: {sum(p.numel() for p in model.parameters()):,}')

    status = {}
    if args.pt_path:
        loader, names = pt_loader(args.pt_path, args.bs)
        tag = os.path.basename(os.path.normpath(args.pt_path))
        base = os.path.dirname(args.best_model_path)
        for s in [s for s in args.steps if s in ('test', 'langmuir')]:
            status[s] = 'skipped (no labels with --pt_path)'
        if 'site' in args.steps:
            status['site'] = run_step('site analysis', lambda: analyze_sites(
                model, args.best_model_path, loader, device, args, names=names, out_dir=os.path.join(base, f'site_analysis_{tag}_{args.seed}')))
        if 'symmetry' in args.steps:
            status['symmetry'] = run_step('symmetry check', lambda: check_symmetry(
                model, args.best_model_path, loader, device, args, names=names, out_dir=os.path.join(base, f'symmetry_check_{tag}')))
    else:
        test_dataset = PygDataset(root=f'{args.data_path}/test', target_name=target_name, gas=args.gas)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
        if 'test' in args.steps:
            train_dataset = PygDataset(root=f'{args.data_path}/train', target_name=target_name, gas=args.gas)
            args.dstd = diff_std(DataLoader(train_dataset, batch_size=args.bs, shuffle=False), device)
            status['test'] = run_step('test metrics', lambda: test_pool_attn_occ(
                model, None, torch.nn.L1Loss(), args.best_model_path, test_loader, device, False, args))
        if 'langmuir' in args.steps:
            status['langmuir'] = run_step('langmuir analysis', lambda: analyze_langmuir(
                model, args.best_model_path, test_loader, device, args, load_weights=True))
        if 'site' in args.steps:
            status['site'] = run_step('site analysis', lambda: analyze_sites(
                model, args.best_model_path, test_loader, device, args))
        if 'symmetry' in args.steps:
            status['symmetry'] = run_step('symmetry check', lambda: check_symmetry(
                model, args.best_model_path, test_loader, device, args))

    print('\n[analysis] summary')
    for k, v in status.items():
        print(f'  {k:10s} {v}')


if __name__ == '__main__':
    main()
