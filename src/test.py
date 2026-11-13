import os
import argparse


parser = argparse.ArgumentParser(description='Digital_Discovery')
parser.add_argument('--model_dir', required=True, type=str, help='folder that holds best_{seed}.pt')
parser.add_argument('--desc', default="base", type=str)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_key', default='WANDB_KEY', type=str)
parser.add_argument('--data_path', default='../dataset/hMOF/full_dataset_clean_void_inc_consistent_sym', type=str)
parser.add_argument('--target', default='heat_co2', type=str)
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
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


import torch
from utils.utils import seed_everything, PygDataset
from torch_geometric.loader import DataLoader
from model.hc_model import Base_Model, Hierarchical_Clustering_Model, Hierarchical_Clustering_Model_With_LA
from utils.test_utils import test_base_heat, test_pool, test_pool_attn, test_pool_attn_occ


def main():
    seed_everything(args.seed)
    best_model_path = os.path.join(args.model_dir, f'best_{args.seed}.pt')
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f'checkpoint not found: {best_model_path}')

    if args.wandb:
        import wandb
        wandb.login(key=args.wandb_key)
        wandb.init(project="Digital_Discovery_hMOF_v2", name=os.path.basename(os.path.normpath(args.model_dir)) + '_test',
                   config=args.__dict__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TARGETS = {
        'heat_co2': ['heat_co2_0.01', 'heat_co2_0.05', 'heat_co2_0.1', 'heat_co2_0.5', 'heat_co2_2.5'],
        'heat_ch4': ['heat_ch4_0.05', 'heat_ch4_0.5', 'heat_ch4_0.9', 'heat_ch4_2.5', 'heat_ch4_4.5'],
        'load_co2': ['load_co2_0.01', 'load_co2_0.05', 'load_co2_0.1', 'load_co2_0.5', 'load_co2_2.5'],
        'load_ch4': ['load_ch4_0.05', 'load_ch4_0.5', 'load_ch4_0.9', 'load_ch4_2.5', 'load_ch4_4.5']
        }
    PRESSURES = {
            'heat_co2': [0.01, 0.05, 0.1, 0.5, 2.5],
            'heat_ch4': [0.05, 0.5, 0.9, 2.5, 4.5],
            'load_co2': [0.01, 0.05, 0.1, 0.5, 2.5],
            'load_ch4': [0.05, 0.5, 0.9, 2.5, 4.5]
        }
    HEAT_STD = {'heat_co2': [6.518, 6.084, 6.014, 6.282, 6.678],          # kJ/mol
                'heat_ch4': [4.917, 4.804, 4.780, 4.733, 4.684]}
    LOAD_STD = {'heat_co2': [0.2098, 0.4950, 0.6960, 1.3995, 2.5644],     # mmol/g
                'heat_ch4': [0.1070, 0.4802, 0.6532, 1.0429, 1.3295]}
    REG_KEYS = ('rmse', 'r2', 'dev_mae', 'dev_r2', 'mean_mae', 'dev_std_true')
    target_name = TARGETS[args.target]
    args.pressures = PRESSURES[args.target]
    args.heat_std = HEAT_STD[args.target]
    args.load_std = LOAD_STD[args.target]
    args.n_out = len(target_name)
    args.gas = args.target.split('_')[-1]
    args.reg_keys = REG_KEYS

    test_dataset = PygDataset(root=f'{args.data_path}/test', target_name=target_name, gas=args.gas)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=0)

    if args.pool:
        if args.attn:
            model = Hierarchical_Clustering_Model_With_LA(args).to(device)
        else:
            model = Hierarchical_Clustering_Model(args).to(device)
    else:
        model = Base_Model(args).to(device)
    print(model)

    criterion = torch.nn.L1Loss()
    print(f'[test] checkpoint: {best_model_path}')

    if args.pool:
        if args.attn:
            if args.occ:
                test_pool_attn_occ(model, None, criterion, best_model_path, test_loader, device, True, args)
            else:
                test_pool_attn(model, None, criterion, best_model_path, test_loader, device, args)
        else:
            test_pool(model, None, criterion, best_model_path, test_loader, device, args)
    else:
        test_base_heat(model, None, criterion, best_model_path, test_loader, device, args)


if __name__=='__main__':
    main()
