import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.utils import to_dense_batch


def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PygDataset(InMemoryDataset):
    GEO_KEYS = ('void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd')
    def __init__(self, root, target_name, gas):
        super().__init__(root=None)
        self.target_name = list(target_name)
        self.load_name = [i.replace('heat', 'load') for i in self.target_name]

        tag = gas
        cache_path = f"{root.rstrip('/')}_{tag}_cache_ori.pt"

        if not os.path.exists(cache_path):
            file_list = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt'))
            data_list = [self._process(torch.load(f, weights_only=False)) for f in file_list]
            torch.save(self.collate(data_list), cache_path)
            print(f'[PygDataset] cache saved: {cache_path} ({len(data_list)} graphs)')

        self.data, self.slices = torch.load(cache_path, weights_only=False)

    def geo_with_density(self, d):
        vol = torch.det(d.cell.reshape(3, 3).double()).abs()
        density = float(d.mass_cell) * 1.66054 / float(vol)
        return torch.tensor([[float(d[k]) for k in self.GEO_KEYS] + [density]], dtype=torch.float)

    def _process(self, data):
        data.y = torch.cat([data[k].view(1, -1) for k in self.target_name], dim=1).float()
        data.geo = torch.stack([data[k] for k in self.GEO_KEYS], dim=1).float()
        data.y_load = torch.cat([data[k].view(1, -1) for k in self.load_name], dim=1).float()
        return data


class PygDataset_Inference(Dataset):
    GEO_KEYS = ('void_fraction', 'surface_area_m2g', 'surface_area_m2cm3', 'pld', 'lcd')
    def __init__(self, root):
        super().__init__(root=None)
        self.files = sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt'))

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        data.geo = torch.stack([data[k] for k in self.GEO_KEYS], dim=1).float()
        return data


def compute_grad_similarity(grad1, grad2):
    if grad1 is None or grad2 is None:
        return 0.0
    g1 = grad1.view(grad1.size(0), -1)
    g2 = grad2.view(grad2.size(0), -1)
    graph_similarities = F.cosine_similarity(g1, g2, dim=1)
    return graph_similarities.mean().item()


def compute_grad_from_init(aux_loss, s, layer_num):
    loss_L_init = aux_loss[layer_num]
    loss_L_init.backward(retain_graph=True)
    grad_from_layer = s[0].grad.clone() if s[0].grad is not None else None
    if s[0].grad is not None:
        s[0].grad.zero_()
    return grad_from_layer


def soft_modularity_batch(A, S, eps=1e-10):
    m = A.sum(dim=(1, 2)) / 2
    m = torch.clamp(m, min=eps)
    d = A.sum(dim=2)
    expected = torch.einsum('bi,bj->bij', d, d) / (2 * m[:, None, None])
    B = A - expected
    BS = torch.bmm(B, S)
    STBS = torch.bmm(S.transpose(1, 2), BS)
    Q = STBS.diagonal(dim1=1, dim2=2).sum(dim=1)
    Q = Q / (2 * m)
    return Q.sum()


def effective_num_clusters_batch(S, N=None, B=None, eps=1e-10):
    if N==None:
        cluster_mass = S.sum(dim=1)
        u = cluster_mass / cluster_mass.sum(dim=-1, keepdim=True).clamp(min=eps)
    else:
        _, mask = to_dense_batch(N, B)
        mask = mask.unsqueeze(-1)
        S_masked = S * mask
        num_nodes = mask.sum(dim=1)
        u = S_masked.sum(dim=1) / num_nodes

    u_safe = torch.clamp(u, min=eps)
    H = -torch.sum(u * torch.log(u_safe), dim=1)
    K_eff = torch.exp(H)
    return K_eff.sum()


class PressureMetrics:
    def __init__(self, pressures, heat_std, load_std):
        self.p = list(pressures)
        self.heat_std = heat_std.detach().float().cpu()
        self.load_std = load_std.detach().float().cpu()
        self.reset()

    def reset(self):
        self.sum_q = torch.zeros(len(self.p))
        self.sum_n = torch.zeros(len(self.p))
        self.count = 0

    @torch.no_grad()
    def update(self, Q_pred, y_heat, N_model, y_load):
        self.sum_q += (Q_pred - y_heat).abs().sum(0).float().cpu()
        self.sum_n += (N_model - y_load).abs().sum(0).float().cpu()
        self.count += Q_pred.shape[0]

    def compute(self):
        n = max(self.count, 1)
        mae_q, mae_n = self.sum_q / n, self.sum_n / n
        return {'heat': mae_q, 'load': mae_n, 'heat_nmae': mae_q / self.heat_std, 'load_nmae': mae_n / self.load_std}

    def summary(self):
        return {k: v.mean().item() for k, v in self.compute().items()}

    def wandb_sections(self, gas):
        out = {}
        for name, v in self.compute().items():
            sec = f'{gas}_{name}'
            out[f'{sec}/avg'] = v.mean().item()
            for p, x in zip(self.p, v):
                out[f'{sec}/{p:g}bar'] = x.item()
        return out

    def line(self, key):
        return ' '.join(f'{p:g}:{x:.3f}' for p, x in zip(self.p, self.compute()[key]))


class HeatMetrics:
    def __init__(self, pressures, heat_std):
        self.pressures = list(pressures)
        self.heat_std = heat_std
        self.reset()

    def reset(self):
        self.abs_sum, self.n = None, 0

    @torch.no_grad()
    def update(self, pred, y):
        e = (pred.detach() - y).abs().sum(0)
        self.abs_sum = e if self.abs_sum is None else self.abs_sum + e
        self.n += y.size(0)

    def compute(self):
        mae = self.abs_sum / max(self.n, 1)
        return {'heat': mae, 'heat_nmae': mae / self.heat_std}

    def summary(self):
        return {k: v.mean().item() for k, v in self.compute().items()}

    def line(self, key):
        return ' '.join(f'{p:g}:{v:.3f}' for p, v in zip(self.pressures, self.compute()[key].tolist()))

    def wandb_sections(self, gas):
        row = {}
        for k, v in self.compute().items():
            row[f'{gas}_{k}/avg'] = v.mean().item()
            for p, x in zip(self.pressures, v.tolist()):
                row[f'{gas}_{k}/{p:g}bar'] = x
        return row


ACTIVATIONS = {
    'elu': nn.ELU,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2),
}


def get_act(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation '{name}', choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()
