import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Linear
from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver


def edge_vectors(data, edge_index) -> torch.Tensor:
    if getattr(data, "edge_vec", None) is not None:
        return data.edge_vec
    j, i = edge_index
    rij = data.pos[j] - data.pos[i]
    if getattr(data, "cell_offsets", None) is not None:
        cell_e = data.cell[data.batch[i]]
        rij = rij + torch.einsum("ek,ekl->el", data.cell_offsets.to(rij.dtype), cell_e)

    return rij


class SinusoidalRBF(nn.Module):
    def __init__(self, n_rbf, cutoff, r_min=0.5):
        super().__init__()
        self.cutoff = cutoff
        self.r_min = r_min
        self.register_buffer("n", torch.arange(1, n_rbf + 1, dtype=torch.float))

    def forward(self, r):
        r = r.clamp_min(self.r_min).unsqueeze(-1)
        return torch.sin(self.n * math.pi * r / self.cutoff) / r


def cosine_cutoff(r, cutoff):
    return 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0) * (r < cutoff).to(r.dtype)


class PaiNNMessage(nn.Module):
    def __init__(self, F, n_rbf):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F))
        self.W = nn.Linear(n_rbf, 3 * F)
        self.F = F

    def forward(self, s, v, edge_index, rbf, fcut, unit):
        j, i = edge_index
        x = self.phi(s)[j] * self.W(rbf) * fcut.unsqueeze(-1)
        ds, dvv, dvs = torch.split(x, self.F, dim=-1)
        dv = v[j] * dvv.unsqueeze(1) + dvs.unsqueeze(1) * unit.unsqueeze(-1)
        N = s.size(0)
        s = s + scatter(ds, i, dim=0, dim_size=N, reduce="sum")
        v = v + scatter(dv, i, dim=0, dim_size=N, reduce="sum")

        return s, v


class PaiNNUpdate(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.U = nn.Linear(F, F, bias=False)
        self.V = nn.Linear(F, F, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F))
        self.F = F

    def forward(self, s, v):
        Uv, Vv = self.U(v), self.V(v)
        Vv_norm = torch.sqrt((Vv ** 2).sum(dim=1) + 1e-8)
        a = self.mlp(torch.cat([s, Vv_norm], dim=-1))
        avv, asv, ass = torch.split(a, self.F, dim=-1)
        v = v + avv.unsqueeze(1) * Uv
        s = s + asv * (Uv * Vv).sum(dim=1) + ass

        return s, v


class PeriodicPaiNN(nn.Module):
    def __init__(self, hidden_channels, num_interactions=3, n_rbf=20, cutoff=6.0, max_z=100, r_min=0.5):
        super().__init__()
        self.F = hidden_channels
        self.cutoff = cutoff
        self.r_min = r_min
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.rbf = SinusoidalRBF(n_rbf, cutoff, r_min)
        self.messages = nn.ModuleList([PaiNNMessage(hidden_channels, n_rbf) for _ in range(num_interactions)])
        self.updates = nn.ModuleList([PaiNNUpdate(hidden_channels) for _ in range(num_interactions)])

    def forward(self, data) -> torch.Tensor:
        edge_index = data.edge_index
        vec = edge_vectors(data, edge_index).float()
        r = vec.norm(dim=-1)
        unit = vec / r.clamp_min(self.r_min).unsqueeze(-1)
        rbf = self.rbf(r)
        fcut = cosine_cutoff(r, self.cutoff)

        s = self.embedding(data.x.long())
        v = torch.zeros(s.size(0), 3, self.F, device=s.device, dtype=s.dtype)
        for msg, upd in zip(self.messages, self.updates):
            s, v = msg(s, v, edge_index, rbf, fcut, unit)
            s, v = upd(s, v)

        return s
