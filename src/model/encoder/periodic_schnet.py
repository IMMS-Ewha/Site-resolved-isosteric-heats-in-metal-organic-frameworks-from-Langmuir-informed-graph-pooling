import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
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


class PeriodicSchNet(SchNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.lin1, self.act, self.lin2, self.readout

    def forward(self, data) -> torch.Tensor:
        edge_index = data.edge_index
        edge_weight = edge_vectors(data, edge_index).float().norm(dim=-1)

        h = self.embedding(data.x.long())
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h


class SchNet_Readout(nn.Module):
    def __init__(self, hidden_channels, n_out, readout='mean'):
        super().__init__()
        self.readout = aggr_resolver(readout)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, n_out)
        self.reset_parameters()

    def forward(self, x, batch):
        x = self.lin2(self.act(self.lin1(x)))
        return self.readout(x, batch, dim=0)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)


class SchNet_Head(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
