import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, LayerNorm
from torch_geometric.nn.models.visnet import (
    ExpNormalSmearing, Sphere, VecLayerNorm,
    NeighborEmbedding, EdgeEmbedding, ViS_MP, ViS_MP_Vertex,
)


def edge_vectors(data, edge_index) -> Tensor:
    if getattr(data, "edge_vec", None) is not None:
        return data.edge_vec
    j, i = edge_index
    rij = data.pos[j] - data.pos[i]
    off = getattr(data, "offsets", None)
    if off is None:
        off = getattr(data, "cell_offsets", None)
    if off is not None:
        cell_e = data.cell[data.batch[i]]
        rij = rij + torch.einsum("ek,ekl->el", off.to(rij.dtype), cell_e)
    return rij


class PeriodicViSNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        lmax: int = 1,
        num_heads: int = 8,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        vecnorm_type: str = None,
        trainable_vecnorm: bool = False,
        cutoff: float = 6.0,
        max_z: int = 100,
        vertex: bool = False,
        add_self_loops: bool = True,
        return_vec: bool = False,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.add_self_loops = add_self_loops
        self.return_vec = return_vec
        self.eps = eps

        self.embedding = Embedding(max_z, hidden_channels)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        self.vis_mp_layers = nn.ModuleList(
            [vis_mp_class(last_layer=False, **vis_mp_kwargs) for _ in range(num_layers - 1)]
            + [vis_mp_class(last_layer=True, **vis_mp_kwargs)]
        )

        self.out_norm = LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(self, data):
        z = data.x.long()
        N = z.size(0)
        edge_index = data.edge_index
        vec = edge_vectors(data, edge_index).float()

        if self.add_self_loops:
            loop = torch.arange(N, device=z.device)
            edge_index = torch.cat([edge_index, torch.stack([loop, loop])], dim=1)
            vec = torch.cat([vec, vec.new_zeros(N, 3)], dim=0)

        edge_weight = vec.norm(dim=-1)
        mask = edge_index[0] != edge_index[1]
        unit = torch.zeros_like(vec)
        unit[mask] = vec[mask] / edge_weight[mask].clamp_min(self.eps).unsqueeze(-1)
        edge_sh = self.sphere(unit)
        edge_attr = self.distance_expansion(edge_weight)

        x = self.embedding(z)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec_feat = torch.zeros(N, (self.lmax + 1) ** 2 - 1, x.size(1), dtype=x.dtype, device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec_feat, edge_index, edge_weight, edge_attr, edge_sh)
            x = x + dx
            vec_feat = vec_feat + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec_feat, edge_index, edge_weight, edge_attr, edge_sh)
        x = x + dx
        vec_feat = vec_feat + dvec

        x = self.out_norm(x)
        vec_feat = self.vec_out_norm(vec_feat)

        if self.return_vec:
            return x, vec_feat
        return x
