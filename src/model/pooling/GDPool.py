import torch
import torch.nn as nn
from utils.utils import get_act
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseSAGEConv, MLP


class Pool_Head(nn.Module):
    def __init__(self, d_model, mult=4, n_out=1, act='silu', drop_rate=0.1):
        super().__init__()
        self.dim = max(1, d_model // mult)
        self.fc = nn.Sequential(nn.Linear(d_model, self.dim),
                                get_act(act),
                                nn.Dropout(p=drop_rate),
                                nn.Linear(self.dim, n_out))

    def forward(self, x):
        out = self.fc(x[0])
        m = x[1].unsqueeze(-1).to(out.dtype)
        out = out * m

        return out, m


class Pool_Head_With_Norm(nn.Module):
    def __init__(self, d_model,  mult=4, n_out=1, act='silu', drop_rate=0.1):
        super().__init__()
        self.dim = max(1, d_model // mult)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, self.dim),
                                get_act(act),
                                nn.Dropout(p=drop_rate),
                                nn.Linear(self.dim, n_out))

    def forward(self, x):
        out = self.fc(self.norm(x[0]))
        m = x[1].unsqueeze(-1).to(out.dtype)
        out = out * m

        return out, m


class GDPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_dense, adj_dense, mask_nodes):
        B, N, D = x_dense.size()

        x_norm = self.norm(x_dense)
        x_norm = x_norm * mask_nodes.unsqueeze(-1).float()
        Q = x_norm
        K = x_norm
        V = x_dense

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        padding_mask = mask_nodes.unsqueeze(1) & mask_nodes.unsqueeze(2)
        scores = scores.masked_fill(~padding_mask, -1e9)

        attn_map = torch.softmax(scores, dim=-1)
        attn_map = attn_map * mask_nodes.unsqueeze(-1).float()

        node_importance = attn_map.sum(dim=-2)
        node_importance = node_importance.masked_fill(~mask_nodes, 0.0)

        threshold = torch.full((B, 1), 1, device=x_dense.device)
        anchor_mask = (node_importance > threshold) & mask_nodes

        max_indices = node_importance.argmax(dim=-1)
        batch_indices = torch.arange(B, device=x_dense.device)
        anchor_mask[batch_indices, max_indices] = True

        num_anchors = anchor_mask.sum(dim=-1)
        C_max = int(num_anchors.max().item())

        anchor_indices = torch.arange(N, device=x_dense.device).unsqueeze(0).expand(B, -1)
        anchor_indices = torch.where(anchor_mask, anchor_indices, torch.full_like(anchor_indices, N))
        anchor_indices = torch.sort(anchor_indices, dim=-1).values
        anchor_indices = anchor_indices[:, :C_max]
        anchor_indices = anchor_indices.clamp_max(N - 1)

        cluster_mask = torch.arange(C_max, device=x_dense.device).unsqueeze(0) < num_anchors.unsqueeze(1)
        gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, N)
        anchor_attn = torch.gather(scores, dim=1, index=gather_index)
        anchor_attn = anchor_attn.masked_fill(~cluster_mask.unsqueeze(-1), -1e9)
        anchor_attn = anchor_attn.transpose(1, 2)
        s = torch.softmax(anchor_attn, dim=-1)

        s = s * mask_nodes.unsqueeze(-1).float()

        x_pooled = torch.matmul(s.transpose(-2, -1), V)
        x_pooled = x_pooled * cluster_mask.unsqueeze(-1).float()
        adj_pooled = torch.matmul(s.transpose(-2, -1), torch.matmul(adj_dense, s))
        adj_pooled = adj_pooled * (cluster_mask.unsqueeze(1) & cluster_mask.unsqueeze(2)).float()

        return x_pooled, adj_pooled, cluster_mask, s, num_anchors


class GFFN(nn.Module):
    def __init__(self, d_model, er=2, act='silu', zero_init=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fcn1 = DenseSAGEConv(d_model, d_model * er)
        self.fcn2 = DenseSAGEConv(d_model * er, d_model)
        self.act = get_act(act)
        if zero_init:
            for p in self.fcn2.parameters():
                nn.init.zeros_(p)

    def forward(self, x, adj, m):
        mf = m.unsqueeze(-1).to(x.dtype)
        h = self.norm(x * mf)
        h = self.act(self.fcn1(h, adj, m))
        h = self.fcn2(h, adj, m) * mf
        return h


class GDPoolNet(nn.Module):
    def __init__(self, d_model, num_layers=3, er=2, act='silu'):
        super().__init__()
        self.gnn_emb_base = DenseSAGEConv(d_model, d_model)
        self.gnn_emb_layers = nn.ModuleList()
        self.gnn_pool_layers = nn.ModuleList()
        self.num_layers = max(1, num_layers)
        for _ in range(self.num_layers):
            self.gnn_emb_layers.append(GFFN(d_model, er, act))
            self.gnn_pool_layers.append(GDPooling(d_model))
        self.entropy_lambda = 0.05

    def forward(self, node, edge_index, batch):
        adjacency_matrix = []
        soft_assignment = []
        auxiliary_loss = []
        mask_list = []
        num_anchors = []

        n, mask_init = to_dense_batch(node, batch)
        adj_init = to_dense_adj(edge_index, batch)
        adjacency_matrix.append(adj_init)

        n = self.gnn_emb_base(n, adj_init, mask_init)

        z_init = self.gnn_emb_layers[0](n, adj_init, mask_init)
        z, adj, m, s_init, n = self.gnn_pool_layers[0](z_init, adj_init, mask_init)
        adjacency_matrix.append(adj)
        soft_assignment.append(s_init)
        mask_list.append(m)
        num_anchors.append(n)

        for gnn_emb, gnn_pool in zip(self.gnn_emb_layers[1:], self.gnn_pool_layers[1:]):
            z = gnn_emb(z, adj, m)
            z, adj, m, s, n = gnn_pool(z, adj, m)
            adjacency_matrix.append(adj)
            soft_assignment.append(s)
            mask_list.append(m)
            num_anchors.append(n)

        mask_bool = mask_init.bool().squeeze(-1)
        pair_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
        pair_mask = pair_mask.float()
        num_pairs = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
        num_nodes = mask_bool.sum(dim=1).clamp_min(1)

        now_dtype = z_init.dtype
        s_mask = (mask_bool.unsqueeze(-1) & mask_list[0].unsqueeze(1)).to(now_dtype)
        s_all = s_init * s_mask

        link = adj_init - torch.matmul(s_all, s_all.transpose(1, 2))
        link = (link ** 2) * pair_mask
        link_loss_per_graph = torch.sqrt(link.sum(dim=(1, 2)) / num_pairs + 1e-8)
        link_loss = link_loss_per_graph.mean()

        entropy_per_node = (-s_all * torch.log(s_all + 1e-8)).sum(dim=-1)
        entropy_per_graph = ((entropy_per_node * mask_bool.to(now_dtype)).sum(dim=1)) / num_nodes
        num_clusters = mask_list[0].sum(dim=1)
        entropy_per_graph = torch.where(num_clusters > 1, entropy_per_graph, torch.zeros_like(entropy_per_graph))
        entropy_loss = entropy_per_graph.mean()

        auxiliary_loss.append(link_loss + entropy_loss * self.entropy_lambda)

        for now_s, now_m in zip(soft_assignment[1:], mask_list[1:]):
            s_all = torch.bmm(s_all, now_s)
            s_mask = (mask_bool.unsqueeze(-1) & now_m.unsqueeze(1)).to(z_init.dtype)
            s_all = s_all * s_mask

            link = adj_init - torch.matmul(s_all, s_all.transpose(1, 2))
            link = (link ** 2) * pair_mask
            link_loss_per_graph = torch.sqrt(link.sum(dim=(1, 2)) / num_pairs + 1e-15)
            link_loss = link_loss_per_graph.mean()

            entropy_per_node = (-s_all * torch.log(s_all + 1e-15)).sum(dim=-1)
            entropy_per_graph = ((entropy_per_node * mask_bool.to(now_dtype)).sum(dim=1)) / num_nodes
            num_clusters = now_m.sum(dim=1)
            entropy_per_graph = torch.where(num_clusters > 1, entropy_per_graph, torch.zeros_like(entropy_per_graph))
            entropy_loss = entropy_per_graph.mean()

            auxiliary_loss.append(link_loss + entropy_loss * self.entropy_lambda)

        return (z, m), auxiliary_loss, soft_assignment, adjacency_matrix, num_anchors, batch
