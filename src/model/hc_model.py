import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from utils.utils import get_act
from .encoder.periodic_schnet import PeriodicSchNet, SchNet_Readout, SchNet_Head
from .encoder.periodic_painn import PeriodicPaiNN
from .encoder.periodic_visnet import PeriodicViSNet
from .pooling.GDPool import Pool_Head, GFFN, GDPoolNet


def build_encoder(args):
    if args.encoder == "SchNet":
        return PeriodicSchNet(hidden_channels=args.d_model, num_filters=args.d_model,
                              num_interactions=6, num_gaussians=61, cutoff=6.0)
    if args.encoder == "PaiNN":
        return PeriodicPaiNN(hidden_channels=args.d_model, num_interactions=3, n_rbf=20, cutoff=6.0)
    if args.encoder == "ViSNet":
        return PeriodicViSNet(lmax=args.lmax, vecnorm_type='max_min', num_heads=args.n_head,
                              num_layers=args.num_layers, hidden_channels=args.d_model,
                              num_rbf=32, cutoff=6.0)
    raise ValueError(f"unknown encoder {args.encoder!r}")


class GMP_Readout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node, batch):
        output = global_mean_pool(node, batch)

        return output


class Base_Head(nn.Module):
    def __init__(self, d_model, mult=4, n_out=1, act='silu', drop_rate=0.1):
        super().__init__()
        self.dim = max(1, d_model // mult)
        self.fc = nn.Sequential(nn.Linear(d_model, self.dim),
                                get_act(act),
                                nn.Dropout(p=drop_rate),
                                nn.Linear(self.dim, n_out))

    def forward(self, x):
        output = self.fc(x)

        return output


class Base_Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.encoder = build_encoder(args)
        self.readout = GMP_Readout()
        self.head = Base_Head(args.d_model, mult=args.mult, n_out=args.n_out, act=args.act, drop_rate=args.drop_rate)

    def forward(self, data):
        node_representation = self.encoder(data)
        graph_representation = self.readout(node_representation, data.batch)
        output = self.head(graph_representation)

        return output


class Base_Model_Heat_Load(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.n_p = len(args.pressures)
        self.encoder = build_encoder(args)
        self.readout = GMP_Readout()
        self.head = Base_Head(args.d_model, mult=args.mult, n_out=2 * self.n_p, act=args.act, drop_rate=args.drop_rate)

    def forward(self, data):
        node_representation = self.encoder(data)
        graph_representation = self.readout(node_representation, data.batch)
        output = self.head(graph_representation)

        return output[:, :self.n_p], output[:, self.n_p:]


class Hierarchical_Clustering_Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.encoder = build_encoder(args)

        self.pool = GDPoolNet(args.d_model, num_layers=args.pool_num)
        self.head = Pool_Head(args.d_model, args.n_out)

    def forward(self, data):
        node_representation  = self.encoder(data)
        graph_representation, aux_loss, sa, adj, na, batch = self.pool(node_representation, data.edge_index, data.batch)
        output, m = self.head(graph_representation)

        return output, m, aux_loss, sa, adj, na, batch


class Geo_Encoder(nn.Module):
    def __init__(self, d_model, act):
        super().__init__()
        self.bn = nn.BatchNorm1d(5)
        self.fc = nn.Sequential(nn.Linear(5, d_model), get_act(act), nn.Linear(d_model, d_model))

    def forward(self, geo):
        g = torch.cat([geo[:, 0:1],
                       torch.log1p(geo[:, 1:3].clamp_min(0)),
                       geo[:, 3:5]], dim=1)

        return self.fc(self.bn(g))


class Pore_Token(nn.Module):
    def __init__(self, d_model, act):
        super().__init__()
        self.geo_enc = Geo_Encoder(d_model, act)
        self.type_emb = nn.Parameter(torch.zeros(d_model))

    def forward(self, z, m, geo):
        g = self.geo_enc(geo.to(z.dtype))
        t = g + self.type_emb

        return t.unsqueeze(1)


class Bounded_Tanh(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return self.c * torch.tanh(x / self.c)


class Langmuir_Sites(nn.Module):
    def __init__(self, d_model, pressures, mult=4, n_out=1, act='silu', drop_rate=0.0, log_K_init=0.0):
        super().__init__()
        hid = max(1, d_model // mult)
        self.register_buffer("log_p", torch.log(torch.tensor(pressures, dtype=torch.float)).view(1, 1, -1))
        self.n_head = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate), nn.Linear(hid, n_out))
        nn.init.zeros_(self.n_head[-1].weight); nn.init.zeros_(self.n_head[-1].bias)
        self.K_head = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate), nn.Linear(hid, n_out))
        nn.init.zeros_(self.K_head[-1].weight); nn.init.constant_(self.K_head[-1].bias, log_K_init)

    def forward(self, x):
        z, m = x
        mb = m.bool()

        log_n = self.n_head(z).squeeze(-1)
        log_K = self.K_head(z).squeeze(-1)

        x_p = log_K.unsqueeze(-1) + self.log_p
        theta = torch.sigmoid(x_p)
        log_w = log_n.unsqueeze(-1) + log_K.unsqueeze(-1) - 2.0 * F.softplus(x_p)
        log_w = log_w.masked_fill(~mb.unsqueeze(-1), float('-inf'))

        return log_n, log_K, theta, log_w, mb


R_J = 8.314462618                                # J/mol/K
R_KJ = R_J * 1e-3                                # kJ/mol/K
S_MAX = {'co2': 210.8, 'ch4': 143.5}             # J/mol/K, entropy the adsorbate can lose (TraPPE, 298 K, 1 bar)


class Langmuir_Sites_Thermo(nn.Module):
    def __init__(self, d_model, pressures, mult=4, n_out=1, act='silu', drop_rate=0.0, gas='co2', T=298.0):
        super().__init__()
        gas = str(gas).lower()
        assert gas in S_MAX, f"gas must be one of {list(S_MAX)}, got {gas!r}"
        self.gas = gas
        hid = max(1, d_model // mult)
        self.register_buffer("log_p", torch.log(torch.tensor(pressures, dtype=torch.float)).view(1, 1, -1))
        self.n_head = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate), nn.Linear(hid, n_out))
        nn.init.zeros_(self.n_head[-1].weight); nn.init.zeros_(self.n_head[-1].bias)
        self.K0_head = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate), nn.Linear(hid, n_out))
        nn.init.zeros_(self.K0_head[-1].weight); nn.init.zeros_(self.K0_head[-1].bias)
        self.RT = R_KJ * T                                                                        # kJ/mol
        self.lnK0_lo = -S_MAX[gas] / R_J

    def forward(self, x, e):
        z, m = x
        mb = m.bool()

        log_n = torch.log(F.softplus(self.n_head(z).squeeze(-1)) + 1e-8)
        ln_K0 = self.lnK0_lo * torch.sigmoid(self.K0_head(z).squeeze(-1))
        log_K = ln_K0 + e / self.RT

        x_p = log_K.unsqueeze(-1) + self.log_p
        theta = torch.sigmoid(x_p)
        log_w = log_n.unsqueeze(-1) + log_K.unsqueeze(-1) - 2.0 * F.softplus(x_p)
        log_w = log_w.masked_fill(~mb.unsqueeze(-1), float('-inf'))

        return log_n, log_K, theta, log_w, mb


class Pore_Heat_Change(nn.Module):
    def __init__(self, d_model, mult=4, n_out=2, act='silu', drop_rate=0.0, coef_max=80.0, g_max=20.0):
        super().__init__()
        hid = max(1, d_model // mult)
        self.mlp = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate),nn.Linear(hid, n_out))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.coef_fn = Bounded_Tanh(coef_max) if (coef_max is not None and coef_max > 0) else nn.Identity()
        self.g_fn = Bounded_Tanh(g_max) if (g_max is not None and g_max > 0) else nn.Identity()
        self.register_buffer("powers", torch.arange(1, n_out + 1, dtype=torch.float))

    def forward(self, h, theta):
        coef = self.coef_fn(self.mlp(h))
        theta_k = theta.unsqueeze(-1).pow(self.powers)
        g = self.g_fn((theta_k * coef.unsqueeze(1)).sum(-1))
        g = g - g[:, :1]

        return g, coef, theta_k


class Mean_Field_Interaction(nn.Module):
    def __init__(self, d_model, mult=4, act='silu', drop_rate=0.0, T=298.0, slope_frac=0.9):
        super().__init__()
        hid = max(1, d_model // mult)
        self.mlp = nn.Sequential(nn.Linear(d_model, hid), get_act(act), nn.Dropout(p=drop_rate), nn.Linear(hid, 2))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.RT = R_KJ * T                                                                   # kJ/mol
        self.slope_max = slope_frac * 4.0 * self.RT
        self.slope_fn = Bounded_Tanh(self.slope_max)
        self.register_buffer("powers", torch.arange(1, 3, dtype=torch.float))

    def coef(self, h):
        s = self.slope_fn(self.mlp(h))
        return torch.stack([s[:, 0], 0.5 * (s[:, 1] - s[:, 0])], dim=-1)

    @staticmethod
    def u(coef, t):
        return coef[:, :1] * t + coef[:, 1:2] * t * t

    @staticmethod
    def du(coef, t):
        return coef[:, :1] + 2.0 * coef[:, 1:2] * t

    def forward(self, coef, theta_bar):
        theta_k = theta_bar.unsqueeze(-1).pow(self.powers)
        return self.u(coef, theta_bar), coef, theta_k


class Langmuir_Sites_Thermo_MF(Langmuir_Sites_Thermo):
    def __init__(self, *args, n_iter=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = n_iter

    def _fill(self, field, s, coef, nw, t):
        th = torch.sigmoid(s + field.u(coef, t).unsqueeze(1) / self.RT)
        return th, (nw * th).sum(1)

    def forward(self, x, e, field, coef):
        z, m = x
        mb = m.bool()

        log_n = torch.log(F.softplus(self.n_head(z).squeeze(-1)) + 1e-8)
        ln_K0 = self.lnK0_lo * torch.sigmoid(self.K0_head(z).squeeze(-1))
        log_K = ln_K0 + e / self.RT
        s = log_K.unsqueeze(-1) + self.log_p
        n = torch.exp(log_n) * m.to(log_n.dtype)
        nw = (n / n.sum(1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1)

        with torch.no_grad():
            s_, c_, w_ = s.detach(), coef.detach(), nw.detach()
            lo = torch.zeros_like(s_[:, 0]); hi = torch.ones_like(lo)
            for _ in range(self.n_iter):
                mid = 0.5 * (lo + hi)
                up = self._fill(field, s_, c_, w_, mid)[1] > mid
                lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
            t_s = 0.5 * (lo + hi)
            th_s = self._fill(field, s_, c_, w_, t_s)[0]
            fp = (w_ * th_s * (1.0 - th_s)).sum(1) * field.du(c_, t_s) / self.RT

        F_s = self._fill(field, s, coef, nw, t_s)[1]
        t_bar = t_s + (F_s - t_s) / (1.0 - fp)
        x_p = s + field.u(coef, t_bar).unsqueeze(1) / self.RT
        theta = torch.sigmoid(x_p)
        log_w = log_n.unsqueeze(-1) + x_p - self.log_p - 2.0 * F.softplus(x_p)
        log_w = log_w.masked_fill(~mb.unsqueeze(-1), float('-inf'))

        return log_n, log_K, theta, log_w, mb


MODE_FLAGS = ('ctx', 'thermo', 'gp', 'mf')


class Hierarchical_Clustering_Model_With_LA(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        pressures = args.pressures
        self.eps = args.eps
        self.mode = args.mode
        flags = args.mode.split('+')
        if flags[0] != 'base' or not set(flags[1:]) <= set(MODE_FLAGS):
            raise ValueError(f"unknown mode {args.mode!r}: 'base' + any of {MODE_FLAGS}")
        self.ctx, self.thermo, self.gp, self.mf = (f in flags for f in ('ctx', 'thermo', 'gp', 'mf'))
        if self.mf and not self.thermo:
            raise ValueError("'mf' needs 'thermo'")

        self.encoder = build_encoder(args)
        self.pool = GDPoolNet(args.d_model, num_layers=args.pool_num, er=2, act=args.act)
        self.head = Pool_Head(args.d_model, mult=args.mult, n_out=1, act=args.act, drop_rate=args.drop_rate)
        self.pore_token = Pore_Token(args.d_model, args.act)

        site_kw = dict(mult=args.mult, n_out=1, act=args.act, drop_rate=args.drop_rate)
        if self.mf:
            self.sites = Langmuir_Sites_Thermo_MF(args.d_model, pressures, gas=args.gas, T=298.0, **site_kw)
        elif self.thermo:
            self.sites = Langmuir_Sites_Thermo(args.d_model, pressures, gas=args.gas, T=298.0, **site_kw)
        else:
            self.sites = Langmuir_Sites(args.d_model, pressures, **site_kw)

        if self.ctx:
            self.site_gffn = GFFN(args.d_model, er=2, act=args.act)

        if self.mf:
            self.pore_inc = Mean_Field_Interaction(args.d_model, mult=args.mult, act=args.act, drop_rate=args.drop_rate,
                                                   T=298.0, slope_frac=getattr(args, 'mf_slope', 0.9))
        else:
            self.pore_inc = Pore_Heat_Change(args.d_model, mult=args.mult, n_out=2, act=args.act, drop_rate=args.drop_rate)

        self.tok_norm = nn.LayerNorm(args.d_model)
        self.pore_norm = nn.LayerNorm(args.d_model)

    def forward(self, data):
        node_representation = self.encoder(data)
        (z, m), aux_loss, sa, adj, na, batch = self.pool(node_representation, data.edge_index, data.batch)
        pore = self.pore_token(z, m, data.geo)

        if self.ctx:
            z = self.site_gffn(z, self.site_context_adj(adj[-1].detach(), m), m)

        tok = torch.cat([z, pore], dim=1)
        tok = self.tok_norm(tok)
        pore_tok = tok[:,-1]

        m = torch.cat([m, torch.ones_like(m[:, :1])], dim=1)
        mf = m.to(tok.dtype)

        e = self.head((tok, m))[0]
        e = F.softplus(e)
        e = e.squeeze(-1)
        mof_total = pore_tok if self.gp else (tok * mf.unsqueeze(-1)).sum(1) / mf.sum(1, keepdim=True).clamp_min(1)

        if self.mf:
            coef = self.pore_inc.coef(mof_total.detach())
            log_n, log_K, theta, log_w, mb = self.sites((tok, m), e, self.pore_inc, coef)
        elif self.thermo:
            log_n, log_K, theta, log_w, mb = self.sites((tok, m), e)
        else:
            log_n, log_K, theta, log_w, mb = self.sites((tok, m))
        n = torch.exp(log_n) * mf

        N_model = (n.unsqueeze(-1) * theta).sum(1)
        theta_total = N_model / n.sum(1, keepdim=True).clamp_min(self.eps)

        if self.mf:
            g, coef, theta_k = self.pore_inc(coef, theta_total.detach())
        else:
            g, coef, theta_k = self.pore_inc(mof_total.detach(), theta_total.detach())
        e_pore_p = e[:, -1:].expand_as(g)
        e_tok = e.unsqueeze(-1) + g.unsqueeze(1)

        log_dNdp = torch.logsumexp(log_w, dim=1, keepdim=True)
        alpha = torch.exp(log_w - log_dNdp) * mf.unsqueeze(-1)
        output = (alpha * e_tok).sum(dim=1)
        return output, (e, coef, log_K[:, -1:], theta_total.unsqueeze(-1), e_pore_p, N_model), m, alpha, aux_loss, sa, adj, na, batch

    def site_context_adj(self, adj, m):
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        a = adj * (1 - eye)
        mf = m.to(adj.dtype)
        a = a * mf.unsqueeze(1) * mf.unsqueeze(2)
        return a / a.sum(-1, keepdim=True).clamp_min(self.eps)
