import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

def get_param(shape):
    """Initialize a trainable parameter tensor with Xavier uniform."""
    param = Parameter(torch.empty(shape, dtype=torch.float))
    nn.init.xavier_uniform_(param.data)
    return param


def ccorr(a, b):
    """
    Circular correlation of two vectors (used in holographic embeddings).
    Implemented via FFT.
    """
    fft_a = torch.fft.rfft(a, dim=-1)
    fft_b = torch.fft.rfft(b, dim=-1)
    conj_b = torch.conj(fft_b)
    ccorr_res = torch.fft.irfft(fft_a * conj_b, n=a.size(-1), dim=-1)
    return ccorr_res

class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None):
        super().__init__(aggr="add")

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = nn.Dropout(self.p.dropout if self.p is not None and hasattr(self.p, "dropout") else 0.0)
        self.bn = nn.BatchNorm1d(out_channels)

        if self.p is not None and getattr(self.p, "bias", False):
            self.register_parameter("bias", Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode="in")
        loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode="loop")
        out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode="out")

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p is not None and getattr(self.p, "bias", False):
            out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # ignore self-loop relation

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == "corr":
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == "sub":
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == "mult":
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError(f"Unknown operator: {self.p.opn}")
        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, f"w_{mode}")
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float("inf")] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, num_rels={self.num_rels})"
