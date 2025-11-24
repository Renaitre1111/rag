import torch
from torch import nn
from .models.gcl import unsorted_segment_sum


class ShiftedSoftplus(nn.Module):
    """
    Activation function.
    """
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0))

    def forward(self, x):
        return nn.functional.softplus(x) - self.shift


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Functions for expanding interatomic distances.
    """
    def __init__(self, n_gaussians=64, cutoff=10.0, start=0.0):
        super(GaussianRBF, self).__init__()
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff
        offsets = torch.linspace(start, cutoff, n_gaussians)
        self.widths = nn.Parameter(torch.abs(offsets[1] - offsets[0]) * torch.ones_like(offsets), requires_grad=False)
        self.centers = nn.Parameter(offsets, requires_grad=False)

    def forward(self, distances):
        """
        Args:
            distances: (..., 1) Tensor, interatomic distances.
        Returns:
            (..., n_gaussians) Tensor, expanded distance features.
        """
        distances = distances.clamp_max(self.cutoff) 
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = distances - self.centers
        exponent = coeff * torch.pow(diff, 2)
        return torch.exp(exponent)


class CFConv(nn.Module):
    """
    Core of SchNet: Continuous-Filter Convolution (CFConv).
    """
    def __init__(self, in_channels, out_channels, n_filters, n_gaussians, cutoff):
        super(CFConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.cutoff = cutoff
        
        self.rbf = GaussianRBF(n_gaussians, cutoff)
        
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters)
        )
        
        self.in_mlp = nn.Linear(in_channels, n_filters, bias=False)
        self.out_mlp = nn.Linear(n_filters, out_channels)
        self.act = ShiftedSoftplus()

    def forward(self, h, coord, edge_index, edge_mask):
        row, col = edge_index
        
        coord_diff = coord[row] - coord[col]
        distances = torch.norm(coord_diff, dim=-1, keepdim=True)
        
        dist_features = self.rbf(distances)
        
        W = self.filter_net(dist_features)
        
        h_j = self.in_mlp(h[col])
        
        msg = h_j * W
        
        msg = msg * edge_mask
        
        agg = unsorted_segment_sum(msg, row, num_segments=h.size(0))
        
        v = self.out_mlp(agg)
        return v


class SchNetInteraction(nn.Module):
    """
    SchNet Interaction Block (CFConv + Residual).
    """
    def __init__(self, n_channels, n_filters, n_gaussians, cutoff):
        super(SchNetInteraction, self).__init__()
        self.cfconv = CFConv(n_channels, n_channels, n_filters, n_gaussians, cutoff)
        self.act = ShiftedSoftplus()

    def forward(self, h, coord, edge_index, edge_mask):
        v = self.cfconv(h, coord, edge_index, edge_mask)
        h = h + v
        return h


class SchNet(nn.Module):
    """
    SchNet Property Prediction Model.
    """
    def __init__(self, in_node_nf=5, hidden_nf=128, n_interactions=3, n_gaussians=64, cutoff=10.0, device='cpu', act_fn=nn.SiLU()):
        super(SchNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_interactions = n_interactions
        self.n_filters = hidden_nf
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff

        self.embedding = nn.Linear(in_node_nf, hidden_nf) 

        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_nf, self.n_filters, n_gaussians, cutoff)
            for _ in range(n_interactions)
        ])

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf)
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, 1)
        )
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        """
        h0: (B*N, in_node_nf)
        x: (B*N, 3)
        edges: (2, n_edges)
        edge_attr: None
        node_mask: (B*N, 1)
        edge_mask: (n_edges, 1)
        n_nodes: int
        """
        
        h = self.embedding(h0)

        for i in range(self.n_interactions):
            h = self.interactions[i](h, x, edges, edge_mask)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)