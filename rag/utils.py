import torch
import torch.nn as nn
import os
import math
from torch.distributions.categorical import Categorical

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class PropertyEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, output_dim), nn.SiLU()
        )
    def forward(self, x):
        feat = self.sinusoidal(x)
        return self.mlp(feat)

class PropertyDistribution:
    def __init__(self, values, num_bins=1000, device='cpu'):
        self.num_bins = num_bins
        self.device = device
        values = torch.tensor([v for v in values if not (np.isnan(v) or np.isinf(v))], dtype=torch.float)
        self.probs, self.params = self._create_prob_dist(values)
        self.m = Categorical(self.probs)
        self.probs = self.probs.to(device)

    def _create_prob_dist(self, values):
        n_bins = self.num_bins
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12

        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            if i == n_bins: i = n_bins - 1
            histogram[i] += 1

        probs = histogram / torch.sum(histogram)
        params = [prop_min, prop_max]
        return probs, params
    
    def _idx2value(self, idx, params, n_bins):
        prop_min = params[0]
        prop_max = params[1]
        prop_range = prop_max - prop_min
        
        left = idx.float() / n_bins * prop_range + prop_min
        right = (idx + 1).float() / n_bins * prop_range + prop_min

        val = torch.rand(idx.shape, device=self.device) * (right - left) + left
        return val
    
    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,)).to(self.device)
        vals = self._idx2value(idx, self.params, self.num_bins)
        return vals.view(-1, 1) # [N, 1]
