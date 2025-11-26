import torch
import torch.nn as nn
import os
import math

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
