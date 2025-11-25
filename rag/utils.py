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
    
class cVAE(nn.Module):
    def __init__(self, input_dim=128, cond_dim=128, latent_dim=64):
        super().__init__()
        
        self.prop_encoder = PropertyEncoder(output_dim=cond_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, prop):
        c = self.prop_encoder(prop)

        encoder_input = torch.cat([x, c], dim=1)
        hidden = self.encoder(encoder_input)
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)

        z = self.reparameterize(mu, logvar)
        
        decoder_input = torch.cat([z, c], dim=1)
        recon_x = self.decoder(decoder_input)

        return recon_x, mu, logvar
    
    def decode(self, z, prop):
        c = self.prop_encoder(prop)
        decoder_input = torch.cat([z, c], dim=1)
        recon_x = self.decoder(decoder_input)
        return recon_x