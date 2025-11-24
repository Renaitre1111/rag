import torch
import torch.nn as nn
import numpy as np
import argparse
import math
from sklearn.cluster import KMeans
import sys
import os

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
            nn.Linear(input_dim + cond_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.SiLU(),
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

    def inference(self, z, prop):
        c = self.prop_encoder(prop)
        decoder_input = torch.cat([z, c], dim=1)
        recon_x = self.decoder(decoder_input)
        return recon_x
    
def generate_reference(model, target_prop, prop_mean, prop_std, 
                         num_samples=1000, num_clusters=5, device='cpu'):
    model.eval()

    target_norm = (target_prop - prop_mean) / prop_std
    prop_tensor = torch.tensor([[target_norm]]).repeat(num_samples, 1).float().to(device)

    z = torch.randn(num_samples, 64).to(device)
    with torch.no_grad():
        gen_embeddings = model.inference(z, prop_tensor) # [P, Input_Dim]
    
    gen_np = gen_embeddings.cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(gen_np)

    cluster_centers = kmeans.cluster_centers_ 
    
    return cluster_centers, gen_np

def main(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    prop_mean = ckpt['prop_mean']
    prop_std = ckpt['prop_std']

    input_dim = 128
    model = cVAE(
        input_dim=input_dim,
        cond_dim=config.get('cond_dim', 128),
        latent_dim=config.get('latent_dim', 64)
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    centers, all_samples = generate_reference(
        model, 
        target_prop_val=args.target_property,
        prop_mean=prop_mean,
        prop_std=prop_std,
        num_samples=args.num_samples,
        num_clusters=args.num_clusters,
        device=device
    )
    save_file = f"references_gap_{args.target_property}.npy"
    np.save(save_file, centers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='cvae_alchemy_gap.pth')
    parser.add_argument('--target_property', type=float, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    main(args)
    
