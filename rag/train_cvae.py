import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import math
import argparse
from utils import cVAE
    
class MoleculeEmbeddingDataset(Dataset):
    def __init__(self, embedding_path, property_path):
        data = np.load(embedding_path)
        self.embeddings = torch.from_numpy(data['embeddings']).float()

        with open(property_path, 'r') as f:
            lines = f.readlines()
        self.properties = torch.tensor([float(line.strip()) for line in lines]).float().unsqueeze(1) # [N, 1]

        self.prop_mean = self.properties.mean()
        self.prop_std = self.properties.std()
        self.properties_norm = (self.properties - self.prop_mean) / (self.prop_std + 1e-6)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.properties_norm[idx]
    
def get_sampler(properties, num_bins=100):
    prop_np = properties.numpy().flatten()

    hist, bin_edges = np.histogram(prop_np, bins=num_bins)
    
    inds = np.digitize(prop_np, bin_edges[:-1]) - 1
    inds = np.clip(inds, 0, num_bins - 1)
    
    bin_weights = 1.0 / (hist + 1e-6)
    
    sample_weights = bin_weights[inds]
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(properties),
        replacement=True
    )
    return sampler

def loss_function(recon_x, x, mu, logvar, current_kl_weight):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = MSE + current_kl_weight * KLD
    return loss, MSE, KLD

def get_kl_weight(epoch, warmup_epochs, max_kl_weight):
    if epoch < warmup_epochs:
        return max_kl_weight * (epoch / warmup_epochs)
    else:
        return max_kl_weight
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MoleculeEmbeddingDataset(args.embedding_path, args.property_path)
    input_dim = dataset.embeddings.shape[1]

    sampler = get_sampler(dataset.properties)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = cVAE(input_dim=input_dim, cond_dim=args.cond_dim, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        current_beta = get_kl_weight(epoch, args.warmup_epochs, args.max_kl_weight)
        total_loss = 0
        total_mse = 0
        total_kld = 0

        for batch_idx, (emb, prop) in enumerate(dataloader):
            emb, prop = emb.to(device), prop.to(device)
            
            optimizer.zero_grad()
            
            recon_emb, mu, logvar = model(emb, prop)
            
            loss, mse, kld = loss_function(recon_emb, emb, mu, logvar, current_beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
        avg_loss = total_loss / len(dataloader.dataset)
        avg_mse = total_mse / len(dataloader.dataset)
        avg_kld = total_kld / len(dataloader.dataset)

        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch}/{args.epochs} | Beta: {current_beta:.5f} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KL: {avg_kld:.4f}")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'prop_mean': dataset.prop_mean,
        'prop_std': dataset.prop_std,
        'config': vars(args)
    }
    torch.save(save_dict, args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cVAE with Inverse Frequency Sampling & KL Annealing")
    
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--property_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='cvae_alchemy_gap.pth')
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--max_kl_weight', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--cond_dim', type=int, default=128)
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_interval', type=int, default=5)

    args = parser.parse_args()
    main(args)

