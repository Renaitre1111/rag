import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import sys
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utils import cVAE

def kmeans(X, num_clusters, num_iters=10):
    B, N, D = X.shape
    device = X.device
    rand_indices = torch.randint(0, N, (B, num_clusters), device=device)
    
    batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, num_clusters)

    centers = X[batch_indices, rand_indices, :] # [B, K, D]
    
    for _ in range(num_iters):
        dists = torch.sum((X.unsqueeze(2) - centers.unsqueeze(1))**2, dim=-1)

        labels = torch.argmin(dists, dim=2)

        # mask: [B, N, K]
        mask = torch.nn.functional.one_hot(labels, num_clusters).float()
        
        # sum_data: [B, K, D] -> mask^T * X
        # [B, K, N] x [B, N, D] -> [B, K, D]
        sum_data = torch.bmm(mask.transpose(1, 2), X)
        
        # counts: [B, K, 1]
        counts = mask.sum(dim=1).unsqueeze(2)
        counts = torch.clamp(counts, min=1.0)
        
        centers = sum_data / counts
        
    return centers

def load_database(database_path, prop_path, device):
    data = np.load(database_path)
    db_emb = data['embeddings']
    with open(prop_path, 'r') as f:
        lines = f.readlines()
    db_prop = np.array([float(line.strip()) for line in lines])

    if len(db_prop.shape) == 1:
        db_prop = db_prop[:, np.newaxis]

    return (torch.from_numpy(db_emb).float().to(device), 
            torch.from_numpy(db_prop).float().to(device))

def get_references(gen_centers, db_embeddings, db_props, exclude_idx=None):
    B, K, D = gen_centers.shape

    gen_centers_norm = torch.nn.functional.normalize(gen_centers, p=2, dim=2)
    db_norm = torch.nn.functional.normalize(db_embeddings, p=2, dim=1)

    similarities = torch.matmul(gen_centers_norm, db_norm.T)
    
    if exclude_idx is not None:
        batch_indices = torch.arange(B, device=gen_centers.device)

        for k in range(K):
            similarities[batch_indices, k, exclude_idx] = -1e9

    top1_indices = torch.argmax(similarities, dim=2)

    real_ref_props = db_props[top1_indices]
    
    return top1_indices, real_ref_props

def main(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    prop_mean = ckpt['prop_mean'].to(device)
    prop_std = ckpt['prop_std'].to(device)

    input_dim = 128

    model = cVAE(
        input_dim=input_dim,
        cond_dim=config.get('cond_dim', 128),
        latent_dim=config.get('latent_dim', 64)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    db_embeddings, db_props = load_database(args.database_path, args.database_prop_path, device)

    with open(args.target_path, 'r') as f:
        target_lines = f.readlines()
    target_props = torch.tensor([float(line.strip()) for line in target_lines]).float()

    dataset = TensorDataset(target_props)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_refs_indices = []
    all_ref_props = []
    
    global_indices = torch.arange(len(target_props)).to(device)
    start_ptr = 0

    for batch in tqdm(dataloader):
        targets_batch = batch[0].to(device) # [B]
        bs = len(targets_batch)
        
        target_norm = (targets_batch - prop_mean) / (prop_std + 1e-6)
        target_norm = target_norm.view(-1, 1) # [B, 1]

        props_expanded = target_norm.unsqueeze(1).repeat(1, args.num_samples, 1).view(-1, 1)
        z = torch.randn(bs * args.num_samples, 64).to(device)
        
        with torch.no_grad():
            gen_embeddings = model.decode(z, props_expanded)

        gen_embeddings = gen_embeddings.view(bs, args.num_samples, input_dim)
        
        centers = kmeans(gen_embeddings, args.num_clusters, num_iters=10)
        
        exclude_batch = None
        if args.remove_self:
            exclude_batch = global_indices[start_ptr : start_ptr + bs]
            
        ref_indices, real_ref_props = get_references(
            centers, db_embeddings, db_props, exclude_idx=exclude_batch
        )
        
        all_refs_indices.append(ref_indices.cpu().numpy())
        all_ref_props.append(real_ref_props.cpu().numpy())
        
        start_ptr += bs

    all_refs_indices = np.concatenate(all_refs_indices, axis=0)         # [N, K]
    all_ref_props = np.concatenate(all_ref_props, axis=0) # [N, K, 1]

    np.savez(args.output_path, indices=all_refs_indices, properties=all_ref_props)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='cvae_alchemy_gap.pth')
    
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--database_prop_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='train_gap_references.npz')
    
    parser.add_argument('--remove_self', action='store_true')

    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)

