import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
from retriever import MoleculeRetriever
from torch.distributions.categorical import Categorical

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class FinetuneRetriever(MoleculeRetriever):
    def generate_finetune_data(self, prop_path, save_path, num_samples=20000, k_pool=100, k_fine=10, batch_size=1024):
        with open(prop_path, 'r') as f:
            raw_values = [float(line.strip().split()[0]) for line in f if line.strip()]

        dist = PropertyDistribution(raw_values, num_bins=1000, device=self.device)

        sampled_targets = dist.sample(num_samples) # [N, 1]

        final_indices = np.zeros((num_samples, k_fine), dtype=np.int32)
        final_scores = np.zeros((num_samples, k_fine), dtype=np.float32)

        for i in tqdm(range(0, num_samples, batch_size), desc="Pre-calculating"):
            end = min(i + batch_size, num_samples)
            batch_queries = sampled_targets[i:end] # [B, 1]

            cand_indices = self._coarse_filter(batch_queries, k_pool=k_pool) 
            
            cand_embs = self.db_embeddings[cand_indices] # [B, k_pool, Dim]

            cand_embs_norm = F.normalize(cand_embs, p=2, dim=2)
            sim_mats = torch.bmm(cand_embs_norm, cand_embs_norm.transpose(1, 2)) # [B, k_pool, k_pool]
            centrality = sim_mats.sum(dim=2) # [B, k_pool]

            top_k_scores, top_k_local_indices = torch.topk(centrality, k=k_fine, dim=1)
            top_k_global_indices = torch.gather(cand_indices, 1, top_k_local_indices)

            final_indices[i:end] = top_k_global_indices.cpu().numpy()
            final_scores[i:end] = top_k_scores.cpu().numpy()

        targets_np = sampled_targets.cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(
            save_path, 
            targets=targets_np,     # [N, 1] 
            indices=final_indices,  # [N, k]
            scores=final_scores
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_emb_path", type=str, required=True, help="Path to DB embeddings .npz")
    parser.add_argument("--db_prop_path", type=str, required=True, help="Path to DB properties .txt")

    parser.add_argument("--prop_path", type=str, required=True, help="Path to source properties .txt")
    parser.add_argument("--save_path", type=str, default='data/finetune_gap.npz')
    parser.add_argument("--num_samples", type=int, default=20000, help="Total prompts to generate")
    parser.add_argument("--k_pool", type=int, default=100)
    parser.add_argument("--k_fine", type=int, default=10, help="Number of references per prompt (usually use index 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    retriever = FinetuneRetriever(args.db_emb_path, args.db_prop_path, device=device)
    
    retriever.generate_finetune_data(
        prop_path=args.prop_path, 
        save_path=args.save_path,
        num_samples=args.num_samples,
        k_pool=args.k_pool,
        k_fine=args.k_fine
    )