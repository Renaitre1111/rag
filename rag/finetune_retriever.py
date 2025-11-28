import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
from retriever import MoleculeRetriever
from rag.utils import PropertyDistribution

class FinetuneRetriever(MoleculeRetriever):
    def gen_finetune_data(self, prop_path, save_path, num_samples=20000, k_pool=100, k_fine=10, batch_size=1024):
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
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    retriever = FinetuneRetriever(args.db_emb_path, args.db_prop_path, device=device)
    
    retriever.generate_finetune_data(
        prop_path=args.prop_path, 
        save_path=args.save_path,
        num_samples=args.num_samples,
        k_pool=args.k_pool,
        k_fine=args.k_fine
    )