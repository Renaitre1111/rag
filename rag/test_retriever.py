import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from retriever import MoleculeRetriever

class InferenceRetriever(MoleculeRetriever):
    def process_queries(self, query_prop_path, save_path, k_pool=100, k_fine=10, batch_size=1024):
        with open(query_prop_path, 'r') as f:
            queries = [float(line.strip().split()[0]) for line in f if line.strip()]

        query_props = torch.tensor(queries, dtype=torch.float32).view(-1, 1).to(self.device)
        num_queries = len(query_props)

        final_indices = np.zeros((num_queries, k_fine), dtype=np.int32)
        final_scores = np.zeros((num_queries, k_fine), dtype=np.float32)

        for i in tqdm(range(0, num_queries, batch_size), desc="Retrieving for Test"):
            end = min(i + batch_size, num_queries)
            batch_queries = query_props[i:end]

            cand_indices = self._coarse_filter(batch_queries, k_pool=k_pool) # [B, k_pool]
            cand_embs = self.db_embeddings[cand_indices] # [B, k_pool, Dim]

            cand_embs_norm = F.normalize(cand_embs, p=2, dim=2)
            sim_mats = torch.bmm(cand_embs_norm, cand_embs_norm.transpose(1, 2))
            centrality = sim_mats.sum(dim=2) # [B, k_pool]

            top_k_scores, top_k_local_indices = torch.topk(centrality, k=k_fine, dim=1)
            top_k_global_indices = torch.gather(cand_indices, 1, top_k_local_indices)

            final_indices[i:end] = top_k_global_indices.cpu().numpy()
            final_scores[i:end] = top_k_scores.cpu().numpy()
        np.savez(save_path, indices=final_indices, sims=final_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run retrieval for external test queries.")
    parser.add_argument("--db_emb_path", type=str, required=True, help="Path to Train DB embeddings .npz")
    parser.add_argument("--db_prop_path", type=str, required=True, help="Path to Train DB properties .txt")
    
    parser.add_argument("--query_prop_path", type=str, required=True, help="Path to target properties (.txt) for generation")
    parser.add_argument("--save_path", type=str, default='cond/test_ret_gap.npz', help="Output path for the indices")

    parser.add_argument("--k_pool", type=int, default=100, help="Pool size for property filtering")
    parser.add_argument("--k_fine", type=int, default=10, help="Number of references to save")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    retriever = InferenceRetriever(args.db_emb_path, args.db_prop_path, device=device)
    
    retriever.process_queries(
        query_prop_path=args.query_prop_path, 
        save_path=args.save_path, 
        k_pool=args.k_pool, 
        k_fine=args.k_fine
    )