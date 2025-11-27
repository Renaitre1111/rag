import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm

class MoleculeRetriever:
    def __init__(self, embedding_path, property_path, device='cuda'):
        self.device = device
        emb_data = np.load(embedding_path)
        self.db_embeddings = torch.from_numpy(emb_data['embeddings']).to(device) # [N, Dim]

        with open(property_path, 'r') as f:
            props = [float(line.strip().split()[0]) for line in f if line.strip()]
        self.db_properties = torch.tensor(props, dtype=torch.float32).view(-1, 1).to(device) # [N, 1]

        self.num_samples = len(self.db_embeddings)

    def _coarse_filter(self, target_val, k_pool=100, exclude_indices=None):
        # target_val [B, 1] or [1]
        if target_val.dim() == 0:
            target_val = target_val.view(1, 1)
        diff = torch.abs(self.db_properties.T - target_val) # [1, N] - [B, 1] = [B, N]

        if exclude_indices is not None:
            # exclude_indices: [B]
            batch_size = target_val.size(0)
            rows = torch.arange(batch_size, device=self.device)
            diff[rows, exclude_indices] = float('inf')

        topk_res = torch.topk(diff, k=k_pool, largest=False, dim=1)
        return topk_res.indices
    
    def preprocess(self, save_path, k_pool=100, k_fine=10, batch_size=1024):
        final_indices = np.zeros((self.num_samples, k_fine), dtype=np.int32)
        final_sims = np.zeros((self.num_samples, k_fine), dtype=np.float32)

        for i in tqdm(range(0, self.num_samples, batch_size), desc="preprocess"):
            end = min(i + batch_size, self.num_samples)
            batch_gt_embs = self.db_embeddings[i:end]   # [B, Dim]
            batch_gt_props = self.db_properties[i:end]   # [B, 1]

            exclude_idxs = torch.arange(i, end, device=self.device)

            cand_indices = self._coarse_filter(batch_gt_props, k_pool=k_pool, exclude_indices=exclude_idxs) # [B, k_pool]

            cand_embs = self.db_embeddings[cand_indices] # [B, k_pool, Dim]

            sims = F.cosine_similarity(batch_gt_embs.unsqueeze(1), cand_embs, dim=2) # [B, k_pool]

            batch_sims, top_k_indices = torch.topk(sims, k=k_fine, dim=1) # [B, k_fine]
            batch_final_indices = torch.gather(cand_indices, 1, top_k_indices) # [B, k_fine]

            final_indices[i:end] = batch_final_indices.cpu().numpy()
            final_sims[i:end] = batch_sims.cpu().numpy()
        
        np.savez(save_path, indices=final_indices, sims=final_sims)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run retrieval preprocessing.")
    parser.add_argument("--db_emb_path", type=str, default='data/alchemy_gap_embeddings.npz', help="Path to the embeddings .npz file")
    parser.add_argument("--db_prop_path", type=str, default='data/gap.txt', help="Path to the property .txt file")
    parser.add_argument("--save_path", type=str, default='data/train_ret_gap.npz', help="Output path for the indices")
    parser.add_argument("--k_pool", type=int, default=100, help="Step 1: Coarse retrieval pool size")
    parser.add_argument("--k_fine", type=int, default=10, help="Step 2: Top-K structural matches to save")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    retriever = MoleculeRetriever(args.db_emb_path, args.db_prop_path, device=device)
    retriever.preprocess(args.save_path, k_pool=args.k_pool, k_fine=args.k_fine)