import os
import random
import math
import copy
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dataset import SimpleTokenizer
from model.poetic import PoeticMamba
from model.mamba import MambaConfig
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
from qm9.property_prediction.prop_utils import get_adj_matrix

ALCHEMY_ATOM_MAP = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 
}
NUM_ATOM_TYPES = 10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate(line: str):
    try:
        toks = line.split()
        if len(toks) % 4 != 0 or len(toks) == 0:
            return None, False

        mol_data = np.array(toks).reshape(-1, 4)
        symbols = mol_data[:, 0]

        if any(s not in ALCHEMY_ATOM_MAP for s in symbols):
            return None, False

        d = mol_data[:, 1].astype(float)
        theta = np.array([float(x.replace('°', '')) for x in mol_data[:, 2]])
        phi = np.array([float(x.replace('°', '')) for x in mol_data[:, 3]])

        xyz = np.stack((
            d * np.sin(theta) * np.cos(phi), 
            d * np.sin(theta) * np.sin(phi), 
            d * np.cos(theta)
        ), axis=1)

        mol = Chem.RWMol()
        conf = Chem.Conformer()
        
        for i, sym in enumerate(symbols):
            atom = Chem.Atom(sym)
            idx = mol.AddAtom(atom)
            conf.SetAtomPosition(idx, (float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
        
        mol.AddConformer(conf)

        try:
            if hasattr(Chem, 'rdDetermineBonds'):
                Chem.rdDetermineBonds.DetermineConnectivity(mol)
                Chem.rdDetermineBonds.DetermineBondOrders(mol)
            else:
                pass
            
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            if len(smi) == 0: return None, False
            return smi, True
            
        except:
            return None, False

    except Exception:
        return None, False
    
def get_stats(prop_path):
    with open(prop_path, 'r') as f:
        vals = [float(line.strip().split()[0]) for line in f.readlines() if line.strip()]
    
    vals = torch.tensor(vals, dtype=torch.float32)
    mean = torch.mean(vals)
    std = torch.std(vals)   
    mad = torch.mean(torch.abs(vals - mean)) 
    
    print(f"Computed Stats -> Mean: {mean.item():.4f}, Std: {std.item():.4f}, MAD: {mad.item():.4f}")
    return mean.item(), std.item(), mad.item() 

class GRPODataset(Dataset):
    def __init__(self, data_path, db_emb_path, db_prop_path, mean, std):
        self.mean = mean
        self.std = std

        data = np.load(data_path)

        raw_targets = torch.from_numpy(data['targets']).float()
        self.targets = (raw_targets - mean) / std
        self.ref_indices = torch.from_numpy(data['indices'][:, 0]).long()

        emb_data = np.load(db_emb_path)
        self.db_embeddings = torch.from_numpy(emb_data['embeddings']).float()

        with open(db_prop_path, 'r') as f:
            raw_db_props = [float(line.strip().split()[0]) for line in f if line.strip()]
        
        raw_db_props = torch.tensor(raw_db_props, dtype=torch.float).view(-1, 1)
        self.db_props = (raw_db_props - mean) / std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target_val = self.targets[idx] 
        
        ref_idx = self.ref_indices[idx]
        ref_vec = self.db_embeddings[ref_idx] 
        ref_prop = self.db_props[ref_idx]     
        
        return ref_vec, ref_prop, target_val

class RewardModel:
    def __init__(self, classifier_path, device, train_mean, train_mad, max_num_atoms=40):
        self.device = device
        self.maxN = max_num_atoms
        self.elem2idx = ALCHEMY_ATOM_MAP
        
        self.mean = torch.tensor(train_mean, device=device)
        self.mad = torch.tensor(train_mad, device=device)
        
        self.classifier = self._load_alchemy_classifier(classifier_path, device)
        self.classifier.eval()

    def _load_alchemy_classifier(self, dir_path, device):
        args_path = os.path.join(dir_path, 'args.pickle')
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"Classifier args not found at {args_path}")
            
        with open(args_path, 'rb') as f:
            args_classifier = pickle.load(f)
            
        args_classifier.device = device
        print(f"Loading classifier config from {dir_path}...")
        
        if args_classifier.model_name == 'egnn':
            classifier = EGNN(in_node_nf=NUM_ATOM_TYPES, in_edge_nf=0, 
                              hidden_nf=args_classifier.nf, device=device, 
                              n_layers=args_classifier.n_layers, coords_weight=1.0,
                              attention=args_classifier.attention, node_attr=args_classifier.node_attr)
        elif args_classifier.model_name == 'naive':
            classifier = Naive(device=device)
        elif args_classifier.model_name == 'numnodes':
            classifier = NumNodes(device=device)
        else:
            raise ValueError(f"Unknown model: {args_classifier.model_name}")

        ckpt_path = os.path.join(dir_path, 'best_checkpoint.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(dir_path, 'best_checkpoint.npy')
        
        print(f"Loading weights from {ckpt_path}")
        classifier_state_dict = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(classifier_state_dict)
        
        return classifier.to(device)
    
    def _parse_mol(self, line: str):
        try:
            toks = np.array(line.split(), dtype=object)
            if len(toks) % 4 != 0 or len(toks) == 0: return None
            mol = toks.reshape(-1, 4)
            
            seq = mol[:, 0]
            idxs = [self.elem2idx[str(s)] for s in seq]
            one_hot = torch.nn.functional.one_hot(torch.tensor(idxs), num_classes=NUM_ATOM_TYPES).float()
            
            d = mol[:, 1].astype(float)
            theta = [str(x).replace('°','') for x in mol[:, 2]]
            theta = np.array(theta).astype(float)
            phi = [str(x).replace('°','') for x in mol[:, 3]]
            phi = np.array(phi).astype(float)
            
            xyz = np.stack((d * np.sin(theta) * np.cos(phi), 
                            d * np.sin(theta) * np.sin(phi), 
                            d * np.cos(theta)), axis=1).astype(np.float32)
            
            N = one_hot.shape[0]
            if N > self.maxN: return None

            oh_out = torch.zeros((self.maxN, NUM_ATOM_TYPES), dtype=torch.float32)
            pos_out = torch.zeros((self.maxN, 3), dtype=torch.float32)
            msk_out = torch.zeros((self.maxN,), dtype=torch.float32)

            oh_out[:N] = one_hot
            pos_out[:N] = torch.from_numpy(xyz)
            msk_out[:N] = 1.0
            return oh_out, pos_out, msk_out
        except:
            return None
    
    @torch.no_grad()
    def predict_batch(self, lines):
        parsed = [self._parse_mol(ln) for ln in lines]
        valid_mask = torch.tensor([1 if p is not None else 0 for p in parsed], dtype=torch.bool, device=self.device)
        
        if valid_mask.sum() == 0:
            return torch.zeros(len(lines), device=self.device), valid_mask

        valid_indices = torch.nonzero(valid_mask).squeeze()
        if valid_indices.dim() == 0: valid_indices = valid_indices.unsqueeze(0)
        
        oh = torch.stack([parsed[i][0] for i in valid_indices]).to(self.device)
        pos = torch.stack([parsed[i][1] for i in valid_indices]).to(self.device)
        am = torch.stack([parsed[i][2] for i in valid_indices]).to(self.device)

        BN, N = am.shape
        edge_mask = (am.unsqueeze(1) * am.unsqueeze(2)).bool()
        diag = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        edge_mask = (edge_mask & diag).view(BN * N * N, 1).float()
        
        nodes = oh.view(BN * N, -1)
        edges = get_adj_matrix(N, BN, self.device)
        
        with autocast(enabled=True):
            pred = self.classifier(h0=nodes, x=pos.view(BN * N, -1), edges=edges, edge_attr=None,
                                   node_mask=am.view(BN * N, -1), edge_mask=edge_mask, n_nodes=N)
        
        y_pred_abs = (pred.float() * self.mad + self.mean).detach().view(BN)
        
        full_preds = torch.zeros(len(lines), device=self.device)
        full_preds[valid_indices] = y_pred_abs
        
        return full_preds, valid_mask
    
class GRPOTrainer:
    def __init__(self, args, train_mean, train_std, train_mad):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        self.device = torch.device(f"cuda:{self.local_rank}")
        set_seed(args.seed + self.local_rank)

        tok_path = os.path.join(args.vocab_dir, "vocab.json")
        self.tok = SimpleTokenizer(args.max_len)
        self.tok.load_vocab(tok_path)
        self.pad_id = self.tok.vocab.get("<pad>", 0)
        self.start_id = self.tok.vocab.get("<s>", 1)
        self.eos_id = self.tok.vocab.get("</s>", 2)
        vocab_size = self.tok.get_vocab_size()

        # Model Init
        mcfg = MambaConfig(
            d_model=args.n_embd,
            n_layer=args.n_layer,
            vocab_size=vocab_size,
            auto_fp16to32=args.auto_fp16to32
        )
        model = PoeticMamba(mcfg, egnn_dim=args.egnn_dim, device=self.device).to(self.device)

        print(f"Loading SFT checkpoint from {args.sft_ckpt}")
        ckpt = torch.load(args.sft_ckpt, map_location="cpu")
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)

        # Reference Model (Frozen)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad = False
        
        # Active Model
        self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        
        # Reward Model
        self.train_mean = train_mean
        self.train_std = train_std
        self.train_mad = train_mad
        self.reward_model = RewardModel(
            classifier_path=args.classifier_path,
            device=self.device,
            train_mean=train_mean,
            train_mad=train_mad
        )

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler(enabled=True)

        self.dataset = GRPODataset(
            data_path=args.finetune_data_path,
            db_emb_path=args.db_emb_path,
            db_prop_path=args.db_prop_path,
            mean=train_mean,
            std=train_std
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=args.batch_conditions,
            shuffle=True, 
            num_workers=8,
            drop_last=True
        )
        self.data_iter = iter(self.dataloader)

        os.makedirs(args.out_dir, exist_ok=True)

    def _get_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return [x.to(self.device) for x in batch]

    def train(self, total_iters=1000):
        if self.local_rank == 0:
            print(f"Starting GRPO Training for {total_iters} steps...")

        for step in range(1, total_iters + 1):
            ref_egnn_vec, ref_prop, target_prop = self._get_batch()
            
            G = self.args.group_size

            ref_egnn_vec = ref_egnn_vec.repeat_interleave(G, dim=0)   # [B*G, Dim]
            ref_prop = ref_prop.repeat_interleave(G, dim=0)           # [B*G, 1]
            target_prop = target_prop.repeat_interleave(G, dim=0)     # [B*G, 1]
            
            current_bs = ref_egnn_vec.size(0)

            with torch.no_grad():
                model_inner = self.model.module
                ref_emb = model_inner.mol_projector(ref_egnn_vec)        
                target_emb = model_inner.prop_projector(target_prop)
                
                # Delta Calculation (Crucial for learning)
                delta_val = target_prop - ref_prop
                delta_emb = model_inner.delta_projector(delta_val)           
                
                prefix_embeds = torch.cat([ref_emb, target_emb, delta_emb], dim=1) 
                
                start_tokens = torch.full((current_bs, 1), self.start_id, dtype=torch.long, device=self.device)
                start_emb = model_inner.backbone.embedding(start_tokens)
                
                initial_embeds = torch.cat([prefix_embeds, start_emb], dim=1) 

            gen_seqs, _ = self._generate_poetic(
                initial_embeds, 
                max_len=self.args.max_len, 
                temp=self.args.temperature, 
                top_k=self.args.topk
            )
            
            # Decode to Text
            gen_texts = []
            for seq in gen_seqs:
                if (seq == self.eos_id).any():
                    end_idx = (seq == self.eos_id).nonzero()[0].item()
                    seq = seq[:end_idx]
                gen_texts.append(self.tok.decode(seq))

            egnn_preds, egnn_valid_mask = self.reward_model.predict_batch(gen_texts)

            target_prop_real = target_prop * self.train_std + self.train_mean
            
            rewards = torch.zeros(current_bs, device=self.device)
            group_smiles_tracker = [set() for _ in range(self.args.batch_conditions)]
            
            valid_count = 0
            mae_list = []

            for i in range(current_bs):
                group_idx = i // G
                
                if not egnn_valid_mask[i]:
                    rewards[i] = -1.0
                    continue 
                
                smi, is_chem_valid = validate(gen_texts[i])
                if not is_chem_valid or smi is None:
                    rewards[i] = -0.5
                    continue
                
                valid_count += 1
                
                if smi in group_smiles_tracker[group_idx]:
                    rewards[i] = -0.2
                    continue
                
                group_smiles_tracker[group_idx].add(smi)

                pred_val = egnn_preds[i]
                target_val = target_prop_real[i].squeeze()
                abs_err = torch.abs(pred_val - target_val).item()
                mae_list.append(abs_err)

                alpha = 0.5 * self.train_mad
                r_acc = math.exp( -abs_err / (alpha + 1e-6) )
                
                r_valid = 0.5 
                
                rewards[i] = (r_valid + r_acc) * self.args.reward_scale

            # Advantage Normalization
            advantages = torch.zeros_like(rewards)
            for g in range(self.args.batch_conditions):
                s = g * G
                e = s + G
                grp_r = rewards[s:e]
                
                mean_r = grp_r.mean()
                std_r = grp_r.std()
                if std_r < 1e-6: std_r = 1.0
                advantages[s:e] = (grp_r - mean_r) / std_r

            # PPO Update Preparation
            prefix_embeds_det = prefix_embeds.detach()
            full_seqs = torch.cat([start_tokens, gen_seqs], dim=1) 
            targets = full_seqs[:, 1:].clone() 
            loss_mask = (targets != self.pad_id)
            
            # Reference LogProbs
            with torch.no_grad():
                ref_seq_emb = self.ref_model.backbone.embedding(full_seqs)
                ref_inputs = torch.cat([prefix_embeds_det, ref_seq_emb], dim=1)
                ref_logits = self.ref_model.lm_head(self.ref_model.backbone(inputs_embeds=ref_inputs))
                gen_logits_ref = ref_logits[:, 3:-1, :] 
                ref_logprobs = F.log_softmax(gen_logits_ref, dim=-1)
                ref_token_logprobs = torch.gather(ref_logprobs, -1, targets.unsqueeze(-1)).squeeze(-1)

            # PPO Epochs
            for _ in range(self.args.ppo_epochs):
                self.opt.zero_grad()
                
                with autocast(enabled=True):
                    # Re-compute Embeddings (Enable Gradients)
                    curr_ref_emb = self.model.module.mol_projector(ref_egnn_vec)
                    curr_tgt_emb = self.model.module.prop_projector(target_prop)
                    
                    curr_delta_val = target_prop - ref_prop
                    curr_delta = self.model.module.delta_projector(curr_delta_val)
                    
                    curr_prefix = torch.cat([curr_ref_emb, curr_tgt_emb, curr_delta], dim=1)
                    
                    curr_seq_emb = self.model.module.backbone.embedding(full_seqs)
                    curr_inputs = torch.cat([curr_prefix, curr_seq_emb], dim=1)
                    
                    logits = self.model.module.lm_head(self.model.module.backbone(inputs_embeds=curr_inputs))
                    gen_logits = logits[:, 3:-1, :]
                    
                    logprobs = F.log_softmax(gen_logits, dim=-1)
                    token_logprobs = torch.gather(logprobs, -1, targets.unsqueeze(-1)).squeeze(-1)
                    
                    # Loss
                    ratio = torch.exp(token_logprobs - ref_token_logprobs.detach())
                    adv_expanded = advantages.unsqueeze(1).expand_as(token_logprobs)
                    
                    pg_loss1 = -adv_expanded * ratio
                    pg_loss2 = -adv_expanded * torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)
                    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / loss_mask.sum()
                    
                    kl_div = ref_token_logprobs.detach() - token_logprobs
                    kl_loss = self.args.kl_coeff * torch.sum(kl_div * loss_mask) / loss_mask.sum()

                    loss = pg_loss + kl_loss
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()

            # Logging
            if self.local_rank == 0:
                avg_r = rewards.mean().item()
                valid_rate = valid_count / current_bs
                avg_mae = np.mean(mae_list) if mae_list else 0.0
                print(f"Step {step} | R: {avg_r:.4f} | Val: {valid_rate:.1%} | MAE: {avg_mae:.4f} | Loss: {loss.item():.4f}")

            # Save
            if step % self.args.save_every == 0 and self.local_rank == 0:
                save_path = os.path.join(self.args.out_dir, f"{self.args.run_name}_step{step}.pt")
                torch.save({'model_state_dict': self.model.module.state_dict()}, save_path)
    
    @torch.no_grad()
    def _generate_poetic(self, initial_embeds, max_len, temp=0.7, top_k=50):
        self.model.eval()
        bs = initial_embeds.size(0)
        curr_embeds = initial_embeds
        gen_tokens = []
        finished = torch.zeros(bs, dtype=torch.bool, device=self.device)
        for _ in range(max_len):
            outputs = self.model.module.backbone(inputs_embeds=curr_embeds)
            last_hidden = outputs[:, -1, :]
            logits = self.model.module.lm_head(last_hidden) 
            logits = logits / temp
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1) 
            is_eos = (next_token.squeeze(1) == self.eos_id)
            finished = finished | is_eos
            next_token = torch.where(finished.unsqueeze(1), torch.tensor(self.pad_id, device=self.device), next_token)
            gen_tokens.append(next_token)
            token_emb = self.model.module.backbone.embedding(next_token)
            curr_embeds = torch.cat([curr_embeds, token_emb], dim=1)
            if finished.all(): break
        return torch.cat(gen_tokens, dim=1), None

def run_ddp(rank, world_size, args, train_mean, train_std, train_mad):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    trainer = GRPOTrainer(args, train_mean, train_std, train_mad)
    
    trainer.train(total_iters=args.iters)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="grpo_test")
    parser.add_argument("--sft_ckpt", type=str, required=True)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument("--finetune_data_path", type=str, required=True, help="Path to .npz from finetune_retriever.py")
    parser.add_argument("--prop_path", type=str, required=True, help="Original prop txt for stats")
    parser.add_argument("--db_emb_path", type=str, required=True)
    parser.add_argument("--db_prop_path", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, required=True)
    
    # Model Args
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--egnn_dim", type=int, default=128)
    parser.add_argument("--auto_fp16to32", action='store_true')
    
    # RL Args
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_conditions", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="grpo_checkpoints")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    
    # Gen Args
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=80) 
    parser.add_argument("--reward_scale", type=float, default=1.0)

    args = parser.parse_args()
    train_mean, train_std, train_mad = get_stats(args.prop_path)

    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(run_ddp, args=(world_size, args, train_mean, train_std, train_mad), nprocs=world_size)
    else:
        run_ddp(0, 1, args, train_mean, train_std, train_mad)
