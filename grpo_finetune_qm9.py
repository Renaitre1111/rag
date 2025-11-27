import os
import re
import math
import json
import copy
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm

import pickle
from qm9.property_prediction.schnet import SchNet
from dataset import SimpleTokenizer
from eval_qm9 import get_classifier, get_args_gen, get_dataloader
from qm9.utils import compute_mean_mad
from qm9.property_prediction.prop_utils import get_adj_matrix

MIN_D_VALUE = 0.000
MAX_D_VALUE = 15.000

def _extract_value(line: str) -> str:
    s = line.strip()
    if not s:
        raise ValueError("empty condition line")

    m = re.match(r'^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
    if m:
        return m.group(1)

    m = re.search(r'\[COND_START\]\s*(\d+\.\d+)\s*\[COND_END\]', s)
    if m:
        return m.group(1)

    m = re.search(r'value\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
    if m:
        return m.group(1)

    m = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
    if m:
        return m.group(1)

    m = re.search(r'\b([+-]?\d+)\b', s)
    if m:
        return m.group(1)

    nums = re.findall(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
    if nums:
        return nums[-1]

    raise ValueError(f"no numeric value found in condition line: {s!r}")

@torch.no_grad()
def generate_autoregressive(model: torch.nn.Module,
                            input_ids: torch.Tensor,
                            max_len: int,
                            temp: float = 1.0,
                            top_k: int = 50,
                            eos_id: int = None,
                            pad_id: int = None,
                            *,
                            tokenizer=None):
    if tokenizer is None:
        raise ValueError("tokenizer is required for distance filtering generation.")

    model.eval()
    device = input_ids.device
    B = input_ids.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    expected_token_type = torch.zeros(B, dtype=torch.long, device=device)

    current_len = input_ids.size(1)
    valid_mask = None

    for _ in range(max_len - current_len):
        outputs = model(input_ids)
        logits = outputs[0][:, -1, :]  # [B, V]，
        if (valid_mask is None) or (valid_mask.size(0) != logits.size(-1)):
            V = logits.size(-1)
            mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            for tok_str, tok_id in tokenizer.get_vocab().items():  # str->id
                if 0 <= tok_id < V:
                    ok = False
                    try:
                        val = float(tok_str)
                        ok = (MIN_D_VALUE <= val <= MAX_D_VALUE)
                    except Exception:
                        pass
                    mask[tok_id] = ok
            valid_mask = mask
            tokenizer._valid_dis_mask = mask.detach().cpu()
        else:
            valid_mask = tokenizer._valid_dis_mask.to(logits.device)

        if eos_id is not None and pad_id is not None:
            logits = torch.where(
                finished.unsqueeze(1),
                torch.full_like(logits, float("-inf")),
                logits
            )
            logits[finished, pad_id] = 0.0

        need_dist = (expected_token_type == 1) & (~finished)
        if need_dist.any():
            rows = need_dist.nonzero(as_tuple=False).squeeze(1)
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)  # [K]
            sub = logits[rows][:, valid_ids]                                 # [R, K]
            t_d = 0.7            
            k_d = 80    
            sub = sub / max(1e-6, t_d)
            if 0 < k_d < sub.size(-1):
                vals,_ = torch.topk(sub, k_d); kth = vals[..., -1, None]
                sub = torch.where(sub < kth, torch.full_like(sub, float('-inf')), sub)
        
            p = F.softmax(sub, dim=-1)
            idx = torch.multinomial(p, 1).squeeze(1)      
            chosen = valid_ids[idx]                        

            logits[rows, :] = float('-inf')
            logits[rows, chosen] = 0.0

        if temp is not None and temp > 0:
            logits = logits / max(1e-6, temp)

        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            values, _ = torch.topk(logits, top_k)
            min_values = values[..., -1, None]
            logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
            next_token = torch.full((B, 1), pad_id if pad_id is not None else 0, dtype=torch.long, device=device)
        else:
            next_token = torch.multinomial(probs, 1)  # [B, 1]

        if eos_id is not None and pad_id is not None:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_id),
                next_token,
            )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        expected_token_type = torch.where(finished, expected_token_type, (expected_token_type + 1) % 4)

    return input_ids

def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state

def _infer_has_per_layer_norm(state):
    pat = re.compile(r"^backbone\.layers\.(\d+)\.norm\.weight$")
    return any(pat.match(k) is not None for k in state.keys())

def _infer_n_layers_from_state(state):
    layer_ids = []
    for k in state.keys():
        m = re.match(r"^backbone\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    return (max(layer_ids) + 1) if layer_ids else None

def _maybe_remap_block_name_keys(state, model_keys):
    has_mixer_in_ckpt = any(".mixer." in k for k in state.keys())
    has_mamba_in_ckpt = any(".mamba." in k for k in state.keys())
    has_mixer_in_model = any(".mixer." in k for k in model_keys)
    has_mamba_in_model = any(".mamba." in k for k in model_keys)

    remapped = state
    if has_mixer_in_ckpt and not has_mixer_in_model and has_mamba_in_model:
        remapped = {k.replace(".mixer.", ".mamba."): v for k, v in state.items()}
    elif has_mamba_in_ckpt and not has_mamba_in_model and has_mixer_in_model:
        remapped = {k.replace(".mamba.", ".mixer."): v for k, v in state.items()}
    return remapped

def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def load_tokenizer(tokenizer_path: str, max_length: int, tokenizer_type: str = "simple"):
    """Loads tokenizer from vocab file."""
    tok = SimpleTokenizer(max_length)
    tok.load_vocab(tokenizer_path)
    return tok

def build_mask_after_prefix(seqs, prefix_lens, pad_id):
    B, T = seqs.shape
    mask = torch.zeros((B, T - 1), dtype=torch.bool, device=seqs.device)
    tgt = seqs[:, 1:]  # targets for the model's prediction
    not_pad = (tgt != pad_id)  # True for non-padding tokens

    for i in range(B):
        eff_len = int((seqs[i] != pad_id).sum().item())
        start_idx = max(prefix_lens[i] - 1, 0)

        if start_idx >= (eff_len - 1):
            continue

        row = torch.zeros(T - 1, dtype=torch.bool, device=seqs.device)
        row[start_idx: eff_len - 1] = True
        mask[i] = (row & not_pad[i])

    return mask  # [B, T-1]

os.environ['GRPO_STEP'] = '0'

def token_logprobs(model, seqs):
    """teacher-forcing per-token logprob for targets = seqs[:,1:]"""
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(seqs[:, :-1])[0]  # [B, T-1, V]

    if torch.isnan(logits).any():
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"[{rank}] WARNING: NaN detected in Mamba model's logits! Step: {os.environ.get('GRPO_STEP', 'N/A')}, Shape: {logits.shape}, Seq_shape: {seqs.shape}")

    logp = F.log_softmax(logits, dim=-1)

    if torch.isnan(logp).any():
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"[{rank}] WARNING: NaN detected in log_softmax output (logp)! Step: {os.environ.get('GRPO_STEP', 'N/A')}, Shape: {logp.shape}")

    tgt = seqs[:, 1:]  # [B, T-1]
    lp = torch.gather(logp, -1, tgt.unsqueeze(-1)).squeeze(-1)

    if torch.isnan(lp).any():
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"[{rank}] WARNING: NaN detected in gathered logprobs (lp)! Step: {os.environ.get('GRPO_STEP', 'N/A')}, Shape: {lp.shape}")

    return lp

class RewardEvaluator:
    """
    Build an in-memory evaluator consistent with eval_conditional_qm9.py
    Lines format: "<value> <tail tokens...>"; value comes from original rag prefix.
    """
    def __init__(self, reward_model_type, classifiers_path, generators_path, prop_name, device, max_num_atoms=30, remove_h=False):
        self.device = device
        self.maxN = max_num_atoms
        # Keep elem2idx as is, assuming it's correct based on your data.
        self.elem2idx = {'H':0, 'C':1, 'C@':1, 'C@@':1, 'N':2, 'O':3, 'F':4} if not remove_h \
                        else {'C':0, 'C@':0, 'C@@':0, 'N':1, 'O':2, 'F':3}
        self.classifier = get_classifier(classifiers_path, device=device, model_type=reward_model_type)
        args_gen = get_args_gen(generators_path)
        dls = get_dataloader(args_gen)
        norms = compute_mean_mad(dls, [prop_name], args_gen.dataset)
        self.mean = norms[prop_name]['mean']; self.mad = norms[prop_name]['mad']
        

    def _parse_one(self, line: str):
        # parse "<value> E d θ° φ° E d θ° φ° ..." → (one_hot[N,T], pos[N,3], mask[N])
        try:
            toks = np.array(line.split(), dtype=object)
            mol = toks[1:]  # mol is a 1D array initially

            try:
                mol = mol.reshape(-1, 4)
            except Exception:
                cut_idx = -1
                for c_idx in range(int(len(mol) / 4) - 1):
                    vals = mol[4 * c_idx: 4 * c_idx + 4]
                    if vals[2][-1] != '°' or vals[3][-1] != '°':
                        mol = mol[:4 * c_idx].reshape(-1, 4)
                        cut_idx = c_idx
                        break
                    else:
                        try:
                            _ = self.elem2idx.get(str(vals[0]))
                            _ = float(vals[1])
                            _ = float(str(vals[2][:-1]))
                            _ = float(str(vals[3][:-1]))
                        except Exception:
                            mol = mol[:4 * c_idx].reshape(-1, 4)
                            cut_idx = c_idx
                            break

                if cut_idx == -1 and len(mol) % 4 == 0:
                    mol = mol.reshape(-1, 4)
                elif cut_idx != -1:
                    pass
                else:
                    return None

            if mol.shape[0] == 0 or mol.shape[1] != 4:
                return None

            seq = mol[:, 0]
            try:
                idxs = [self.elem2idx[str(s)] for s in seq]
            except Exception:
                return None
            one_hot = torch.nn.functional.one_hot(torch.tensor(idxs), num_classes=max(self.elem2idx.values()) + 1).float()

            d = mol[:, 1].astype(float)
            theta = mol[:, 2].astype(str)
            phi = mol[:, 3].astype(str)

            try:
                theta = np.array([s[:-1] for s in theta]).astype(float)
                phi = np.array([s[:-1] for s in phi]).astype(float)
                xyz = np.stack((d * np.sin(theta) * np.cos(phi), d * np.sin(theta) * np.sin(phi), d * np.cos(theta)), axis=1).astype(np.float32)
            except Exception:
                return None

            if one_hot.shape[0] > self.maxN:
                return None

            N = one_hot.shape[0]; types = one_hot.shape[1]
            oh = torch.zeros((self.maxN, types), dtype=torch.float32)
            pos = torch.zeros((self.maxN, 3), dtype=torch.float32)
            msk = torch.zeros((self.maxN,), dtype=torch.float32)

            oh[:N] = one_hot; pos[:N] = torch.from_numpy(xyz); msk[:N] = 1.0
            return oh, pos, msk
        except Exception as e:
            return None

    @torch.no_grad()
    def predict(self, lines: List[str]):
        B = len(lines)
        parsed = [self._parse_one(ln) for ln in lines]
        valid = torch.tensor([0 if p is None else 1 for p in parsed], dtype=torch.bool, device=self.device)

        if valid.sum() == 0:
            return torch.full((B,), float('nan'), device=self.device), valid

        # Stack valid parsed data, and use zeros for invalid (None) entries
        oh = torch.stack([p[0] if p is not None else torch.zeros(self.maxN, max(self.elem2idx.values()) + 1) for p in parsed]).to(self.device)
        pos = torch.stack([p[1] if p is not None else torch.zeros(self.maxN, 3) for p in parsed]).to(self.device)
        am = torch.stack([p[2] if p is not None else torch.zeros(self.maxN) for p in parsed]).to(self.device)

        BN, N = am.shape

        # Construct edge mask
        edge_mask = (am.unsqueeze(1) * am.unsqueeze(2)).bool()
        diag = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0)
        edge_mask = (edge_mask & diag).view(BN * N * N, 1).float()

        nodes = oh.view(BN * N, -1)
        edges = get_adj_matrix(N, BN, self.device)

        # EGNN Classifier prediction
        with torch.cuda.amp.autocast(enabled=True):
            pred = self.classifier(h0=nodes, x=pos.view(BN * N, -1), edges=edges, edge_attr=None,
                                   node_mask=am.view(BN * N, -1), edge_mask=edge_mask, n_nodes=N)

        if torch.isnan(pred).any():
            rank = os.environ.get("LOCAL_RANK", "N/A")
            print(f"[{rank}] WARNING: NaN detected in EGNN classifier's prediction! Step: {os.environ.get('GRPO_STEP', 'N/A')}, Shape: {pred.shape}")

        y = (self.mad * pred + self.mean).detach().view(BN)

        if torch.isnan(y).any():
            rank = os.environ.get("LOCAL_RANK", "N/A")
            print(f"[{rank}] WARNING: NaN detected in de-normalized EGNN output (y)! Step: {os.environ.get('GRPO_STEP', 'N/A')}, Shape: {y.shape}")

        y[~valid] = float('nan')
        return y, valid

@dataclass
class GRPOConfig:
    run_name: str
    vocab_dir: str
    tokenizer: str = "simple"    
    max_len: int = 640
    n_layer: int = 24
    n_embd: int = 768
    sft_ckpt: str = "" 
    model: str = "mamba"
    reward_model: str = "egnn"
    n_head: int = 8
    # generation
    temperature: float = 1.0
    topk: int = 80
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    # GRPO
    group_size: int = 8
    batch_conditions: int = 8
    clip_eps: float = 0.2
    kl_coeff: float = 0.03
    ppo_epochs: int = 2
    # reward
    invalid_penalty: float = 1.0
    reward_scale: float = 1.0
    # optim
    lr: float = 5e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # io
    save_every: int = 200
    out_dir: str = "cond/grpo_weights"
    seed: int = 42

class GRPOTrainerDDP:
    def __init__(self, cfg: GRPOConfig, prop: str, rag_path: str, classifiers_path: str, generators_path: str):
        self.cfg = cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        self.device = torch.device(f"cuda:{self.local_rank}")
        set_seed(cfg.seed, rank=self.local_rank)

        tok_path = os.path.join(cfg.vocab_dir, "vocab.json")
        self.tok = load_tokenizer(tok_path, max_length=cfg.max_len, tokenizer_type=cfg.tokenizer)
        self.vocab = self.tok.get_vocab()
        self.pad_id = self.vocab.get(cfg.pad_token, 0)
        self.eos_id = self.vocab.get(cfg.eos_token, None)

        # Build model with ckpt-aware config
        ckpt = torch.load(cfg.sft_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = _strip_module_prefix(state)

        ckpt_n_layers = _infer_n_layers_from_state(state)
        if ckpt_n_layers is not None and ckpt_n_layers != cfg.n_layer:
            print(f"[info] override cfg.n_layer: {cfg.n_layer} -> {ckpt_n_layers}")
            cfg.n_layer = ckpt_n_layers

        if cfg.model == 'mamba':
            from model.mamba import MambaLMHeadModel, MambaConfig

            has_pln = _infer_has_per_layer_norm(state)
            fused_add_norm = not has_pln
            mcfg = MambaConfig(
                d_model=cfg.n_embd,
                n_layer=cfg.n_layer,
                vocab_size=len(self.vocab),
                ssm_cfg={},  
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=fused_add_norm,
                pad_vocab_size_multiple=8,  
                num_props=0,  
                scaffold=None,  
                isconditional=True,  
                auto_fp16to32=True,  
            )
            model = MambaLMHeadModel(mcfg).to(self.device)

            model_keys = list(model.state_dict().keys())
            state_try = _maybe_remap_block_name_keys(state, model_keys)

            res = model.load_state_dict(state_try, strict=False)
            print(f"[load_state_dict] missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            if len(res.missing_keys) > 0:
                print("[warn] missing keys:", res.missing_keys[:5])
            if len(res.unexpected_keys) > 0:
                print("[warn] unexpected keys:", res.unexpected_keys[:5])
            if len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0:
                model.load_state_dict(state_try, strict=True)
            else:
                if fused_add_norm and any(".norm." in k for k in res.unexpected_keys):
                    print("[warn] Filtering per-layer norm params from ckpt (fused_add_norm=True, no corresponding modules).")
                    filtered = {k: v for k, v in state_try.items() if (".layers." not in k or ".norm." not in k)}
                    res2 = model.load_state_dict(filtered, strict=False)
                    print(f"[reload after filtering norm] missing={len(res2.missing_keys)}, unexpected={len(res2.unexpected_keys)}")
                    if len(res2.missing_keys) == 0 and len(res2.unexpected_keys) == 0:
                        model.load_state_dict(filtered, strict=True)
                    else:
                        print("[warn] Mismatch keys still exist after filtering, check d_model/vocab/isconditional/ssm_cfg consistency; continuing non-strict.")
                else:
                    print("[warn] Mismatch keys exist, check d_model/vocab/isconditional/ssm_cfg consistency; continuing non-strict.")

        elif cfg.model == 'gpt':
            from model.gpt import GPT, GPTConfig
            gcfg = GPTConfig(
                vocab_size=len(self.vocab), block_size=cfg.max_len, n_layer=cfg.n_layer,
                n_head=cfg.n_head, n_embd=cfg.n_embd, isconditional=True, lstm=False
            )
            model = GPT(gcfg).to(self.device)
            model.load_state_dict(state, strict=True)

        self.ref_model = copy.deepcopy(model).to(self.device).eval()
        self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)

        with open(rag_path, "r", encoding="utf-8") as f:
            self.rag_lines = [ln.strip() for ln in f if ln.strip()]

        self.reward_model = RewardEvaluator(reward_model_type=cfg.reward_model, classifiers_path=classifiers_path, generators_path=generators_path, prop_name=prop, device=self.device, max_num_atoms=30)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = GradScaler(enabled=True)
        os.makedirs(cfg.out_dir, exist_ok=True)
        dist.barrier()

    def _pick_conditions(self, B: int) -> List[str]:
        """Samples a batch of conditions from the RAG lines."""
        return random.sample(self.rag_lines, B)

    def _encode_prefix_batch(self, prefixes: List[str], repeats_each: int):
        """Encodes prefixes and extracts target values for a batch."""
        ids_list, prefix_lens, targets = [], [], []
        failed_extract_count = 0
        for s in prefixes:
            try:
                val = float(_extract_value(s))
            except Exception as e:
                if self.local_rank == 0:
                    print(f"[{self.local_rank}] [warn] Value extract failed for prefix '{s[:50]}...': {e}. Default to 0.0")
                val = 0.0
                failed_extract_count += 1
            for _ in range(repeats_each):
                # generation_encode adds <s> token
                seq = torch.tensor(self.tok.generation_encode(s), dtype=torch.long)
                ids_list.append(seq)
                prefix_lens.append(len(seq))
                targets.append(val)

        if failed_extract_count == len(prefixes):
            if self.local_rank == 0:
                print(f"[{self.local_rank}] [warn] All values extract failed in batch_conditions; signaling to skip iter.")
            return None, None, None

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=self.pad_id).to(self.device)
        return input_ids, prefix_lens, torch.tensor(targets, dtype=torch.float32, device=self.device)

    def _decode_tails(self, seqs: torch.Tensor, prefix_lens: List[int]) -> List[str]:
        """Decodes the generated tail part of the sequences."""
        tails = []
        for i, seq in enumerate(seqs):
            eff_len = int((seq != self.pad_id).sum().item())
            pre = min(prefix_lens[i], eff_len)
            tail = seq[pre:eff_len]

            if self.eos_id is not None and tail.numel() > 0:
                eos_pos = (tail == self.eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    tail = tail[: int(eos_pos[0].item())]

            tail = tail[tail != self.pad_id]

            if tail.numel() == 0:
                tails.append("")
            else:
                tails.append(self.tok.decode(tail))
        return tails

    def _save_if_main(self, step):
        """Saves model checkpoint if it's the main process and at a save interval."""
        if dist.get_rank() == 0 and (step % self.cfg.save_every == 0):
            out = os.path.join(self.cfg.out_dir, f"{self.cfg.run_name}_grpo_step{step}.pt")
            torch.save({"model_state_dict": self.model.module.state_dict()}, out)

    def train(self, total_iters: int = 2000):
        """Main GRPO training loop."""
        cfg = self.cfg
        if dist.get_rank() == 0:
            print(f"[GRPO-DDP] world_size={self.world_size}")

        for step in range(1, total_iters + 1):
            os.environ['GRPO_STEP'] = str(step)

            cond_batch = self._pick_conditions(cfg.batch_conditions)
            in_ids, prefix_lens, targets = self._encode_prefix_batch(cond_batch, cfg.group_size)
            if in_ids is None:
                if self.local_rank == 0:
                    print(f"[{self.local_rank}] [it {step}] Skip due to all value extract failed in batch.")
                continue

            with torch.no_grad():
                self.model.eval()
                seqs = generate_autoregressive(
                    self.model.module, in_ids, max_len=cfg.max_len,
                    temp=cfg.temperature, top_k=cfg.topk,
                    eos_id=self.eos_id, pad_id=self.pad_id,
                    tokenizer=self.tok,   
                )

            tails = self._decode_tails(seqs, prefix_lens)
            lines = [f"{targets[i].item()} {tails[i]}" for i in range(len(tails))]
            pred_y, valid_mask = self.reward_model.predict(lines)

            mae = torch.abs(pred_y - targets)
            is_nan = torch.isnan(mae) | (~valid_mask)

            vmae = mae[~is_nan]
            if vmae.numel() > 0 and not torch.isnan(vmae).any():
                sigma = (0.8 * vmae.detach().median()).clamp_min(1e-6)
            else:
                sigma = torch.tensor(1.0, device=self.device)

            raw_reward = torch.empty_like(mae).fill_(-cfg.invalid_penalty)
            if vmae.numel() > 0:
                raw_reward[~is_nan] = torch.exp(-(vmae / sigma))
            raw_reward = torch.nan_to_num(cfg.reward_scale * raw_reward, nan=-cfg.invalid_penalty)

            adv_tokens_flat = torch.empty_like(raw_reward)
            for g in range(len(cond_batch)):
                s = g * cfg.group_size; e = s + cfg.group_size
                grp_rewards = raw_reward[s:e]
                mu = grp_rewards.mean()
                sd = grp_rewards.std(unbiased=False)

                normalized_grp = (grp_rewards - mu) / (sd + 1e-6)
                adv_tokens_flat[s:e] = torch.clamp(normalized_grp, -5.0, 5.0)  # Clamp advantage values

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    old_lp = token_logprobs(self.model.module, seqs)
                    ref_lp = token_logprobs(self.ref_model, seqs)

            adv_tokens = adv_tokens_flat.unsqueeze(1).expand_as(old_lp).detach()

            mask = build_mask_after_prefix(seqs, prefix_lens, self.pad_id)

            self.model.train()
            for ei in range(cfg.ppo_epochs):
                self.opt.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    new_lp = token_logprobs(self.model, seqs)

                    old_lp_det = old_lp.detach()
                    ref_lp_det = ref_lp.detach()
                    mask_det = mask.detach()

                    ratio = torch.exp((new_lp - old_lp_det))[mask_det]
                    adv = adv_tokens[mask_det]

                    obj1 = ratio * adv
                    obj2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv

                    ppo_policy_loss = -torch.mean(torch.min(obj1, obj2))

                    log_diff_ref_new = ref_lp_det - new_lp
                    kl_per_token = (torch.exp(log_diff_ref_new) - log_diff_ref_new - 1.0)[mask_det]
                    kl_reg_loss = cfg.kl_coeff * torch.mean(kl_per_token)

                    loss = ppo_policy_loss + kl_reg_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()

            if dist.get_rank() == 0:
                valid_mae = mae[~is_nan]
                avg_mae = float(valid_mae.mean().item()) if valid_mae.numel() > 0 else float('nan')
                med_mae = float(valid_mae.median().item()) if valid_mae.numel() > 0 else float('nan')
                avg_r = float(raw_reward.mean().item())
                valid_rate = 1.0 - float(is_nan.float().mean().item())
                print(f"[{self.local_rank}] [it {step}] MAE: {avg_mae:.4f} (Med {med_mae:.4f}) | Valid: {valid_rate:.2%} | Raw R: {avg_r:.4f} | Sigma: {sigma:.4f} | KL_loss: {kl_reg_loss.item():.4f} | Total_loss: {loss.item():.4f}")

            self._save_if_main(step)

        if dist.get_rank() == 0:
            out = os.path.join(self.cfg.out_dir, f"{self.cfg.run_name}_grpo_last.pt")
            torch.save({"model_state_dict": self.model.module.state_dict()}, out)
            print(f"[GRPO-DDP] saved to: {out}")
        dist.barrier()  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Base configuration parameters
    parser.add_argument("--run_name", type=str, required=True, help="Name for the GRPO run.")
    parser.add_argument("--vocab_dir", type=str, required=True, help="Path to tokenizer vocab directory (contains vocab.json).")
    parser.add_argument("--tokenizer", type=str, default="simple", choices=["simple","subch"], help="Type of tokenizer to use.")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length for model input.")
    # Model architecture parameters
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "mamba"])
    parser.add_argument("--reward_model", type=str, default="egnn", choices=["egnn", "schnet"], help="Reward model type to use.")
    parser.add_argument("--n_head", type=int, default=8, help="Heads for GPT")
    parser.add_argument("--n_layer", type=int, default=16, help="Number of layers in the Mamba model.")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension of the Mamba model.")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="Path to the Supervised Fine-Tuned (SFT) model checkpoint.")
    # Data and Reward parameters
    parser.add_argument("--rag_path", type=str, required=True, help="Path to the RAG prompts file (conditions + hints).")
    parser.add_argument("--prop", type=str, default="alpha", help="Property name to optimize (e.g., 'alpha', 'homo').")
    parser.add_argument("--classifiers_path", type=str, required=True, help="Path to the EGNN classifier model directory.")
    parser.add_argument("--generators_path", type=str, required=True, help="Path to the EGNN generator arguments directory (for property norms).")
    # Sampling and GRPO specific parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for molecular generation.")
    parser.add_argument("--topk", type=int, default=80, help="Top-K sampling parameter for molecular generation.")
    parser.add_argument("--group_size", type=int, default=12, help="Number of samples per condition in a group (G in GRPO).")
    parser.add_argument("--batch_conditions", type=int, default=16, help="Number of unique conditions processed in each training step.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon (epsilon in GRPO Equation 3).")
    parser.add_argument("--kl_coeff", type=float, default=0.03, help="KL regularization coefficient (beta in GRPO Equation 3).")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Number of policy update epochs per training step.")
    parser.add_argument("--invalid_penalty", type=float, default=1.0, help="Negative reward assigned to invalid/unparseable generated samples.")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Scaling factor for the calculated rewards.")
    parser.add_argument("--lr", type=float, default=1.2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--save_every", type=int, default=200, help="Save model checkpoint every N training steps.")
    parser.add_argument("--out_dir", type=str, default="cond/grpo_weights", help="Directory to save GRPO checkpoints.")
    parser.add_argument("--iters", type=int, default=800, help="Total number of training iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    cfg = GRPOConfig(
        run_name=args.run_name, vocab_dir=args.vocab_dir, tokenizer=args.tokenizer, max_len=args.max_len,
        n_layer=args.n_layer, n_embd=args.n_embd, sft_ckpt=args.sft_ckpt, model=args.model, reward_model=args.reward_model, n_head=args.n_head,
        temperature=args.temperature, topk=args.topk, group_size=args.group_size, batch_conditions=args.batch_conditions,
        clip_eps=args.clip_eps, kl_coeff=args.kl_coeff, ppo_epochs=args.ppo_epochs,
        invalid_penalty=args.invalid_penalty, reward_scale=args.reward_scale,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        save_every=args.save_every, out_dir=args.out_dir, seed=args.seed
    )

    trainer = GRPOTrainerDDP(cfg, prop=args.prop, rag_path=args.rag_path,
                             classifiers_path=args.classifiers_path, generators_path=args.generators_path)

    trainer.train(total_iters=args.iters)
