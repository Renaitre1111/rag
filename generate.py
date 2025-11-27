import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataset import SimpleTokenizer, SoftRAGDataset
from model.poetic import PoeticMamba
from model.mamba import MambaConfig

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_tokenizer(tokenizer_path: str, max_length: int):
    tok = SimpleTokenizer(max_length)
    tok.load_vocab(tokenizer_path)
    return tok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help="Run name for loading weights")
    parser.add_argument('--tokenizer_dir', type=str, required=True, help="Directory containing vocab.json")
    parser.add_argument('--save_path', type=str, default='cond/gens', help="Directory to save generated molecules")

    parser.add_argument('--target_prop_path', type=str, required=True, help="Path to the txt file containing target properties (absolute values)")
    parser.add_argument('--test_retrieval_path', type=str, required=True, help="Path to the retrieval .npz file for these targets")
    
    parser.add_argument('--db_emb_path', type=str, required=True, help="Path to training set EGNN embeddings (.npz)")
    parser.add_argument('--db_prop_path', type=str, required=True, help="Path to training set properties (.txt)")
    
    parser.add_argument('--egnn_dim', type=int, default=128, help="Dimension of EGNN embeddings (128 or 256)")
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--max_len', type=int, default=160, help="Max sequence length")

    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--repeats', type=int, default=1, help="Number of molecules to generate per target property")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_path', type=str, default=None, help="Explicit path to checkpoint.")
    parser.add_argument('--auto_fp16to32', action='store_true', default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tok_path = os.path.join(args.tokenizer_dir, 'vocab.json')
    print(f"Loading tokenizer from {tok_path}")
    tokenizer = load_tokenizer(tok_path, max_length=args.max_len)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.vocab.get('<pad>', 0)
    start_id = tokenizer.vocab.get('<s>', 1)
    eos_id = tokenizer.vocab.get('</s>', 2)

    with open(args.target_prop_path, 'r') as f:
        target_props = [float(line.strip().split()[0]) for line in f.readlines() if line.strip()]

    dummy_texts = [""] * len(target_props)

    test_dataset = SoftRAGDataset(
        target_texts=dummy_texts,
        target_props=target_props,
        tokenizer=tokenizer,
        db_emb_path=args.db_emb_path,
        db_prop_path=args.db_prop_path,
        retrieval_path=args.test_retrieval_path,
        split='test', 
        max_len=args.max_len,
        mean=None, std=None 
    )

    print(f"Normalization Stats: Mean={test_dataset.mean:.4f}, Std={test_dataset.std:.4f}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8
    )

    mcfg = MambaConfig(
        d_model=args.n_embd,
        n_layer=args.n_layer,
        vocab_size=vocab_size,
        auto_fp16to32=args.auto_fp16to32
    )
    
    model = PoeticMamba(mcfg, egnn_dim=args.egnn_dim, device=device).to(device)

    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    os.makedirs(args.save_path, exist_ok=True)
    output_file = os.path.join(args.save_path, f"{args.run_name}_generated.txt")
    print(f"Starting generation. Output: {output_file}.")

    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            ref_egnn_vec, _, ref_prop, target_prop = [x.to(device) for x in batch]
            
            curr_bs = ref_egnn_vec.size(0)
            if args.repeats > 1:
                ref_egnn_vec = ref_egnn_vec.repeat_interleave(args.repeats, dim=0)
                target_prop = target_prop.repeat_interleave(args.repeats, dim=0)
                ref_prop = ref_prop.repeat_interleave(args.repeats, dim=0)
                curr_bs *= args.repeats

            # 1. Reference [B, 1, H]
            ref_emb = model.mol_projector(ref_egnn_vec)     
            
            # 2. Target Prop [B, 1, H]
            target_prop_emb = model.prop_projector(target_prop) 
            
            # 3. Delta [B, 1, H]
            delta = target_prop - ref_prop
            delta_emb = model.delta_projector(delta)        
            
            prefix_embeds = torch.cat([ref_emb, target_prop_emb, delta_emb], dim=1)
            
            start_tokens = torch.full((curr_bs, 1), start_id, dtype=torch.long, device=device)
            start_emb = model.backbone.embedding(start_tokens)
            
            inputs_embeds = torch.cat([prefix_embeds, start_emb], dim=1)
            
            generated_seqs = []
            
            finished = torch.zeros(curr_bs, dtype=torch.bool, device=device)
            
            for _ in range(args.max_len):
                outputs = model.backbone(inputs_embeds=inputs_embeds)
                
                last_hidden = outputs[:, -1, :] 
                logits = model.lm_head(last_hidden) 
                
                logits = logits / args.temperature
                if args.topk > 0:
                    v, _ = torch.topk(logits, args.topk)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
                
                is_eos = (next_token == eos_id).squeeze(1)
                finished = finished | is_eos
                
                if finished.all():
                    generated_seqs.append(next_token)
                    break
                
                generated_seqs.append(next_token)
                
                next_token_emb = model.backbone.embedding(next_token)
                inputs_embeds = torch.cat([inputs_embeds, next_token_emb], dim=1)

            generated_ids = torch.cat(generated_seqs, dim=1)
            
            for i in range(curr_bs):
                seq = generated_ids[i]
                
                if (seq == eos_id).any():
                    end_idx = (seq == eos_id).nonzero()[0].item()
                    seq = seq[:end_idx]
                
                mol_seq = tokenizer.decode(seq)
                
                results.append(f"{mol_seq}\n")

    with open(output_file, 'w') as f:
        f.writelines(results)

    print(f"Done! Generated {len(results)} molecules.")

if __name__ == '__main__':
    main()