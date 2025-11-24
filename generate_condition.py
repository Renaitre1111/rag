import argparse
import os
import random
import re
from functools import partial
from collections import Counter

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dataset import SimpleTokenizer

MIN_D_VALUE = 0.000
MAX_D_VALUE = 20.000


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(tokenizer_path: str, max_length: int, tokenizer_type: str = "simple"):
    tok = SimpleTokenizer(max_length)
    tok.load_vocab(tokenizer_path)
    return tok


def safe_load_state_dict(model: torch.nn.Module, model_path: str, device="cpu", strict=True):
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=strict)
    return model


def top_k_filtering(logits: torch.Tensor, k: int):
    if (k is None) or (k <= 0) or (k >= logits.size(-1)):
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _extract_value(line: str) -> str:
    s = line.strip()
    if not s:
        raise ValueError("empty condition line")

    m = re.search(r'\[COND_START\]\s*(\d+\.\d+)\s*\[COND_END\]', s)
    if m:
        return m.group(1)

    m = re.search(r'value\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
    if m:
        return m.group(1)

    m = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', s)
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
                            top_k: int = 80,
                            eos_id: int = None,
                            pad_id: int = None,
                            start_token_type: int = 0,
                            tokenizer=None,
                            use_amp: bool = True,
                            amp_dtype: torch.dtype = torch.float16): 
    model.eval()
    device = input_ids.device
    B, cur_len = input_ids.shape
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    expected_token_type = torch.full((B,), start_token_type, dtype=torch.long, device=device)

    vocab = tokenizer.get_vocab() if tokenizer is not None else {}
    mask = None
    vocab = tokenizer.get_vocab() if tokenizer is not None else {}
    valid_mask = None
    if tokenizer is not None and len(vocab) > 0:
        max_id = max(vocab.values()) + 1
        base_mask = torch.zeros(max_id, dtype=torch.bool, device=device)
        for tok_str, tok_id in vocab.items():
            if tok_id < max_id:
                ok = False
                try:
                    val = float(tok_str)
                    ok = (MIN_D_VALUE <= val <= MAX_D_VALUE)
                except Exception:
                    pass
                base_mask[tok_id] = ok
        vocab_size_logits = model.config.vocab_size if hasattr(model, "config") else max_id
        if base_mask.size(0) < vocab_size_logits:
            pad = torch.zeros(vocab_size_logits - base_mask.size(0),
                              dtype=torch.bool, device=device)
            valid_mask = torch.cat([base_mask, pad], dim=0)
        else:
            valid_mask = base_mask[:vocab_size_logits]

    for _ in range(max_len - cur_len):
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(input_ids)
                logits = outputs[0][:, -1, :]
        else:
            outputs = model(input_ids)
            logits = outputs[0][:, -1, :]

        if eos_id is not None and pad_id is not None:
            logits = torch.where(
                finished.unsqueeze(1),
                torch.full_like(logits, float("-inf")),
                logits
            )
            logits[finished, pad_id] = 0.0

        need_dist = (expected_token_type == 1) & (~finished)
        if need_dist.any() and valid_mask is not None:
            rows = need_dist.nonzero(as_tuple=False).squeeze(1)
            valid_ids = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

            if valid_ids.numel() > 0:
                sub = logits[rows][:, valid_ids]
                t_d = 0.7
                k_d = 80
                sub = sub / max(1e-6, t_d)
                if 0 < k_d < sub.size(-1):
                    vals, _ = torch.topk(sub, k_d)
                    kth = vals[..., -1, None]
                    sub = torch.where(sub < kth, torch.full_like(sub, float("-inf")), sub)
                p = F.softmax(sub, dim=-1).clamp_min(1e-12)
                p = p / p.sum(dim=-1, keepdim=True)

                if torch.isnan(p).any() or (p.sum(dim=-1) == 0).any():
                    idx = torch.randint(0, valid_ids.size(0),
                                        (rows.size(0),), device=device)
                else:
                    idx = torch.multinomial(p, num_samples=1).squeeze(1)

                chosen = valid_ids[idx]
                logits[rows, :] = float("-inf")
                logits[rows, chosen] = 0.0


        if temp is not None and temp > 0:
            logits = logits / max(1e-6, temp)
        logits = top_k_filtering(logits, top_k)
        probs = F.softmax(logits, dim=-1).clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs, num_samples=1)


        if eos_id is not None and pad_id is not None:
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_id),
                next_token,
            )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_id)

        if bool(finished.all().item()):
            break

        expected_token_type = torch.where(
            finished, expected_token_type, (expected_token_type + 1) % 4
        )

    return input_ids



def get_first_tokens_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--output_tokenizer_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='cond/gens')
    parser.add_argument('--conditions_path', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='simple', choices=['simple'])
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, required=True)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--num_props', type=int, default=0)
    parser.add_argument('--scaffold', action='store_true', default=False)
    parser.add_argument('--auto_fp16to32', action='store_true', default=False)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repeats', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--model', type=str, default='mamba')
    parser.add_argument('--grpo_pt', type=str, default=None)
    parser.add_argument('--prop', type=str, default='alpha')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tok_path = os.path.join(args.output_tokenizer_dir, 'vocab.json')
    tokenizer = load_tokenizer(tok_path, max_length=args.max_len, tokenizer_type=args.tokenizer)
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    pad_id = vocab.get('<pad>', 0)
    eos_id = vocab.get('</s>', None)

    if args.model == 'mamba':
        from model.mamba import MambaConfig, MambaLMHeadModel

        mcfg = MambaConfig(
            d_model=args.n_embd,
            n_layer=args.n_layer,
            vocab_size=vocab_size,
            num_props=args.num_props,
            scaffold=args.scaffold,
            isconditional=True,
            auto_fp16to32=args.auto_fp16to32,
        )
        model = MambaLMHeadModel(mcfg).to(device)
    elif args.model == 'gpt':
        from model.gpt import GPT, GPTConfig

        gcfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.max_len, 
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            isconditional=True,
            scaffold=args.scaffold,
            scaffold_maxlen=0,
            lstm=False,
        )
        model = GPT(gcfg).to(device)

    if args.grpo_pt is not None:
        grpo_path = args.grpo_pt
        if not os.path.isfile(grpo_path):
            raise FileNotFoundError(f'GRPO checkpoint not found: {grpo_path}')
        ckpt = torch.load(grpo_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
    else:
        ckpt_path = f'cond/{args.prop}_weights/{args.run_name}.pt'
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f'[warn] load_state_dict loose: missing={len(missing)}, unexpected={len(unexpected)}')
        model = model.to(device).eval()

    first_tokens = get_first_tokens_from_file(args.conditions_path)
    condition = os.path.splitext(os.path.basename(args.conditions_path))[0].partition('_')[0]

    os.makedirs(args.save_path, exist_ok=True)
    merged_out = os.path.join(args.save_path, condition + '_generated.txt')
    with open(merged_out, 'w', encoding='utf-8') as fout:
        for idx, first_token in enumerate(first_tokens):
            total = args.repeats
            written = 0
            prefix_value = _extract_value(first_token)
            while written < total:
                cur_bs = min(args.batch_size, total - written)
                ids_list = [torch.tensor(tokenizer.generation_encode(first_token), dtype=torch.long)
                            for _ in range(cur_bs)]
                input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id).to(device)
                prefix_len = input_ids.size(1)

                out_ids = generate_autoregressive(
                    model, input_ids,
                    max_len=args.max_len,
                    temp=args.temperature,
                    top_k=args.topk,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    tokenizer=tokenizer,
                    use_amp=True,
                    amp_dtype=torch.float16,
                )
                lines = []
                for seq in out_ids:
                    tail = tokenizer.decode(seq[prefix_len:].cpu())
                    lines.append(f"{prefix_value} {tail}\n")
                fout.writelines(lines)
                written += cur_bs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f'[cond {idx}] wrote {written} samples')

    print(f'All samples saved to: {merged_out}')


if __name__ == '__main__':
    main()
