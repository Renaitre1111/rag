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

@torch.no_grad()
def generate_rag(model, ref_egnn_vec, target_prop, tokenizer, 
                 max_len=160, temperature=1.0, top_k=50, device='cuda'):
    model.eval()
    