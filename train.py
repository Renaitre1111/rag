import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from trainer import Trainer, TrainerConfig
from model.mamba import MambaLMHeadModel, MambaConfig
from model.poetic import PoeticMamba                
from dataset import Mol3DDataset, SimpleTokenizer, SoftRAGDataset
import math
import re
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_dataset_from_files(root_path, split, ids):
    dataset = []
    for id in range(ids):
        with open(root_path+'_'+split+'_'+str(id)+'.txt', 'r') as file:
            dataset.extend(file.readlines())
            print('loaded dataset from '+root_path+'_'+split+'_'+str(id)+'.txt')
    return dataset

def load_tokenizer(tokenizer_path,max_length):
    tokenizer = SimpleTokenizer(max_length)
    tokenizer.load_vocab(tokenizer_path)
    return tokenizer

def run_DDP(rank, world_size, args):
    setup(rank, world_size)
    run(args, rank)
    cleanup()

def run(args, rank=None):
    set_seed(args.seed)
    os.environ["WANDB_MODE"] = "dryrun"

    print("making tokenizer")
    max_len = args.max_len
    tokenizer_path = args.tokenizer_dir
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    tokenizer_path = args.tokenizer_dir + "/vocab.json"

    tokenizer = SimpleTokenizer(max_length=max_len)
    tokenizer.fit_on_file(args.root_path)
    if rank is None or rank == 0:
        tokenizer.save_vocab(tokenizer_path)
        print(f"tokenizer saved to {tokenizer_path}")

    vocab_size = tokenizer.get_vocab_size()

    print("making dataset")
    with open(args.root_path, 'r') as f:
        train_texts = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(args.prop_path, 'r') as f:
        train_props = f.readlines()

    if args.load_checkpoint_path:
        load_checkpoint_path = args.load_checkpoint_path
    else:
        load_checkpoint_path = None
    
    if args.model == 'poetic':
        train_dataset = SoftRAGDataset(
            target_texts=train_texts,
            target_props=train_props,
            tokenizer=tokenizer,
            db_emb_path=args.db_emb_path,     
            db_prop_path=args.db_prop_path,    
            retrieval_path=args.retrieval_path,
            split='train',
            max_len=max_len
        )
    else:
        train_dataset = Mol3DDataset(train_texts, tokenizer, max_len)

    print(f"train dataset size: {len(train_dataset)}")

    print("loading model")
    if args.dist:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'poetic':
        from model.mamba import MambaConfig
        from model.poetic import PoeticMamba
        mamba_config = MambaConfig(d_model=args.n_embd, n_layer=args.n_layer, vocab_size=vocab_size,
                                   auto_fp16to32=args.auto_fp16to32)
        
        model = PoeticMamba(mamba_config, egnn_dim=args.egnn_dim, device=device)

    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        state_dict = torch.load(model_path)
        if args.model == 'poetic' and not list(state_dict.keys())[0].startswith('backbone.'):
             new_state_dict = {'backbone.' + k: v for k, v in state_dict.items() if 'lm_head' not in k}
             new_state_dict['lm_head.weight'] = state_dict['lm_head.weight']
             model.load_state_dict(new_state_dict, strict=False)
        else:
             model.load_state_dict(state_dict, strict=False)

    if load_checkpoint_path is not None:
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
    print('total params:', sum(p.numel() for p in model.parameters()))

    if rank is None or rank == 0:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    tconf = TrainerConfig(
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=0.1 * len(train_dataset) * max_len,
        final_tokens=args.max_epochs * len(train_dataset) * max_len,
        num_workers=args.num_workers, 
        ckpt_path=args.save_path,
        run_name=args.run_name, 
        block_size=max_len, 
        generate=False, 
        save_start_epoch=args.save_start_epoch,
        grad_norm_clip=args.grad_norm_clip, 
        load_checkpoint_path=load_checkpoint_path,
        save_interval_epoch=args.save_interval_epoch, 
        dist=args.dist, 
        rank=rank
    )
                          
    trainer = Trainer(model, train_dataset, None, tconf)
    df = trainer.train(wandb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--prop', type=str, default='alpha', help="propertie to be used for condition", required=False)
    parser.add_argument('--model', type=str, default='poetic', help="name of the model", required=False)
    parser.add_argument('--tokenizer', type=str, default='simple', help="name of the tokenizer", required=False)
    parser.add_argument('--n_layer', type=int, default=16, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=60, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=32, help="batch size", required=False)
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers for data loaders", required=False)
    parser.add_argument('--save_start_epoch', type=int, default=10, help="save model start epoch", required=False)
    parser.add_argument('--save_interval_epoch', type=int, default=10, help="save model epoch interval", required=False)
    parser.add_argument('--learning_rate', type=float, default=4e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0, help="number of layers in lstm", required=False)
    parser.add_argument('--max_len', type=int, default=512, help="max_len", required=False)
    parser.add_argument('--seed', type=int, default=42, help="seed", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help="gradient norm clipping", required=False)
    parser.add_argument('--auto_fp16to32', action='store_true', default=False, help='Auto casting fp16 tensors to fp32')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help="Path to load training checkpoint", required=False)
    parser.add_argument('--pre_root_path', default=None, help="Path to the pretrain data directory", required=False)
    parser.add_argument('--pre_model_path', default=None, help="Path to the pretrain model", required=False)
    parser.add_argument('--root_path', help="Path to the root data directory", required=True)
    parser.add_argument('--prop_path', help="Path to target properties (.txt)", required=True)
    parser.add_argument('--tokenizer_dir', help="Path to the saved tokenizer directory", required=True)
    parser.add_argument('--save_path', help="Path to save the model checkpoint", required=True)
    parser.add_argument('--dist', action='store_true', default=False, help='use torch.distributed')

    parser.add_argument('--db_emb_path', help="Path to DB EGNN embeddings (.npz)", required=True)
    parser.add_argument('--db_prop_path', help="Path to DB properties (.txt/npy)", required=True)
    parser.add_argument('--retrieval_path', help="Path to retrieval indices & sims (.npz)", required=True)
    parser.add_argument('--egnn_dim', type=int, default=128, help="Dimension of EGNN embeddings")

    args = parser.parse_args()

    if args.dist:
        world_size = torch.cuda.device_count()
        mp.spawn(run_DDP,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        run(args)