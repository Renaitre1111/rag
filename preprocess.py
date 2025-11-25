import argparse
import numpy as np
import os
from tqdm import tqdm
from dataset import SimpleTokenizer

def tokenize(txt_path, tokenizer, save_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    token_ids = []
    for line in tqdm(lines):
        ids = tokenizer.encode(line.strip())
        token_ids.append(ids)
    
    data = np.array(token_ids, dtype=np.uint16)
    np.save(save_path, data)
    print(f"Saved encoded data to {save_path} (Shape: {data.shape})")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True, help="Path to input txt (without .txt extension)")
    parser.add_argument('--tokenizer_dir', required=True)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()

    tokenizer = SimpleTokenizer(args.max_len)
    vocab_path = os.path.join(args.tokenizer_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        print("Error: Vocab not found. Please run train.py once to generate tokenizer vocab.")
        return

    train_txt = args.root_path + '.txt'
    train_npy = args.root_path + '_ids.npy'
    tokenize(train_txt, tokenizer, train_npy)

if __name__ == "__main__":
    main()