import torch
from torch.utils.data import Dataset
import numpy as np
import re
import json
import torch.nn.functional as F

class SoftRAGDataset(Dataset):
    def __init__(self, target_texts, target_props, tokenizer, db_emb_path, db_prop_path, retrieval_path, split='train', max_len=512, mean=None, std=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        self.target_texts = target_texts 

        raw_target_props = torch.tensor([float(str(p).strip()) for p in target_props], dtype=torch.float)

        emb_data = np.load(db_emb_path)
        self.db_embeddings = torch.from_numpy(emb_data['embeddings']).float()

        with open(db_prop_path, 'r') as f:
            raw_db_props = [float(line.strip().split()[0]) for line in f if line.strip()]
        raw_db_props = torch.tensor(raw_db_props, dtype=torch.float)

        if mean is None or std is None:
            self.mean = torch.mean(raw_db_props)
            self.std = torch.std(raw_db_props)
        else:
            self.mean = mean
            self.std = std

        self.target_props = (raw_target_props - self.mean) / self.std
        self.db_props = (raw_db_props - self.mean) / self.std

        ret_data = np.load(retrieval_path)
        self.retrieved_indices = torch.from_numpy(ret_data['indices']).long() # [N, K]

        if 'sims' in ret_data:
            self.retrieved_sims = torch.from_numpy(ret_data['sims']).float()  # [N, K]
        else:
            self.retrieved_sims = None
    
    def __len__(self):
        return len(self.target_texts)
    
    def __getitem__(self, idx):
        tgt_ids_list = self.tokenizer.encode(self.target_texts[idx])
        target_input_ids = torch.tensor(tgt_ids_list, dtype=torch.long)

        target_prop = self.target_props[idx].view(1) # [1]

        indices = self.retrieved_indices[idx] # [K]

        if self.retrieved_sims is not None:
            sims = self.retrieved_sims[idx] # [K]
            temperature = 0.5 
            probs = F.softmax(sims / temperature, dim=0)
            
            sample_idx = torch.multinomial(probs, 1).item()
            final_ref_idx = indices[sample_idx]
        else:
            final_ref_idx = indices[0]

        ref_egnn_vec = self.db_embeddings[final_ref_idx] # [EGNN_Dim]
        ref_prop = self.db_props[final_ref_idx].view(1)  # [1]

        return ref_egnn_vec, target_input_ids, ref_prop, target_prop



class Mol3DDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, conditions=None, conditions_split_id=None, prefix_sep=' '):
        self.texts = texts
        self.conditions = conditions
        self.conditions_split_id = conditions_split_id
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix_sep = prefix_sep

    def __len__(self):
        return len(self.texts)
    
    def _token_count(self, text: str) -> int:
        if hasattr(self.tokenizer, "split_text"):
            return len(self.tokenizer.split_text(text))
        else:
            return len(text.split())

    def __getitem__(self, idx):
        text = self.texts[idx].strip()
        prefix = self.conditions[idx].strip() if self.conditions is not None else ""
        
        if prefix:
            full_text = prefix + self.prefix_sep + text
            prefix_for_count = prefix
        else:
            full_text = text
            prefix_for_count = ""
        
        if self.conditions_split_id is not None:
            condition_split_id = int(self.conditions_split_id[idx].strip())
        else:
            condition_split_id = self._token_count(prefix_for_count) if prefix_for_count else 0
        
        encoded_text = self.tokenizer.batch_encode_plus([full_text])
        raw_input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.long).squeeze(0)
        input_ids = raw_input_ids[:-1]
        targets = raw_input_ids[1:]
        return input_ids, targets, condition_split_id


class SimpleTokenizer:
    def __init__(self, max_length):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.count = 4
        self.max_length = max_length

    def fit_on_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.fit_on_text(line.strip())

    def fit_on_text(self, text):
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.count
                self.count += 1

    def encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence + [self.vocab["</s>"]]
        padding_length = self.max_length - len(sequence)

        if padding_length > 0:
            sequence.extend([self.vocab["<pad>"]] * padding_length)

        return sequence[:self.max_length]

    def decode(self, token_ids):
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (token_ids == self.vocab["</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"
        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence
        return sequence

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file)

    def token_decode(self, token_id):
        return self.reverse_vocab.get(token_id, "<unk>")

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            self.vocab = json.load(file)
            self.count = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_decoder_func = np.vectorize(self.token_decode)

    def batch_encode_plus(self, texts):
        encodings = self.encode_batch(texts)
        attention_masks = [[float(token != self.vocab["<pad>"]) for token in encoding] for encoding in encodings]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks
        }

