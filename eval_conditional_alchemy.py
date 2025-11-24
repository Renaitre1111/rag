import argparse
from os.path import join
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join('../../')))
from qm9.property_prediction import main_qm9_prop
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes

ALCHEMY_ATOM_MAP = {
    'H': 0,
    'C': 1, 
    'N': 2,
    'O': 3, 
    'F': 4,
    'P': 5,
    'S': 6,
    'Cl': 7,
    'Br': 8,
    'I': 9,
}
NUM_ATOM_TYPES = 10

def get_classifier(dir_path='', device='cpu'):
    args_path = join(dir_path, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Classifier args not found at {args_path}")
        
    with open(args_path, 'rb') as f:
        args_classifier = pickle.load(f)
        
    args_classifier.device = device
    print(f"Loading classifier from {dir_path}...")
    
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

    ckpt_path = join(dir_path, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = join(dir_path, 'best_checkpoint.npy')
    
    print(f"Loading weights from {ckpt_path}")
    classifier_state_dict = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(classifier_state_dict)
    return classifier.to(device).eval()


def parse_generated_samples(input_path, max_num_atoms=100):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    num_samples = len(lines)

    one_hot = torch.zeros((num_samples, max_num_atoms, NUM_ATOM_TYPES), dtype=torch.float32)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=torch.float32)
    node_mask = torch.zeros((num_samples, max_num_atoms), dtype=torch.float32)
    y = torch.zeros((num_samples), dtype=torch.float32)

    idx = 0
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0

    for i, line in enumerate(tqdm(lines, desc="Parsing generated samples")):
        line = line.strip()
        if not line: continue
        
        split = np.array(line.split())
        
        try:
            prop = float(split[0])
            mol_data = split[1:]
            
            mol_data = mol_data.reshape(-1, 4)
        except:
            count_invalid_len += 1
            continue

        seq = mol_data[:, 0]
        
        try:
            atom_indices = [ALCHEMY_ATOM_MAP[a] for a in seq]
            one_hot_emb = torch.nn.functional.one_hot(torch.tensor(atom_indices), NUM_ATOM_TYPES)
        except:
            count_invalid_seq += 1
            continue

        try:
            spherical_coords = mol_data[:, 1:]
            
            d = spherical_coords[:, 0].astype(float)
            
            theta = np.array([s[:-1] for s in spherical_coords[:, 1]]).astype(float)
            phi = np.array([s[:-1] for s in spherical_coords[:, 2]]).astype(float)
            
            invariant_coords = np.stack((
                d * np.sin(theta) * np.cos(phi), 
                d * np.sin(theta) * np.sin(phi), 
                d * np.cos(theta)
            )).T
        except Exception as e:
            # print(f"Coords error: {e}")
            count_invalid_coords += 1
            continue
            
        # 5. 存入 Tensor
        num_nodes = len(seq)
        if num_nodes > max_num_atoms:
            continue
            
        one_hot[idx, :num_nodes] = one_hot_emb
        x[idx, :num_nodes] = torch.tensor(invariant_coords, dtype=torch.float32)
        node_mask[idx, :num_nodes] = 1.
        y[idx] = prop
        idx += 1

    one_hot = one_hot[:idx]
    x = x[:idx]
    node_mask = node_mask[:idx]
    y = y[:idx]

    print(f'Successfully parsed {idx} molecules.')
    print(f'Invalid counts -> Length: {count_invalid_len}, Seq: {count_invalid_seq}, Coords: {count_invalid_coords}')
    
    if idx == 0:
        raise RuntimeError("CRITICAL: No valid molecules parsed! Check format.")

    molecules = {'one_hot': one_hot, 'positions': x, 'atom_mask': node_mask, 'y': y}
    return molecules


edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)
    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    edges_dic[n_nodes][batch_size] = edges
    return edges


class CondMol(Dataset):
    def __init__(self, txt_file, max_num_atoms=40):
        self.data = parse_generated_samples(txt_file, max_num_atoms=max_num_atoms)

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


def collate_fn(batch):
    batch_size = len(batch)
    keys = batch[0].keys()
    collated = {k: torch.stack([b[k] for b in batch]) for k in keys}
    
    atom_mask = collated['atom_mask']
    n_nodes = atom_mask.size(1)
    
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(n_nodes, device=edge_mask.device, dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    
    collated['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    return collated


def eval(model, loader, mean, mad, device, log_interval=20):
    loss_l1 = nn.L1Loss()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    
    print(f"Evaluating with Mean: {mean:.4f}, MAD: {mad:.4f}")
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            model.eval()
            
            batch_size, n_nodes = data['atom_mask'].size()
            
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
            edge_mask = data['edge_mask'].to(device, torch.float32)
            nodes = data['one_hot'].view(batch_size * n_nodes, -1).to(device, torch.float32)
            
            label = data['y'].to(device, torch.float32)
            
            edges = get_adj_matrix(n_nodes, batch_size, device)

            pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            
            if pred.shape != label.shape:
                label = label.view_as(pred)

            pred_real = pred * mad + mean
            
            loss = loss_l1(pred_real, label)
            
            if i == 0:
                 print(f"Sample 0 Pred: {pred_real[0].item():.4f} | True Cond: {label[0].item():.4f}")

            res['loss'] += loss.item() * batch_size
            res['counter'] += batch_size
            res['loss_arr'].append(loss.item())

            if i % log_interval == 0:
                 if len(res['loss_arr']) > 0:
                    curr_loss = sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])
                    print("Iteration %d \t MAE: %.4f" % (i, curr_loss))
    
    if res['counter'] == 0:
        return 0.0

    return res['loss'] / res['counter']


def main_quantitative(args):
    classifier = get_classifier(args.classifiers_path, args.device)
    
    mean = torch.tensor(args.train_mean, device=args.device)
    mad = torch.tensor(args.train_mad, device=args.device)

    dataset = CondMol(txt_file=args.generated_path, max_num_atoms=args.max_num_atoms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Evaluating {len(dataset)} samples...")
    loss = eval(classifier, dataloader, mean, mad, args.device, args.log_interval)

    print(f"Final MAE ({args.property}): {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_path', type=str, required=True)
    parser.add_argument('--classifiers_path', type=str, required=True)
    parser.add_argument('--property', type=str, default='gap')
    parser.add_argument('--train_mean', type=float, required=True)
    parser.add_argument('--train_mad', type=float, required=True)
    parser.add_argument('--max_num_atoms', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.device}")
    else:
        args.device = torch.device("cpu")

    main_quantitative(args)