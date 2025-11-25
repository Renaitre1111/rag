import torch
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
from os.path import join
import argparse

def get_classifier(dir_path, atom_types, device='cpu'):
    args_path = join(dir_path, 'args.pickle')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Classifier args not found at {args_path}")
        
    with open(args_path, 'rb') as f:
        args_classifier = pickle.load(f)
        
    args_classifier.device = device
    print(f"Loading classifier from {dir_path}...")
    
    if args_classifier.model_name == 'egnn':
        classifier = EGNN(in_node_nf=atom_types, in_edge_nf=0, 
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
    
    print(f"Loading weights from {ckpt_path}")
    classifier_state_dict = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(classifier_state_dict)
    return classifier.to(device).eval()

def parse_molecules(input_path, atom_map, atom_types, max_num_atoms=100):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    num_samples = len(lines)

    one_hot = torch.zeros((num_samples, max_num_atoms, atom_types), dtype=torch.float32)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=torch.float32)
    node_mask = torch.zeros((num_samples, max_num_atoms), dtype=torch.float32)

    idx = 0

    for i, line in enumerate(tqdm(lines, desc="Parsing molecules")):
        line = line.strip()
        if not line: continue
        
        mol_data = np.array(line.split())
        mol_data = mol_data.reshape(-1, 4)

        seq = mol_data[:, 0]
        
        atom_indices = [atom_map[a] for a in seq]
        one_hot_emb = torch.nn.functional.one_hot(torch.tensor(atom_indices), atom_types)

        spherical_coords = mol_data[:, 1:]
            
        d = spherical_coords[:, 0].astype(float)
            
        theta = np.array([s[:-1] for s in spherical_coords[:, 1]]).astype(float)
        phi = np.array([s[:-1] for s in spherical_coords[:, 2]]).astype(float)
            
        invariant_coords = np.stack((
            d * np.sin(theta) * np.cos(phi), 
            d * np.sin(theta) * np.sin(phi), 
            d * np.cos(theta)
        )).T
            
        num_nodes = len(seq)
        if num_nodes > max_num_atoms:
            continue
            
        one_hot[idx, :num_nodes] = one_hot_emb
        x[idx, :num_nodes] = torch.tensor(invariant_coords, dtype=torch.float32)
        node_mask[idx, :num_nodes] = 1.
        idx += 1

    one_hot = one_hot[:idx]
    x = x[:idx]
    node_mask = node_mask[:idx]

    print(f'Successfully parsed {idx} molecules.')
    molecules = {'one_hot': one_hot, 'positions': x, 'atom_mask': node_mask}
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

class CondMol(Dataset):
    def __init__(self, txt_file, atom_map, atom_types, max_num_atoms=40):
        self.data = parse_molecules(txt_file, atom_map, atom_types, max_num_atoms=max_num_atoms)

    def __len__(self):
        return len(self.data['one_hot'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

def get_embedding(model, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
    h = model.embedding(h0)
    
    for i in range(0, model.n_layers):
        if model.node_attr:
            h, _, _ = model._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
        else:
            h, _, _ = model._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=None, n_nodes=n_nodes)

    h = model.node_dec(h)
    
    h = h * node_mask
    h = h.view(-1, n_nodes, model.hidden_nf)
    
    h = torch.sum(h, dim=1)
    
    return h

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a model.")
    parser.add_argument("--dataset", type=str, default='alchemy', help="Dataset to use for embeddings.")
    parser.add_argument("--classifiers_path", type=str, default='./qm9/property_prediction/outputs/alchemy_gap', help="Path to the classifiers directory.")
    parser.add_argument("--input_path", type=str, default='./data/alchemy_seq.txt', help="Path to the generated data file.")
    parser.add_argument("--save_path", type=str, default='alchemy_gap_embeddings.npz', help="Path to save the embeddings.")
    parser.add_argument("--max_num_atoms", type=int, default=40, help="Maximum number of atoms in a molecule.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use for processing.")

    args = parser.parse_args()
    if args.dataset == 'alchemy':
        atom_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
        atom_types = 10

    classifier = get_classifier(args.classifiers_path, atom_types, args.device)

    dataset = CondMol(txt_file=args.input_path, atom_map=atom_map, atom_types=atom_types, max_num_atoms=args.max_num_atoms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    all_embeddings = []

    print("Extracting embeddings...")
    classifier.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            batch_size, n_nodes = data['atom_mask'].size()
            
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(args.device, torch.float32)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(args.device, torch.float32)
            edge_mask = data['edge_mask'].to(args.device, torch.float32)
            nodes = data['one_hot'].view(batch_size * n_nodes, -1).to(args.device, torch.float32)
            
            edges = get_adj_matrix(n_nodes, batch_size, args.device)

            embedding = get_embedding(
                classifier, 
                h0=nodes, 
                x=atom_positions, 
                edges=edges, 
                edge_attr=None, 
                node_mask=atom_mask, 
                edge_mask=edge_mask, 
                n_nodes=n_nodes
            )
            
            all_embeddings.append(embedding.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"Total Molecules: {all_embeddings.shape[0]}")
    print(f"Embedding Shape: {all_embeddings.shape}")
    
    np.savez(args.save_path, embeddings=all_embeddings)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()
