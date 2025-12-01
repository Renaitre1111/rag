import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import json
import pickle

sys.path.append(os.path.abspath(os.path.join('../../')))
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
from qm9.property_prediction.models import SchNet
from qm9.property_prediction import prop_utils

PROPERTY_MAP = {"alpha": 0, "gap": 1, "homo": 2, "lumo": 3, "mu": 4, "Cv": 5}
ALLOWED_CHARGES_ALCHEMY = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
ALLOWED_CHARGES_QM9 = [1, 6, 7, 8, 9]

class AlchemyDataset(Dataset):
    def __init__(self, npz_path, property_name, dataset='alchemy'):
        super().__init__()
        if dataset == 'qm9':
            self.allowed_charges = ALLOWED_CHARGES_QM9
        else:
            self.allowed_charges = ALLOWED_CHARGES_ALCHEMY
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}")
            
        data = np.load(npz_path)
        self.positions = torch.tensor(data['positions'], dtype=torch.float32).clone()
        self.charges = torch.tensor(data['charges'], dtype=torch.long).clone()
        self.num_atoms = torch.tensor(data['num_atoms'], dtype=torch.long).clone()
        self.props = torch.tensor(data['props'], dtype=torch.float32).clone()
        
        if property_name not in PROPERTY_MAP:
            raise ValueError(f"Property {property_name} not found in {list(PROPERTY_MAP.keys())}")
            
        self.prop_idx = PROPERTY_MAP[property_name]
        self.charge_to_idx = {z: i for i, z in enumerate(self.allowed_charges)}
        self.num_atom_types = len(self.allowed_charges)

    def __len__(self):
        return len(self.props)

    def __getitem__(self, idx):
        n_nodes = self.num_atoms[idx]
        pos = self.positions[idx].clone() 
        charges = self.charges[idx] 
        label = self.props[idx, self.prop_idx]

        if pos.size(0) > n_nodes:
            pos[n_nodes:] = 1000.0

        one_hot = torch.zeros((charges.size(0), self.num_atom_types), dtype=torch.float32)
        for k, z in enumerate(charges):
            if k >= n_nodes: break 
            z_val = z.item()
            if z_val in self.charge_to_idx:
                one_hot[k, self.charge_to_idx[z_val]] = 1.0

        atom_mask = torch.zeros((pos.size(0), 1), dtype=torch.float32)
        atom_mask[:n_nodes] = 1.0
        
        edge_mask = atom_mask.mm(atom_mask.t()) 
        edge_mask = edge_mask.view(-1, 1) 

        return {
            'positions': pos,
            'one_hot': one_hot,
            'atom_mask': atom_mask,
            'edge_mask': edge_mask,
            'charges': charges,
            'label': label, 
            'num_atoms': n_nodes
        }

loss_l1 = nn.L1Loss()
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
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)
    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    edges_dic[n_nodes][batch_size] = edges
    return edges

def compute_mean_mad(dataset):
    values = dataset.props[:, dataset.prop_idx]
    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    return mean, mad

def run_epoch(model, epoch, loader, mean, mad, device, partition='train', optimizer=None, lr_scheduler=None, log_interval=20):
    if partition == 'train':
        model.train()
        if optimizer: optimizer.zero_grad()
    else:
        model.eval()

    res = {'loss': 0, 'counter': 0, 'loss_arr': []}
    
    for i, data in enumerate(loader):
        batch_size, n_nodes, _ = data['positions'].size()
        
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, torch.float32)
        nodes = data['one_hot'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        label = data['label'].to(device, torch.float32)

        edges = get_adj_matrix(n_nodes, batch_size, device)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)

        if pred.shape != label.shape:
            label = label.view_as(pred)

        if partition == 'train':
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad() 
        else:
            loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            avg_loss = sum(res['loss_arr'][-10:]) / len(res['loss_arr'][-10:])
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, avg_loss))

    if partition == 'train' and lr_scheduler:
        lr_scheduler.step()
        
    return res['loss'] / res['counter']

def get_model(args, in_node_nf):
    if args.model_name == 'egnn':
        model = EGNN(in_node_nf=in_node_nf, in_edge_nf=0, hidden_nf=args.nf, device=args.device, 
                     n_layers=args.n_layers, coords_weight=1.0,
                     attention=args.attention, node_attr=args.node_attr)
    elif args.model_name == 'schnet':
        model = SchNet(in_node_nf=in_node_nf, hidden_nf=args.nf, device=args.device,
                       n_interactions=args.n_interactions, 
                       n_gaussians=args.n_gaussians, 
                       cutoff=args.cutoff,
                       act_fn=nn.SiLU())
    elif args.model_name == 'naive':
        model = Naive(device=args.device)
    elif args.model_name == 'numnodes':
        model = NumNodes(device=args.device)
    else:
        raise Exception("Wrong model name %s" % args.model_name)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='alchemy_run')
    parser.add_argument('--data_path', type=str, default='alchemy')
    parser.add_argument('--property', type=str, default='gap', choices=list(PROPERTY_MAP.keys()))
    parser.add_argument('--dataset', type=str, default='alchemy', choices=['alchemy', 'qm9'])
    
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-16)
    
    parser.add_argument('--model_name', type=str, default='egnn')
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--attention', type=int, default=1)
    parser.add_argument('--node_attr', type=int, default=0)
    # SchNet
    parser.add_argument('--n_interactions', type=int, default=6, help='number of interaction blocks for SchNet')
    parser.add_argument('--n_gaussians', type=int, default=50, help='number of gaussians for SchNet distance expansion')
    parser.add_argument('--cutoff', type=float, default=10.0, help='cutoff radius for SchNet')
    
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--outf', type=str, default='outputs')
    parser.add_argument('--save_model', type=eval, default=True)
    
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    print(args)

    prop_utils.makedir(args.outf)
    prop_utils.makedir(args.outf + "/" + args.exp_name)

    train_file = os.path.join(args.data_path, "alchemy_train_eval.npz")
    test_file = os.path.join(args.data_path, "alchemy_train_gen.npz")
    val_file = os.path.join(args.data_path, "alchemy_valid.npz")

    print(f"Loading Train: {train_file}")
    train_set = AlchemyDataset(train_file, args.property)
    
    print(f"Loading Test: {test_file}")
    test_set = AlchemyDataset(test_file, args.property)
    
    print(f"Loading Valid: {val_file}")
    val_set = AlchemyDataset(val_file, args.property)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    mean, mad = compute_mean_mad(train_set)
    mean = mean.to(device)
    mad = mad.to(device)
    
    print(f"Property: {args.property}")
    print(f"Mean (Train): {mean.item():.4f}")
    print(f"MAD (Train):  {mad.item():.4f}")
    
    in_node_nf = train_set.num_atom_types
    model = get_model(args, in_node_nf=in_node_nf)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    for epoch in range(0, args.epochs):
        train_loss = run_epoch(model, epoch, train_loader, mean, mad, device, 
                               partition='train', optimizer=optimizer, lr_scheduler=lr_scheduler, 
                               log_interval=args.log_interval)
        
        if epoch % args.test_interval == 0:
            val_loss = run_epoch(model, epoch, val_loader, mean, mad, device, 
                                 partition='valid', log_interval=args.log_interval)
            test_loss = run_epoch(model, epoch, test_loader, mean, mad, device, 
                                  partition='test', log_interval=args.log_interval)
            
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                if args.save_model:
                    save_path = os.path.join(args.outf, args.exp_name, "best_checkpoint.pth")
                    torch.save(model.state_dict(), save_path)
                    
                    pickle_path = os.path.join(args.outf, args.exp_name, "args.pickle")
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(args, f)
            
            print(f"Val loss: {val_loss:.4f} \t Test loss: {test_loss:.4f} \t Epoch {epoch}")
            print(f"Best: Val loss: {res['best_val']:.4f} \t Test loss: {res['best_test']:.4f} \t Epoch {res['best_epoch']}")

        json_path = os.path.join(args.outf, args.exp_name, "losses.json")
        with open(json_path, "w") as outfile:
            json.dump(res, outfile, indent=4)