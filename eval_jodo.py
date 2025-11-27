try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import argparse
import torch
import numpy as np
from tqdm import tqdm
from eval_follow_edm.datasets_config import get_dataset_info
from eval_follow_jodo.stability import get_edm_metric, get_2D_edm_metric
from eval_follow_jodo.mose_metric import get_moses_metrics
import re
import math
import logging
from rdkit.Chem import rdDetermineBonds, rdmolops
from eval_follow_edm import rdkit_functions

dict_alchemy = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 
                5: 'P', 6: 'S', 7: 'Cl', 8: 'Br', 9: 'I'}

def write_xyz_file(atom_types, atom_coordinates, file_path):
    with open(file_path, 'w') as file:
        num_atoms = len(atom_types)
        file.write(f"{num_atoms}\n")
        file.write('\n')

        for atom_type, coords in zip(atom_types, atom_coordinates):
            x, y, z = coords
            file.write(f"{atom_type} {np.format_float_positional(x)} {np.format_float_positional(y)} {np.format_float_positional(z)}\n")

def spherical_seq_for_eval(dataset_name='alchemy',
                           input_path='generated_samples.txt',
                           remove_h=False):
    dict_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
                'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
    current_atom_dict = dict_alchemy 
    
    mol_for_jodo_eval_rdkit_e = []

    try:
        with open(input_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return None, []

    lines = [l for l in lines if l.strip()]
    
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    
    for num_line, line in enumerate(tqdm(lines, disable=False, desc="Parsing Generated File")):
        mol = np.array(line.split())

        if len(mol) % 4 != 0:
            count_invalid_len += 1
            continue
            
        try:
            mol = mol.reshape(-1, 4)
        except:
            count_invalid_len += 1
            continue
            
        seq = mol[:, 0]
        try:
            x = torch.tensor([dict_map[key] for key in seq])
        except KeyError:
            count_invalid_seq += 1
            continue
            
        try:
            spherical_coords = mol[:, 1:]
            d = np.array([s.replace('°', '') for s in spherical_coords[:, 0]]).astype(float)
            theta = np.array([s.replace('°', '') for s in spherical_coords[:, 1]]).astype(float)
            phi = np.array([s.replace('°', '') for s in spherical_coords[:, 2]]).astype(float)
            
            invariant_coords = np.stack((d * np.sin(theta) * np.cos(phi), d * np.sin(theta) * np.sin(phi), d * np.cos(theta))).T
        except:
            count_invalid_coords += 1
            continue   
        
        pos = torch.tensor(invariant_coords, dtype=torch.float32)
    
        file_path = 'temp_eval_mol.xyz' 
        ori_z = x
        ori_coords = pos
        
        write_xyz_file([current_atom_dict[key.item()] for key in ori_z], ori_coords, file_path)
        
        raw_mol = Chem.MolFromXYZFile(file_path)
        if raw_mol is None:
            continue
            
        mol = Chem.Mol(raw_mol)
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except:
            pass
        
        e_rdkit = torch.tensor(rdmolops.GetAdjacencyMatrix(mol), dtype=torch.float32)
        mol_for_jodo_eval_rdkit_e.append((pos, x, e_rdkit, x))
    
    print('Parsing Stats | Invalid Len: %d, Invalid Type: %d, Invalid Coords: %d' % 
          (count_invalid_len, count_invalid_seq, count_invalid_coords))
    print('Successfully Parsed Molecules:', len(mol_for_jodo_eval_rdkit_e))
    return None, mol_for_jodo_eval_rdkit_e

def eval(data, dataset_info, train_smiles_ref, test_smiles_ref=None):
    metric_fn = get_edm_metric(dataset_info, train_smiles_ref) 
    
    try:
        stability_res, rdkit_res, sample_rdmols = metric_fn(data)
    except ValueError as e:
        print(f"Error in metric calculation: {e}")
        print("CRITICAL ERROR: Dataset info missing bond radius keys.")
        return

    print('\n==================== Results (Validity) ====================')
    print(f"Validity: {rdkit_res['Validity']:.4f}") 
    print(f"Complete: {rdkit_res['Complete']:.4f}")
    print(f"Atom Stability: {stability_res['atom_stable']:.4f}")
    print(f"Mol Stability:  {stability_res['mol_stable']:.4f}")

    # 2. 计算 Unique 和 Novelty
    metric_2d_fn = get_2D_edm_metric(dataset_info, train_smiles_ref)
    _, rdkit_res_2d, complete_rdmols = metric_2d_fn(data)
    
    print('\n==================== Results (Unique & Novelty) ====================')
    print(f"Unique (of Valid): {rdkit_res_2d['Unique']:.4f}")
    print(f"Novelty (of Unique): {rdkit_res_2d['Novelty']:.4f}")

    # 3. 计算 MOSES 指标
    if len(complete_rdmols) > 0:
        ref_smiles = test_smiles_ref if test_smiles_ref is not None else train_smiles_ref
        mose_metric = get_moses_metrics(ref_smiles, n_jobs=16, device='cpu')
        mose_res = mose_metric(complete_rdmols)
        
        print('\n==================== Results (MOSES / Distribution) ====================')
        print(f"FCD: {mose_res['FCD']:.4f}")
        print(f"IntDiv: {mose_res['IntDiv']:.4f}")
    else:
        print("\n[Warning] No valid molecules found to compute MOSES metrics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="ours") 
    parser.add_argument('--input_path', type=str, required=True, help="Path to generated samples (.txt)")
    parser.add_argument('--dataset', type=str, default="alchemy")
    parser.add_argument('--ref_smiles_path', type=str, default="alchemy_train_gen_smiles.npy", help="Path to reference smiles")
    args = parser.parse_args()

    dataset_info = {
        'name': 'alchemy',
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9},
        'atom_decoder': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        'atom_types': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1, 5: 5, 6: 6, 7: 1, 8: 1, 9: 1}, 
        'max_n_nodes': 40,
        'with_h': True,
        'bond1_radius': {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57, 'P': 107, 'S': 105, 'Cl': 102, 'Br': 120, 'I': 139},
        'bond1_stdv': {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3, 'P': 3, 'S': 3, 'Cl': 3, 'Br': 3, 'I': 3},
        'bond2_radius': {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59, 'P': 96, 'S': 94, 'Cl': 90, 'Br': 110, 'I': 130},
        'bond3_radius': {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53, 'P': 88, 'S': 85, 'Cl': -1000, 'Br': -1000, 'I': -1000},
        'n_nodes': {10: 1}, 'distances': [0], 'colors_dic': [], 'radius_dic': []
    }
    
    print('============================================================')
    print('Using Hardcoded Alchemy Dataset Info (Fixed Bond Radius).')

    processed_mols, processed_mols_e_rdkit = spherical_seq_for_eval(
        dataset_name=args.dataset, 
        input_path=args.input_path, 
        remove_h=False
    )

    print(f"Loading reference SMILES from {args.ref_smiles_path}...")
    try:
        ref_smiles_data = np.load(args.ref_smiles_path, allow_pickle=True)
        if isinstance(ref_smiles_data, np.ndarray):
            ref_smiles_list = ref_smiles_data.tolist()
        else:
            ref_smiles_list = list(ref_smiles_data)
        
        ref_smiles_list = [s for s in ref_smiles_list if s is not None]
        print(f"Loaded {len(ref_smiles_list)} reference SMILES.")
    except Exception as e:
        print(f"[Error] Failed to load reference SMILES: {e}")
        exit()

    if processed_mols_e_rdkit is not None and len(processed_mols_e_rdkit) > 0:
        eval(processed_mols_e_rdkit, dataset_info, train_smiles_ref=ref_smiles_list, test_smiles_ref=ref_smiles_list)
    else:
        print("[Error] No valid molecules parsed from input file.")