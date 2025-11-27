import numpy as np
from rdkit import Chem
from eval_follow_jodo.bond_analyze import get_bond_order, geom_predictor, allowed_bonds, allowed_fc_bonds
from eval_follow_jodo.rdkit_metric import eval_rdmol
from rdkit.Geometry import Point3D
import copy
import torch

bond_list = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
stability_bonds = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3,
                   Chem.rdchem.BondType.AROMATIC: 1.5}

def get_bond_order_local(atom1_sym, atom2_sym, dist, dataset_info):
    r1 = dataset_info['bond1_radius']
    r2 = dataset_info['bond2_radius']
    r3 = dataset_info['bond3_radius']
    st = dataset_info['bond1_stdv'] 

    if atom1_sym not in r1 or atom2_sym not in r1:
        return 0 
    margin1 = st.get(atom1_sym, 0) + st.get(atom2_sym, 0)
    thresh1 = (r1[atom1_sym] + r1[atom2_sym] + margin1) / 100.0
    
    thresh2 = (r2.get(atom1_sym, -1000) + r2.get(atom2_sym, -1000) + margin1) / 100.0
    thresh3 = (r3.get(atom1_sym, -1000) + r3.get(atom2_sym, -1000) + margin1) / 100.0

    if dist < thresh3:
        return 3
    elif dist < thresh2:
        return 2
    elif dist < thresh1:
        return 1
    return 0

# Stability and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    """Look up for bond types and construct a Rdkit Mol"""
    
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # convert to RDKit Mol, add atom first
    mol = Chem.RWMol()
    for atom in atom_type:
        # atom 是 tensor index
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            
            atom1_sym = atom_decoder[atom_type[i]]
            atom2_sym = atom_decoder[atom_type[j]]

            if 'qm9' in dataset_info['name']:
                order = get_bond_order(atom1_sym, atom2_sym, dist)
            elif 'geom' in dataset_info['name']:
                # geom logic
                pair = sorted([atom_type[i], atom_type[j]])
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            elif 'alchemy' in dataset_info['name']:
                order = get_bond_order_local(atom1_sym, atom2_sym, dist, dataset_info)
            else:
                if 'bond1_radius' in dataset_info:
                    order = get_bond_order_local(atom1_sym, atom2_sym, dist, dataset_info)
                else:
                    raise ValueError('Fail to get dataset bond info.')

            nr_bonds[i] += order
            nr_bonds[j] += order
            
            # add bond to RDKIT Mol
            if order > 0:
                mol.AddBond(i, j, bond_list[order])

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        atom_sym = atom_decoder[atom_type_i]

        idx = atom_type_i.item()
        if 'atom_types' in dataset_info and idx in dataset_info['atom_types']:
            allowed_valency = dataset_info['atom_types'][idx]
            is_stable = (allowed_valency == nr_bonds_i)
        elif atom_sym in allowed_bonds:
            possible_bonds = allowed_bonds[atom_sym]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == nr_bonds_i
            else:
                is_stable = nr_bonds_i in possible_bonds
        else:
            is_stable = False 

        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_sym, nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x), mol

def check_2D_stability(positions, atom_types, formal_charges, edge_types, dataset_info):
    """Convert the generated tensors to rdkit mols and check stability"""
    dataset_name = dataset_info['name']
    atom_decoder = dataset_info['atom_decoder']
    if 'atom_fc_num' in dataset_info:
        atom_fcs = dataset_info['atom_fc_num']
    else:
        atom_fcs = {}
    atom_num = atom_types.size(0)

    # convert to rdkit mol
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    if formal_charges.shape[-1] == 0:
        formal_charges = torch.zeros_like(atom_types)

    for atom_id, fc in enumerate(formal_charges):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_str = atom.GetSymbol()
        if fc != 0:
            atom_fc = atom_str + str(fc.item())
            if atom_fc in atom_fcs:
                atom.SetFormalCharge(fc.item())

    if positions is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

    edge_index = torch.nonzero(edge_types)
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        if src < dst:
            order = edge_types[src, dst]
            mol.AddBond(src.item(), dst.item(), bond_list[int(order)])

    if dataset_name not in ['GeomDrug', 'QM9', 'qm9', 'geom']:
        return 0, 0, atom_num, mol
    return 0, 0, atom_num, mol


def get_edm_metric(dataset_info, train_smiles=None):
    def edm_metric(processed_list):
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        rd_mols = []
        for mol in processed_list:
            pos, atom_type = mol[:2]
            # 调用修改后的 check_stability
            validity_res = check_stability(pos, atom_type, dataset_info)

            molecule_stable += int(validity_res[0])
            nr_stable_bonds += int(validity_res[1])
            n_atoms += int(validity_res[2])
            rd_mols.append(validity_res[3])

        fraction_mol_stable = molecule_stable / float(len(processed_list)) if len(processed_list) > 0 else 0
        fraction_atm_stable = nr_stable_bonds / float(n_atoms) if n_atoms > 0 else 0
        stability_dict = {
            'mol_stable': fraction_mol_stable,
            'atom_stable': fraction_atm_stable,
        }

        rdkit_dict = eval_rdmol(rd_mols, train_smiles)
        return stability_dict, rdkit_dict, rd_mols

    return edm_metric


def get_2D_edm_metric(dataset_info, train_smiles=None):
    def edm_metric_2D(processed_list):
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        rd_mols = []
        for mol in processed_list:
            pos, atom_types, edge_types, fc = mol
            validity_res = check_2D_stability(pos, atom_types, fc, edge_types, dataset_info)
            molecule_stable += int(validity_res[0])
            nr_stable_bonds += int(validity_res[1])
            n_atoms += int(validity_res[2])
            rd_mols.append(validity_res[3])

        fraction_mol_stable = molecule_stable / float(len(processed_list)) if len(processed_list) > 0 else 0
        fraction_atm_stable = nr_stable_bonds / float(n_atoms) if n_atoms > 0 else 0
        stability_dict = {
            'mol_stable': fraction_mol_stable,
            'atom_stable': fraction_atm_stable,
        }

        rdkit_dict = eval_rdmol(rd_mols, train_smiles)
        return stability_dict, rdkit_dict, rd_mols

    return edm_metric_2D