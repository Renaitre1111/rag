import os
import numpy as np
import pandas as pd
from rdkit import Chem

root = "alchemy"
sdf_root = root 

csv_path = os.path.join(root, "final_version.csv")
df = pd.read_csv(csv_path)

df = df.rename(columns={
    "alpha\n(a_0^3, Isotropic polarizability)": "alpha",
    "gap\n(Ha, LUMO-HOMO)": "gap",
    "HOMO\n(Ha, energy of HOMO)": "homo",
    "LUMO\n(Ha, energy of LUMO)": "lumo",
    "mu\n(D, dipole moment)": "mu",
    "Cv\n(cal/molK, heat capacity at 298.15 K)": "Cv",
})

energy_cols = ["gap", "homo", "lumo"]
for col in energy_cols:
    df[col] = df[col] * 27.2114

df["Cv"] = df["Cv"] * 627509.47

target_cols = ["alpha", "gap", "homo", "lumo", "mu", "Cv"]
df["gdb_idx"] = df["gdb_idx"].astype(int)
df = df[["gdb_idx"] + target_cols]

idx2path = {}
for folder in os.listdir(sdf_root):
    if not folder.startswith("atom_"):
        continue
    folder_path = os.path.join(sdf_root, folder)
    if not os.path.isdir(folder_path):
        continue

    for fname in os.listdir(folder_path):
        if not fname.endswith(".sdf"):
            continue
        name = os.path.splitext(fname)[0]
        try:
            gdb_idx = int(name)
        except ValueError:
            continue
        idx2path[gdb_idx] = os.path.join(folder_path, fname)

records = []
for _, row in df.iterrows():
    gdb_idx = int(row["gdb_idx"])
    if gdb_idx not in idx2path:
        continue
    sdf_path = idx2path[gdb_idx]

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        continue

    mol_no_h = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol_no_h, isomericSmiles=True)

    conf = mol.GetConformer()
    coords = conf.GetPositions()              # (N, 3)
    charges = [atom.GetAtomicNum() for atom in mol.GetAtoms()]  # (N,)

    num_atoms = len(charges)
    props = row[target_cols].values.astype(np.float32)

    records.append({
        "gdb_idx": gdb_idx,
        "num_atoms": num_atoms,
        "charges": np.array(charges, dtype=np.int64),
        "positions": coords.astype(np.float32),
        "props": props,
        "smiles": smiles,
    })

print("Total valid molecules:", len(records))

num_mol = len(records)
max_n = max(r["num_atoms"] for r in records)

charges_arr = np.zeros((num_mol, max_n), dtype=np.int64)
pos_arr = np.zeros((num_mol, max_n, 3), dtype=np.float32)
num_atoms_arr = np.zeros(num_mol, dtype=np.int64)
index_arr = np.zeros(num_mol, dtype=np.int64)
props_arr = np.zeros((num_mol, len(target_cols)), dtype=np.float32)
smiles_arr = np.empty(num_mol, dtype=object)

for i, r in enumerate(records):
    n = r["num_atoms"]
    charges_arr[i, :n] = r["charges"]
    pos_arr[i, :n, :] = r["positions"]
    num_atoms_arr[i] = n
    index_arr[i] = r["gdb_idx"]
    props_arr[i] = r["props"]
    smiles_arr[i] = r["smiles"]

np.random.seed(42)  
perm = np.random.permutation(num_mol)

n_valid = int(num_mol * 0.1)
n_test = int(num_mol * 0.1)

valid_idx = perm[:n_valid]
test_idx = perm[n_valid : n_valid + n_test]

train_all_idx = perm[n_valid + n_test :]
n_train_all = len(train_all_idx)

n_train_gen = n_train_all // 2 

train_gen_idx = train_all_idx[:n_train_gen]
train_eval_idx = train_all_idx[n_train_gen:]

print(f"Total: {num_mol}")
print(f"  train_gen: {len(train_gen_idx)}")
print(f"  train_eval: {len(train_eval_idx)}")
print(f"  valid: {len(valid_idx)}")
print(f"  test: {len(test_idx)}")


def save_split(name, idx):
    np.savez_compressed(
        f"alchemy_{name}.npz",
        num_atoms=num_atoms_arr[idx],
        charges=charges_arr[idx],
        positions=pos_arr[idx],
        index=index_arr[idx],
        props=props_arr[idx],
    )
    
    smiles_filename = f"alchemy_{name}_smiles.npy"
    np.save(smiles_filename, smiles_arr[idx])
    
    print(f"Saved: alchemy_{name}.npz AND {smiles_filename}")

save_split("valid", valid_idx)
save_split("test", test_idx)
save_split("train_gen", train_gen_idx)
save_split("train_eval", train_eval_idx)
