import os
import numpy as np
from rdkit.Chem.rdchem import GetPeriodicTable

def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True))
    )

def process_alchemy_train_gen(input_npz="alchemy_train_gen.npz",
                              output_dir="data_alchemy_seq/"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading NPZ file: {input_npz}")
    data = np.load(input_npz)

    charges_arr = data["charges"]     # (N_mol, max_n)
    pos_arr = data["positions"]   # (N_mol, max_n, 3)
    num_atoms_arr = data["num_atoms"]   # (N_mol,)
    props_arr = data["props"]       # (N_mol, 6)  [alpha, gap, homo, lumo, mu, Cv]
    index_arr = data["index"]   

    num_mol = len(num_atoms_arr)
    print(f"Total molecules in train_gen: {num_mol}")

    attr_names = ["alpha", "gap", "homo", "lumo", "mu", "Cv"]

    lines_seq = []
    lines_attr = {name: [] for name in attr_names}

    pt = GetPeriodicTable()

    for idx in range(num_mol):
        n = int(num_atoms_arr[idx])
        if n <= 0:
            continue

        charges = charges_arr[idx][:n]      # (n,)
        coords = pos_arr[idx][:n]          # (n, 3)
        props = props_arr[idx].copy()     # (6,)
        props[1] *= 27.2114    # gap
        props[2] *= 27.2114    # homo
        props[3] *= 27.2114    # lumo
        
        atom_types = [pt.GetElementSymbol(int(z)) for z in charges]

        centered = coords - coords[0]    
        invariant = np.zeros_like(centered)
        spherical = np.zeros_like(centered)

        if n == 1:
            pass
        elif n == 2:
            d = np.linalg.norm(centered[1])
            invariant[1, 0] = d
            spherical[1, 0] = d
        else:
            v1 = centered[1]
            flag = False
            for k in range(2, n):
                v2 = centered[k]
                if np.linalg.norm(np.cross(v1, v2)) != 0:
                    flag = True
                    break
            if (not flag) and k == n - 1:
                invariant = centered
            else:
                x = _normalize(v1)
                y = _normalize(np.cross(v1, v2))
                z = np.cross(x, y)
                invariant = np.dot(centered, np.stack((x, y, z)).T)

            d = np.linalg.norm(invariant, axis=-1)
            theta = np.zeros_like(d)
            mask = d > 1e-6
            theta[mask] = np.arccos(
                np.clip(invariant[mask, 2] / d[mask], -1.0, 1.0)
            )
            phi = np.arctan2(invariant[:, 1], invariant[:, 0])
            spherical = np.stack((d, theta, phi)).T

        spherical_str = [["{:.3f}".format(v) for v in row] for row in spherical]

        tokens = []
        for j in range(n):
            tokens.append(atom_types[j])
            tokens.append(spherical_str[j][0])           # r
            tokens.append(spherical_str[j][1] + "°")     # theta
            tokens.append(spherical_str[j][2] + "°")     # phi

        lines_seq.append(" ".join(tokens) + "\n")

        for a_i, name in enumerate(attr_names):
            lines_attr[name].append(f"{props[a_i]:.2f}\n")

    seq_path = os.path.join(output_dir, "alchemy_seq.txt")
    with open(seq_path, "w") as f:
        f.writelines(lines_seq)
    print(f"Saved: {seq_path}")

    for name in attr_names:
        p = os.path.join(output_dir, f"{name}.txt")
        with open(p, "w") as f:
            f.writelines(lines_attr[name])
        print(f"Saved: {p}")


if __name__ == "__main__":
    process_alchemy_train_gen(
        input_npz="alchemy/alchemy_train_gen.npz",
        output_dir="data/"
    )
