# Property-aware Reinforcement Learning with Retrieval Enhancement for Controllable 3D Molecule Generation

This repository implements **POETIC** based on the source code of [Geo2Seq](https://github.com/divelab/AIRS/tree/7acc5357befe7493c335fb94e34829423a6eacba/OpenMol/Geo2Seq).

Our work introduces **POETIC** (Property-aware Reinforcement Learning with Retrieval Enhancement), a novel framework for controllable 3D molecule generation. Our work unifies two complementary components:

\- **Retrieval-augmented conditioning**: retrieves property- and structure-similar molecules as compact prefixes, providing chemically meaningful context.  

\- **Property-aware reinforcement learning**: uses a frozen property predictor to deliver explicit rewards, ensuring validity and alignment with target properties.  

This design enables POETIC to achieve precise controllability on in-distribution properties while maintaining robust generalization to unseen ones. 

![image](media/overview.png)

## Environment setup

Please refer to `enviroment_full.yml`.

## Process data

We first tokenize the QM9 dataset using Geo2Seq, which converts 3D molecular structures into SE(3)-invariant sequences and attaches property labels as generation conditions.
```bash
python process_data_qm9.py --raw_path qm9/temp/qm9/ --write_path data/ --split valid
```


## Train cVAE

```bash
python train_cvae.py --embedding_path alchemy_gap_embeddings.npz --property_path data/gap.txt --save_path cvae_alchemy_gap.pth --epochs 150
```