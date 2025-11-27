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
python rag/generate_reference.py --checkpoint_path cond/gap_weights/cvae_alchemy_gap.pth --database_path data/alchemy_gap_embeddings.npz --database_prop_path data/gap.txt --target_path data/gap.txt --output_path data/train_gap_ref.npz --remove_self --batch_size 128 --device cuda
```
```bash
python train.py --run_name conditional_alchemy --prop gap --model poetic --root_path ./data/alchemy_seq.txt --prop_path ./data/gap.txt --db_emb_path ./data/alchemy_gap_embeddings.npz --db_prop_path ./data/gap.txt --retrieval_path data/train_ret_gap.npz --tokenizer_dir ./data/tokenizer --batch_size 160 --learning_rate 6e-4 --max_epochs 200 --num_workers 8 --max_len 160
```