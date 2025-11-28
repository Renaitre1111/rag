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
python rag/process_seq.py --dataset alchemy --classifier_path qm9/property_prediction/outputs/alchemy_gap --input_path data/alchemy_seq.txt --save_path data/gap/alchemy_gap_embeddings.npz
```
```bash
python rag/retriever.py --db_emb_path data/gap/alchemy_gap_embeddings.npz --db_prop_path data/gap/gap.txt --save_path data/gap/train_ret_gap.npz --k_pool 100 --k_fine 10
```
```bash
python rag/finetune_retriever.py --db_emb_path data/gap/alchemy_gap_embeddings.npz --db_prop_path data/gap/gap.txt --prop_path data/gap/gap.txt --save_path data/gap/finetune_ret_gap.npz --k_pool 100 --k_fine 10
```
```bash
python rag/test_retriever.py --db_emb_path data/gap/alchemy_gap_embeddings.npz --db_prop_path data/gap/gap.txt --query_prop_path data/gap/sampled_gap.txt --save_path data/gap/test_ret_gap.npz --k_pool 100 --k_fine 10
```
```bash
python train.py --run_name conditional_alchemy --prop gap --model poetic --root_path ./data/alchemy_seq.txt --prop_path ./data/gap.txt --db_emb_path ./data/gap/alchemy_gap_embeddings.npz --db_prop_path ./data/gap/gap.txt --retrieval_path data/gap/train_ret_gap.npz --tokenizer_dir ./data/tokenizer --batch_size 160 --learning_rate 6e-4 --max_epochs 200 --num_workers 8 --max_len 160
```
```bash
python sample.py --prop_path data/gap/gap.txt --save_path data/gap/sampled_gap.txt --num_samples 10000 --num_bins 1000 --seed 42
```
```bash
python grpo_finetune_alchemy.py --run_name finetune_alchemy --sft_ckpt ./cond/gap_weights/conditional_alchemy.pt --vocab_dir ./data/tokenizer --finetune_data_path ./data/gap/finetune_ret_gap.npz --prop_path ./data/gap/gap.txt --db_emb_path ./data/gap/alchemy_gap_embeddings.npz --db_prop_path ./data/gap/gap.txt --classifier_path qm9/property_prediction/outputs/alchemy_gap --out_dir cond/gap_weights --lr 5e-6 --save_every 100 --batch_conditions 8 --iters 500
```
```bash
python generate.py --run_name gap --tokenizer_dir ./data/tokenizer --save_path cond/gap_gens --target_prop_path data/gap/sampled_gap.txt --test_retrieval_path data/gap/test_ret_gap.npz --db_emb_path data/gap/alchemy_gap_embeddings.npz --db_prop_path data/gap/gap.txt --temperature 0.7 --topk 80 --repeats 1 --batch_size 320 --ckpt_path cond/gap_weights/conditional_alchemy.pt
```
```bash
python generate.py --run_name gap --tokenizer_dir ./data/tokenizer --save_path cond/gap_gens --target_prop_path data/gap/sampled_gap.txt --test_retrieval_path data/gap/test_ret_gap.npz --db_emb_path data/gap/alchemy_gap_embeddings.npz --db_prop_path data/gap/gap.txt --temperature 0.7 --topk 80 --repeats 1 --batch_size 320 --ckpt_path cond/gap_weights/finetune_alchemy_step500.pt 
```
```bash
python eval_alchemy.py --generated_path cond/gap_gens/gap_generated.txt --target_path data/gap/sampled_gap.txt --classifier_path qm9/property_prediction/outputs/alchemy_gap --property gap --train_prop_path data/gap/gap.txt
```