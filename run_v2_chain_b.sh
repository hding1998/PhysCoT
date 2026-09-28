#!/bin/bash
# v2 全量评测链 B（GPU1）：noise + ood_mv + ood_tx
set -x
source /d/miniconda/etc/profile.d/conda.sh
conda activate physcot
cd /c/Users/Administrator/Desktop/因果推理论文设计/02_代码框架
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
CKPT=runs/sft_v2_20k/final
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_noise_pn100.jsonl --out results/v2_noise_pn.json --bs 8
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_ood_mv_pn100.jsonl --out results/v2_ood_mv_pn.json --bs 8
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_ood_tx_pn100.jsonl --out results/v2_ood_tx_pn.json --bs 8
echo V2_CHAIN_B_DONE
