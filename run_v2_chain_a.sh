#!/bin/bash
# v2 全量评测链 A（GPU0）：id + holdout + abstain
set -x
source /d/miniconda/etc/profile.d/conda.sh
conda activate physcot
cd /c/Users/Administrator/Desktop/因果推理论文设计/02_代码框架
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
CKPT=runs/sft_v2_20k/final
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_id_pn100.jsonl --out results/v2_id_pn.json --bs 8
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_holdout.jsonl --out results/v2_holdout.json --bs 8
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_abstain.jsonl --out results/v2_abstain.json --bs 8
echo V2_CHAIN_A_DONE
