#!/bin/bash
# P0-1b 口径修复重评链（GPU1）：分层抽样子集，全部模型同口径
set -x
source /d/miniconda/etc/profile.d/conda.sh
conda activate physcot
cd /c/Users/Administrator/Desktop/因果推理论文设计/02_代码框架
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
# v1 四集
python -u evaluation/evaluate.py --ckpt runs/sft_v1/final --data datasets/eval_id_pn100.jsonl      --out results/v1_id_pn.json     --bs 8
python -u evaluation/evaluate.py --ckpt runs/sft_v1/final --data datasets/eval_noise_pn100.jsonl   --out results/v1_noise_pn.json  --bs 8
python -u evaluation/evaluate.py --ckpt runs/sft_v1/final --data datasets/eval_ood_mv_pn100.jsonl  --out results/v1_ood_mv_pn.json --bs 8
python -u evaluation/evaluate.py --ckpt runs/sft_v1/final --data datasets/eval_ood_tx_pn100.jsonl  --out results/v1_ood_tx_pn.json --bs 8
# 消融家族 id 子集
for m in sft_a0_base sft_a1_nochain sft_a2_shuffled sft_a3_cheat; do
  python -u evaluation/evaluate.py --ckpt runs/$m/final --data datasets/eval_id_pn100.jsonl --out results/${m}_id_pn.json --bs 8
done
echo REEVAL_PN_DONE
