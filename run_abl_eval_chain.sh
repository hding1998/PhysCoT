#!/bin/bash
# 消融家族评测链（GPU1，晨审版：eval_id 限量 400 + holdout 全量供 ICS 对比）
set -x
source /d/miniconda/etc/profile.d/conda.sh
conda activate physcot
cd /c/Users/Administrator/Desktop/因果推理论文设计/02_代码框架
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1
for m in sft_a0_base sft_a2_shuffled sft_a3_cheat; do
  python -u evaluation/evaluate.py --ckpt runs/$m/final --data datasets/eval_id.jsonl --out results/${m}_id.json --bs 8 --limit 400
done
python -u evaluation/evaluate.py --ckpt runs/sft_a0_base/final --data datasets/eval_holdout.jsonl --out results/sft_a0_base_holdout.json --bs 8
echo ABL_EVAL_DONE
