#!/bin/bash
# v1 评测链（GPU0，晨审版：先保 E1 主表 + E4c ICS + E4b 探针，限量提速；全量集后续补跑）
set -x
source /d/miniconda/etc/profile.d/conda.sh
conda activate physcot
cd /c/Users/Administrator/Desktop/因果推理论文设计/02_代码框架
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
CKPT=runs/sft_v1/final
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_id.jsonl --out results/v1_id.json --bs 8 --limit 400
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_holdout.jsonl --out results/v1_holdout.json --bs 8
python -u evaluation/faithfulness_probe.py --ckpt $CKPT --data datasets/eval_id.jsonl --out results/v1_probe.json --n 200
python -u evaluation/evaluate.py --ckpt $CKPT --data datasets/eval_ood_tx.jsonl --out results/v1_ood_tx.json --bs 8 --limit 300
echo V1_EVAL_CHAIN_DONE
