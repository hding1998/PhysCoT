# PhysCoT: Physics-Verifiable Reasoning Steps for Small Language Models in Distribution Networks

Code, evaluation records, and analysis scripts accompanying the paper:

> H. Ding, W. Huang, Y. Wang, L. Zong, "Voltage violation screening under measurement scarcity: a small language model with physics-verifiable reasoning steps", submitted to *Electrical Engineering* (Springer).

## What this is

A post-training framework that supervises a 1.5B-parameter language model
(Qwen2.5-1.5B) to reason through an explicit mechanistic chain — network
topology → branch power flow → nodal voltages → violation verdict — for
voltage-violation screening in measurement-scarce distribution networks.
Every intermediate quantity can be substituted back into the power-flow
equations for verification.

## Repository layout

| Path | Content |
|---|---|
| `data_engine/` | scenario and training/evaluation data generation (pandapower-based) |
| `build_datasets.py` | dataset construction with held-out intervention assertions |
| `training/` | SFT (chunked cross-entropy trainer), DPO, GRPO (per-stage / terminal) |
| `evaluation/` | evaluator, physics verifier, DSSE / solver / Chronos baselines |
| `configs/` | run configurations |
| `results/` | all evaluation summaries and per-sample records reported in the paper |
| `remediation/` | post-hoc re-scoring scripts (strict intervention metric, reward-hacking re-verification, DPO median recompute) |
| `paper_figures/` | `make_figures_v2.py` — regenerates all 8 paper figures from `results/` |
| `docs_MANIFEST.md` | model-checkpoint inventory and rebuild instructions |

## Quick start

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
# evaluate a checkpoint on a stratified evaluation set
python evaluation/evaluate.py --ckpt <checkpoint> --data <eval_set.jsonl> --out results/x.json
# re-aggregate all results
python evaluation/summarize_results.py --results results
```

Datasets (13 GB) and model checkpoints (28 GB) are not stored in this
repository; they will be released on Hugging Face. Until then, they are
available from the corresponding author (hding1998@foxmail.com) upon
reasonable request.

## Reproducibility notes

- All numbers in the paper are computed from `results/` (per-sample
  records included); `remediation/` scripts reproduce the post-hoc
  re-scoring reported in Sections 5 and 6.
- Evaluation-set construction enforces zero scenario-level overlap with
  the training corpus (assertions in `build_datasets.py`).
- Held-out intervention combinations are withheld in form; see the paper
  for the exact protocol scope.

## Licence

MIT (see `LICENSE`). Public benchmark data remain under their original
providers' licences (SimBench, CIGRE MV, Ausgrid, CER, Low Carbon London).
