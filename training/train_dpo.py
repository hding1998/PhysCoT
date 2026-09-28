# -*- coding: utf-8 -*-
"""
train_dpo.py — 阶段2备选：物理残差偏好对 DPO（build 子命令构造偏好对，train 子命令训练）
==========================================================================================
用途：
  1) GRPO 不稳定/显存不足时的降级路线；
  2) 论文消融："物理偏好 vs 纯任务偏好"的对照。

构造偏好对（build）：
  对同一场景采样 SFT 模型的多个输出 → physics_verifier 打分 →
  最高者为 chosen、最低者为 rejected（需最小差距 --min_gap）。
    --strategy physics : 按 phys_score 排序 ——"物理偏好"
    --strategy task    : 按 task_score 排序 ——"纯任务偏好"对照

用法：
  python training/train_dpo.py build --ckpt runs/sft_v1/final --data datasets/rl_scenarios.jsonl \
      --out datasets/dpo_phys.jsonl --n_sample 6 --strategy physics
  python training/train_dpo.py train --pairs datasets/dpo_phys.jsonl --ckpt runs/sft_v1/final --out runs/dpo_phys
（--ckpt 可为 adapter 目录或合并目录；adapter 会先合并到 bf16 基座）
"""
import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))


def build_pairs():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sample", type=int, default=6)
    ap.add_argument("--strategy", default="physics", choices=["physics", "task"])
    ap.add_argument("--min_gap", type=float, default=0.10)
    ap.add_argument("--max_pairs", type=int, default=5000)
    ap.add_argument("--max_new", type=int, default=3584)
    ap.add_argument("--cpu", action="store_true", help="CPU 冒烟")
    args = ap.parse_args()

    import torch
    from physics_verifier import verify
    from render import render_sample
    from model_loading import load_causal_lm

    model, tok = load_causal_lm(args.ckpt, dtype=torch.float32 if args.cpu else torch.bfloat16,
                                device_map="cpu" if args.cpu else "auto", merge=True)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.max_pairs * 2]
    n_out = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for rr in rows:
            if n_out >= args.max_pairs:
                break
            r = render_sample(rr)
            inputs = tok([r["prompt"] + "\n"], return_tensors="pt", padding=True,
                         return_token_type_ids=False).to(model.device)
            outs = model.generate(**inputs, do_sample=True, temperature=0.9,
                                  num_return_sequences=args.n_sample,
                                  max_new_tokens=args.max_new, pad_token_id=tok.eos_token_id)
            cands = [tok.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                     for o in outs]
            scored = []
            for c in cands:
                try:
                    v = verify(r, c)
                    score = (v["phys_score"] if args.strategy == "physics" else v["task_score"])
                    scored.append((score, v["reward"], c))
                except Exception:              # noqa: BLE001
                    continue
            if len(scored) < 3:
                continue
            scored.sort(key=lambda x: x[0])
            if scored[-1][0] - scored[0][0] < args.min_gap:
                continue                      # 差距太小不构造（信息量不足）
            fo.write(json.dumps({
                "prompt": r["prompt"] + "\n",
                "chosen": scored[-1][2], "rejected": scored[0][2],
                "gap": scored[-1][0] - scored[0][0],
                "strategy": args.strategy}, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 50 == 0:
                print(f"  [pairs] {n_out}", flush=True)
    print(f"偏好对: {n_out} -> {args.out}")


def train_dpo():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="CPU 冒烟")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    from model_loading import load_causal_lm

    model, tok = load_causal_lm(args.ckpt, dtype=torch.float32 if args.cpu else torch.bfloat16,
                                device_map={"": "cpu"} if args.cpu else {"": 0}, merge=True)
    # merge 后新建 LoRA 的 adapter_config.base_model_name_or_path 会继承原始基座路径，
    # 导致产出 adapter 评测时被挂到裸基座（已实测踩坑）——改写为本任务基座（SFT 合并体）。
    model.config._name_or_path = os.path.abspath(args.ckpt + "-merged") \
        if os.path.exists(args.ckpt + "-merged") else os.path.abspath(args.ckpt)
    ds = Dataset.from_list([json.loads(l) for l in open(args.pairs, encoding="utf-8")])
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])

    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
    cfg = DPOConfig(output_dir=args.out, per_device_train_batch_size=1,
                    gradient_accumulation_steps=16, learning_rate=args.lr,
                    num_train_epochs=args.epochs, bf16=not args.cpu, use_cpu=args.cpu,
                    beta=args.beta,
                    gradient_checkpointing=True, logging_steps=10,
                    save_strategy="epoch", report_to="none",
                    max_length=args.max_len, max_prompt_length=args.max_len // 2,
                    seed=args.seed)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    print("DPO done ->", final)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("build", "train"):
        mode = sys.argv.pop(1)
        build_pairs() if mode == "build" else train_dpo()
    else:
        print("用法: train_dpo.py build|train ...（见文件头）")
        sys.exit(1)
