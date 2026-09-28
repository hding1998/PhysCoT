# -*- coding: utf-8 -*-
"""
train_grpo.py — 阶段2：物理奖励强化学习（GRPO；Unsloth 优先，TRL+PEFT 回退）
================================================================================
奖励（--reward_mode）：
  per_stage（默认，现状）= physics_verifier.verify(...)["reward"]
     = 0.40*task + 0.30*physics + 0.20*format + 0.10*order
     + 0.05*(1 - (confidence - correctness)^2)      ← v2 校准项（实验手册 V5，防"永远高置信"）
  terminal（消融，对标"终答约束奖励"基线，arXiv 2603.23004 式）
     = physics_verifier.terminal_reward(...)：只用 <stage:answer> 的 state/v_min/n_viol
       正确性（T3 混入 ICS），忽略 topo/flow/volt 中间阶段一致性残差与格式分；
       Brier 校准项同口径叠加（correctness 取 terminal 的 state_ok）。

3080-20GB 工程要点：
  * 默认 num_generations=8, max_prompt=3072, max_completion=3584（完整链 SimBench-urban completion max≈2830 token，实测）
  * OOM 时：--gens 4 → --max_completion 1536 → --bs 1；仍不行走 train_dpo.py
  * use_vllm=False（Windows 原生无 vLLM；Linux 双卡如需 vLLM 加速请自行改 GRPOConfig，见 TRL 文档）

用法：
  python training/train_grpo.py --sft runs/sft_v1/final --data datasets/rl_scenarios.jsonl --out runs/grpo_v1
  python training/train_grpo.py --sft runs/sft_v1/final --data datasets/rl_scenarios.jsonl --out runs/grpo_terminal --reward_mode terminal
  （--sft 可以是 adapter 目录 或 final-merged 完整目录）
"""
import os
import sys
import json
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "evaluation"))
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT checkpoint（adapter 目录或合并目录）")
    ap.add_argument("--data", required=True, help="rl_scenarios.jsonl（原始样本，含 truth）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--max_prompt", type=int, default=3072)    # prompt p99≈2305（实测）
    ap.add_argument("--max_completion", type=int, default=3584) # completion max≈2830（实测）
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    ap.add_argument("--calib_w", type=float, default=0.05, help="Brier 校准奖励权重（0 关闭）")
    ap.add_argument("--reward_mode", default="per_stage", choices=["per_stage", "terminal"],
                    help="per_stage=逐阶段综合物理奖励（现状）；terminal=终答约束奖励消融（仅 answer 阶段）")
    ap.add_argument("--no_unsloth", action="store_true")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="CPU 冒烟（关闭 unsloth/4bit/bf16）")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()
    if args.cpu:
        args.no_unsloth = True; args.no_4bit = True

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from physics_verifier import verify, terminal_reward
    from render import render_sample
    from model_loading import is_adapter_dir, base_model_of

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]
    ds = Dataset.from_list([
        {"prompt": render_sample(rr)["prompt"] + "\n",
         "sample": json.dumps({"net": rr.get("net"), "task": rr.get("task"),
                               "obs": rr.get("obs"), "inj": rr.get("inj"),
                               "truth": {"chain": rr["chain"], "answer": rr["answer"]}},
                              ensure_ascii=False)}
        for rr in rows
    ]).shuffle(seed=args.seed)

    def reward_fn(prompts=None, completions=None, sample=None, **kw):
        """TRL 会把数据集其余列（这里是 sample）按名传入。"""
        score = terminal_reward if args.reward_mode == "terminal" else verify
        out = []
        for c, sm in zip(completions, sample or []):
            try:
                s = json.loads(sm)
                v = score(s, c)
                r = float(v["reward"])
                conf = v.get("confidence_pred")
                if args.calib_w > 0 and isinstance(conf, (int, float)):
                    correctness = float(v.get("state_ok") or 0.0)
                    r += args.calib_w * (1.0 - (float(conf) - correctness) ** 2)
                out.append(r)
            except Exception:                  # noqa: BLE001
                out.append(-0.1)               # 解析/校验失败给负奖励
        return out
    reward_fn.__name__ = f"physcot_reward_{args.reward_mode}"

    # ---- 模型加载：Unsloth 优先；否则 TRL + PEFT ----
    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except Exception:                      # noqa: BLE001
            use_unsloth = False
            print("[warn] unsloth 不可用，回退 TRL+PEFT GRPO（速度较慢；显存吃紧时减小 --gens）")

    peft_config = None
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.sft, max_seq_length=args.max_prompt + args.max_completion,
            load_in_4bit=not args.no_4bit, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
        dev = {"": "cpu"} if args.cpu else {"": 0}
        dt = torch.float32 if args.cpu else torch.bfloat16
        if is_adapter_dir(args.sft):
            # 未合并的 adapter：先合并到基座，再在其上挂新 LoRA 做 RL
            from model_loading import load_causal_lm
            model, tokenizer = load_causal_lm(args.sft, dtype=dt, device_map=dev, merge=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.sft)
            model = AutoModelForCausalLM.from_pretrained(args.sft, torch_dtype=dt, device_map=dev)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"],
                                 task_type="CAUSAL_LM")

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_generations=args.gens,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        max_steps=args.steps,
        logging_steps=5,
        save_strategy="steps", save_steps=100,
        bf16=not args.cpu,
        use_cpu=args.cpu,
        use_vllm=False,
        gradient_checkpointing=True,
        report_to="none",
        temperature=0.9,
        beta=args.beta,
        seed=args.seed,
    )
    trainer = GRPOTrainer(model=model, args=cfg, reward_funcs=reward_fn,
                          train_dataset=ds, processing_class=tokenizer,
                          peft_config=peft_config)
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print("GRPO done ->", final)


if __name__ == "__main__":
    main()
