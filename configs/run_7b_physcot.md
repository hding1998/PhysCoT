# 7B+PhysCoT 对照实验配置（回应魔鬼审稿 M1）

## 目的
审稿人指出：1.5B 微调 vs 7B 零样本是训练数据不对称，不能回答"机理内嵌能否替代规模"。
核心对照改为：**7B+PhysCoT vs 1.5B+PhysCoT vs 7B-prompt**（同数据、同配方、仅规模不同）。

## 训练命令（7B QLoRA，2×3080 可行；bs=1 + 梯度累积）
```bash
# 卡0（约 9.3GB 显存，seq 2048）
python training/train_sft.py --data datasets/sft_train.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --out runs/sft_7b \
    --epochs 2 --bs 1 --accum 32 --lr 5e-5 --max_len 1792

# 若 OOM：--max_len 1536 --bs 1 --accum 64；或 4bit + flash-attn 确认开启
# GRPO 7B 不做（10GB 不可行）；7B 的 RL 后训练用 DPO 路线：
python training/train_dpo.py build --ckpt runs/sft_7b/final \
    --data datasets/rl_scenarios.jsonl --out datasets/dpo7b.jsonl \
    --strategy physics --n_sample 4 --max_pairs 3000
python training/train_dpo.py train --pairs datasets/dpo7b.jsonl \
    --ckpt runs/sft_7b/final --out runs/dpo_7b
```

## 评测
```bash
python evaluation/evaluate.py --ckpt runs/sft_7b/final --data datasets/eval_id.jsonl \
    --out results/sft7b_id.json --backend vllm
python evaluation/evaluate.py --ckpt Qwen/Qwen2.5-7B-Instruct --data datasets/eval_id.jsonl \
    --out results/qwen7b_prompt_id.json --backend vllm   # 零样本对照
```

## 预算
SFT 7B：约 4–6 天（2 epoch）；DPO：约 2 天。纳入 W9–W10（见实验手册 v2）。

## 论文叙事（三种结果的讲法都预先注册）
- 7B+PhysCoT ≈ 1.5B+PhysCoT → "机理内嵌使性能对规模饱和"（规模效率主张成立）
- 7B+PhysCoT > 1.5B+PhysCoT 明显 → "机理与规模正交、可叠加"（诚实报告，卖点转为
  物理一致性/ICS 维度的规模无关性——检查 phys_score 与 ICS 是否也随规模增长）
- 7B+PhysCoT < 1.5B+PhysCoT → 小模型更易拟合结构化监督（引用 NAACL'25 小模型
  CoT 微调现象），转为"结构化监督与小容量的匹配性"分析
