# PhysCoT 交付包清单（MANIFEST）

> 组装时间：2026-09-26（第 3 天）。组装脚本：`02_代码框架/assemble_package.sh`。
> 目标：在另一终端独立完成论文写作与结果复现。

## 目录结构

```
PhysCoT/
├── 01_数据/                    # 源数据（ausgrid/CER/LCL/sgcc/simbench 原始 + 网络模型与场景真值）
├── 02_代码框架/
│   ├── configs/                # 数据构建配置（含 cfg_abstain.json 拒识增强配置）
│   ├── data_engine/            # scenario_gen / render / build_curriculum / 预处理
│   ├── datasets/               # 全部训练/评测数据（含 D: 冷回迁的 60k 全量文件）
│   ├── evaluation/             # evaluate / eval_api / solver / dsse / timesfm / physics_verifier /
│   │                           #   faithfulness_probe / summarize_results / model_loading
│   ├── training/               # train_sft（chunked CE）/ train_dpo / train_grpo
│   ├── results/                # 全部结果 JSON + records/preds + 图 + 日志 + 报告 + 案例
│   └── runs/                   # 全部模型（adapter 必备；merged 部分需重建，见下）
├── 04_实验文档/                # 实验手册 / 查新 / 投稿策略 / 魔鬼审稿与回应 / 定位说明 / 方案 v4
└── 05_环境/                    # requirements / environment.yml / 部署说明
```

## 硬件与环境

- Windows 11 + Git Bash；conda 环境 `physcot`（Python 3.11，torch 2.x + CUDA）
- GPU：2×RTX 3080 20GB；模型权重缓存 `D:\hf_cache\models\`（Qwen2.5-0.5B/1.5B/3B/7B-Instruct + chronos-bolt-small）
- 复现评测需先下载同等 HF 模型到本地并设 `HF_HUB_OFFLINE=1`

## 关键口径（务必遵守）

1. **论文数字一律以 pn 分层口径或 full 全量为准**；旧 `--limit N` 口径（文件名无 `_pn`/`_full` 后缀）存在网络偏斜，仅作趋势参考。汇总表 `results/summary_main.md` 的"来源"列可审计每个数字的文件出处。
2. **v_mae / phys_resid 报 median + 离群率**：CIGRE_MV 存在 21/333 的"数量级幻觉"离群（预测 10× 电压），mean 被拉爆。案例见 `results/v2_noise_full_preds.jsonl` 中 CIGRE_MV-000004。
3. ICS 协议：Δv 结论字段缺失按 0 计（保守）；另报条件 ICS（见 `实验日志.md` CP46-pre D2 节）。
4. IEEE145 因病态收敛域已从 OOD 集剔除（文档化于实验手册增补）。

## runs/ 模型清单与 merged 重建

adapter（final/）全部在位。**部分 merged 权重因 C: 盘容量删除，可用一行命令重建**：

```bash
python evaluation/model_loading.py --adapter runs/<run>/final --out runs/<run>/final-merged
```

adapter 的 `base_model_name_or_path` 均已修正为有效基座（原始 HF 本地路径或 runs 内相对路径）。
依赖链注意：dpo_v3 依赖 sft_v1/final-merged；grpo_ps60/tm60 依赖 sft_a0_base/final-merged；sft_v3b_6k 依赖 sft_v2_20k/final-merged。若需重建，请先重建其基座。

| 模型 | 角色 | 关键结果 |
|---|---|---|
| sft_v2_20k | 主线 SFT（20k+课程） | id viol_f1 0.661 / ICS 0.434 |
| sft_v3b_6k | **最终 SFT（v2+6.6k 刷新）** | id viol_f1 0.688 / ICS 0.596 |
| sft_v3_22k | 拒识增强从零训练（对照） | abstain 拒识率 49.3% |
| dpo_v3 | DPO 主线（v1 基座） | 六项全正向小幅 |
| grpo_ps60 / tm60 | E5 reward 消融 | per_stage 物理一致 / terminal 奖励黑客 |
| sft15_* / sft_a* / sft30_3k | 消融与规模家族 | 见 e6_scale_table.json 与 summary_main.md |

## 结果文件索引

- 主表与图：`results/summary_main.md`、`fig_*.png`
- 规模曲线：`results/e6_scale_table.json`
- 敏感性：`results/analysis_phys_score_sensitivity.json`、`results/analysis_ics_tolerance.json`
- 案例文本：`results/e7_cases.md`（E7）、`results/tm60_probe4_preds.jsonl`（奖励黑客案例）
- 实验日志（全部检查点与纠偏决策）：`results/实验日志.md`（CP0–CP51）
- 结论报告：`results/阶段性结论报告.md`（v4 修订版为最终口径）

## 已知作废文件（勿引用）

- `results/v1_probe.json` / `v1_probe_swap.json`：探针解析失败版（null 97%+），由 CP48-b 证伪存档
- `results/dpo_v3_id_pn.json`（无 fix 后缀）：adapter 基座指针 bug 的"假崩坏"版；以 `dpo_v3fix_id_pn.json` 为准
- `results/probe_v2_*.json` / `probe_a1_*.json`：修复后第二轮探针（仍证伪 teacher-forcing 范式，作为负结果存档）

## 安全

包内不含任何 API key（已全量 grep 验证）。DeepSeek 对照实验复现需自备 OpenAI 兼容 key，以环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 注入。
