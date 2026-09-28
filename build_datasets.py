# -*- coding: utf-8 -*-
"""
build_datasets.py — 一键构建全量训练/评测数据集（实验机上执行）
================================================================
产出（02_代码框架/datasets/）：
  sft_train.jsonl          全链 SFT 样本（ID 网络）
  sft_train_curriculum.jsonl  单步课程样本
  sft_nochain.jsonl        无链消融基线
  sft_shuffled.jsonl       乱序链消融基线
  sft_volt_only.jsonl / sft_topo_volt.jsonl / sft_topo_flow_volt.jsonl
                           中间变量组合消融（实验手册 §1.3b，--variants 开启才产出）
  sft_cheat_visible.jsonl  “作弊可见”对照组训练集（答案拼入 prompt）
  rl_scenarios.jsonl       GRPO 场景（原始样本）
  eval_id.jsonl            ID 评测
  eval_ood_mv.jsonl / eval_ood_tx.jsonl   OOD 评测（同域放大 MV / 跨域输电）
  eval_noise.jsonl         高噪声评测
  eval_holdout.jsonl       扣留干预泛化评测（combo/extreme）
  （各 eval_* 均有对应 raw_eval_*.jsonl 原始样本，供 dsse/solver/timesfm 基线使用）

用法：
  python build_datasets.py --scale full   # 50k SFT + 10k RL + 各1k评测
  python build_datasets.py --scale small  # 5k/1k/300 快速冒烟
  python build_datasets.py --scale tiny   # 120/40/30 管线冒烟（约 1–2 分钟，先跑通再放大）
  python build_datasets.py --scale small --variants no_chain,shuffled,volt_only,topo_volt,topo_flow_volt
      # --variants：逗号分隔的渲染变体列表（默认 no_chain,shuffled，保持现状行为）；
      # 可选值 = render.py 的 --mode 去掉 full_chain；产出命名为 sft_<变体>.jsonl
      # （历史兼容：no_chain → sft_nochain.jsonl）
"""
import os
import sys
import json
import subprocess
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
DE = os.path.join(BASE, "data_engine")
OUT = os.path.join(BASE, "datasets")
sys.path.insert(0, DE)                      # 供 render_sample 导入（须在 import 之前）

SIZES = {
    "tiny":  {"sft": 120, "rl": 40, "eval": 30},        # 管线冒烟（约 1–2 分钟）
    "small": {"sft": 5000, "rl": 1000, "eval": 300},
    "full":  {"sft": 50000, "rl": 10000, "eval": 1000},
}

WORKERS = max(1, (os.cpu_count() or 2) - 1)

# mode → 产出文件名后缀（no_chain 历史命名为 nochain，保持兼容）
VARIANT_SUFFIX = {"no_chain": "nochain", "shuffled": "shuffled",
                  "volt_only": "volt_only", "topo_volt": "topo_volt",
                  "topo_flow_volt": "topo_flow_volt"}


def sh(cmd):
    cmd = [str(c) for c in cmd]             # 修复：seed 等整数参数需转为 str
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True, cwd=BASE)


def _filter_holdout(src, dst):
    """只保留扣留类样本（combo / extreme 干预）——训练集从未出现这些标签。"""
    kept = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            s = json.loads(line)
            tag = s.get("do_tag", "none")
            if tag.startswith("combo") or tag.startswith("extreme"):
                fo.write(line)
                kept += 1
    print(f">> holdout filter: {kept} 条扣留样本 -> {dst}")


def _assert_no_holdout_leak(train_path):
    """训练集断言：不得含 combo/extreme 标签（E4c 协议可信性，实验手册 v2 核对项）。"""
    bad = 0
    with open(train_path, encoding="utf-8") as fi:
        for line in fi:
            tag = json.loads(line).get("do_tag", "none")
            if tag.startswith("combo") or tag.startswith("extreme"):
                bad += 1
    assert bad == 0, f"训练集含 {bad} 条扣留标签样本（combo/extreme）—— 检查 cfg_train.json combo_ratio"
    print(">> 训练集扣留断言通过：无 combo/extreme")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=list(SIZES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=WORKERS, help="scenario_gen 并行进程数")
    ap.add_argument("--variants", default="no_chain,shuffled",
                    help="逗号分隔的消融渲染变体（render.py 的 mode），"
                         "可选：no_chain,shuffled,volt_only,topo_volt,topo_flow_volt")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANT_SUFFIX]
    if bad:
        raise SystemExit(f"未知变体: {bad}（可选：{sorted(VARIANT_SUFFIX)}）")
    sz = SIZES[args.scale]
    os.makedirs(OUT, exist_ok=True)
    GEN = os.path.join(DE, "scenario_gen.py")
    CFG_TRAIN = os.path.join(BASE, "configs", "cfg_train.json")
    W = ["--workers", args.workers]

    def out(name):
        return os.path.join(OUT, name)

    # 1) ID 域全量样本（SFT+RL 用）——训练扣留组合干预（回应审稿 C3：combo_ratio=0）
    sh([GEN, "--net", "all_id", "--n", sz["sft"] + sz["rl"], "--out", out("raw_id.jsonl"),
        "--seed", args.seed, "--cfg", CFG_TRAIN] + W)

    # 2) 评测：ID / OOD 分层(M4: 同域放大 vs 跨域) / 高噪声 / 扣留干预泛化(C3)
    #    注意：各评测集 seed 互不相同且与训练 seed 不同（原版 ood_mv/ood_tx 共用 seed+2，已改为独立 seed）
    sh([GEN, "--net", "all_id", "--n", sz["eval"], "--out", out("raw_eval_id.jsonl"),
        "--seed", args.seed + 1, "--cfg", CFG_TRAIN] + W)
    sh([GEN, "--net", "SimBench_1-MV-semiurb--0-sw,SimBench_1-MV-rural--0-sw",
        "--n", sz["eval"], "--out", out("raw_eval_ood_mv.jsonl"),
        "--seed", args.seed + 2, "--cfg", CFG_TRAIN] + W)
    sh([GEN, "--net", "PEGASE89,IEEE118,Illinois200",   # IEEE145 已剔除（刀刃型收敛域，见 scenario_gen.OOD_NETS 注释）
        "--n", sz["eval"], "--out", out("raw_eval_ood_tx.jsonl"),
        "--seed", args.seed + 5, "--cfg", CFG_TRAIN] + W)
    sh([GEN, "--net", "all_id", "--n", sz["eval"], "--out", out("raw_eval_noise.jsonl"),
        "--seed", args.seed + 3, "--cfg", os.path.join(BASE, "configs", "cfg_high_noise.json")] + W)
    # 扣留干预泛化评测（组合+极端，训练从未见过）
    sh([GEN, "--net", "all_id", "--n", sz["eval"], "--out", out("raw_eval_holdout_raw.jsonl"),
        "--seed", args.seed + 4, "--cfg", os.path.join(BASE, "configs", "cfg_holdout_eval.json")] + W)
    _filter_holdout(out("raw_eval_holdout_raw.jsonl"), out("raw_eval_holdout.jsonl"))
    _assert_no_holdout_leak(out("raw_id.jsonl"))

    # 3) 渲染变体
    sh([os.path.join(DE, "render.py"), "--in", out("raw_id.jsonl"),
        "--out", out("sft_train.jsonl"), "--mode", "full_chain"])
    sh([os.path.join(DE, "build_curriculum.py"), "--in", out("raw_id.jsonl"),
        "--out", out("sft_train_curriculum.jsonl")])
    for mode in variants:
        sh([os.path.join(DE, "render.py"), "--in", out("raw_id.jsonl"),
            "--out", out(f"sft_{VARIANT_SUFFIX[mode]}.jsonl"), "--mode", mode])

    # 4) RL 场景（取尾部 rl 条，渲染为 prompt+truth 供 reward 用）
    rows = [json.loads(l) for l in open(out("raw_id.jsonl"), encoding="utf-8")]
    rl_rows = rows[-sz["rl"]:]
    with open(out("rl_scenarios.jsonl"), "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5) 评测集渲染 + “作弊可见”对照组
    for tag in ["eval_id", "eval_ood_mv", "eval_ood_tx", "eval_noise", "eval_holdout"]:
        sh([os.path.join(DE, "render.py"), "--in", out(f"raw_{tag}.jsonl"),
            "--out", out(f"{tag}.jsonl"), "--mode", "full_chain"])

    # 作弊可见：训练用（把真值 answer 放进 prompt —— 消融 §5.3(c)）
    from render import render_sample          # sys.path 已在文件头部插入 DE
    with open(out("sft_cheat_visible.jsonl"), "w", encoding="utf-8") as f, \
         open(out("raw_id.jsonl"), encoding="utf-8") as fi:
        for line in fi:
            s = json.loads(line)
            r = render_sample(s)
            r["prompt"] += ("\n<leaked_answer>" + json.dumps(s["answer"]) +
                            "</leaked_answer>")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n全部数据集构建完成 ->", OUT)
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        print(f"  {fn:40s} {os.path.getsize(p)/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
