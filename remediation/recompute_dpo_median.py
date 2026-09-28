# -*- coding: utf-8 -*-
"""
recompute_dpo_median.py -- DPO 段 median v_mae 重算（论文 P1-8）

目的：results.tex 的 DPO 段落原只报告 mean v_mae (0.114 -> 0.098)，
审稿要求补统一口径的 median 对。本脚本从交付包逐样本 records 重算
mean 与 median，口径与正文其他处一致：median across samples，
parse 失败样本已在 records 中以哨兵值 v_mae=1.0 记录
（见 evaluation/evaluate.py 失败分支），无需额外处理。

用法（在 ee_paper/ 下运行）：
  python remediation/recompute_dpo_median.py

数据来源（只读）：
  PhysCoT_论文交付包_20260926/04_评测原始数据/v1_id_pn_records.jsonl        (DPO 前, v1 @ id pn300)
  PhysCoT_论文交付包_20260926/04_评测原始数据/dpo_v3fix_id_pn_records.jsonl (DPO 后, v3fix @ id pn300)

自检：v1 的 median 必须复现 tab:main 中 v1 行的 0.026；mean 必须复现
正文 0.114 -> 0.098 的口径。
"""
import json
import os
import statistics

BASE = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(
    BASE, "..", "..", "..",
    "PhysCoT_论文交付包_20260926", "04_评测原始数据")

FILES = {
    "v1 (DPO 前)": "v1_id_pn_records.jsonl",
    "dpo_v3fix (DPO 后)": "dpo_v3fix_id_pn_records.jsonl",
}


def load_vmae(path):
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(json.loads(line)["v_mae"]))
    return vals


def main():
    out = {}
    for label, fname in FILES.items():
        vals = load_vmae(os.path.join(RECORDS_DIR, fname))
        out[label] = {
            "file": fname,
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "median": statistics.median(vals),
        }
        print(f"{label}: n={len(vals)}  "
              f"mean={out[label]['mean']:.4f}  median={out[label]['median']:.4f}")

    # 自检：与正文既有数字对齐
    assert abs(out["v1 (DPO 前)"]["mean"] - 0.114) < 0.005, "v1 mean 与正文 0.114 不符"
    assert abs(out["dpo_v3fix (DPO 后)"]["mean"] - 0.098) < 0.005, "dpo mean 与正文 0.098 不符"
    assert abs(out["v1 (DPO 前)"]["median"] - 0.026) < 0.002, "v1 median 与 tab:main 0.026 不符"

    out_path = os.path.join(BASE, "dpo_median_recompute.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已落盘 {out_path}")


if __name__ == "__main__":
    main()
