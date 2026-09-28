# -*- coding: utf-8 -*-
"""
build_curriculum.py — 单步课程样本构造（阶段因果掩码的数据侧实现）
====================================================================
把全链样本拆成"给定前序阶段、预测下一阶段"的单步样本：
  step1: obs → topo
  step2: obs + topo → flow
  step3: obs + topo + flow → volt
  step4: obs + topo + flow + volt → answer
每个单步样本的 completion 只含目标阶段 —— 训练时天然不可"偷看"后续阶段
（CausalWM stage-ordered attention mask 在语言模型中的等价物之一，方案 §5.3(a)）。

用法：
  python build_curriculum.py --in sft_full.jsonl.src(smoke/全量原始样本) --out sft_full_curriculum.jsonl
"""
import os
import json
import argparse
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from render import TASK_DESC, render_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="scenario_gen 输出的原始 jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="topo,flow,volt,answer")
    args = ap.parse_args()
    steps = args.steps.split(",")

    n = 0
    with open(args.inp, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        for line in fi:
            s = json.loads(line)
            base_prompt = render_prompt(s)
            ctx = ""
            stage_json = {
                "topo": json.dumps(s["chain"]["topo"], ensure_ascii=False),
                "flow": json.dumps(s["chain"]["flow"], ensure_ascii=False),
                "volt": json.dumps(s["chain"]["volt"], ensure_ascii=False),
                "answer": json.dumps(s["answer"], ensure_ascii=False),
            }
            for st in steps:
                prompt = base_prompt + ("\n" + ctx if ctx else "")
                completion = f"<stage:{st}>{stage_json[st]}</stage:{st}>"
                fo.write(json.dumps({
                    "sample_id": f"{s['sample_id']}-cur-{st}", "net": s["net"],
                    "task": s["task"], "step": st,
                    "prompt": prompt, "completion": completion,
                    "truth": {"chain": s["chain"], "answer": s["answer"],
                              "inj": s.get("inj")},
                }, ensure_ascii=False) + "\n")
                ctx += f"<stage:{st}>{stage_json[st]}</stage:{st}>"
                n += 1
    print(f"课程样本 {n} 条 -> {args.out}")


if __name__ == "__main__":
    main()
