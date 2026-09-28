# -*- coding: utf-8 -*-
"""
render.py — 样本文本化：把 scenario_gen 的 JSONL 样本渲染为 (prompt, completion) 对。
==============================================================================
格式设计原则：
  1. <stage:> 特殊标记划分因果链阶段 —— 供阶段因果掩码与分段评测使用；
  2. 数值序列化统一（电压 3 位小数、电流 1 位、缺失记 "NA"）；
  3. prompt 只含 Observation (+do 干预)，completion 含 Topo→Flow→Volt→Answer 全链；
  4. 同一场景渲染多种"变体"用于消融：
     full_chain (完整链) / no_chain (直接给答案) / shuffled (乱序链)；
     中间变量组合消融（实验手册 §1.3b）：volt_only (仅V+answer) /
     topo_volt (T+V+answer) / topo_flow_volt (T+F+V+answer) —— completion 只保留
     被选中的中间阶段段块 + answer，prompt 不变，评测/奖励接口不受影响
     （physics_verifier 对缺失阶段按缺省处理）。

用法：
  python render.py --in smoke_test.jsonl --out sft_full.jsonl --mode full_chain
  python render.py --in smoke_test.jsonl --out sft_nochain.jsonl --mode no_chain
  python render.py --in smoke_test.jsonl --out sft_volt_only.jsonl --mode volt_only
"""
import os
import json
import argparse
import random

TASK_DESC = {
    "T1": "You are a distribution-grid state reasoning engine. Given partial, noisy measurements "
          "(some channels are 'NA'), infer the full physical state step by step: first the topology, "
          "then branch currents, then the voltage profile, and finally the conclusion. "
          "Follow the exact JSON stage format.",
    "T3": "You are a distribution-grid what-if reasoning engine. Given partial, noisy measurements "
          "and ONE intervention (do-action), infer the post-intervention physical state step by step: "
          "topology, branch currents, voltage profile, then the conclusion with deltas vs. the pre-state. "
          "Follow the exact JSON stage format.",
}


def render_prompt(s):
    parts = [TASK_DESC[s["task"]],
             "<observation>" + json.dumps(s["obs"], ensure_ascii=False) + "</observation>"]
    if s["task"] == "T3":
        parts.append("<do>" + json.dumps(s["do"], ensure_ascii=False) + "</do>")
    return "\n".join(parts)


# 中间变量组合消融（§1.3b）：completion 保留的中间阶段子集（answer 恒在末尾）
STAGE_SUBSETS = {
    "volt_only": ["volt"],
    "topo_volt": ["topo", "volt"],
    "topo_flow_volt": ["topo", "flow", "volt"],
}


def render_chain_stages(chain, answer, mode="full_chain"):
    """full_chain: topo→flow→volt→answer；no_chain: 仅 answer；shuffled: 乱序中间阶段；
    volt_only / topo_volt / topo_flow_volt: 只保留指定中间阶段 + answer（§1.3b）。"""
    topo = json.dumps(chain["topo"], ensure_ascii=False)
    flow = json.dumps(chain["flow"], ensure_ascii=False)
    volt = json.dumps(chain["volt"], ensure_ascii=False)
    ans = json.dumps(answer, ensure_ascii=False)
    if mode == "no_chain":
        return f"<stage:answer>{ans}</stage:answer>"
    if mode in STAGE_SUBSETS:
        bodies = {"topo": topo, "flow": flow, "volt": volt}
        out = "".join(f"<stage:{k}>{bodies[k]}</stage:{k}>" for k in STAGE_SUBSETS[mode])
        return out + f"<stage:answer>{ans}</stage:answer>"
    if mode == "shuffled":
        stages = [topo, flow, volt]
        names = ["topo", "flow", "volt"]
        idx = list(range(3))
        random.shuffle(idx)
        out = "".join(f"<stage:{names[i]}>{stages[i]}</stage:{names[i]}>" for i in idx)
        return out + f"<stage:answer>{ans}</stage:answer>"
    return (f"<stage:topo>{topo}</stage:topo>"
            f"<stage:flow>{flow}</stage:flow>"
            f"<stage:volt>{volt}</stage:volt>"
            f"<stage:answer>{ans}</stage:answer>")


def render_sample(s, mode="full_chain"):
    return {
        "sample_id": s["sample_id"],
        "net": s["net"],
        "task": s["task"],
        "prompt": render_prompt(s),
        "completion": render_chain_stages(s["chain"], s["answer"], mode),
        "truth": {"chain": s["chain"], "answer": s["answer"], "chain_pre": s["chain_pre"],
                  "inj": s.get("inj")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full_chain",
                    choices=["full_chain", "no_chain", "shuffled",
                             "volt_only", "topo_volt", "topo_flow_volt"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    with open(args.inp, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        for line in fi:
            s = json.loads(line)
            fo.write(json.dumps(render_sample(s, args.mode), ensure_ascii=False) + "\n")
    print(f"rendered [{args.mode}] -> {args.out}")


if __name__ == "__main__":
    main()
