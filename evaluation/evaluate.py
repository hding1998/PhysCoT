# -*- coding: utf-8 -*-
"""
evaluate.py — 批量推理 + 全指标评测（跑在卡B，或评测机）
=========================================================
指标：v_mae / violation_f1 / state_acc / format_rate / order_rate /
      physics_residual / phys_score / ICS(T3) / 综合表（按 net×task 分组）

用法：
  python evaluate.py --ckpt runs/sft_v1/final --data eval_id.jsonl --out results/sft_v1.json
  python evaluate.py --ckpt Qwen/Qwen2.5-7B-Instruct --data eval_id.jsonl --out results/qwen7b.json
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))

from physics_verifier import verify          # noqa: E402
from render import render_sample             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=3584,       # completion max≈2830（实测）
                    help="完整链 SimBench-urban 可达 2000 token，900 会截断 answer 阶段")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条（注意：数据按网络排序，前N条有偏！）")
    ap.add_argument("--per_net", type=int, default=None,
                    help="按网络分层等数抽样（每个 net 取前 N 条，固定顺序内偏移）——推荐用于限量评测，避免网络维度偏斜")
    ap.add_argument("--dump_predictions", default=None)
    args = ap.parse_args()

    from data_io import load_rows, ensure_rendered
    rows = load_rows(args.data)              # raw 或 rendered 均可
    if args.per_net:
        # 分层等数抽样：每个 net 在组内按固定 seed 洗牌后取前 per_net 条（口径可比、可复现）
        import collections, random as _rnd
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[r.get("net", "?")].append(i)
        sel = []
        for net in sorted(groups):
            g = groups[net]
            _rnd.Random(2026).shuffle(g)
            sel.extend(g[:args.per_net])
        rows = [rows[i] for i in sorted(sel)]
        print(f"[per_net] {args.per_net}/net × {len(groups)} 网 = {len(rows)} 条")
    elif args.limit:
        rows = rows[:args.limit]

    # ---- 推理 ----
    rendered = ensure_rendered(rows)
    prompts = [rd["prompt"] + "\n" for rd in rendered]      # 与 SFT 训练文本拼接方式一致
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from model_loading import is_adapter_dir
        if is_adapter_dir(args.ckpt):
            raise SystemExit(f"vLLM 需要合并后的完整模型目录，请先: python evaluation/model_loading.py "
                             f"--adapter {args.ckpt} --out {args.ckpt}-merged")
        llm = LLM(model=args.ckpt, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=6656)
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new)
        outs = llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    else:
        import torch
        from model_loading import load_causal_lm
        model, tok = load_causal_lm(args.ckpt, device_map="auto", merge=True)
        texts = []
        bs = args.bs
        model.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                chunk = prompts[i:i + bs]
                enc = tok(chunk, return_tensors="pt", padding=True, return_token_type_ids=False).to(model.device)
                o = model.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=args.temperature > 0,
                                   temperature=max(args.temperature, 1e-5),
                                   pad_token_id=tok.eos_token_id)
                texts.extend(tok.decode(x[enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True) for x in o)
                print(f"  [infer] {i + len(chunk)}/{len(prompts)}")

    # ---- 评测 ----
    records = []
    n_verify_fail = 0
    for r, rd, t in zip(rows, rendered, texts):
        try:
            v = verify(rd, t)
        except Exception as e:                   # 模型输出再怪也不许崩掉整批评测
            n_verify_fail += 1
            v = {"v_mae": 1.0, "violation_f1": 0.0, "state_ok": 0.0, "format_ok": 0.0,
                 "stages": [], "physics_residual": None, "phys_score": 0.0,
                 "order_ok": 0.0, "ics": 0.0, "task_score": 0.0, "dv_pred": None,
                 "dv_true": None, "confidence_pred": None, "abstain_pred": None,
                 "reward": 0.0, "verify_error": f"{type(e).__name__}: {e}"[:200]}
        v.update({"sample_id": r["sample_id"], "net": r["net"], "task": r["task"]})
        records.append(v)
    if n_verify_fail:
        print(f"[warn] verify 异常样本数: {n_verify_fail}/{len(records)}（已按最坏分计入）")
    if args.dump_predictions:
        with open(args.dump_predictions, "w", encoding="utf-8") as f:
            for r, rd, t in zip(rows, rendered, texts):
                f.write(json.dumps({"sample_id": r["sample_id"], "prompt": rd["prompt"],
                                    "prediction": t}, ensure_ascii=False) + "\n")

    # ICS 容差敏感性扫描（M2）+ 置信度分桶（M3 校准的原始材料）
    def ics_sweep(recs, tols=(0.002, 0.005, 0.01)):
        out = {}
        for tol in tols:
            vals = []
            for x in recs:
                dp, dt = x.get("dv_pred"), x.get("dv_true")
                if dp is None or dt is None:
                    continue
                same = (dp > 0) == (dt > 0)
                close = abs(dp - dt) <= tol
                vals.append((0.4 if same else 0.0) + (0.4 if close else 0.0))
            out[f"tol_{tol}"] = float(np.mean(vals)) if vals else None
        return out

    def conf_bins(recs):
        rows = [(x.get("confidence_pred"), x.get("v_mae")) for x in recs
                if isinstance(x.get("confidence_pred"), (int, float))
                and x.get("v_mae") is not None]
        bins = {}
        for c, m in rows:
            k = min(int(float(c) * 5), 4)
            bins.setdefault(k, []).append(m)
        return {f"c[{k/5:.1f}-{(k+1)/5:.1f}]": float(np.mean(v)) for k, v in sorted(bins.items())}

    # 分组汇总
    def agg(recs):
        m = lambda k: float(np.mean([x[k] for x in recs if x.get(k) is not None])) \
            if any(x.get(k) is not None for x in recs) else None
        return {"n": len(recs), "v_mae": m("v_mae"), "viol_f1": m("violation_f1"),
                "state_acc": m("state_ok"), "format_rate": m("format_ok"),
                "order_rate": m("order_ok"),
                "phys_resid": m("physics_residual"), "phys_score": m("phys_score"),
                "reward": m("reward"), "ics": m("ics")}

    t3recs = [r for r in records if r["task"] == "T3"]
    summary = {"overall": agg(records),
               "ics_sensitivity": ics_sweep(t3recs),
               "confidence_vs_error": conf_bins(records)}
    for net in sorted({r["net"] for r in records}):
        summary[net] = agg([r for r in records if r["net"] == net])
    for task in sorted({r["task"] for r in records}):
        summary[f"task_{task}"] = agg([r for r in records if r["task"] == task])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_records.jsonl"), "w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
