# -*- coding: utf-8 -*-
"""
eval_api.py — API 大模型纯提示对照（GPT-4o / DeepSeek-V3 / Qwen-Max 等 OpenAI 兼容接口）
==========================================================================================
实验手册 §4.1 要求补的"薄封装"：prompt → API → physics_verifier.verify，输出与 evaluate.py 同构。
公平性约定（写进论文）：同一 prompt 模板、temperature=0、默认零样本；--fewshot 3 给附录 3-shot 版本。

环境变量：OPENAI_API_KEY（必填）、OPENAI_BASE_URL（DeepSeek: https://api.deepseek.com；可选）
用法：
  python evaluation/eval_api.py --model deepseek-chat --data datasets/eval_id.jsonl \
      --out results/deepseek_v3_id.json --limit 500
"""
import os
import sys
import json
import time
import argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))

from physics_verifier import verify          # noqa: E402
from render import render_sample             # noqa: E402

SYSTEM = ("You are a distribution-grid physical reasoning engine. Answer ONLY with the four stages "
          "<stage:topo>{json}</stage:topo><stage:flow>{json}</stage:flow>"
          "<stage:volt>{json}</stage:volt><stage:answer>{json}</stage:answer>, no prose.")


def call(client, model, prompt, fewshot, max_tokens, retries=5):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in fewshot:
        msgs.append({"role": "user", "content": ex["prompt"]})
        msgs.append({"role": "assistant", "content": ex["completion"]})
    msgs.append({"role": "user", "content": prompt})
    for k in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msgs,
                                               temperature=0, max_tokens=max_tokens)
            return r.choices[0].message.content or ""
        except Exception as e:                 # noqa: BLE001
            wait = 2 ** k
            print(f"  [api] {type(e).__name__}: {str(e)[:80]} → retry in {wait}s", flush=True)
            time.sleep(wait)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--fewshot", type=int, default=0, help="从 --shots_from 取 k 条教师样本做 few-shot")
    ap.add_argument("--shots_from", default=None, help="如 datasets/sft_train.jsonl")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dump_predictions", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"))

    from data_io import load_rows, ensure_rendered
    rows = load_rows(args.data)[:args.limit]
    rendered_rows = ensure_rendered(rows)
    fewshot = []
    if args.fewshot and args.shots_from:
        shots = [json.loads(l) for l in open(args.shots_from, encoding="utf-8")][:args.fewshot]
        fewshot = [{"prompt": s["prompt"], "completion": s["completion"]} for s in shots]

    records, dump = [], []
    for i, (r, rd) in enumerate(zip(rows, rendered_rows)):
        text = call(client, args.model, rd["prompt"], fewshot, args.max_tokens)
        v = verify(rd, text)
        v.update({"sample_id": r["sample_id"], "net": r["net"], "task": r["task"]})
        records.append(v)
        dump.append({"sample_id": r["sample_id"], "prediction": text})
        if (i + 1) % 20 == 0:
            print(f"  [api] {i + 1}/{len(rows)}", flush=True)

    def agg(recs):
        m = lambda k: float(np.mean([x[k] for x in recs if x.get(k) is not None])) \
            if any(x.get(k) is not None for x in recs) else None
        return {"n": len(recs), "v_mae": m("v_mae"), "viol_f1": m("violation_f1"),
                "state_acc": m("state_ok"), "format_rate": m("format_ok"),
                "order_rate": m("order_ok"), "phys_resid": m("physics_residual"),
                "phys_score": m("phys_score"), "reward": m("reward"), "ics": m("ics")}

    summary = {"model": args.model, "fewshot": args.fewshot, "overall": agg(records)}
    for task in sorted({x["task"] for x in records}):
        summary[f"task_{task}"] = agg([x for x in records if x["task"] == task])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_records.jsonl"), "w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    if args.dump_predictions:
        with open(args.dump_predictions, "w", encoding="utf-8") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
