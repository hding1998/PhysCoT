# -*- coding: utf-8 -*-
"""
data_io.py — 评测/基线/探针脚本共用的样本读入
================================================
两种文件格式：
  raw_*.jsonl   scenario_gen 原始样本：obs / do / chain / answer / inj / chain_pre …
  eval_*.jsonl  render.py 渲染样本：prompt / completion / truth{chain, answer, inj, chain_pre}

* evaluate.py 只需 prompt + truth → 两种都可；
* solver/dsse/faithfulness 需要 obs/chain → 必须原始样本；若传入的是渲染文件，
  自动回退到同目录的 raw_<name>.jsonl（build_datasets 总是成对产出）。
"""
import os
import json


def load_rows(path, need_raw=False):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not rows:
        return rows
    is_rendered = "prompt" in rows[0] and "obs" not in rows[0]
    if need_raw and is_rendered:
        d, fn = os.path.split(path)
        raw = os.path.join(d, "raw_" + fn)
        if os.path.exists(raw):
            print(f"[data_io] {fn} 为渲染文件，自动改读原始样本 {os.path.basename(raw)}")
            rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
        else:
            raise SystemExit(f"{path} 缺少 obs/chain 字段且找不到 {raw}；请传入 scenario_gen 原始样本")
    return rows


def ensure_rendered(rows, mode="full_chain"):
    """原始样本 → 渲染样本；已渲染则原样返回。"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_engine"))
    from render import render_sample
    out = []
    for r in rows:
        if "prompt" in r and "truth" in r:
            out.append(r)
        else:
            out.append(render_sample(r, mode))
    return out
