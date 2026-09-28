# -*- coding: utf-8 -*-
"""
rescore_ics.py — ICS 指标审稿前补救重算（纯 CPU，不重跑模型）
================================================================
从交付包 04_评测原始数据/*_records.jsonl 重算四种口径的 ICS：

  ics_orig    原口径：mean(ics) over T3（解析失败=0 计入），应复现 summary json
  ics_cond    条件 ICS：仅在 <stage:answer> 解析成功（records.stages[3]=True）的
              T3 样本上取均值，同时报告解析率 parse_rate
  ics_strict  严格 ICS：|dv_pred| >= STRICT_MIN(=0.002, 中位真值量级) 才给方向分
              (0.4) 与幅值分(0.4)，state 分(0.2) 照常
  ics_base    平凡基线 ICS：同一评测集上 dummy 预测 {dv_min=c, state_after=训练
              先验多数类(abnormal)}，按同一公式打分；报告 c=-0.001 与网格最优 c

口径与 physics_verifier.ics_score 完全一致（逐字复刻，见下）。records 中存有
ics / dv_pred / dv_true / stages，由 ics - (方向分+幅值分) 可精确分解出
state_after 分量（0 或 0.2），因此无需原始文本即可精确重算全部口径。

用法：python rescore_ics.py
产出：ics_rescore_table.csv / ics_rescore_detail.json / 终端打印校验与判定表
"""
import os
import re
import json
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PKG = os.path.join(ROOT, "PhysCoT_论文交付包_20260926")
REC_DIR = os.path.join(PKG, "04_评测原始数据")
DS_DIR = os.path.join(PKG, "07_完整可运行工程", "02_代码框架", "datasets")

TOL_DV = 0.005          # 与 ics_score 默认 tol_dv 一致
STRICT_MIN = 0.002      # 严格口径：|dv_pred| 阈值（中位真值量级）

# (模型, records 基名, 评测集, 评测集 jsonl)
COMBOS = [
    ("v3b",            "v3b_id_pn",          "id",      "eval_id_pn100.jsonl"),
    ("v3b",            "v3b_holdout",        "holdout", "eval_holdout.jsonl"),
    ("v2",             "v2_id_pn",           "id",      "eval_id_pn100.jsonl"),
    ("v2",             "v2_holdout",         "holdout", "eval_holdout.jsonl"),
    ("sft15_a0",       "sft15_a0_id_pn",     "id",      "eval_id_pn100.jsonl"),
    ("sft15_a0",       "sft15_a0_holdout",   "holdout", "eval_holdout.jsonl"),
    ("sft15_a1",       "sft15_a1_id_pn",     "id",      "eval_id_pn100.jsonl"),
    ("sft15_a1",       "sft15_a1_holdout",   "holdout", "eval_holdout.jsonl"),
    ("sft_a0_base",    "sft_a0_base_id_pn",  "id",      "eval_id_pn100.jsonl"),
    ("sft_a0_base",    "sft_a0_base_holdout","holdout", "eval_holdout.jsonl"),
    ("sft_a1_nochain", "sft_a1_nochain_id_pn","id",     "eval_id_pn100.jsonl"),
    ("sft_a1_nochain", "sft_a1_nochain_holdout","holdout","eval_holdout.jsonl"),
    ("deepseek_zs",    "deepseek_zs_id",     "id",      "eval_id_pn100.jsonl"),
    ("deepseek_zs",    "deepseek_zs_holdout","holdout", "eval_holdout.jsonl"),
    ("deepseek_fs",    "deepseek_fs_id",     "id",      "eval_id_pn100.jsonl"),
    ("deepseek_fs",    "deepseek_fs_holdout","holdout", "eval_holdout.jsonl"),
]


# ----------------------------------------------------------------------------
# 逐字复刻 physics_verifier.ics_score（不 import，以免拖入 pandapower）
def ics_score(pred_delta, true_delta, tol_dv=TOL_DV):
    if not isinstance(pred_delta, dict) or not isinstance(true_delta, dict):
        return 0.0
    s = 0.0
    pdv, tdv = pred_delta.get("dv_min"), true_delta.get("dv_min")
    if pdv is not None and tdv is not None:
        same_sign = (pdv > 0) == (tdv > 0)
        close = abs(pdv - tdv) <= tol_dv
        s += (0.4 if same_sign else 0.0) + (0.4 if close else 0.0)
    ps, ts = pred_delta.get("state_after"), true_delta.get("state_after")
    if ps is not None and ps == ts:
        s += 0.2
    return s


def ics_strict(pred_dv, true_dv, state_comp, tol_dv=TOL_DV, vmin=STRICT_MIN):
    """严格口径：|dv_pred| >= vmin 才给方向/幅值分；state 分量照旧。"""
    s = state_comp
    if pred_dv is not None and true_dv is not None and abs(pred_dv) >= vmin:
        same_sign = (pred_dv > 0) == (true_dv > 0)
        close = abs(pred_dv - true_dv) <= tol_dv
        s += (0.4 if same_sign else 0.0) + (0.4 if close else 0.0)
    return s


def dir_mag(dv_pred, dv_true, tol_dv=TOL_DV):
    """由 records 的 dv_pred/dv_true 重算方向+幅值分（ics_score 前半）。"""
    if dv_pred is None or dv_true is None:
        return 0.0
    same_sign = (dv_pred > 0) == (dv_true > 0)
    close = abs(dv_pred - dv_true) <= tol_dv
    return (0.4 if same_sign else 0.0) + (0.4 if close else 0.0)


# ----------------------------------------------------------------------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def train_prior_state():
    """训练先验：sft_train.jsonl 中 T3 样本 answer.delta.state_after 多数类。"""
    c = collections.Counter()
    for l in open(os.path.join(DS_DIR, "sft_train.jsonl"), encoding="utf-8"):
        s = json.loads(l)
        if s.get("task") == "T3":
            d = ((s.get("truth") or {}).get("answer") or {}).get("delta") or {}
            c[d.get("state_after")] += 1
    return c.most_common(1)[0][0], dict(c)


def eval_t3_truth(eval_file):
    """评测集 T3 真值：[(sample_id, dv_true, state_after_true)]。"""
    out = []
    for s in load_jsonl(os.path.join(DS_DIR, eval_file)):
        if s.get("task") != "T3":
            continue
        d = ((s.get("truth") or {}).get("answer") or {}).get("delta") or {}
        out.append((s["sample_id"], d.get("dv_min"), d.get("state_after")))
    return out


def trivial_baseline(truth, state_const):
    """平凡基线：恒定 dv_min=c + 恒定 state。网格搜索最优 c（原口径/严格口径）。"""
    grid = [round(x, 4) for x in np.arange(-0.05, 0.0201, 0.0005)]
    best = {"orig": (-1.0, None), "strict": (-1.0, None)}
    for c in grid + [-0.001]:
        so, ss = [], []
        for _, tdv, ts in truth:
            pd = {"dv_min": c, "state_after": state_const}
            td = {"dv_min": tdv, "state_after": ts}
            so.append(ics_score(pd, td))
            st = 0.2 if ts == state_const else 0.0
            ss.append(ics_strict(c, tdv, st))
        mo, ms = float(np.mean(so)), float(np.mean(ss))
        if mo > best["orig"][0]:
            best["orig"] = (mo, c)
        if ms > best["strict"][0]:
            best["strict"] = (ms, c)
    fix = float(np.mean([ics_score({"dv_min": -0.001, "state_after": state_const},
                                   {"dv_min": tdv, "state_after": ts})
                         for _, tdv, ts in truth]))
    return {"base_fix_-0.001": fix,
            "base_best_orig": best["orig"][0], "c_best_orig": best["orig"][1],
            "base_best_strict": best["strict"][0], "c_best_strict": best["strict"][1]}


def boot_ci(a, b, n_boot=10000, seed=2026):
    """配对 bootstrap：mean(a-b) 的 95% 百分位 CI。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = [float(np.mean(a[idx] - b[idx])) for idx in
             rng.integers(0, n, size=(n_boot, n))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(a - b)), float(lo), float(hi)


# ----------------------------------------------------------------------------
def rescore_combo(model, base, es, eval_file, prior_state):
    rp = os.path.join(REC_DIR, base + "_records.jsonl")
    if not os.path.exists(rp):
        return None
    recs = load_jsonl(rp)
    t3 = [r for r in recs if r["task"] == "T3"]

    per = []           # 每 T3 样本: (orig_ics, strict_ics, answer_parsed, dv_parsed)
    for r in t3:
        ics = r.get("ics") or 0.0
        dp, dt = r.get("dv_pred"), r.get("dv_true")
        base_dm = dir_mag(dp, dt)
        state_comp = 0.2 if (ics - base_dm) > 0.1 else 0.0   # 精确分解 state 分量
        stages = r.get("stages") or []
        ans_ok = bool(stages[3]) if len(stages) > 3 else False
        per.append({"ics": ics, "strict": ics_strict(dp, dt, state_comp),
                    "ans_ok": ans_ok, "dv_ok": dp is not None,
                    "sample_id": r["sample_id"]})

    n = len(per)
    ics_orig = float(np.mean([p["ics"] for p in per]))
    parsed = [p for p in per if p["ans_ok"]]
    parse_rate = len(parsed) / n
    cond = float(np.mean([p["ics"] for p in parsed])) if parsed else None
    dv_parsed = [p for p in per if p["dv_ok"]]
    cond_dv = float(np.mean([p["ics"] for p in dv_parsed])) if dv_parsed else None
    strict = float(np.mean([p["strict"] for p in per]))

    # --- 平凡基线（同一评测集 T3 真值）---
    truth = eval_t3_truth(eval_file)
    bl = trivial_baseline(truth, prior_state)
    # 每样本基线分数（最优 c，原口径），用于配对 bootstrap
    c_best = bl["c_best_orig"]
    base_per = [ics_score({"dv_min": c_best, "state_after": prior_state},
                          {"dv_min": tdv, "state_after": ts})
                for _, tdv, ts in truth]
    # records 与评测集真值按 sample_id 对齐（两者 T3 集合一致，已核验）
    truth_map = {sid: (tdv, ts) for sid, tdv, ts in truth}
    base_aligned = [ics_score({"dv_min": c_best, "state_after": prior_state},
                              {"dv_min": truth_map[p["sample_id"]][0],
                               "state_after": truth_map[p["sample_id"]][1]})
                    for p in per]
    d_diff, d_lo, d_hi = boot_ci([p["ics"] for p in per], base_aligned)
    # 严格口径的基线须用严格口径下的最优常数 c_best_strict（否则口径不一致）
    cs_best = bl["c_best_strict"]
    s_diff, s_lo, s_hi = boot_ci([p["strict"] for p in per],
                                 [ics_strict(cs_best, truth_map[p["sample_id"]][0],
                                             0.2 if truth_map[p["sample_id"]][1] == prior_state else 0.0)
                                  for p in per])

    # --- 与 summary json 校验 ---
    sp = os.path.join(REC_DIR, base + ".json")
    summ_ics = None
    if os.path.exists(sp):
        sd = json.load(open(sp, encoding="utf-8"))
        summ_ics = (sd.get("task_T3") or {}).get("ics")
        if summ_ics is None:
            summ_ics = (sd.get("overall") or {}).get("ics")

    return {"model": model, "set": es, "records": base, "n_t3": n,
            "ics_orig": ics_orig, "summary_ics": summ_ics,
            "ics_diff_vs_summary": None if summ_ics is None else ics_orig - summ_ics,
            "parse_rate": parse_rate, "ics_cond": cond,
            "dv_parse_rate": len(dv_parsed) / n, "ics_cond_dv": cond_dv,
            "ics_strict": strict, **bl,
            "boot_orig_vs_base": [d_diff, d_lo, d_hi],
            "boot_strict_vs_base": [s_diff, s_lo, s_hi],
            "_per_ics": [p["ics"] for p in per],
            "_per_strict": [p["strict"] for p in per],
            "_per_sample_id": [p["sample_id"] for p in per]}


def main():
    prior_state, prior_dist = train_prior_state()
    print(f"[训练先验] state_after 多数类 = {prior_state}  分布 = {prior_dist}")

    rows = []
    for model, base, es, ef in COMBOS:
        r = rescore_combo(model, base, es, ef, prior_state)
        if r is None:
            print(f"[skip] {base} 无 records")
            continue
        rows.append(r)

    # ---- 口径校验 ----
    print("\n== (a) 口径校验：重算 ics_orig vs summary json ==")
    for r in rows:
        d = r["ics_diff_vs_summary"]
        print(f"  {r['records']:26s} rescore={r['ics_orig']:.6f} "
              f"summary={r['summary_ics'] if r['summary_ics'] is None else round(r['summary_ics'],6)} "
              f"diff={'NA' if d is None else f'{d:+.2e}'}")

    # ---- 主表 ----
    hdr = ["model", "set", "n_t3", "ics_orig", "parse", "ics_cond", "ics_strict",
           "base(-0.001)", "base_best(c)", "c*", "base_strict(c*)", "cs*"]
    print("\n== (b) 任务A主表 ==\n" + "\t".join(hdr))
    for r in rows:
        print(f"{r['model']}\t{r['set']}\t{r['n_t3']}\t{r['ics_orig']:.3f}\t"
              f"{r['parse_rate']:.3f}\t"
              f"{('NA' if r['ics_cond'] is None else format(r['ics_cond'],'.3f'))}\t"
              f"{r['ics_strict']:.3f}\t{r['base_fix_-0.001']:.3f}\t"
              f"{r['base_best_orig']:.3f}\t{r['c_best_orig']}\t"
              f"{r['base_best_strict']:.3f}\t{r['c_best_strict']}")

    # ---- 显著性：v3b vs 平凡基线 / v3b vs sft15_a1（id，配对 bootstrap）----
    print("\n== 显著性（配对 bootstrap, mean diff [95% CI]）==")
    for r in rows:
        d, lo, hi = r["boot_orig_vs_base"]
        sd, slo, shi = r["boot_strict_vs_base"]
        print(f"  {r['records']:26s} orig-基线: {d:+.3f} [{lo:+.3f},{hi:+.3f}]   "
              f"strict-基线: {sd:+.3f} [{slo:+.3f},{shi:+.3f}]")
    by = {(r["model"], r["set"]): r for r in rows}
    for opp in ["sft15_a1", "sft15_a0", "v2", "sft_a1_nochain"]:
        for es in ["id", "holdout"]:
            a, b = by.get(("v3b", es)), by.get((opp, es))
            if a and b and a["_per_sample_id"] == b["_per_sample_id"]:
                d, lo, hi = boot_ci(a["_per_ics"], b["_per_ics"])
                sdx, slo, shi = boot_ci(a["_per_strict"], b["_per_strict"])
                print(f"  v3b - {opp} ({es}): orig {d:+.3f} [{lo:+.3f},{hi:+.3f}]   "
                      f"strict {sdx:+.3f} [{slo:+.3f},{shi:+.3f}]")

    # ---- holdout 衰减 ----
    print("\n== id→holdout 衰减比（holdout/id）==")
    for m in ["v3b", "v2", "sft15_a0", "sft_a0_base", "deepseek_zs", "deepseek_fs"]:
        a, b = by.get((m, "id")), by.get((m, "holdout"))
        if a and b:
            print(f"  {m:12s} orig {b['ics_orig']/a['ics_orig']:.3f}   "
                  f"strict {b['ics_strict']/a['ics_strict']:.3f}"
                  if a["ics_orig"] and a["ics_strict"] else f"  {m}: 分母为0")

    # ---- 落盘 ----
    import csv
    with open(os.path.join(HERE, "ics_rescore_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "set", "n_t3", "ics_orig", "summary_ics", "parse_rate",
                    "ics_cond", "dv_parse_rate", "ics_cond_dv", "ics_strict",
                    "base_fix_-0.001", "base_best_orig", "c_best_orig",
                    "base_best_strict", "c_best_strict",
                    "boot_orig_diff", "boot_orig_lo", "boot_orig_hi",
                    "boot_strict_diff", "boot_strict_lo", "boot_strict_hi"])
        for r in rows:
            w.writerow([r["model"], r["set"], r["n_t3"], r["ics_orig"], r["summary_ics"],
                        r["parse_rate"], r["ics_cond"], r["dv_parse_rate"], r["ics_cond_dv"],
                        r["ics_strict"], r["base_fix_-0.001"], r["base_best_orig"],
                        r["c_best_orig"], r["base_best_strict"], r["c_best_strict"],
                        *r["boot_orig_vs_base"], *r["boot_strict_vs_base"]])
    detail = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    json.dump({"prior_state": prior_state, "prior_dist": prior_dist,
               "tol_dv": TOL_DV, "strict_min": STRICT_MIN, "rows": detail},
              open(os.path.join(HERE, "ics_rescore_detail.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n-> ics_rescore_table.csv / ics_rescore_detail.json")


if __name__ == "__main__":
    main()
