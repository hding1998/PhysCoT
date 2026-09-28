# -*- coding: utf-8 -*-
"""
summarize_results.py — 汇总 results/ 下的评测 JSON → Markdown 主表 + 论文图（实验手册阶段 5）
====================================================================================
约定文件名：results/<run>_<evalset>.json，例如 sft_v1_id.json / abl_nochain_ood_mv.json /
            grpo_v1_holdout.json / dsse_id.json / solver_id.json / qwen7b_id.json
产出：
  results/summary_main.md          E1 主表（viol_f1 / v_mae / state_acc / format / phys_resid / ICS）
  results/fig_e1_bars.png          各 run 在各评测集上的 viol_f1
  results/fig_e2_residual.png      物理残差箱线图（需 *_records.jsonl）
  results/fig_e3_degradation.png   id → noise 的退化（viol_f1 保持率）
  results/fig_e4_ics.png           ICS：id(T3) vs holdout（衰减比，E4c）
  results/fig_calibration.png      置信度-误差分桶（M3）
用法：python evaluation/summarize_results.py --results results
"""
import os
import re
import json
import glob
import argparse
import numpy as np

EVAL_SETS = ["id", "ood_mv", "ood_tx", "noise", "holdout", "abstain"]

# 口径优先级：full(去limit) > pn(分层抽样) > 无前缀(旧 limit 偏斜口径)
_SUFFIX_PRIORITY = {"full": 3, "pn": 2, "": 1}


def load_all(rd):
    runs = {}
    for p in sorted(glob.glob(os.path.join(rd, "*.json"))):
        name = os.path.basename(p)[:-5]
        if name.endswith("_records") or name.startswith(("e7_", "analysis_", "probe_", "tm60_probe")):
            continue
        m = re.match(r"^(.*)_(id|ood_mv|ood_tx|noise|holdout|abstain)(_pn\d*|_full)?$", name)
        if not m:
            continue
        run, es, suf = m.group(1), m.group(2), (m.group(3) or "").lstrip("_")
        suf = "pn" if suf.startswith("pn") else suf
        d = json.load(open(p, encoding="utf-8"))
        if "overall" in d:                                   # evaluate.py / eval_api.py
            o = d["overall"]
            rec = {"viol_f1": o.get("viol_f1"), "v_mae": o.get("v_mae"),
                   "state_acc": o.get("state_acc"), "format": o.get("format_rate"),
                   "phys_resid": o.get("phys_resid"), "ics": (d.get("task_T3") or {}).get("ics") or o.get("ics"),
                   "n": o.get("n"), "ics_sens": d.get("ics_sensitivity"),
                   "conf_bins": d.get("confidence_vs_error"), "_src": name,
                   "_prio": _SUFFIX_PRIORITY.get(suf, 1)}
        elif "profile" in d or "full" in d:                   # dsse / solver
            for cond, o in d.items():
                if not isinstance(o, dict) or "viol_f1" not in o:
                    continue
                runs.setdefault(f"{run}[{cond}]", {})[es] = {
                    "viol_f1": o.get("viol_f1"), "v_mae": o.get("v_mae"), "n": o.get("n"),
                    "state_acc": None, "format": None, "phys_resid": None, "ics": None,
                    "solved": o.get("solved_rate"), "_src": name, "_prio": 9}
            continue
        else:
            continue
        prev = runs.setdefault(run, {}).get(es)
        if prev is None or rec["_prio"] > prev.get("_prio", 0):
            runs[run][es] = rec
    return runs


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    runs = load_all(args.results)
    if not runs:
        raise SystemExit(f"{args.results}/ 下没有可识别的 <run>_<evalset>.json")

    # ---- 主表 ----
    lines = ["# E1 主表（自动汇总；口径优先级 full > pn分层 > 旧limit，来源列可审计）", "",
             "| run | set | n | viol_f1 | v_mae | state_acc | format | phys_resid | ICS(T3) | 来源 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run, sets in runs.items():
        for es in EVAL_SETS:
            if es in sets:
                r = sets[es]
                lines.append(f"| {run} | {es} | {r.get('n')} | {fmt(r['viol_f1'])} | {fmt(r['v_mae'],4)} | "
                             f"{fmt(r['state_acc'])} | {fmt(r['format'])} | {fmt(r['phys_resid'],4)} | {fmt(r['ics'])} | {r.get('_src','')} |")
    # 退化保持率与 ICS 衰减比
    lines += ["", "## 退化与衰减", "", "| run | viol_f1 保持率 noise/id | ICS 衰减比 holdout/id |", "|---|---|---|"]
    for run, sets in runs.items():
        keep = (sets["noise"]["viol_f1"] / sets["id"]["viol_f1"]) \
            if "noise" in sets and "id" in sets and sets["id"].get("viol_f1") else None
        dec = (sets["holdout"]["ics"] / sets["id"]["ics"]) \
            if "holdout" in sets and "id" in sets and sets["id"].get("ics") else None
        lines.append(f"| {run} | {fmt(keep)} | {fmt(dec)} |")
    out_md = os.path.join(args.results, "summary_main.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] 无 matplotlib，跳过画图"); return

    names = list(runs)
    # E1 bars
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(names)), 4))
    w = 0.8 / len(EVAL_SETS)
    for j, es in enumerate(EVAL_SETS):
        ys = [runs[n].get(es, {}).get("viol_f1") or 0 for n in names]
        ax.bar(np.arange(len(names)) + j * w, ys, w, label=es)
    ax.set_xticks(np.arange(len(names)) + 0.4); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("violation F1"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig_e1_bars.png"), dpi=200); plt.close(fig)

    # E2 residual boxplot（从 records 读）
    data, labels = [], []
    for n in names:
        p = os.path.join(args.results, f"{n}_id_records.jsonl")
        if os.path.exists(p):
            res = [json.loads(l).get("physics_residual") for l in open(p, encoding="utf-8")]
            res = [x for x in res if isinstance(x, (int, float))]
            if res:
                data.append(res); labels.append(n)
    if data:
        fig, ax = plt.subplots(figsize=(max(5, 1.0 * len(labels)), 4))
        ax.boxplot(data, showfliers=False); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("power-flow self-consistency residual (pu)"); fig.tight_layout()
        fig.savefig(os.path.join(args.results, "fig_e2_residual.png"), dpi=200); plt.close(fig)

    # E3 degradation
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in names:
        s = runs[n]
        xs = [es for es in ["id", "noise", "ood_mv", "ood_tx"] if es in s and s[es].get("viol_f1") is not None]
        if len(xs) >= 2:
            ax.plot(xs, [s[es]["viol_f1"] for es in xs], marker="o", label=n)
    ax.set_ylabel("violation F1"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig_e3_degradation.png"), dpi=200); plt.close(fig)

    # E4 ICS
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [runs[n].get("id", {}).get("ics") or 0 for n in names], 0.4, label="ICS id (T3)")
    ax.bar(xs + 0.2, [runs[n].get("holdout", {}).get("ics") or 0 for n in names], 0.4, label="ICS holdout")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig_e4_ics.png"), dpi=200); plt.close(fig)

    # calibration
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in names:
        cb = runs[n].get("id", {}).get("conf_bins")
        if cb:
            ks = sorted(cb); ax.plot(ks, [cb[k] for k in ks], marker="o", label=n)
    ax.set_xlabel("predicted confidence bin"); ax.set_ylabel("mean v_mae"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.results, "fig_calibration.png"), dpi=200); plt.close(fig)
    print("figures ->", args.results)


if __name__ == "__main__":
    main()
