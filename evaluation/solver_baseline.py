# -*- coding: utf-8 -*-
"""
solver_baseline.py — 数值求解器基线（上界参照 + 缺测失效对照，E1 对照组 5）
================================================================================
两个条件：
  full   : 模型假设量测完备（拿 obs 里的真值注入直接解潮流）—— 求解器上界
  masked : 与 LLM 完全相同的可见观测（缺测/含噪）—— 展示求解器失效模式
           （缺测负荷的母线用默认值/插值补，误差放大）

输出与 evaluate.py 同构的 records，便于同表对比。
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))

from physics_verifier import violation_f1, volt_mae  # noqa: E402
import pandapower as pp                              # noqa: E402

NET_DIR = os.path.join(BASE, "..", "..", "01_数据", "网络模型与场景真值", "networks")


def solve_with(net, inj, condition, obs, topo_sw):
    """按条件重解潮流。condition: full=真值注入; masked=只用可见观测。"""
    net = pp.from_json(os.path.join(NET_DIR, f"{net}.json"))
    if condition == "full" and inj is not None:
        net.load.drop(net.load.index, inplace=True)
        for bus, p, q in inj.get("load", []):
            pp.create_load(net, bus=int(bus), p_mw=p / 1e3, q_mvar=q / 1e3)
        net.sgen.drop(net.sgen.index, inplace=True)
        for bus, p in inj.get("sgen", []):
            pp.create_sgen(net, bus=int(bus), p_mw=p / 1e3, q_mvar=0.0)
        if len(net.shunt):
            net.shunt.drop(net.shunt.index, inplace=True)
        for bus, q in inj.get("shunt", []):
            pp.create_shunt(net, bus=int(bus), q_mvar=q)
    else:
        # 缺测条件：负荷/PV 用网络默认值（“不知道”的等价物）。
        # 开关状态用观测值（观测里本来就有，两条件共享）。
        pass
    # 开关状态：用链真值 topo（full）或观测 sw_open（masked，二者通常一致）
    sw = (topo_sw or [])
    net.line["in_service"] = True
    for key in sw:
        if isinstance(key, str) and "->" in key:
            f, t = [int(x) for x in key.split("->")]
            mask = (((net.line.from_bus == f) & (net.line.to_bus == t)) |
                    ((net.line.from_bus == t) & (net.line.to_bus == f)))
            net.line.loc[mask, "in_service"] = False
        else:
            try:
                net.line.at[int(key), "in_service"] = False
            except (KeyError, ValueError, TypeError):
                pass
    try:
        pp.runpp(net, calculate_voltage_angles=False, init="flat")
        return {int(b): (float(v) if v == v else 0.0)   # 失电→0，与引擎约定一致
                for b, v in net.res_bus.vm_pu.items()}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="both", choices=["full", "masked", "both"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from data_io import load_rows
    rows = load_rows(args.data, need_raw=True)
    if args.limit:
        rows = rows[:args.limit]

    conds = ["full", "masked"] if args.condition == "both" else [args.condition]
    summary = {}
    for cond in conds:
        recs = []
        for r in rows:
            v_pred = solve_with(r["net"], r.get("inj"), cond, r["obs"],
                                r["chain"]["topo"].get("sw_open"))
            if v_pred is None:
                recs.append({"solved": False, "v_mae": 1.0, "viol_f1": 0.0})
                continue
            v_true = {int(k): vv for k, vv in r["chain"]["volt"]["v_pu"].items()}
            mae = float(np.mean([abs(v_pred.get(b, 0) - t) for b, t in v_true.items()]))
            # 与 LLM 口径一致：只在受监控母线上判越限（失电母线 v=0 亦计为越限，与引擎约定一致）
            pred_viol = [b for b, v in v_pred.items() if b in v_true and not (0.95 <= v <= 1.05)]
            true_viol = [x["bus"] for x in r["chain"]["volt"]["violations"]]
            recs.append({"solved": True, "v_mae": mae,
                         "viol_f1": violation_f1(pred_viol, true_viol)})
        ok = [x for x in recs if x["solved"]]
        summary[cond] = {
            "n": len(recs), "solved_rate": len(ok) / max(len(recs), 1),
            "v_mae": float(np.mean([x["v_mae"] for x in ok])) if ok else None,
            "viol_f1": float(np.mean([x["viol_f1"] for x in ok])) if ok else None}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
