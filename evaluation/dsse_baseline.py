# -*- coding: utf-8 -*-
"""
dsse_baseline.py — 配电网状态估计基线（回应魔鬼审稿意见 C1 的"公平对手"）
================================================================================
动机：原 solver_baseline 的 masked 条件（铭牌默认值硬跑）是稻草人。真正的经典对手是
带伪量测的 WLS 状态估计（DSSE）——配电网处理缺测的标准做法：
  * 可见电压 → 电压量测（std 按噪声水平设定，坏数据放大 std 或被 LNR 剔除）
  * 不可见负荷 → 伪量测：先验 = 铭牌值 × 日曲线形状因子(hour)，std 取先验不确定度
  * PV → 伪量测：先验 = pv_est 按容量分摊，std 较大
  * 零注入母线 → 高置信伪量测（std=1e-3）
  * 馈线首端 P → 注入量测
条件：
  oracle    : 伪量测用真值注入、紧容差（API/符号校验用；注：在含噪+坏数据的真实
              观测下 oracle 反而更差——WLS 过度信任被污染量测，这正是无鲁棒化的
              WLS 的已知弱点，洁净受控测试 v_mae=0.00000 已验证实现正确性）
  profile   : 伪量测用曲线先验 + 工程典型容差（公平对手 —— 标准工程实践）
  （鲁棒化变体 LNR/Huber 为标准扩展，主对比用 WLS）
符号约定（实测确诊）：pandapower bus 注入量测 = 消耗为正；PV 取负；
零注入伪量测必须排除松弛母线，否则强迫净注入为零、全网被压平。

用法：
  python dsse_baseline.py --data datasets/eval_id.jsonl --out results/dsse.json
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))

from physics_verifier import violation_f1        # noqa: E402
from scenario_gen import parametric_daily_curve  # noqa: E402
import pandapower as pp                          # noqa: E402
import pandapower.estimation as est              # noqa: E402

NET_DIR = os.path.join(BASE, "..", "..", "01_数据", "网络模型与场景真值", "networks")

V_STD_BASE = 0.005          # 电压量测 std（与噪声2%量级匹配，工程典型值）
PSEUDO_STD_REL = 0.50       # 伪量测相对 std（50%先验不确定度，DSSE 文献典型）


def build_and_estimate(net_name, obs, inj=None, condition="profile"):
    v_std = 0.003 if condition == "oracle" else V_STD_BASE
    pseudo_rel = 0.03 if condition == "oracle" else PSEUDO_STD_REL
    """构建量测集并跑 WLS。返回估计电压 dict 或 None。
    注：鲁棒化变体（LNR 剔除 / Huber）为标准扩展，主对比用 WLS 即可；
    pandapower 的 estimate 内部即 WLS（含坏数据敏感性——这正是与 LLM 对比的点）。"""
    try:
        net = pp.from_json(os.path.join(NET_DIR, f"{net_name}.json"))
        # 拓扑：观测的开关状态
        net.line["in_service"] = True
        for key in obs.get("sw_open", []):
            try:
                net.line.at[int(key), "in_service"] = False
            except (KeyError, ValueError, TypeError):
                pass
        if len(net.measurement):
            net.measurement.drop(net.measurement.index, inplace=True)
        hour = float(obs.get("hour", 12.0))
        shape = float(parametric_daily_curve(hour, "residential"))
        pv_shape = max(float(parametric_daily_curve(hour, "pv")), 0.0)

        # ---- 负荷/PV 伪量测 ----
        load_buses = set(net.load.bus.tolist())
        load_by_bus = {}
        for i in net.load.index:
            b = int(net.load.at[i, "bus"])
            load_by_bus.setdefault(b, [0.0, 0.0])
            load_by_bus[b][0] += float(net.load.at[i, "p_mw"])
            load_by_bus[b][1] += float(net.load.at[i, "q_mvar"])
        if condition == "oracle" and inj is not None:
            load_by_bus = {}
            for b, p, q in inj.get("load", []):
                load_by_bus.setdefault(int(b), [0.0, 0.0])
                load_by_bus[int(b)][0] += p / 1e3
                load_by_bus[int(b)][1] += q / 1e3
            pv_by_bus = {}
            for b, p in inj.get("sgen", []):
                pv_by_bus[int(b)] = pv_by_bus.get(int(b), 0.0) + p / 1e3
        else:
            # profile 先验：铭牌 × 日曲线形状（不知道真实个体差异 —— 与调度实用做法一致）
            load_by_bus = {b: [v[0] * shape, v[1] * shape] for b, v in load_by_bus.items()}
            pv_by_bus = {}
            pv_total = float(obs.get("pv_est_kw", 0.0)) / 1e3
            cap = net.sgen.p_mw.sum() if len(net.sgen) else 0.0
            if cap > 0 and pv_total > 0:
                for i in net.sgen.index:
                    pv_by_bus[int(net.sgen.at[i, "bus"])] = \
                        pv_total * float(net.sgen.at[i, "p_mw"]) / float(cap)

        for b, (p, q) in load_by_bus.items():
            std = max(p * pseudo_rel, 1e-4)
            pp.create_measurement(net, "p", "bus", p, std, element=int(b))
            if q > 0:
                pp.create_measurement(net, "q", "bus", q, max(q * pseudo_rel, 1e-4),
                                      element=int(b))
        for b, p in pv_by_bus.items():
            if p > 0:
                pp.create_measurement(net, "p", "bus", -p, max(p * 0.6, 1e-4), element=int(b))

        # ---- 零注入母线：高置信伪量测（必须排除松弛母线，否则强迫净注入为零）----
        slack_buses = set(int(b) for b in net.ext_grid.bus)
        for b in net.bus.index:
            b = int(b)
            if b not in slack_buses and b not in load_by_bus and b not in pv_by_bus:
                pp.create_measurement(net, "p", "bus", 0.0, 1e-3, element=b)

        # ---- 可见电压量测 ----
        v_meas = obs.get("v_meas_pu", {})
        for b_str, v in v_meas.items():
            if v == "NA" or not isinstance(v, (int, float)):
                continue
            try:
                b = int(b_str)
            except (TypeError, ValueError):
                continue
            if b in set(net.bus.index.astype(int)):
                pp.create_measurement(net, "v", "bus", float(v), v_std, element=b)
        for b in net.ext_grid.bus:
            pp.create_measurement(net, "v", "bus", 1.0, 1e-4, element=int(b))

        # ---- WLS 求解（profile_lnr：先做最大标准化残差坏数据剔除，再 WLS —— 鲁棒变体，手册 V1）----
        if condition == "profile_lnr":
            try:
                est.remove_bad_data(net, init="flat", tolerance=1e-5, maximum_iterations=50,
                                    rn_max_threshold=3.0)
            except Exception:               # noqa: BLE001  剔除失败则退回普通 WLS
                pass
        ok = est.estimate(net, init="flat", tolerance=1e-5, maximum_iterations=50)
        v = net.res_bus_est.vm_pu
        if ok is False or len(v) == 0 or not np.isfinite(v.values).all():
            return None                     # 未收敛/空结果 → 记为未求解，而不是当 0 pu 参与 MAE
        return {int(b): float(x) for b, x in v.items()}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="oracle,profile",
                    help="oracle | profile | profile_lnr（LNR 坏数据剔除的鲁棒变体）")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from data_io import load_rows
    rows = load_rows(args.data, need_raw=True)
    if args.limit:
        rows = rows[:args.limit]

    summary = {}
    for cond in args.conditions.split(","):
        recs = []
        for r in rows:
            v_pred = build_and_estimate(r["net"], r["obs"], r.get("inj"), cond)
            if v_pred is None:
                recs.append({"solved": False, "v_mae": 1.0, "viol_f1": 0.0})
                continue
            v_true = {int(k): vv for k, vv in r["chain"]["volt"]["v_pu"].items()}
            mae = float(np.mean([abs(v_pred.get(b, 0) - t) for b, t in v_true.items()]))
            # 与 LLM 口径一致：只在受监控母线（truth 的 v_pu 键）上判越限
            pred_viol = [b for b, v in v_pred.items() if b in v_true and not (0.95 <= v <= 1.05)]
            true_viol = [x["bus"] for x in r["chain"]["volt"]["violations"]]
            recs.append({"solved": True, "v_mae": mae, "net": r["net"],
                         "viol_f1": violation_f1(pred_viol, true_viol)})
        ok = [x for x in recs if x["solved"]]
        summary[cond] = {"n": len(recs),
                        "solved_rate": len(ok) / max(len(recs), 1),
                        "v_mae": float(np.mean([x["v_mae"] for x in ok])) if ok else None,
                        "viol_f1": float(np.mean([x["viol_f1"] for x in ok])) if ok else None,
                        "v_mae_unsolved_as_1": float(np.mean([x["v_mae"] for x in recs])),
                        "by_net": {}}
        for net in sorted({x.get("net") for x in ok if x.get("net")}):
            sub = [x for x in ok if x.get("net") == net]
            summary[cond]["by_net"][net] = {"n": len(sub),
                                            "v_mae": float(np.mean([x["v_mae"] for x in sub])),
                                            "viol_f1": float(np.mean([x["viol_f1"] for x in sub]))}
        print(cond, "->", {k: v for k, v in summary[cond].items() if k != "by_net"})
        for net, s in summary[cond]["by_net"].items():
            print(f"    {net:28s} n={s['n']:3d} v_mae={s['v_mae']:.4f} viol_f1={s['viol_f1']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
