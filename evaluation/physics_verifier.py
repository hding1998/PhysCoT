# -*- coding: utf-8 -*-
"""
physics_verifier.py — 物理校验器与奖励函数（GRPO/DPO 共用，也可独立评测）
================================================================================
设计（对应论文 E2 与 R_physics 奖励）：

  parse_model_output(text)
      → 从模型输出中按 <stage:xxx> 拆出各阶段 JSON（鲁棒解析，容忍前后缀噪声）

  verify(sample, parsed) → dict:
      format_ok      : 阶段齐全 + JSON 合法
      task_scores    : 数值/分类与真值的一致性（v MAE、越限集合 F1、结论 state 正确）
      physics        : 把模型声称的拓扑代回潮流方程重解，比对其声称的电压/电流
                       → self_consistency_residual（模型"链"是否物理自洽）
      ics            : 干预一致性得分（T3）

奖励组装：
      R = 0.40*task + 0.30*physics + 0.20*format + 0.10*chain_order

  terminal_reward(sample, model_text) → dict:
      消融用"终答约束奖励"（arXiv 2603.23004 式）：复用同一解析器，但只评
      <stage:answer> 的结论正确性，忽略中间阶段一致性残差与格式分。
"""
import os
import re
import json
import numpy as np
import pandapower as pp

BASE = os.path.dirname(os.path.abspath(__file__))
NET_DIR = os.path.join(BASE, "..", "..", "01_数据", "网络模型与场景真值", "networks")

STAGE_RE = re.compile(r"<stage:(topo|flow|volt|answer)>(.*?)</stage:\1>", re.S)


# ----------------------------------------------------------------------------
def parse_model_output(text):
    """鲁棒解析模型输出。返回 {stage: dict}；解析失败的阶段缺省。"""
    out = {}
    for m in STAGE_RE.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        try:
            out[name] = json.loads(body)
        except json.JSONDecodeError:
            # 容错：截取第一个 { 到最后一个 }
            i, j = body.find("{"), body.rfind("}")
            if i >= 0 and j > i:
                try:
                    out[name] = json.loads(body[i:j + 1])
                except json.JSONDecodeError:
                    pass
    return out


# ----------------------------------------------------------------------------
def volt_mae(pred_v: dict, true_v: dict):
    """电压字典的 MAE（按真值键对齐；预测缺键按 NA 记 0.05 pu 惩罚）。
    预测若非 dict（如模型误输出为 list），视为整体解析失败：全部按缺失惩罚。"""
    if not isinstance(pred_v, dict):
        pred_v = {}
    errs = []
    for b, vt in true_v.items():
        vp = pred_v.get(str(b), pred_v.get(int(b), None))
        if vp is None or not isinstance(vp, (int, float)):
            errs.append(0.05)          # 缺失惩罚
        else:
            errs.append(abs(float(vp) - float(vt)))
    return float(np.mean(errs)) if errs else 1.0


def violation_f1(pred_viol, true_viol):
    """越限母线集合的 F1。输入为 bus 列表。双空集=完全一致记 1。"""
    p, t = set(int(x) for x in pred_viol), set(int(x) for x in true_viol)
    if not p and not t:
        return 1.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / max(len(p), 1), tp / max(len(t), 1)
    return 2 * prec * rec / (prec + rec)


def physics_self_consistency(net_name, claimed_topo, claimed_volt, inj=None, obs=None):
    """核心物理校验：以模型声称的拓扑为准 + 场景注入真值（评审端持有，模型不可见），
    重解潮流并比对模型声称的电压。残差小 = 链物理自洽（KVL/KCL 意义）。
    inj 缺失时退化为观测粗重构（误差较大，仅调试用）。"""
    obs = obs or {}
    try:
        net = pp.from_json(os.path.join(NET_DIR, f"{net_name}.json"))
        if inj is not None:
            # 由注入真值重建负荷与 PV（先清空再按母线建，保证与场景一致）
            net.load.drop(net.load.index, inplace=True)
            for bus, p_kw, q_kvar in inj.get("load", []):
                pp.create_load(net, bus=int(bus), p_mw=p_kw / 1e3, q_mvar=q_kvar / 1e3)
            net.sgen.drop(net.sgen.index, inplace=True)
            for bus, p_kw in inj.get("sgen", []):
                pp.create_sgen(net, bus=int(bus), p_mw=p_kw / 1e3, q_mvar=0.0)
            if len(net.shunt):
                net.shunt.drop(net.shunt.index, inplace=True)
            for bus, q_mvar in inj.get("shunt", []):
                pp.create_shunt(net, bus=int(bus), q_mvar=q_mvar)
        else:
            p_feeder = obs.get("p_feeder_kw"); pv_est = obs.get("pv_est_kw", 0.0)
            base_p = net.load.p_mw.sum()
            if base_p > 0 and p_feeder:
                scale = float(np.clip((p_feeder / 1e3 + pv_est / 1e3) / base_p, 0.05, 3.0))
            else:
                scale = 1.0
            net.load.p_mw = net.load.p_mw.values * scale
            net.load.q_mvar = net.load.q_mvar.values * scale
            if len(net.sgen) and pv_est:
                s0 = net.sgen.p_mw.sum()
                if s0 > 0:
                    net.sgen.p_mw = net.sgen.p_mw.values * (pv_est / 1e3 / s0)
        # 代入模型声称的开关状态
        if claimed_topo and "sw_open" in claimed_topo:
            net.line["in_service"] = True
            for key in (claimed_topo.get("sw_open") or []):
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
        pp.runpp(net, calculate_voltage_angles=False, init="flat")
        v_solved = {int(b): float(vv) for b, vv in net.res_bus.vm_pu.items()}
    except Exception:
        return None
    if not claimed_volt:
        return None
    vp_map = claimed_volt.get("v_pu", {})
    if not isinstance(vp_map, dict):             # 模型误输出 list/str：无可用声称电压
        return None
    errs = []
    for b, vc in vp_map.items():
        try:
            b = int(b)
        except (TypeError, ValueError):
            continue
        if b in v_solved and isinstance(vc, (int, float)) and v_solved[b] > 0:
            errs.append(abs(float(vc) - v_solved[b]))
    return float(np.mean(errs)) if errs else None     # 残差越小越自洽


def ics_score(pred_delta, true_delta, tol_dv=0.005, tol_state=0):
    """干预一致性得分（T3）：方向正确 + 幅值接近 + 状态判断正确。
    模型把 delta 误输出为标量/其他类型时按 0 分（实测 0.5B 消融模型会出现）。"""
    if not isinstance(pred_delta, dict) or not isinstance(true_delta, dict):
        return 0.0
    if pred_delta is None or true_delta is None:
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


# ----------------------------------------------------------------------------
def verify(sample, model_text):
    parsed = parse_model_output(model_text)
    # 类型消毒：模型可能把某阶段输出成 list/str（实测 0.5B 偶发），统一归并为空 dict 防崩溃
    for _k in ("topo", "flow", "volt", "answer"):
        if _k in parsed and not isinstance(parsed[_k], dict):
            parsed[_k] = {}
    if isinstance(parsed.get("volt"), dict) and not isinstance(parsed["volt"].get("v_pu", {}), dict):
        parsed["volt"]["v_pu"] = {}              # v_pu 被误输出为 list：按无电压处理
    truth = sample.get("truth", sample)          # render 后或原始样本均可
    chain, answer = truth["chain"], truth["answer"]

    stages_present = [k in parsed for k in ("topo", "flow", "volt", "answer")]
    format_ok = all(stages_present)

    # --- task ---
    v_mae = volt_mae(parsed.get("volt", {}).get("v_pu", {}),
                     chain["volt"]["v_pu"]) if "volt" in parsed else 1.0
    pv = parsed.get("volt", {}).get("violations", [])
    if not isinstance(pv, list):                 # 模型误把 violations 输出为计数等标量
        pv = []
    pred_viol_buses = [v.get("bus") for v in pv if isinstance(v, dict)]
    f1 = violation_f1(pred_viol_buses,
                      [v["bus"] for v in chain["volt"]["violations"]])
    state_ok = float(parsed.get("answer", {}).get("state") == answer["state"]) \
        if "answer" in parsed else 0.0

    # --- physics ---
    truth_all = sample.get("truth", sample)
    resid = physics_self_consistency(sample.get("net"), parsed.get("topo"),
                                     parsed.get("volt"), inj=truth_all.get("inj"),
                                     obs=sample.get("obs", {}))
    phys_score = float(np.exp(-80.0 * resid)) if resid is not None else 0.0

    # --- chain order（阶段顺序正确性，防止乱序"装饰链"）---
    order_idx = [model_text.find(f"<stage:{k}>") for k in ("topo", "flow", "volt", "answer")]
    order_ok = float(all(order_idx[i] != -1 and
                         order_idx[i] < order_idx[i + 1] for i in range(3)))

    # --- ICS (T3)（记录 dv 原值，供容差敏感性扫描 M2）---
    ics = None
    dv_pred = dv_true = None
    if sample.get("task") == "T3":
        pd_ = (parsed.get("answer", {}) or {}).get("delta") or {}
        td_ = answer.get("delta") or {}
        ics = ics_score(pd_, td_)
        dv_pred = pd_.get("dv_min"); dv_true = td_.get("dv_min")

    task_score = float(np.exp(-30.0 * v_mae)) * 0.4 + f1 * 0.4 + state_ok * 0.2
    reward = 0.40 * task_score + 0.30 * phys_score + 0.20 * float(format_ok) + 0.10 * order_ok

    return {
        "format_ok": format_ok, "stages": stages_present,
        "v_mae": v_mae, "violation_f1": f1, "state_ok": state_ok,
        "physics_residual": resid, "phys_score": phys_score,
        "order_ok": order_ok, "ics": ics, "task_score": task_score,
        "dv_pred": dv_pred, "dv_true": dv_true,
        "confidence_pred": (parsed.get("answer", {}) or {}).get("confidence"),
        "abstain_pred": (parsed.get("answer", {}) or {}).get("abstain"),
        "reward": reward,
    }


def terminal_reward(sample, model_text):
    """终答约束奖励（GRPO 消融模式，对标 arXiv 2603.23004 式 terminal-constraint reward）。
    复用 parse_model_output 的解析，但只评 <stage:answer> 的结论正确性：
        R = 0.50*state_ok + 0.30*exp(-30*|Δv_min|) + 0.20*1[n_viol 命中]
    T3 样本再按 0.7/0.3 混入 ICS（干预差分 delta 属于终答的一部分）。
    中间阶段（topo/flow/volt）的物理一致性残差、越限集合 F1、格式与顺序分全部忽略。"""
    parsed = parse_model_output(model_text)
    truth = sample.get("truth", sample)
    answer = truth["answer"]
    ans = parsed.get("answer") or {}
    if not isinstance(ans, dict):                # 模型误输出 list/str 时按空处理
        ans = {}

    state_ok = float(ans.get("state") == answer["state"]) if "answer" in parsed else 0.0
    pv, tv = ans.get("v_min"), answer.get("v_min")
    if isinstance(pv, (int, float)) and isinstance(tv, (int, float)):
        v_min_score = float(np.exp(-30.0 * abs(float(pv) - float(tv))))
    else:
        v_min_score = 0.0
    n_viol_ok = float(ans.get("n_viol") == answer.get("n_viol")) if "answer" in parsed else 0.0

    reward = 0.50 * state_ok + 0.30 * v_min_score + 0.20 * n_viol_ok

    ics = None
    if sample.get("task") == "T3":
        ics = ics_score(ans.get("delta") or {}, answer.get("delta") or {})
        reward = 0.7 * reward + 0.3 * ics

    return {
        "state_ok": state_ok, "v_min_score": v_min_score, "n_viol_ok": n_viol_ok,
        "ics": ics,
        "confidence_pred": ans.get("confidence"),
        "abstain_pred": ans.get("abstain"),
        "reward": reward,
    }


def grpo_reward_fn(sample, model_text):
    """GRPO 训练直接调用的标量奖励。"""
    return verify(sample, model_text)["reward"]


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 自检：用真值文本喂 verify，各项应接近满分（教师信号可被解析器完整还原）
    import sys
    sys.path.insert(0, os.path.join(BASE, '..', 'data_engine'))
    from render import render_sample

    path = os.path.join(BASE, "..", "..", "01_数据", "网络模型与场景真值", "smoke_test.jsonl")
    samples = [json.loads(l) for l in open(path, encoding="utf-8")][:20]
    rows = []
    for s in samples:
        r = render_sample(s, "full_chain")
        v = verify(r, r["completion"])          # 教师输出应拿高分
        rows.append(v)
    m = lambda k: float(np.mean([x[k] if x[k] is not None else 0 for x in rows]))
    print("== 教师信号自检（应接近满分）==")
    print(f"format_ok {m('format_ok'):.2f} | v_mae {m('v_mae'):.5f} | violF1 {m('violation_f1'):.2f} "
          f"| state_ok {m('state_ok'):.2f} | order {m('order_ok'):.2f} | reward {m('reward'):.3f}")
    r_phys = [x["physics_residual"] for x in rows if x["physics_residual"] is not None]
    print(f"physics_residual(拓扑重解) n={len(r_phys)} mean={np.mean(r_phys):.5f}")
    # 反例检验：把答案 stage 换成错误电压，残差应显著增大
    s = samples[0]
    r = render_sample(s, "full_chain")
    bad = json.loads(r["completion"].split("<stage:volt>")[1].split("</stage:volt>")[0])
    bad["v_pu"] = {k: round(vv - 0.03, 3) for k, vv in bad["v_pu"].items()}
    comp_bad = r["completion"].split("<stage:volt>")[0] + "<stage:volt>" + \
        json.dumps(bad) + "</stage:volt>" + r["completion"].split("</stage:volt>")[1]
    v_bad = verify(r, comp_bad)
    print(f"反例(电压整体-0.03): v_mae={v_bad['v_mae']:.4f} reward={v_bad['reward']:.3f} (应显著低于教师)")
