# -*- coding: utf-8 -*-
"""
scenario_gen.py — PhysCoT 数据引擎核心
========================================
职责：把配电网运行场景 → 带因果链真值的训练样本（T1 前向推演 / T3 干预推演）。

因果链（对齐 CausalWM 的 Observation → Motion → Geometry → Future）：
  Observation(部分含噪量测) → Topo(拓扑状态) → Flow(功率流) → Volt(电压分布) → Answer(结论)

关键性质：
  * 链上每一环的真值都由 pandapower 潮流过程自动导出 —— 零人工标注；
  * 观测与真值分离：观测做缺测/噪声/坏数据注入，真值保持完整（教师信号）；
  * T3 干预：do(action) 后重解潮流，得到干预后链真值与前后差分。

用法：
  python scenario_gen.py --net IEEE33 --n 200 --out smoke_test.jsonl
  python scenario_gen.py --net all_id --n 20000 --out sft_train.jsonl --workers 8
"""
import os
import json
import copy
import argparse
import numpy as np
import pandas as pd
import pandapower as pp

BASE = os.path.dirname(os.path.abspath(__file__))
NET_DIR = os.path.join(BASE, "..", "..", "01_数据", "网络模型与场景真值", "networks")
CURVE_DIR = os.path.join(BASE, "..", "..", "01_数据", "curve_library")  # 真实曲线库(可选)

ID_NETS = ["IEEE33", "CIGRE_MV", "SimBench_1-MV-urban--0-sw"]
# IEEE145 已剔除（2026-09-21 本机排查）：该网收敛域为刀刃型（负荷 ±1% 收敛、±20% 在
# nr/iwamoto/bfsw/gs 四种算法与放宽无功限值下均发散），任何场景扰动都无法产出样本。
OOD_NETS = ["PEGASE89", "IEEE118", "Illinois200",
            "SimBench_1-MV-semiurb--0-sw", "SimBench_1-MV-rural--0-sw"]

V_LOW, V_HIGH = 0.95, 1.05          # 电压合格边界 (pu)
LOADING_LIMIT = 100.0                # 线路载流上限 (%)
RNG = np.random.default_rng


# ----------------------------------------------------------------------------
# 参数化日曲线（真实曲线库存在时被替换/混合）
# ----------------------------------------------------------------------------
def parametric_daily_curve(hour_h, kind="residential", season="summer", weekend=False):
    """返回 0-1 归一化负荷/光伏形状因子。hour_h: 浮点小时。"""
    h = np.asarray(hour_h, dtype=float)
    if kind == "pv":
        # 太阳辐射钟形：日出 6h 日落 18h，正午峰值
        s = np.clip(1 - ((h - 12.5) / 5.5) ** 2, 0, None)
        if season == "winter":
            s = np.clip(1 - ((h - 13.0) / 4.0) ** 2, 0, None) * 0.7
        return s
    if kind == "residential":
        base = 0.35 + 0.10 * np.sin(2 * np.pi * (h - 14.0) / 24.0)
        morning = 0.35 * np.exp(-((h - 7.5) / 1.3) ** 2)
        evening = 0.55 * np.exp(-((h - 19.5) / 2.0) ** 2)
        extra = 0.15 * np.exp(-((h - 12.0) / 1.5) ** 2) if weekend else 0.0
        return base + morning + evening + extra
    # commercial: 白天平顶
    return 0.25 + 0.7 * (np.exp(-((h - 9.0) / 1.8) ** 2) + np.exp(-((h - 18.0) / 2.2) ** 2))


class CurveLibrary:
    """真实表计曲线库（由 preprocess_real_curves.py 生成）。存在则按小时采样真实曲线。"""

    def __init__(self):
        self.lib = None
        p = os.path.join(CURVE_DIR, "curves.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            self.lib = df

    def sample_factor(self, rng, hour_h, kind="residential"):
        if self.lib is None or rng.random() < 0.5:   # 一半用参数曲线（鲁棒性）
            return float(parametric_daily_curve(hour_h, kind))
        sub = self.lib[self.lib["kind"] == kind]
        if len(sub) == 0:
            return float(parametric_daily_curve(hour_h, kind))
        row = sub.iloc[rng.integers(len(sub))]
        h0 = int(hour_h)
        f = max(0.0, float(row[f"h{h0:02d}"]))
        return f / max(1e-6, float(row["peak"])) if row["peak"] > 0 else 0.3


# ----------------------------------------------------------------------------
# 场景构建
# ----------------------------------------------------------------------------
def load_net(name):
    return pp.from_json(os.path.join(NET_DIR, f"{name}.json"))


def ensure_pv(net, rng, pv_penetration):
    """给无 PV 网络按渗透率添加分布式光伏（挂在负荷母线子集）。"""
    if len(net.sgen) == 0 or "PhysCoT_pv" not in getattr(net, "_physcot", {}):
        net._physcot = {"PhysCoT_pv": True}
        load_buses = net.load.bus.values
        n_pv = max(1, int(len(load_buses) * pv_penetration))
        sel = rng.choice(load_buses, size=min(n_pv, len(load_buses)), replace=False)
        base_p = net.load.p_mw.values.max() if len(net.load) else 0.1
        for i, b in enumerate(sel):
            pp.create_sgen(net, bus=int(b), p_mw=float(base_p * rng.uniform(0.3, 0.8)),
                       q_mvar=0.0, name=f"PV_{i}", type="PV")
    return net


def sample_scenario(net, rng, curve_lib, cfg):
    """在 net 上采样一个运行断面。含三种模式：
    normal(常态) / peak_stress(晚高峰重载→低电压) / pv_stress(正午高PV轻载→过电压)。"""
    net = copy.deepcopy(net)
    mode = str(rng.choice(["normal", "peak_stress", "pv_stress"],
                          p=cfg.get("mode_probs", (0.5, 0.3, 0.2))))
    if mode == "peak_stress":
        hour = float(rng.uniform(18.0, 21.5)); load_rng = (1.2, 2.0); cloud = (0.3, 0.8)
    elif mode == "pv_stress":
        hour = float(rng.uniform(10.5, 14.5)); load_rng = (0.3, 0.7); cloud = (0.95, 1.0)
    else:
        hour = float(rng.uniform(0, 24)); load_rng = cfg["load_level_range"]; cloud = (0.7, 1.0)
    season = rng.choice(["summer", "winter", "spring"])
    weekend = bool(rng.random() < 0.3)
    pv_penetration = float(rng.uniform(*cfg["pv_penetration_range"]))

    net = ensure_pv(net, rng, pv_penetration)
    n_load = len(net.load)
    if n_load:
        fac = np.array([curve_lib.sample_factor(rng, hour, "residential")
                        for _ in range(n_load)])
        fac = np.clip(fac * rng.uniform(*load_rng), 0.05, None)
        net.load.p_mw = net.load.p_mw.values * fac
        net.load.q_mvar = net.load.q_mvar.values * fac
    if len(net.sgen):
        pv_fac = curve_lib.sample_factor(rng, hour, "pv")
        if mode == "pv_stress":
            pv_fac = max(pv_fac, 0.85)          # 强制正午出力
        cl = rng.uniform(*cloud)
        net.sgen.p_mw = net.sgen.p_mw.values * pv_fac * cl
    meta = {"hour": round(hour, 2), "season": str(season), "weekend": weekend,
            "pv_penetration": round(pv_penetration, 3), "mode": mode}
    return net, meta


def solve(net):
    pp.runpp(net, calculate_voltage_angles=False, init="flat")
    return net


# ----------------------------------------------------------------------------
# 因果链各环节真值提取
# ----------------------------------------------------------------------------
def topo_truth(net):
    """阶段1：拓扑状态（由开关/线路 in_service 推出，不依赖潮流）。"""
    open_lines = net.line.index[~net.line.in_service].tolist()
    g = net.line[["from_bus", "to_bus"]].values[net.line.in_service.values]
    n = len(net.bus)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in g:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for bus in net.bus.index:
        comps.setdefault(find(int(bus)), []).append(int(bus))
    islands = [sorted(c) for root, c in comps.items()
               if not any(int(b) in net.ext_grid.bus.values for b in c)]
    n_edges = int(net.line.in_service.sum())
    # 辐射判据（精确）：森林 ⇔ 活动边数 == 节点数 - 连通分量数（含变压器支路时按线路近似）
    n_comp = len(comps)
    return {
        "sw_open": [int(i) for i in open_lines],
        "n_active_lines": n_edges,
        "n_bus": n,
        "radial": bool(n_edges == n - n_comp),
        "islands": islands,
        "deenergized": sorted([b for isl in islands for b in isl]),
    }


def flow_truth(net, top_k=20):
    """阶段2：功率流（潮流收敛后的支路电流/载流/损耗）。
    部分标准网络(如 case33bw)无热稳参数 → loading 恒 0，因此以支路电流 i_a 为主指标，
    loading 仅在额定值有效时给出。"""
    i_ka = net.res_line.i_ka.copy()
    i_ka = i_ka.fillna(0.0)
    rated = net.line.max_i_ka.values if "max_i_ka" in net.line else np.zeros(len(net.line))
    rated_ok = np.isfinite(rated) & (rated > 0)
    loading = np.where(rated_ok, i_ka.values / np.where(rated_ok, rated, 1.0) * 100.0, np.nan)
    order = i_ka.sort_values(ascending=False).index[:top_k]
    flows = {}
    for i in order:
        f, t = int(net.line.at[i, "from_bus"]), int(net.line.at[i, "to_bus"])
        flows[f"{f}->{t}"] = round(float(i_ka[i]) * 1000.0, 1)   # 单位: A
    li_valid = loading[np.isfinite(loading)]
    over = []
    if rated_ok.any():
        for i in net.line.index[rated_ok]:
            if loading[i] > LOADING_LIMIT:
                over.append(f"{int(net.line.at[i,'from_bus'])}->{int(net.line.at[i,'to_bus'])}")
    loss_kw = float((net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * 1e3)
    return {
        "i_a": {k: v for k, v in flows.items()},
        "max_i_a": round(float(i_ka.max() * 1000.0), 1),
        "max_loading_pct": round(float(li_valid.max()), 1) if len(li_valid) else None,
        "overloaded": over,
        "loss_kw": round(loss_kw, 1),
    }


def volt_truth(net, monitored=None):
    """阶段3：电压分布与越限清单。孤岛/失电母线(vm=NaN)记 0 并标 outage。"""
    v = net.res_bus.vm_pu.copy()
    v.index = net.bus.index.astype(int)
    v = v.fillna(0.0)                                   # 失电 → 0 pu
    if monitored is not None:
        v = v[v.index.isin(monitored)]
    viol = []
    for b, val in v.items():
        if val == 0.0:
            viol.append({"bus": int(b), "v_pu": 0.0, "type": "outage"})
        elif not (V_LOW <= val <= V_HIGH):
            viol.append({"bus": int(b),
                         "v_pu": round(float(val), 3),
                         "type": "under" if val < V_LOW else "over"})
    vv = {int(b): round(float(x), 3) for b, x in v.items()}
    vmax = float(v[v > 0].max()) if (v > 0).any() else 0.0
    vmin = float(v[v > 0].min()) if (v > 0).any() else 0.0
    weakest = int(v[v > 0].idxmin()) if (v > 0).any() else int(v.index[0])
    return {
        "v_pu": vv,
        "v_min": round(vmin, 3),
        "v_max": round(vmax, 3),
        "weakest_bus": weakest,
        "violations": viol,
        "n_viol": len(viol),
    }


def answer_from(volt, flow, task, delta=None):
    """最终结论：由链上变量组装（保证 answer 与链一致 —— 教师信号内部自洽）。"""
    a = {
        "v_min": volt["v_min"], "weakest_bus": volt["weakest_bus"],
        "n_viol": volt["n_viol"],
        "max_i_a": flow["max_i_a"],
        "state": "normal" if (volt["n_viol"] == 0 and not flow["overloaded"]) else "abnormal",
    }
    if task == "T3" and delta is not None:
        a["delta"] = delta
    return a



def inj_truth(net):
    """注入真值（仅评审/奖励端使用，模型不可见）：含母线号，便于校验器重建网络。"""
    return {
        "load": [[int(net.load.at[i, "bus"]), round(float(net.load.at[i, "p_mw"]) * 1e3, 2),
                  round(float(net.load.at[i, "q_mvar"]) * 1e3, 2)] for i in net.load.index],
        "sgen": [[int(net.sgen.at[i, "bus"]), round(float(net.sgen.at[i, "p_mw"]) * 1e3, 2)]
                 for i in net.sgen.index if net.sgen.at[i, "p_mw"] > 0],
        "shunt": [[int(net.shunt.at[i, "bus"]), round(float(net.shunt.at[i, "q_mvar"]), 3)]
                  for i in net.shunt.index if net.shunt.at[i, "in_service"]],
    }


# ----------------------------------------------------------------------------
# 观测生成（缺测/噪声/坏数据）
# ----------------------------------------------------------------------------
def make_observation(net, meta, rng, cfg):
    """模型实际看到的观测：部分电压 + 部分支路电流 + 全网汇总 + 开关状态，含噪。"""
    v = net.res_bus.vm_pu
    vmeas, vtrue = {}, {}
    for b, val in v.items():
        if rng.random() < cfg["v_meas_ratio"]:
            noise = float(rng.normal(0, cfg["noise_std"]))
            x = float(val) * (1 + noise)
            if rng.random() < cfg["bad_data_ratio"]:      # 坏数据：粗差
                x = x * float(rng.uniform(0.7, 1.4))
            vmeas[int(b)] = round(x, 3)
        else:
            vmeas[int(b)] = None                          # 缺测
        vtrue[int(b)] = round(float(val), 3)
    imeas = {}
    i_ka_all = net.res_line.i_ka.fillna(0.0)
    for i in net.line.index:
        if rng.random() < cfg["i_meas_ratio"]:
            f, t = int(net.line.at[i, "from_bus"]), int(net.line.at[i, "to_bus"])
            x = float(i_ka_all[i]) * 1000.0 * (1 + float(rng.normal(0, cfg["noise_std"])))
            imeas[f"{f}->{t}"] = round(x, 1)
    p_head = float((net.res_line.pl_mw.sum() + net.load.p_mw.sum() -
                    net.sgen.p_mw.sum()) * 1e3) if len(net.res_line) else 0.0
    obs = {
        "hour": meta["hour"], "season": meta["season"], "weekend": meta["weekend"],
        "sw_open": net.line.index[~net.line.in_service].tolist(),
        "v_meas_pu": {str(k): (vv if vv is not None else "NA") for k, vv in vmeas.items()},
        "i_meas_a": imeas,
        "p_feeder_kw": round(abs(p_head), 1),
        "pv_est_kw": round(float(net.sgen.p_mw.sum() * 1e3), 1),
        "hint": (f"PV penetration≈{meta['pv_penetration']:.0%}, "
                 f"{meta['season']} {'weekend' if meta['weekend'] else 'workday'}"),
    }
    return obs


# ----------------------------------------------------------------------------
# T3 干预动作（v2：新增 combo 组合干预，用于扣留干预组合泛化测试 C3）
# ----------------------------------------------------------------------------
def sample_intervention(net, rng, cfg=None):
    """随机 do-动作：开关变位 / PV缩放 / 负荷缩放 / 无功补偿 / 组合(两步)。
    cfg["combo_ratio"] 控制组合干预占比（默认 0，训练/扣留切分用）。"""
    cfg = cfg or {}
    exclude_extreme = bool(cfg.get("exclude_extreme", False))   # 训练集：不生成 extreme 档
    sub_cfg = {"combo_ratio": 0.0, "exclude_extreme": exclude_extreme}
    if rng.random() < cfg.get("combo_ratio", 0.0):
        a1 = sample_intervention(net, rng, sub_cfg)
        a2 = sample_intervention(net, rng, sub_cfg)
        return {"type": "combo", "actions": [a1, a2]}
    choices = ["switch", "pv_scale", "load_scale", "var_support"]
    kind = rng.choice(choices)
    act = {"type": str(kind)}

    def _draw_factor(lo, hi):
        # extreme 判据与 do_tag 一致：f>=1.3 或 f<=0.55 —— 训练集需排除（E4c 扣留协议）
        for _ in range(50):
            f = round(float(rng.uniform(lo, hi)), 2)
            if not exclude_extreme or (0.55 < f < 1.3):
                return f
        return 1.0

    if kind == "switch":
        # 从在运线路中选一条断开（或反之）
        active = net.line.index[net.line.in_service].tolist()
        if len(active) == 0:
            return sample_intervention(net, rng, cfg)
        i = int(rng.choice(active))
        act["line"] = f"{int(net.line.at[i,'from_bus'])}->{int(net.line.at[i,'to_bus'])}"
        act["action"] = "open"
    elif kind == "pv_scale":
        act["scope"] = str(rng.choice(["all", "bus"]))
        act["factor"] = _draw_factor(0.4, 1.4)
    elif kind == "load_scale":
        act["scope"] = str(rng.choice(["all", "bus"]))
        act["factor"] = _draw_factor(0.7, 1.5)
    else:
        act["bus"] = int(rng.choice(net.load.bus.values))
        act["q_kvar"] = int(rng.choice([-800, -400, 400, 800]))
    return act


def do_tag(act):
    """干预标签：供扣留切分（build_datasets holdout 用）。"""
    if act is None:
        return "none"
    if act["type"] == "combo":
        return "combo:" + "+".join(sorted(a["type"] for a in act["actions"]))
    f = act.get("factor")
    if f is not None and (f >= 1.3 or f <= 0.55):
        return f"extreme:{act['type']}"
    return f"single:{act['type']}"


def apply_intervention(net, act):
    net = copy.deepcopy(net)
    if act["type"] == "combo":                      # 顺序施加子动作
        for sub in act["actions"]:
            net = apply_intervention(net, sub)
        return net
    if act["type"] == "switch":
        f, t = [int(x) for x in act["line"].split("->")]
        mask = ((net.line.from_bus == f) & (net.line.to_bus == t)) | \
               ((net.line.from_bus == t) & (net.line.to_bus == f))
        net.line.loc[mask, "in_service"] = False
    elif act["type"] == "pv_scale":
        net.sgen.p_mw = net.sgen.p_mw.values * act["factor"]
    elif act["type"] == "load_scale":
        net.load.p_mw = net.load.p_mw.values * act["factor"]
        net.load.q_mvar = net.load.q_mvar.values * act["factor"]
    elif act["type"] == "var_support":
        pp.create_shunt(net, bus=act["bus"],
                       q_mvar=-act["q_kvar"] / 1e3)  # 容性补偿取负
    return net


# ----------------------------------------------------------------------------
# 样本组装
# ----------------------------------------------------------------------------
def monitored_buses(net, cap=40, rng=None):
    idx = list(net.bus.index.astype(int))
    if len(idx) <= cap:
        return set(idx)
    return set(rng.choice(idx, size=cap, replace=False).tolist())


def evidence_confidence(obs, volt):
    """教师置信度与可拒识（对应申报书"证据锚定与可拒识"，回应审稿意见 M3）。
    依据观测可得的信息量构造代理值（模型推理时同样可得 → 不泄漏）：
      * 量测覆盖率 meas_frac：可见电压母线占比
      * 越限裕度 margin：v_min 距 0.95 / v_max 距 1.05 的边界距离（贴边=易判错=低置信）
    规则（确定性、可审计）：
      * meas_frac < 0.35 → abstain=True（证据不足，输出"insufficient_evidence"）
      * confidence = clip(0.45 + 0.45*meas_frac + 0.10*min(margin/0.02,1), 0.05, 0.97)
    """
    v_meas = obs.get("v_meas_pu", {})
    n_total = max(len(v_meas), 1)
    n_seen = sum(1 for v in v_meas.values() if v != "NA")
    meas_frac = n_seen / n_total
    margins = []
    for b, v in volt.get("v_pu", {}).items():
        if isinstance(v, (int, float)) and v > 0:
            margins.append(min(abs(v - V_LOW), abs(v - V_HIGH)))
    margin = min(margins) if margins else 0.0
    abstain = meas_frac < 0.35
    conf = float(np.clip(0.45 + 0.45 * meas_frac + 0.10 * min(margin / 0.02, 1.0),
                         0.05, 0.97))
    return {"confidence": round(conf, 2), "abstain": bool(abstain),
            "meas_frac": round(meas_frac, 3)}


def gen_one(net_template, net_name, sample_id, cfg, curve_lib, rng):
    task = "T1" if rng.random() < cfg.get("t1_ratio", 0.6) else "T3"
    net0, meta = sample_scenario(net_template, rng, curve_lib, cfg)
    try:
        solve(net0)
    except Exception:
        return None

    mon = monitored_buses(net0, cfg.get("monitor_cap", 40), rng)
    topo0 = topo_truth(net0)
    flow0 = flow_truth(net0, cfg.get("flow_topk", 20))
    volt0 = volt_truth(net0, mon)
    obs = make_observation(net0, meta, rng, cfg)

    if task == "T1":
        chain = {"topo": topo0, "flow": flow0, "volt": volt0}
        ans = answer_from(volt0, flow0, "T1")
        ans.update(evidence_confidence(obs, volt0))
        act = None
    else:
        act = sample_intervention(net0, rng, cfg)
        net1 = apply_intervention(net0, act)
        try:
            solve(net1)
        except Exception:
            return None
        topo1, flow1 = topo_truth(net1), flow_truth(net1, cfg.get("flow_topk", 20))
        volt1 = volt_truth(net1, mon)
        delta = {"dv_min": round(volt1["v_min"] - volt0["v_min"], 3),
                 "d_viol": volt1["n_viol"] - volt0["n_viol"],
                 "d_loss_kw": round(flow1["loss_kw"] - flow0["loss_kw"], 1),
                 "state_after": "normal" if (volt1["n_viol"] == 0 and not flow1["overloaded"]) else "abnormal"}
        chain = {"topo": topo1, "flow": flow1, "volt": volt1}
        ans = answer_from(volt1, flow1, "T3", delta)
        ans.update(evidence_confidence(obs, volt1))
    if ans.get("abstain"):
        ans["conclusion"] = "insufficient_evidence"
        ans["state"] = "unknown"

    return {
        "sample_id": sample_id, "net": net_name, "task": task,
        "obs": obs, "do": act, "do_tag": do_tag(act),
        "chain": chain, "answer": ans,
        "inj": inj_truth(net1 if task == "T3" else net0),      # 注入真值(评审端)
        "chain_pre": {"topo": topo0, "flow": flow0, "volt": volt0} if task == "T3" else None,
    }


DEFAULT_CFG = {
    "pv_penetration_range": (0.1, 0.6),
    "load_level_range": (0.5, 1.6),
    "cloud_range": (0.6, 1.0),
    "v_meas_ratio": 0.6, "i_meas_ratio": 0.3,
    "noise_std": 0.02, "bad_data_ratio": 0.02,
    "monitor_cap": 32, "flow_topk": 15,   # cap 40→32：SimBench-urban 样本 total p99 超 4096（本机 verify_env [E] 实测）
    "t1_ratio": 0.6,
    "combo_ratio": 0.15,          # 组合干预占比（扣留泛化测试用；置0则纯单动作）
}


def _gen_chunk(job):
    """多进程工作单元：(net_name, start_idx, count, seed, cfg) → 样本列表。
    每个 chunk 有独立 seed（由主 seed 与 chunk 序号派生），保证并行结果可复现。"""
    name, start, count, seed, cfg = job
    template = load_net(name)
    curve_lib = CurveLibrary()
    rng = RNG(seed)
    outs = []
    # 潮流不收敛的场景被丢弃并重采样，直到凑满 count（最多尝试 max_retry 倍），
    # 保证各网络样本量与 --n 一致（原版直接丢弃 → 输电风格 OOD 网络产出不足一半）。
    max_attempts = int(count * cfg.get("max_retry", 5))
    k, attempts = start, 0
    while len(outs) < count and attempts < max_attempts:
        attempts += 1
        s = gen_one(template, name, f"{name}-{k:06d}", cfg, curve_lib, rng)
        if s is None:
            continue
        outs.append(json.dumps(s, ensure_ascii=False))
        k += 1
    n_fail = attempts - len(outs)
    return name, outs + [None] * n_fail          # None 仅用于统计失败次数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="IEEE33", help="网络名或 all_id/all_ood/all")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", default=None, help="JSON 覆盖配置文件")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（>1 时按 chunk 派生子 seed）")
    ap.add_argument("--chunk", type=int, default=200, help="每个并行任务的样本数")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.cfg:
        cfg.update(json.load(open(args.cfg, encoding="utf-8")))

    if args.net == "all_id":
        nets = ID_NETS
    elif args.net == "all_ood":
        nets = OOD_NETS
    elif args.net == "all":
        nets = ID_NETS + OOD_NETS
    else:
        nets = args.net.split(",")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    per = max(args.n // len(nets), 1)

    # 任务切分：每张网络 per 条，按 chunk 切块；seed = 主seed*10007 + 网络序号*1000 + 块序号
    jobs = []
    for ni, name in enumerate(nets):
        for ci, start in enumerate(range(0, per, args.chunk)):
            cnt = min(args.chunk, per - start)
            jobs.append((name, start, cnt, args.seed * 10007 + ni * 1000 + ci, cfg))

    n_ok, n_fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as fh:
        if args.workers > 1 and len(jobs) > 1:
            from multiprocessing import Pool
            with Pool(processes=min(args.workers, len(jobs))) as pool:
                for name, outs in pool.imap(_gen_chunk, jobs):
                    for s in outs:
                        if s is None:
                            n_fail += 1
                        else:
                            fh.write(s + "\n"); n_ok += 1
                    print(f"[gen] {name}: +{len(outs)} (累计 {n_ok}, 失败 {n_fail})", flush=True)
        else:
            for job in jobs:
                name, outs = _gen_chunk(job)
                for s in outs:
                    if s is None:
                        n_fail += 1
                    else:
                        fh.write(s + "\n"); n_ok += 1
                print(f"[gen] {name}: +{len(outs)} (累计 {n_ok}, 失败 {n_fail})", flush=True)
    print(f"完成: {n_ok} 样本 -> {out_path}")


if __name__ == "__main__":
    main()
