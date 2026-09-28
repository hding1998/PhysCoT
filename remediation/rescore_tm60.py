# -*- coding: utf-8 -*-
"""
rescore_tm60.py — tm60 物理校验补救（任务 B）
================================================================
诊断 grpo_tm60 phys_resid 300/300 全 null 的机制，并给出与 ps60 可比的补救数字。

机制结论（由 records 字段即可判定，无需原始文本）：
  verify() 对每个样本无条件调用 physics_self_consistency（physics_verifier.py:197），
  不存在"管线没跑重解"的分支。返回 None 的只有四种情况：潮流异常 / 无 volt 阶段 /
  v_pu 非 dict（标量占位，:136-137）/ 无对齐母线。
  records 中 v_mae==0.05 恰为"volt 阶段在但 v_pu 被消毒为空 dict"（:175-176 的
  类型消毒 + volt_mae 全缺失惩罚 0.05），v_mae==1.0 为"volt 阶段缺失"（:184-185）。

用法（pandapower 重解段需要 py312 环境）：
  C:/Users/admin/miniconda3/envs/py312/python.exe rescore_tm60.py
产出：tm60_remediation.json + 终端打印
"""
import os
import re
import json
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PKG = os.path.join(ROOT, "PhysCoT_论文交付包_20260926")
REC_DIR = os.path.join(PKG, "04_评测原始数据")
CASE_DIR = os.path.join(PKG, "03_案例与文本证据")
DS_DIR = os.path.join(PKG, "07_完整可运行工程", "02_代码框架", "datasets")
EVAL_DIR = os.path.join(PKG, "07_完整可运行工程", "02_代码框架", "evaluation")

STAGE_RE = re.compile(r"<stage:(topo|flow|volt|answer)>(.*?)</stage:\1>", re.S)


def parse_model_output(text):
    """逐字复刻 physics_verifier.parse_model_output。"""
    out = {}
    for m in STAGE_RE.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        try:
            out[name] = json.loads(body)
        except json.JSONDecodeError:
            i, j = body.find("{"), body.rfind("}")
            if i >= 0 and j > i:
                try:
                    out[name] = json.loads(body[i:j + 1])
                except json.JSONDecodeError:
                    pass
    return out


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def classify(run):
    """按 records 字段把每个样本归入 null 机制类别。"""
    recs = load_jsonl(os.path.join(REC_DIR, f"{run}_records.jsonl"))
    cat = collections.Counter()
    for r in recs:
        stages = r.get("stages") or []
        volt_present = bool(stages[2]) if len(stages) > 2 else False
        vm = r.get("v_mae")
        if r.get("physics_residual") is not None:
            cat["resid_ok"] += 1
        elif not volt_present:
            cat["null_volt_stage_missing"] += 1        # 截断/未生成 volt 阶段
        elif vm is not None and abs(vm - 0.05) < 1e-9:
            cat["null_vpu_scalar_placeholder"] += 1    # v_pu 标量→消毒空 dict→全缺失惩罚
        else:
            cat["null_other"] += 1
    n = len(recs)
    return {"n": n, **dict(cat),
            "phys_score_mean": sum(r.get("phys_score") or 0 for r in recs) / n}


def text_evidence():
    """文本证据：tm60 有预测文本的 8 个样本（E5 四例 + probe4 四例）中 v_pu 的形态。"""
    out = {}
    seen = set()
    for f in [os.path.join(CASE_DIR, "E5_奖励黑客案例_tm60.jsonl"),
              os.path.join(REC_DIR, "tm60_probe4_preds.jsonl")]:
        for r in load_jsonl(f):
            if r["sample_id"] in seen:
                continue
            seen.add(r["sample_id"])
            parsed = parse_model_output(r["prediction"])
            volt = parsed.get("volt")
            if not isinstance(volt, dict):
                out[r["sample_id"]] = "volt 阶段缺失/不可解析(截断)"
                continue
            vp = volt.get("v_pu")
            out[r["sample_id"]] = ("标量占位 v_pu=" + json.dumps(vp)) \
                if not isinstance(vp, dict) else f"dict({len(vp)} buses)"
    return out


def broadcast_resolve(sample_ids):
    """补救实验：把 tm60 的标量占位 v_pu=1.0 按面值广播到全部母线，
    复用 physics_verifier.physics_self_consistency（claimed topo + inj 真值重解），
    得到"即使宽宏解释占位"时的残差/phys_score，并与 ps60 同样本实测值对比。"""
    try:
        sys.path.insert(0, EVAL_DIR)
        import physics_verifier as pv
        # 交付包内 07 工程不含 01_数据；网络模型改指向 08_源数据（只读）
        alt = os.path.join(PKG, "08_源数据", "网络模型与场景真值", "networks")
        if not os.path.isdir(pv.NET_DIR):
            pv.NET_DIR = alt
    except Exception as e:
        return {"error": f"pandapower 不可用，跳过广播重解: {e}"}
    truth = {s["sample_id"]: s for s in load_jsonl(os.path.join(DS_DIR, "eval_id_pn100.jsonl"))}
    texts = {}
    for f in [os.path.join(CASE_DIR, "E5_奖励黑客案例_tm60.jsonl"),
              os.path.join(REC_DIR, "tm60_probe4_preds.jsonl")]:
        for r in load_jsonl(f):
            texts.setdefault(r["sample_id"], r["prediction"])
    ps60 = {r["sample_id"]: r for r in load_jsonl(os.path.join(REC_DIR, "grpo_ps60_id_pn_records.jsonl"))}
    rows = []
    for sid in sample_ids:
        if sid not in texts or sid not in truth:
            continue
        s = truth[sid]
        parsed = parse_model_output(texts[sid])
        volt = parsed.get("volt") or {}
        vp = volt.get("v_pu")
        scalar = float(vp) if isinstance(vp, (int, float)) else None
        if scalar is None:
            rows.append({"sample_id": sid, "note": "volt 阶段缺失，无法广播"})
            continue
        import pandapower as pp
        net = pv.pp.from_json(os.path.join(pv.NET_DIR, f"{s['net']}.json"))
        claimed_volt = {"v_pu": {str(b): scalar for b in net.bus.index}}
        resid = pv.physics_self_consistency(s["net"], parsed.get("topo"),
                                            claimed_volt, inj=s["truth"].get("inj"),
                                            obs={})
        import math
        ps = math.exp(-80.0 * resid) if resid is not None else 0.0
        p = ps60.get(sid) or {}
        rows.append({"sample_id": sid, "scalar": scalar,
                     "tm60_broadcast_resid": resid, "tm60_broadcast_phys_score": ps,
                     "ps60_resid": p.get("physics_residual"),
                     "ps60_phys_score": p.get("phys_score")})
    return rows


def main():
    rep = {}
    print("== (1) null 机制分类（records 字段判定）==")
    for run in ["grpo_tm60_id_pn", "grpo_ps60_id_pn"]:
        c = classify(run)
        rep[run] = c
        print(f"  {run}: {c}")

    print("\n== (2) 文本证据：tm60 有原始文本样本的 v_pu 形态 ==")
    te = text_evidence()
    rep["text_evidence"] = te
    for k, v in te.items():
        print(f"  {k}: {v}")

    print("\n== (3) 广播重解补救（标量占位按面值解释；需 pandapower）==")
    br = broadcast_resolve(list(te.keys()))
    rep["broadcast_resolve"] = br
    if isinstance(br, list):
        vals = [r["tm60_broadcast_phys_score"] for r in br if "tm60_broadcast_phys_score" in r]
        for r in br:
            print(" ", json.dumps(r, ensure_ascii=False))
        if vals:
            print(f"  tm60 广播解释 phys_score mean={sum(vals)/len(vals):.4f} (n={len(vals)})"
                  f"  vs ps60 全量实测 phys_score=0.1188 / resid=0.0438")
    else:
        print(" ", br)

    json.dump(rep, open(os.path.join(HERE, "tm60_remediation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n-> tm60_remediation.json")


if __name__ == "__main__":
    main()
