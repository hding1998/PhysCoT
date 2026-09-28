# -*- coding: utf-8 -*-
"""
timesfm_baseline.py — Chronos 时序基线（E1 对照组：零样本时序预测 vs 物理因果链推理）
================================================================================
动机：E1 需要一个"纯数据驱动时序预测"对手——Chronos-Bolt（zero-shot）只看历史量测
序列，不接触网络拓扑/潮流方程，用于衬托 PhysCoT 在含噪缺测观测下的物理约束推理优势。

数据来源：data_io 读 raw_eval_*.jsonl 原始样本。注意 obs 是"当前断面快照"
（v_meas_pu 为单点值），不含真实历史序列，因此每个可见监测点的历史按如下规则构造：
  1) 若 obs 含真实序列字段（v_meas_series: {bus: [v_t-H+1, ..., v_t]}，预留接口）→ 直接用；
  2) 否则用参数化日曲线（scenario_gen.parametric_daily_curve）把当前量测沿时间轴反推：
        v_hist(h - k) = v_obs * shape(h - k) / shape(h),  k = 1..H-1
     即"同一日曲线形状下的历史断面"——快照观测下唯一不泄漏真值的历史代理；
  3) 两者都不可得（无 hour / 无可见量测母线 / 日曲线形状因子退化）→ 跳过该样本并计数报告。
Chronos 对每条母线历史做 1 步前预测，作为该母线目标时刻的电压预测。

指标与 E1 对齐：v_mae / viol_f1 复用 physics_verifier.volt_mae / violation_f1
（volt_mae 口径含缺预测母线 0.05 pu 惩罚；越限判定与 dsse_baseline 相同——只在
受监控母线上判），state_acc = 预测越限集空→normal / 非空→abnormal。
输出 JSON 与 evaluate.py 同构（overall + 按 net 分组 + *_records.jsonl），
可被 summarize_results.py 直接识别（文件名形如 chronos_<evalset>.json）。

注意：T3 干预样本的链真值是"干预后"断面，Chronos 看不到 do 动作（时序基线不具备
干预条件化能力），这正是该对照组要展示的能力缺口，不是实现遗漏。

用法：
  python timesfm_baseline.py --data datasets/eval_id.jsonl --out results/chronos_id.json
  python timesfm_baseline.py --data datasets/eval_id.jsonl --out results/chronos_id.json --model_size base
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "data_engine"))

from physics_verifier import volt_mae, violation_f1    # noqa: E402
from scenario_gen import parametric_daily_curve        # noqa: E402

MODEL_IDS = {"small": "amazon/chronos-bolt-small",
             "base": "amazon/chronos-bolt-base"}

V_LOW, V_HIGH = 0.95, 1.05          # 与 scenario_gen 的电压合格边界一致


def load_chronos(model_size, device="auto"):
    """加载 Chronos-Bolt pipeline；缺依赖时给出明确安装提示。"""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as e:
        raise SystemExit(
            "未安装 chronos-forecasting（或 torch），无法运行 Chronos 基线。\n"
            "请先执行：  pip install chronos-forecasting\n"
            f"原始错误：{e}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 本机 HF 直连不可用：优先使用本地缓存目录（ModelScope 下载），否则回退 HF repo id
    import os
    local = os.environ.get("CHRONOS_LOCAL_DIR") or os.path.join(
        "D:/hf_cache/models", f"chronos-bolt-{model_size}")
    model_id = local if os.path.exists(os.path.join(local, "model.safetensors")) \
        else MODEL_IDS[model_size]
    print(f"[chronos] loading {model_id} on {device}")
    return BaseChronosPipeline.from_pretrained(model_id, device_map=device)


def extract_histories(obs, context_len):
    """构造 {bus: [v_1..v_H]} 历史序列。返回 (histories, mode) 或 (None, skip_reason)。"""
    series = obs.get("v_meas_series")           # 预留接口：真实历史序列
    if isinstance(series, dict) and series:
        out = {}
        for b, seq in series.items():
            seq = [float(x) for x in seq if isinstance(x, (int, float))]
            if len(seq) >= context_len:
                out[int(b)] = seq[-context_len:]
        if out:
            return out, "real_series"
        return None, "real_series_too_short"

    hour = obs.get("hour")
    if not isinstance(hour, (int, float)):
        return None, "no_hour"
    h0 = float(hour)
    s0 = float(parametric_daily_curve(h0, "residential"))
    if s0 < 1e-3:
        return None, "degenerate_shape"         # 形状因子≈0，无法按比例反推
    out = {}
    for b, v in obs.get("v_meas_pu", {}).items():
        if not isinstance(v, (int, float)):
            continue                            # "NA" 缺测母线：无历史 → 无预测
        out[int(b)] = [float(v) * float(parametric_daily_curve(h0 - k, "residential")) / s0
                       for k in range(context_len - 1, 0, -1)] + [float(v)]
    if not out:
        return None, "no_visible_meas"
    return out, "curve_reconstructed"


def predict_voltage(pipe, histories):
    """对每条母线历史做 1 步预测，取中位数，裁剪到物理合理区间。返回 {bus: v_pu}。"""
    import torch
    buses = sorted(histories)
    ctx = [torch.tensor(histories[b], dtype=torch.float32) for b in buses]
    try:
        quantiles, _ = pipe.predict_quantiles(ctx, prediction_length=1,
                                              quantile_levels=[0.5])
        med = quantiles[:, -1, 0]
    except (AttributeError, TypeError):
        # 旧版接口：predict 返回 [batch, pred_len, quantiles] 或 [batch, pred_len]
        preds = pipe.predict(ctx, prediction_length=1)
        arr = preds.cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
        med = np.median(arr[:, -1, :], axis=-1) if arr.ndim == 3 else arr[:, -1]
    med = np.asarray(med, dtype=float).ravel()
    return {b: float(np.clip(m, 0.0, 1.3)) for b, m in zip(buses, med)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="raw_eval_*.jsonl 或 eval_*.jsonl（自动回退 raw）")
    ap.add_argument("--out", required=True, help="输出 JSON，命名建议 chronos_<evalset>.json")
    ap.add_argument("--model_size", default="small", choices=list(MODEL_IDS),
                    help="Chronos-Bolt 规格（zero-shot）")
    ap.add_argument("--context", type=int, default=48, help="历史序列长度（小时点数）")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from data_io import load_rows
    rows = load_rows(args.data, need_raw=True)
    if args.limit:
        rows = rows[:args.limit]

    pipe = load_chronos(args.model_size, args.device)

    recs, skipped, mode_cnt = [], {}, {}
    for r in rows:
        hist, mode = extract_histories(r.get("obs", {}), args.context)
        if hist is None:
            skipped[mode] = skipped.get(mode, 0) + 1
            continue
        mode_cnt[mode] = mode_cnt.get(mode, 0) + 1
        v_pred = predict_voltage(pipe, hist)
        v_true = {str(k): vv for k, vv in r["chain"]["volt"]["v_pu"].items()}
        # 与 dsse_baseline 同口径：只在受监控母线（truth 的 v_pu 键）上判越限
        v_true_int = {int(k): vv for k, vv in v_true.items()}
        pred_viol = [b for b, v in v_pred.items() if b in v_true_int and not (V_LOW <= v <= V_HIGH)]
        true_viol = [x["bus"] for x in r["chain"]["volt"]["violations"]]
        state_pred = "normal" if not pred_viol else "abnormal"
        recs.append({"sample_id": r["sample_id"], "net": r["net"], "task": r["task"],
                     "v_mae": volt_mae({str(b): v for b, v in v_pred.items()}, v_true),
                     "violation_f1": violation_f1(pred_viol, true_viol),
                     "state_ok": float(state_pred == r["answer"].get("state"))})

    def agg(recs_):
        m = lambda k: float(np.mean([x[k] for x in recs_])) if recs_ else None
        return {"n": len(recs_), "v_mae": m("v_mae"), "viol_f1": m("violation_f1"),
                "state_acc": m("state_ok"),
                "format_rate": None, "phys_resid": None, "ics": None}

    summary = {"overall": agg(recs),
               "history_mode": mode_cnt,
               "skipped": skipped,
               "model": MODEL_IDS[args.model_size],
               "context_len": args.context}
    for net in sorted({x["net"] for x in recs}):
        summary[net] = agg([x for x in recs if x["net"] == net])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_records.jsonl"), "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"history_mode: {mode_cnt} | skipped: {skipped}")
    print("->", args.out)


if __name__ == "__main__":
    main()
