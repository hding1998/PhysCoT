# -*- coding: utf-8 -*-
"""
preprocess_real_curves.py — 真实表计数据 → 统一曲线库 curves.parquet
=====================================================================
输入（三个公开数据集，均为真实量测）：
  01_数据/CER/    train-*.parquet   (爱尔兰 CER, 6429 户, 30min, HF镜像版)
  01_数据/LCL/    LCL-FullData.zip  (伦敦 5567 户, 30min, 官方 CSV ~10GB)
  01_数据/ausgrid/Ausgrid_solar_home_data.zip (悉尼 300 户 PV+负荷, 30min)

输出（供 scenario_gen.CurveLibrary 使用）：
  01_数据/curve_library/curves.parquet
  列: id, kind(residential/pv/commercial), h00..h23(24小时归一化形状), peak(日峰值系数), source

用法（可分别跑，缺哪个跳过哪个）：
  python preprocess_real_curves.py --source cer
  python preprocess_real_curves.py --source lcl          # 慢：解压+扫描 10GB CSV
  python preprocess_real_curves.py --source ausgrid
  python preprocess_real_curves.py --source all
"""
import os
import io
import sys
import json
import argparse
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "..", "01_数据")
OUT_DIR = os.path.join(DATA, "curve_library")
OUT = os.path.join(OUT_DIR, "curves.parquet")

N_KEEP = 400          # 每源保留曲线数
SEED = 7


def hours_profile(ts_values, vals):
    """时间戳+值 → 24 小时平均形状（归一化到 0-1）与峰值系数。"""
    df = pd.DataFrame({"h": ts_values.dt.hour, "v": vals}).dropna()
    if len(df) < 24:
        return None
    prof = df.groupby("h")["v"].mean().reindex(range(24))
    if prof.isna().any():
        prof = prof.interpolate()
    pk = prof.max()
    if pk <= 0:
        return None
    return prof.values / pk, float(pk)


def emit(rows, kind, source, out_list):
    for i, (shape, peak) in enumerate(rows):
        rec = {"id": f"{source}_{kind}_{i}", "kind": kind, "peak": peak, "source": source}
        rec.update({f"h{h:02d}": round(float(shape[h]), 4) for h in range(24)})
        out_list.append(rec)


# ----------------------------------------------------------------------------
def process_cer():
    """CER parquet (GluonTS 风格: item_id, start, target[序列])。"""
    files = [os.path.join(DATA, "CER", f) for f in os.listdir(os.path.join(DATA, "CER"))
             if f.endswith(".parquet")]
    if not files:
        print("[skip] CER parquet 不存在"); return []
    rng = np.random.default_rng(SEED)
    out, kept = [], 0
    for fp in files:
        df = pd.read_parquet(fp, columns=["item_id", "start", "target"])
        for _, row in df.iterrows():
            if kept >= N_KEEP:
                break
            if rng.random() > 0.05:            # 抽样
                continue
            t0 = pd.Timestamp(row["start"])
            n = len(row["target"])
            idx = pd.date_range(t0, periods=n, freq="30min")
            r = hours_profile(pd.Series(idx), np.asarray(row["target"], dtype=float))
            if r is None:
                continue
            out.append(r); kept += 1
    recs = []
    emit(out, "residential", "CER", recs)
    print(f"[CER] 保留 {len(out)} 条居民曲线")
    return recs


def process_lcl():
    """LCL zip 内 CC_LCL-FullData.csv (~10GB)：先解压到磁盘，再分块向量化聚合。"""
    zp = os.path.join(DATA, "LCL", "LCL-FullData.zip")
    if not os.path.exists(zp):
        print("[skip] LCL zip 不存在"); return []
    lcl_dir = os.path.join(DATA, "LCL")
    csv_path = os.path.join(lcl_dir, "CC_LCL-FullData.csv")
    if not os.path.exists(csv_path):
        import subprocess
        print("  [LCL] 解压 zip（一次性，~10GB，需数分钟）...")
        subprocess.run(["unzip", "-o", "-q", "LCL-FullData.zip"], cwd=lcl_dir, check=True)
    col_kwh = "KWH/hh (per half hour) "
    usecols = ["LCLid", "DateTime", col_kwh]
    # 第一遍：抽样户号（前几个大块即可）
    want_ids = set()
    for ch in pd.read_csv(csv_path, chunksize=5_000_000, usecols=["LCLid"], dtype={"LCLid": str}):
        want_ids.update(ch["LCLid"].drop_duplicates().sample(frac=0.03, random_state=SEED).tolist())
        if len(want_ids) > 6000:
            break
    want_ids = set(sorted(want_ids)[:6000])
    print(f"  [LCL] 抽样 {len(want_ids)} 户")
    # 第二遍：向量化按 户×小时 聚合
    sums = {}
    for k, ch in enumerate(pd.read_csv(csv_path, chunksize=5_000_000, usecols=usecols,
                                       dtype={"LCLid": str})):
        ch = ch[ch["LCLid"].isin(want_ids)]
        if not len(ch):
            continue
        ch["DateTime"] = pd.to_datetime(ch["DateTime"], errors="coerce")
        ch = ch.dropna(subset=["DateTime"])
        ch["h"] = ch["DateTime"].dt.hour
        ch[col_kwh] = pd.to_numeric(ch[col_kwh], errors="coerce")
        g = ch.groupby(["LCLid", "h"])[col_kwh].agg(["sum", "count"])
        for (lid, h), row in g.iterrows():
            d = sums.setdefault(lid, {"s": np.zeros(24), "c": np.zeros(24)})
            d["s"][h] += row["sum"]; d["c"][h] += row["count"]
        print(f"  [LCL] chunk {k} 完成")
    out = []
    rng = np.random.default_rng(SEED)
    lids = sorted(sums.keys())
    rng.shuffle(lids)
    for lid in lids:
        if len(out) >= N_KEEP:
            break
        d = sums[lid]
        if d["c"].sum() < 500:
            continue
        prof = np.divide(d["s"], np.maximum(d["c"], 1))
        pk = prof.max()
        if pk <= 0:
            continue
        out.append((prof / pk, float(pk)))
    recs = []
    emit(out, "residential", "LCL", recs)
    print(f"[LCL] 保留 {len(out)} 条居民曲线")
    return recs


def process_ausgrid():
    """Ausgrid zip: 年度 CSV, 宽表(行=日, 列=30min), Channel 列区分 GG(PV)/GC(负荷)。"""
    zp = os.path.join(DATA, "ausgrid", "Ausgrid_solar_home_data.zip")
    if not os.path.exists(zp):
        print("[skip] Ausgrid zip 不存在"); return []
    import zipfile
    zf = zipfile.ZipFile(zp)
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csvs:
        print("[skip] Ausgrid zip 内无 csv:", zf.namelist()[:5]); return []
    rows_load, rows_pv = [], []
    for name in csvs:
        try:
            df = pd.read_csv(io.BytesIO(zf.read(name)), skiprows=1, low_memory=False)
        except Exception as e:
            print("  [Ausgrid] 读取失败", name, e); continue
        chan_col = next((c for c in df.columns
                         if "Channel" in c or "Category" in c), None)
        cust_col = next((c for c in df.columns if "Customer" in c), None)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        val_cols = [c for c in df.columns if c.startswith("0:") or ":" in str(c)][:48]
        if not (chan_col and cust_col and val_cols):
            continue
        # 30min 列 → 小时映射
        hour_map = {}
        for c in val_cols:
            h = int(str(c).split(":")[0])
            hour_map.setdefault(h, []).append(c)
        rng = np.random.default_rng(SEED)
        for (cust, chan), g in df.groupby([cust_col, chan_col]):
            target_list = rows_pv if str(chan).startswith("GG") else rows_load
            if len(target_list) >= N_KEEP:
                continue
            vals = g[val_cols].apply(pd.to_numeric, errors="coerce").mean().to_dict()
            hourly = {h: np.nanmean([vals.get(c, np.nan) for c in hour_map[h]])
                      for h in hour_map}
            s = pd.Series(hourly).reindex(range(24)).interpolate()
            pk = s.max()
            if pk <= 0:
                continue
            target_list.append((s.values / pk, float(pk)))
    recs = []
    emit(rows_load, "residential", "Ausgrid", recs)
    emit(rows_pv, "pv", "Ausgrid", recs)
    print(f"[Ausgrid] 负荷 {len(rows_load)} 条 / PV {len(rows_pv)} 条")
    return recs


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["all", "cer", "lcl", "ausgrid"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_recs = []
    fns = {"cer": process_cer, "lcl": process_lcl, "ausgrid": process_ausgrid}
    todo = list(fns) if args.source == "all" else [args.source]
    for src in todo:
        try:
            all_recs.extend(fns[src]())
        except Exception as e:
            print(f"[{src}] 处理失败: {e}")
    # 合并已有库（分批跑时不覆盖）
    if os.path.exists(OUT):
        old = pd.read_parquet(OUT)
        old = old[~old["id"].isin({r["id"] for r in all_recs})]
        all_recs = old.to_dict("records") + all_recs
    pd.DataFrame(all_recs).to_parquet(OUT, index=False)
    print(f"curve library: {len(all_recs)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
