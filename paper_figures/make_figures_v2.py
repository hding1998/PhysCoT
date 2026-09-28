# -*- coding: utf-8 -*-
"""
Figure set v2 for the EE paper -- implements the 2026-09-28 beautification list:
  * one global colour-blind-safe palette with fixed semantic roles
  * sans-serif type, TrueType-embedded vector PDFs (fonttype 42)
  * fig1: layer bands + line icons + verifier->DPO/GRPO feedback arrows
  * fig2: semantic recolour, dagger floor markers, +7.3-pt gap bracket, n= in ticks
  * fig3: DSSE-profile reference bars per distribution, abstain caveat note
  * fig4: floor bars hollow+hatched (+ dagger), scale = light/deep blue, divider
  * fig5: no ICS acronym, sft15_a1 added with n.s. bracket, step-line ladder
  * fig6: single 100% stacked composition chart (replaces redundant panels)
  * fig7: half-up rounding (0.471), legend wording, holdout/id ratios
  * fig8: broken y axis, limit labels with leaders, open measurement markers

Run:  python make_figures_v2.py
Outputs fig1..fig8 as PDF (vector) + PNG (300 dpi), same filenames as before.
"""
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Patch, Polygon)

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
PKG = PROJ / "PhysCoT_论文交付包_20260926"
RAW = PKG / "04_评测原始数据"
DATASETS = PKG / "07_完整可运行工程" / "02_代码框架" / "datasets"
REMED = HERE.parent / "remediation"

# ---------------------------------------------------------------- style ----
# Semantic colour roles (Okabe-Ito, colour-blind safe), identical in all figures
C_OURS = "#0072B2"    # deep blue   -- the proposed model
C_OURS2 = "#56B4E9"   # sky         -- secondary series of ours (state acc, holdout)
C_DARK = "#5A5A5A"    # dark grey   -- fair classical rival (DSSE-profile)
C_MID = "#9A9A9A"     # mid grey    -- masked solver / neutral baselines
C_LIGHT = "#D9D9D9"   # light grey  -- hollow references
C_FLOOR = "#E69F00"   # orange      -- floor values / degenerate outputs
C_WARN = "#D55E00"    # vermillion  -- failure/hallucination traces
C_AUX = "#009E73"     # teal        -- auxiliary (strict rescore) evidence
INK = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
    "axes.edgecolor": INK,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DOUBLE = 6.85  # 174 mm
SINGLE = 3.30  # 84 mm


def fmt3(v):
    """Half-up 3-decimal formatting, consistent with the manuscript tables."""
    return f"{math.floor(v * 1000 + 0.5) / 1000:.3f}"


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ------------------------------------------------------------ data loading --
def parse_summary():
    out = {}
    lines = (PKG / "01_最终结论与报告" / "summary_main.md").read_text(encoding="utf-8").splitlines()
    header = None
    for ln in lines:
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells[0] == "run":
            header = cells if "set" in cells else None
            continue
        if header is None or set(cells[0]) <= set("-: "):
            continue
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        rec = {}
        for k in ("n", "viol_f1", "v_mae", "state_acc", "format", "phys_resid", "ICS(T3)"):
            v = row.get(k, "—")
            rec[k] = None if v in ("—", "", None) else float(v)
        out[(row["run"], row["set"])] = rec
    return out


def median_vmae(run_set):
    vals = []
    with open(RAW / f"{run_set}_records.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v = json.loads(line).get("v_mae")
                if v is not None:
                    vals.append(v)
    return statistics.median(vals), len(vals)


SUM = parse_summary()


def vf(run, set_="id"):
    return SUM[(run, set_)]["viol_f1"]


# ------------------------------------------------------------------ fig1 ----
# Vector icons: Tabler Icons (MIT licence, https://tabler.io/icons), fetched
# once into icons/ and rendered as stroke paths (no raster embedding).
ICON_DIR = HERE / "icons"
ICON_BASE = "https://raw.githubusercontent.com/tabler/tabler-icons/master/icons/outline"


def _load_icon(name):
    """Fetch (once) a Tabler outline SVG and return a unit-normalised Path."""
    from matplotlib.path import Path as MplPath
    from svgpath2mpl import parse_path
    cache = ICON_DIR / f"{name}.svg"
    if not cache.exists():
        import urllib.request
        ICON_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(f"{ICON_BASE}/{name}.svg", cache)
    svg = cache.read_text(encoding="utf-8")
    paths = [parse_path(d) for d in re.findall(r'\bd="([^"]+)"', svg)]
    verts = np.concatenate([p.vertices for p in paths])
    codes = np.concatenate([p.codes for p in paths]).astype(np.uint8)
    verts[:, 1] = 24.0 - verts[:, 1]  # SVG y-axis points down
    verts -= 12.0
    verts /= 24.0
    return MplPath(verts, codes)


def _aspect_corr(ax):
    """x-stretch factor so that icons render undistorted in data coords."""
    bb = ax.get_position()
    fig = ax.figure
    x_upi = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (fig.get_figwidth() * bb.width)
    y_upi = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (fig.get_figheight() * bb.height)
    return x_upi / y_upi


def _icon(ax, name, cx, cy, h, color, lw=1.4, chip=None, zorder=6):
    """Stroke-render a Tabler icon with height h (y-data units) at (cx, cy)."""
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    asp = _aspect_corr(ax)
    if chip:
        ax.add_patch(FancyBboxPatch((cx - 0.70 * h * asp, cy - 0.70 * h),
                                    1.40 * h * asp, 1.40 * h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=chip, edgecolor="none",
                                    zorder=zorder - 1))
    t = Affine2D().scale(h * asp, h).translate(cx, cy) + ax.transData
    ax.add_patch(PathPatch(_load_icon(name), transform=t, fill=False,
                           edgecolor=color, linewidth=lw, capstyle="round",
                           joinstyle="round", zorder=zorder))


def _badge(ax, x, y, n, color=C_OURS):
    ax.scatter([x], [y], s=100, marker="o", facecolor=color, edgecolor="white",
               linewidth=0.8, zorder=7)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=6.2,
            fontweight="bold", color="white", zorder=8)


def _card(ax, x, y, w, h, lw=1.0, pad=0.04):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                                linewidth=lw, edgecolor=INK, facecolor="white",
                                zorder=3))


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.5))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    # ---- layer bands -------------------------------------------------------
    bands = [
        ("Input", 0.25, 2.45, 3.55, 6.55, "#F4F6F8"),
        ("Mechanistic chain (generation order)", 2.65, 11.55, 3.55, 6.55, "#EDF3F9"),
        ("Physics verification", 2.65, 11.55, 2.00, 3.25, "#EEF6F1"),
        ("Post-training", 0.25, 11.55, 0.15, 1.75, "#F7F5F0"),
    ]
    for lab, bx0, bx1, by0, by1, tint in bands:
        ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                    boxstyle="round,pad=0.02", linewidth=0.7,
                                    edgecolor="#C8C8C8", facecolor=tint, zorder=1))
        ax.text(bx0 + 0.10, by1 - 0.10, lab, fontsize=7.5, style="italic",
                color="#666666", ha="left", va="top", zorder=5,
                bbox=dict(facecolor=tint, edgecolor="none", pad=1.4))

    # ---- observation card --------------------------------------------------
    ow, ox, oy = 1.95, 0.45, 4.05
    _card(ax, ox, oy, ow, 1.7, pad=0.06)
    _icon(ax, "search", ox + ow / 2, oy + 1.16, 0.34, C_OURS, chip="#E7F1F9")
    ax.text(ox + ow / 2, oy + 0.56, "Observations", ha="center", va="center",
            fontsize=8.3, fontweight="bold", zorder=4)
    ax.text(ox + ow / 2, oy + 0.30, "(partial, noisy)", ha="center",
            va="center", fontsize=7.0, color="#666666", zorder=4)

    # ---- chain cards -------------------------------------------------------
    xs = [2.95, 4.67, 6.39, 8.11, 9.83]
    ws, y0, hh = 1.52, 4.05, 1.7
    stage_info = [("hierarchy-2", "Topology", "$T$"),
                  ("arrows-split-2", "Branch flow", "$F$"),
                  ("activity", "Voltage", "$V$"),
                  ("alert-triangle", "Violation", "& answer")]
    for k, (x, (icn, t1, t2)) in enumerate(zip(xs[1:], stage_info)):
        _card(ax, x, y0, ws, hh)
        _badge(ax, x + 0.15, y0 + hh - 0.15, k + 1)
        _icon(ax, icn, x + ws / 2, y0 + 1.16, 0.34, C_OURS, chip="#E7F1F9")
        ax.text(x + ws / 2, y0 + 0.56, t1, ha="center", va="center",
                fontsize=8.0, fontweight="bold", zorder=4)
        ax.text(x + ws / 2, y0 + 0.30, t2, ha="center", va="center",
                fontsize=7.4, color="#666666", zorder=4)
    # arrow from observations into stage 1 (same height as the chain arrows)
    ax.add_patch(FancyArrowPatch((ox + ow, y0 + hh / 2), (xs[1], y0 + hh / 2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=INK, zorder=4))
    for i in range(1, 4):
        ax.add_patch(FancyArrowPatch((xs[i] + ws, y0 + hh / 2),
                                     (xs[i + 1], y0 + hh / 2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color=INK, zorder=4))

    # ---- physics verification bar ------------------------------------------
    pc_x0, pc_x1, pc_y0, pc_h = 2.95, 11.35, 2.18, 0.62
    ax.add_patch(FancyBboxPatch((pc_x0, pc_y0), pc_x1 - pc_x0, pc_h,
                                boxstyle="round,pad=0.08", linewidth=1.1,
                                edgecolor=C_AUX, facecolor="white",
                                linestyle=(0, (5, 3)), zorder=3))
    _icon(ax, "shield-check", pc_x0 + 0.55, pc_y0 + pc_h / 2, 0.30, C_AUX,
          chip="#E3F4EE")
    ax.text((pc_x0 + pc_x1) / 2 + 0.35, pc_y0 + pc_h / 2,
            "substitute each stage back into the power-flow equations",
            ha="center", va="center", fontsize=8.3, zorder=4)
    for x in xs[1:]:
        xm = x + ws / 2
        ax.add_patch(FancyArrowPatch((xm, y0), (xm, pc_y0 + pc_h + 0.08),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color="#777777",
                                     linestyle=(0, (4, 3)), zorder=2))

    # deployment guard: verifier acts also at inference time
    ax.add_patch(FancyArrowPatch((pc_x1, pc_y0 + pc_h / 2), (12.55, pc_y0 + pc_h / 2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color=C_AUX,
                                 linestyle=(0, (5, 3)), zorder=2))
    ax.text(12.62, pc_y0 + pc_h / 2, "deployment\nguard", fontsize=7.3,
            ha="left", va="center", color=C_AUX, zorder=2)

    # ---- post-training strip ----------------------------------------------
    pt_y0, pt_h = 0.42, 1.05
    stages = [(2.95, 2.55, "database-cog", "SFT", "20k + curriculum"),
              (6.05, 2.55, "scale", "DPO", "109 physics pairs"),
              (9.05, 2.30, "target", "per-stage GRPO", None)]
    for x, w, icn, t1, t2 in stages:
        _card(ax, x, pt_y0, w, pt_h, pad=0.06)
        cxm = x + w / 2
        _icon(ax, icn, cxm, pt_y0 + pt_h - 0.30, 0.32, "#555555", lw=1.3,
              chip="#F2F0EB")
        if t2:
            ax.text(cxm, pt_y0 + 0.36, t1, ha="center", va="center",
                    fontsize=7.8, fontweight="bold", zorder=4)
            ax.text(cxm, pt_y0 + 0.15, t2, ha="center", va="center",
                    fontsize=6.6, color="#666666", zorder=4)
        else:
            ax.text(cxm, pt_y0 + 0.26, t1, ha="center", va="center",
                    fontsize=7.8, fontweight="bold", zorder=4)
    for i in range(2):
        x_end = stages[i][0] + stages[i][1]
        ax.add_patch(FancyArrowPatch((x_end, pt_y0 + pt_h / 2),
                                     (stages[i + 1][0], pt_y0 + pt_h / 2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color=INK, zorder=4))

    # ---- verifier feedback into the trainers -------------------------------
    fb = [(6.05 + 2.55 / 2, "preference pairs"), (9.05 + 2.30 / 2, "per-stage reward")]
    for xm, lab in fb:
        ax.add_patch(FancyArrowPatch((xm, pc_y0), (xm, pt_y0 + pt_h),
                                     arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.1, color=C_AUX, zorder=2))
        ax.text(xm + 0.13, (pc_y0 + pt_y0 + pt_h) / 2, lab, fontsize=7.3,
                ha="left", va="center", color=C_AUX, zorder=2)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ fig2 ----
def fig2_main_results():
    rows = [
        ("DSSE profile", vf("dsse[profile]"), C_DARK, "", False),
        ("DSSE oracle", vf("dsse[oracle]"), "white", "", True),
        ("PhysCoT v3b\n(ours)", vf("v3b"), C_OURS, "", False),
        ("solver\nmasked", vf("solver[masked]"), C_MID, "", False),
        ("DeepSeek\n0-shot †", vf("deepseek_zs"), C_FLOOR, "xx", False),
        ("DeepSeek\n2-shot †", vf("deepseek_fs"), C_FLOOR, "xx", False),
        ("Chronos\n0-shot", vf("chronos"), C_MID, "", False),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.05))
    x = np.arange(len(rows))
    for i, (lab, val, c, hatch, hollow) in enumerate(rows):
        ec = C_MID if hollow else INK
        ax.bar(i, val, width=0.62, color=c, hatch=hatch, edgecolor=ec,
               linewidth=(1.1 if i == 2 else 0.9), zorder=3)
        weight = "bold" if i == 2 else "normal"
        ax.text(i, val + 0.018, fmt3(val), ha="center", va="bottom", fontsize=8,
                fontweight=weight)
    # upper reference
    ax.axhline(vf("solver[full]"), color=INK, linewidth=0.9,
               linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(rows) - 0.45, 0.965, "solver full (upper bound) = 1.000",
            ha="right", va="top", fontsize=8, style="italic")
    # gap bracket: ours vs masked solver (verticals clear both value labels:
    # 8pt text is ~0.05 data units tall here, so keep >=0.03 clearance)
    y_v3b, y_masked = vf("v3b"), vf("solver[masked]")
    br = 0.83
    ax.plot([2, 2, 3, 3], [y_v3b + 0.095, br, br, y_masked + 0.115],
            color="#555555", linewidth=0.9, zorder=2)
    ax.text(2.5, br + 0.010, "+7.3 pts", ha="center", va="bottom", fontsize=8,
            color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.6)
    ax.set_ylabel("violation F1 (id set)")
    ax.set_ylim(0, 1.10)
    ax.text(0.0, -0.30,
            "PhysCoT v3b and DeepSeek: stratified n = 300; solver/DSSE/Chronos: full n = 999.\n"
            "† per-set F1 coincides with the zero-detection floor to within 0.006 "
            "(floor 0.353 here); see Section 5.6.",
            transform=ax.transAxes, fontsize=7.2, color="#444444")
    fig.subplots_adjust(bottom=0.22)
    clean_axes(ax)
    save(fig, "fig2_main_results")


# ------------------------------------------------------------------ fig3 ----
def fig3_degradation():
    sets = ["id", "noise", "holdout", "abstain"]
    viol = [vf("v3b", s) for s in sets]
    acc = [SUM[("v3b", s)]["state_acc"] for s in sets]
    dsse = [vf("dsse[profile]", s) for s in sets]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.75))
    x = np.arange(len(sets))
    w = 0.26
    b1 = ax.bar(x - w, viol, w, color=C_OURS, edgecolor=INK, linewidth=0.9,
                label="violation F1 (PhysCoT v3b)", zorder=3)
    b2 = ax.bar(x, acc, w, color=C_OURS2, edgecolor=INK, linewidth=0.9,
                label="state accuracy (v3b)", zorder=3)
    b3 = ax.bar(x + w, dsse, w, color=C_LIGHT, edgecolor=INK, linewidth=0.9,
                label="violation F1 (DSSE profile, reference)", zorder=3)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    fmt3(b.get_height()), ha="center", va="bottom", fontsize=7.4)
    # abstain caveat folded into the tick label (collision-free by construction)
    ax.annotate("intervention combinations\nunseen in training",
                xy=(2 - w - 0.16, 0.60), xytext=(0.95, 0.91), ha="center",
                fontsize=7.6,
                arrowprops=dict(arrowstyle="->", color="#666666", linewidth=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xticks(x)
    ax.set_xticklabels(["id", "noise", "holdout", "abstain"], fontsize=7.4)
    ax.text(0.0, -0.16,
            "abstain: measurement-starved; the model answers everything and never refuses.",
            transform=ax.transAxes, fontsize=7.0, color="#444444", ha="left",
            va="top")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(top=0.84, bottom=0.20)
    clean_axes(ax)
    save(fig, "fig3_degradation")


# ------------------------------------------------------------------ fig4 ----
FIG4_VARIANTS = [
    ("A0\nfull chain", "sft_a0_base", "sft15_a0"),
    ("A1\nno chain", "sft_a1_nochain", "sft15_a1"),
    ("A2\nshuffled", "sft_a2_shuffled", None),
    ("A3\ncheat", "sft_a3_cheat", "sft15_a3"),
    ("no\ncurriculum\n(1.5B only)", None, "sft15_nocurr"),
]
FLOOR_RUNS = {"sft_a1_nochain", "sft_a3_cheat", "sft15_a1", "sft15_a3"}


def fig4_ablation():
    labels = [v[0] for v in FIG4_VARIANTS]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(DOUBLE, 3.05))
    x = np.arange(len(labels))
    w = 0.36

    def is_floor(run):
        return run in FLOOR_RUNS

    labelled = set()
    for off, scale, color in ((-w / 2, "0.5B", C_OURS2), (w / 2, "1.5B", C_OURS)):
        for i, (_, r05, r15) in enumerate(FIG4_VARIANTS):
            run = r05 if scale == "0.5B" else r15
            if run is None:
                continue
            v = vf(run)
            if is_floor(run):  # floor value: hollow + hatched, single shared label
                axl.bar(i + off, v, w, facecolor="white", edgecolor=C_FLOOR,
                        hatch="///", linewidth=0.9, zorder=3)
            else:
                axl.bar(i + off, v, w, color=color, edgecolor=INK,
                        linewidth=0.9, zorder=3)
                both = r05 is not None and r15 is not None
                ha = ("right" if scale == "0.5B" else "left") if both else "center"
                dx = (-0.03 if scale == "0.5B" else 0.03) if both else 0.0
                axl.text(i + off + dx, v + 0.012, f"{v:.2f}", ha=ha,
                         va="bottom", fontsize=7.6)
    # one shared dagger label per equal-value floor pair (A1, A3)
    for i in (1, 3):
        axl.text(i, vf("sft15_a1" if i == 1 else "sft15_a3") + 0.012,
                 "0.35 †", ha="center", va="bottom", fontsize=7.6)
        labelled.add(i)
    axl.set_title("(a) violation F1 (id set)", fontsize=9)
    axl.set_ylabel("violation F1")
    axl.set_ylim(0, 0.70)
    axl.legend(handles=[
        Patch(facecolor=C_OURS2, edgecolor=INK, linewidth=0.9, label="0.5B"),
        Patch(facecolor=C_OURS, edgecolor=INK, linewidth=0.9, label="1.5B"),
    ], loc="upper right")
    axl.axvline(3.5, color="#BBBBBB", linewidth=0.8, linestyle=(0, (3, 3)))
    axl.text(0.0, -0.30,
             "† zero-detection floor: the variant flags no violations (F1 = 1 − positive rate).",
             transform=axl.transAxes, fontsize=7.0, color="#444444")

    # panel (b): median v_mae; median == 1.0 sentinel => parse failure
    parse_fail = {}
    for off, scale, color in ((-w / 2, "0.5B", C_OURS2), (w / 2, "1.5B", C_OURS)):
        for i, (_, r05, r15) in enumerate(FIG4_VARIANTS):
            run = r05 if scale == "0.5B" else r15
            if run is None:
                continue
            med, _ = median_vmae(f"{run}_id_pn")
            if med >= 0.999:
                parse_fail.setdefault(i, []).append(i + off)
                continue
            axr.bar(i + off, med, w, color=color, edgecolor=INK, linewidth=0.9,
                    zorder=3)
            axr.text(i + off, med + 0.004, fmt3(med), ha="center", va="bottom",
                     fontsize=7.6)
    for i, xps in parse_fail.items():
        for xp in xps:
            axr.plot([xp], [0.0], marker="x", color=INK, markersize=6,
                     zorder=4, clip_on=False)
        axr.text(sum(xps) / len(xps) - 0.13, 0.011, "parse\nfail", ha="center",
                 va="bottom", fontsize=6.4, color="#444444", linespacing=0.95)
    axr.set_title("(b) median voltage MAE (id set)", fontsize=9)
    axr.set_ylabel("median $v_\\mathrm{mae}$ (p.u.)")
    axr.set_ylim(0, 0.115)
    axr.axvline(3.5, color="#BBBBBB", linewidth=0.8, linestyle=(0, (3, 3)))

    for ax in (axl, axr):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_xlim(-0.85, 4.55)  # left margin so edge labels clear the spine
    fig.subplots_adjust(wspace=0.30, bottom=0.20)
    clean_axes(axl)
    clean_axes(axr)
    save(fig, "fig4_ablation")


# ------------------------------------------------------------------ fig5 ----
def fig5_ladder():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) ladder with step-line overlay
    ladder_runs = [("volt only", "sft15_volt_only_id_pn"),
                   ("topo + volt", "sft15_topo_volt_id_pn"),
                   ("full chain", "sft15_nocurr_id_pn")]
    meds = [median_vmae(r)[0] for _, r in ladder_runs]
    x = np.arange(3)
    axl.bar(x, meds, 0.55, color=C_AUX, edgecolor=INK, linewidth=0.9, zorder=3,
            alpha=0.85)
    axl.plot(x, meds, "-o", color="#00694F", linewidth=1.2, markersize=4,
             zorder=4)
    for i, m in enumerate(meds):
        axl.text(i, m + 0.0022, fmt3(m), ha="center", va="bottom", fontsize=8)
    axl.set_xticks(x)
    axl.set_xticklabels([t for t, _ in ladder_runs])
    axl.set_ylabel("median $v_\\mathrm{mae}$ (p.u.)")
    axl.set_ylim(0, 0.075)
    axl.set_title("(a) intermediate-variable ladder\n(1.5B, 3k, no curriculum)",
                  fontsize=9)

    # (b) strict rescore, relative ranking only -- now includes the 1.5B no-chain
    with open(REMED / "ics_rescore_table.csv", encoding="utf-8-sig") as f:
        rows = {(r["model"], r["set"]): r for r in csv.DictReader(f)}
    order = [("v3b\n(intact chain)", "v3b", C_AUX, ""),
             ("A1 1.5B\n(no chain)", "sft15_a1", C_AUX, "//"),
             ("v2", "v2", "#7FCDB4", ""),
             ("A0 1.5B", "sft15_a0", "#7FCDB4", ""),
             ("A1 0.5B", "sft_a1_nochain", "#7FCDB4", "")]
    vals = [float(rows[(m, "id")]["ics_strict"]) for _, m, _, _ in order]
    base = float(rows[("v3b", "id")]["base_best_strict"])
    x = np.arange(len(order))
    for i, ((lab, m, c, h), v) in enumerate(zip(order, vals)):
        axr.bar(i, v, 0.58, color=c, hatch=h, edgecolor=INK, linewidth=0.9,
                zorder=3)
        axr.text(i, v + 0.012, fmt3(v), ha="center", va="bottom", fontsize=8)
    axr.axhline(base, color=INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axr.text(len(order) - 0.45, base + 0.012,
             f"constant predictor = {base:.2f}", ha="right", va="bottom",
             fontsize=7.6, style="italic")
    # n.s. bracket v3b vs A1 1.5B
    yb = max(vals[0], vals[1]) + 0.075
    axr.plot([0, 0, 1, 1], [vals[0] + 0.045, yb, yb, vals[1] + 0.045],
             color="#555555", linewidth=0.9, zorder=2)
    axr.text(0.5, yb + 0.008, "n.s.", ha="center", va="bottom", fontsize=8,
             color="#555555")
    axr.set_xticks(x)
    axr.set_xticklabels([t for t, _, _, _ in order], fontsize=7.4)
    axr.set_ylabel("strict post-hoc intervention score")
    axr.set_ylim(0, 0.82)
    axr.set_title("(b) strict metric, relative ranking only", fontsize=9)
    fig.subplots_adjust(wspace=0.34)
    clean_axes(axl)
    clean_axes(axr)
    save(fig, "fig5_ladder")


# ------------------------------------------------------------------ fig6 ----
def fig6_reward_hacking():
    with open(REMED / "tm60_remediation.json", encoding="utf-8") as f:
        d = json.load(f)
    tm, ps = d["grpo_tm60_id_pn"], d["grpo_ps60_id_pn"]
    ps_res = 100 * ps["resid_ok"] / ps["n"]
    ps_tru = 100 * ps["null_volt_stage_missing"] / ps["n"]
    tm_ph = 100 * tm["null_vpu_scalar_placeholder"] / tm["n"]
    tm_tru = 100 * tm["null_volt_stage_missing"] / tm["n"]

    fig, ax = plt.subplots(figsize=(DOUBLE, 1.95))
    ypos = [1.0, 0.0]
    h = 0.52
    seg_kw = dict(height=h, edgecolor=INK, linewidth=0.9, zorder=3)
    # per-stage row
    ax.barh(ypos[0], ps_res, left=0, color=C_AUX, **seg_kw)
    ax.barh(ypos[0], ps_tru, left=ps_res, color=C_MID, **seg_kw)
    # terminal row
    ax.barh(ypos[1], tm_ph, left=0, color=C_WARN, **seg_kw)
    ax.barh(ypos[1], tm_tru, left=tm_ph, color=C_MID, **seg_kw)
    # in-segment labels (white)
    ax.text(ps_res / 2, ypos[0], f"re-solvable\n{ps['resid_ok']}/{ps['n']} ({ps_res:.0f}%)",
            ha="center", va="center", fontsize=7.6, color="white", zorder=4)
    ax.text(ps_res + ps_tru / 2, ypos[0],
            f"truncated\n{ps['null_volt_stage_missing']}/{ps['n']} ({ps_tru:.0f}%)",
            ha="center", va="center", fontsize=7.6, color="white", zorder=4)
    ax.text(tm_ph / 2, ypos[1] + 0.10,
            f"scalar placeholder {tm['null_vpu_scalar_placeholder']}/{tm['n']} ({tm_ph:.1f}%)",
            ha="center", va="center", fontsize=7.6, color="white", zorder=4)
    ax.text(tm_ph / 2, ypos[1] - 0.13, '"v_pu": 1.0', ha="center", va="center",
            fontsize=7.2, color="white", family="monospace", zorder=4)
    ax.text(tm_ph + tm_tru / 2, ypos[1],
            f"truncated\n{tm['null_volt_stage_missing']}/{tm['n']} ({tm_tru:.1f}%)",
            ha="center", va="center", fontsize=7.6, color="white", zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["per-stage GRPO (ps60)\n70% re-solvable",
                        "terminal GRPO (tm60)\n0% re-solvable"], fontsize=7.8)
    ax.set_xlabel("share of evaluated samples (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    ax.legend(handles=[
        Patch(facecolor=C_AUX, edgecolor=INK, linewidth=0.9,
              label="re-solvable by power flow"),
        Patch(facecolor=C_WARN, edgecolor=INK, linewidth=0.9,
              label="scalar placeholder"),
        Patch(facecolor=C_MID, edgecolor=INK, linewidth=0.9,
              label="volt stage truncated"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
        columnspacing=1.0, handletextpad=0.45, fontsize=7.6)
    fig.subplots_adjust(top=0.84, left=0.24)
    clean_axes(ax)
    save(fig, "fig6_reward_hacking")


# ------------------------------------------------------------------ fig7 ----
def fig7_scale():
    with open(PKG / "02_关键数据表与图" / "e6_scale_table.json", encoding="utf-8") as f:
        e6 = json.load(f)
    keys = ["0.5B_3k_nocurr", "1.5B_3k_nocurr", "3B_3k_nocurr"]
    sizes = ["0.5B", "1.5B", "3B"]
    id_v = [e6[k]["id"]["viol_f1"] for k in keys]
    ho_v = [e6[k]["holdout"]["viol_f1"] for k in keys]
    ratios = [h / i for h, i in zip(ho_v, id_v)]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    ax.plot(x, id_v, "o-", color=C_OURS, linewidth=1.3, markersize=5,
            label="id", zorder=3)
    ax.plot(x, ho_v, "s--", color=C_OURS2, linewidth=1.3, markersize=5,
            label="holdout (combination\nforms held out)", zorder=3)
    for i in range(3):
        dx = 0.06 if i == 0 else 0.0
        ax.text(x[i] + dx, id_v[i] + 0.018, fmt3(id_v[i]), ha="center",
                va="bottom", fontsize=8)
        ax.text(x[i] + dx, ho_v[i] - 0.022, fmt3(ho_v[i]), ha="center",
                va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.set_ylabel("violation F1")
    ax.set_ylim(0.15, 0.74)
    ax.legend(loc="upper left", handlelength=2.6, labelspacing=0.4)
    ax.text(0.98, 0.04,
            "holdout/id: " + " → ".join(f"{r:.2f}" for r in ratios),
            ha="right", va="bottom", transform=ax.transAxes, fontsize=7.6,
            style="italic", color="#444444")
    clean_axes(ax)
    save(fig, "fig7_scale")


# ------------------------------------------------------------------ fig8 ----
STAGE_VOLT_RE = re.compile(r"<stage:volt>(\{.*?\})</stage:volt>")


def fig8_hallucination():
    sid = "CIGRE_MV-000004"
    pred_rec = None
    with open(RAW / "v2_noise_full_preds.jsonl", encoding="utf-8") as f:
        for line in f:
            if sid in line:
                rec = json.loads(line)
                if rec.get("sample_id") == sid:
                    pred_rec = rec
                    break
    truth_rec = None
    with open(DATASETS / "raw_eval_noise.jsonl", encoding="utf-8") as f:
        for line in f:
            if sid in line:
                rec = json.loads(line)
                if rec.get("sample_id") == sid:
                    truth_rec = rec
                    break
    pred_v = json.loads(STAGE_VOLT_RE.search(pred_rec["prediction"]).group(1))["v_pu"]
    true_v = truth_rec["chain"]["volt"]["v_pu"]
    meas_v = {k: v for k, v in truth_rec["obs"]["v_meas_pu"].items() if v != "NA"}

    buses = sorted(int(b) for b in true_v)
    tv = [true_v[str(b)] for b in buses]
    pv = [pred_v[str(b)] for b in buses]

    fig, (axt, axb) = plt.subplots(
        2, 1, sharex=True, figsize=(SINGLE, 3.2),
        gridspec_kw={"height_ratios": [1, 2.1], "hspace": 0.10})

    # upper band: hallucinated prediction (~10 p.u.)
    axt.plot(buses, pv, "^--", color=C_WARN, linewidth=1.2, markersize=4.5,
             label="model prediction", zorder=3)
    axt.set_ylim(9.5, 11.7)
    ayt = axt.get_yticks()
    axt.annotate("prediction $\\approx$ 10×\ntrue magnitude",
                 xy=(6.25, pv[6] + 0.05), xytext=(11.3, 9.60), fontsize=7.4,
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="#666666",
                                 linewidth=0.8,
                                 connectionstyle="arc3,rad=0.12"))
    axt.legend(loc="upper right", handlelength=1.8, fontsize=7.6)

    # lower band: truth, measurements, limits
    axb.plot(buses, tv, "o-", color=C_OURS, linewidth=1.3, markersize=4,
             label="ground truth", zorder=3)
    axb.plot([int(b) for b in meas_v], list(meas_v.values()), "s",
             markerfacecolor="none", markeredgecolor=INK, markersize=4.5,
             linestyle="none", markeredgewidth=0.9, label="noisy measurements",
             zorder=4)
    axb.axhline(1.05, color="#888888", linewidth=0.8, linestyle=(0, (2, 2)),
                zorder=1)
    axb.axhline(0.95, color="#888888", linewidth=0.8, linestyle=(0, (2, 2)),
                zorder=1)
    axb.set_ylim(0.85, 1.25)
    axb.legend(loc="upper right", handlelength=1.8, fontsize=7.6)
    axb.set_yticks([0.9, 1.0, 1.1, 1.2])
    axb.set_xlim(-0.6, 16.6)
    axb.text(15.0, 1.055, "upper limit 1.05", fontsize=7.0, va="bottom",
             ha="left", color="#555555")
    axb.text(15.0, 0.945, "lower limit 0.95", fontsize=7.0, va="top",
             ha="left", color="#555555")

    # conventional diagonal break marks on both sides of the interrupted axis
    axt.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axt.tick_params(bottom=False)
    kw = dict(marker=[(-1, -1), (1, 1)], markersize=10, linestyle="none",
              color=INK, mec=INK, mew=1.0, clip_on=False)
    axt.plot([0, 1], [0, 0], transform=axt.transAxes, **kw)
    axb.plot([0, 1], [1, 1], transform=axb.transAxes, **kw)

    axb.set_xlabel("bus index")
    axb.set_ylabel("voltage (p.u.)")
    axb.set_xticks(range(0, 15, 2))
    clean_axes(axt)
    clean_axes(axb)
    save(fig, "fig8_hallucination")


# ------------------------------------------------------------------- main ----
if __name__ == "__main__":
    fig1_pipeline()
    fig2_main_results()
    fig3_degradation()
    fig4_ablation()
    fig5_ladder()
    fig6_reward_hacking()
    fig7_scale()
    fig8_hallucination()
    print("all figures v2 done")
