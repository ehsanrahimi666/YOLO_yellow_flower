"""Figure 2 (workflow), Figure 6 (accuracy-cost trade-off),
Figure 9 (capitulum density, exact 1-m cells), Figure A1 (equivalence of detector pairs)
and Figure A2 (highest-scoring unmatched detections).

Outputs PNG (300 dpi, for the manuscript), TIFF (600 dpi, LZW) and PDF (vector graphs)
in figures/.
"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures")
TILES = os.path.join(os.path.dirname(ROOT), "data", "tiles")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.facecolor": "white"})
MM = 1 / 25.4
COL = {"YOLOv9t": "#440154", "YOLO11n": "#3b528b", "YOLOv8n": "#21918c", "YOLO11s": "#5ec962", "YOLOv10n": "#e3c800"}
ORDER = ["YOLOv9t", "YOLO11n", "YOLOv8n", "YOLO11s", "YOLOv10n"]


def save(fig, name, vector=True):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    if vector:
        fig.savefig(os.path.join(FIG, name + ".pdf"), facecolor="white")
    plt.close(fig)
    with Image.open(os.path.join(FIG, name + ".tif")) as im:   # drop any alpha channel
        if im.mode != "RGB":
            im.convert("RGB").save(os.path.join(FIG, name + ".tif"), compression="tiff_lzw", dpi=(600, 600))


# ---------------------------------------------------------------- Figure 2
def figure2():
    geo = json.load(open(os.path.join(OUT, "flight_geometry.json")))
    ms = json.load(open(os.path.join(OUT, "map_statistics.json")))
    stages = {"acq": ("#2f6db3", "#e8f0fa", "Data acquisition"), "data": ("#2e8b57", "#e8f4ec", "Dataset construction"),
              "bench": ("#d9782d", "#fbeee3", "Detector benchmark"), "map": ("#7b4fa6", "#f1eaf7", "Deployment")}
    boxes = [
        ("acq", "UAV RGB survey", ["DJI Mavic 3 Multispectral", f"{geo['rel_alt_median']:.1f} m above take-off",
                                   f"~{100 * geo['forward_overlap']:.0f}% forward/side overlap"]),
        ("acq", "SfM orthomosaic", ["OpenDroneMap 3.6.0", "2 mm GSD (0.2 cm/px)", "WGS 84 / UTM 52N"]),
        ("data", "Expert annotation", ["Capitulum polygons", "1,235 digitised objects", "→ 1,194 bounding boxes"]),
        ("data", "Tiling", ["640 × 640 px patches", "1.28 m on the ground", "122 tiles · native scale"]),
        ("data", "Spatial blocking", ["5 contiguous strips", "along the long axis", "no tile shared across folds"]),
        ("bench", "Benchmark", ["5 detectors × 5 folds", "25 runs, one protocol", "mAP, precision, recall"]),
        ("bench", "Model comparison", ["Mixed model, TOST ±0.02", "CPU latency, GFLOPs", "YOLO11n: best compromise"]),
        ("map", "Wall-to-wall map", ["YOLOv9t, all 122 tiles", "1,944 tiles · global NMS",
                                    f"{ms['n']:,} capitula · {ms['density_valid']:.2f} m$^{{-2}}$"]),
    ]
    fig = plt.figure(figsize=(174 * MM, 80 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, 80); ax.axis("off")
    w, h = 37.5, 22
    xs = [3, 47, 91, 135]
    ys = [42, 4]
    centers = []
    for i, (stage, title, lines) in enumerate(boxes):
        r, c = divmod(i, 4)
        x, y = xs[c], ys[r]
        edge, fill, _ = stages[stage]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fill, ec=edge, lw=1.0))
        ax.add_patch(Rectangle((x, y + 0.6), 1.6, h - 1.2, fc=edge, ec="none"))
        ax.text(x + 4, y + h - 5, title, fontsize=7.6, fontweight="bold", va="center")
        for k, line in enumerate(lines):
            ax.text(x + 4, y + h - 10.5 - 3.6 * k, line, fontsize=6.2, va="center", color="#222222")
        centers.append((x, y, w, h))
    arrow = dict(arrowstyle="-|>", color="#555555", lw=1.0, mutation_scale=8)
    for i in [0, 1, 2, 4, 5, 6]:
        x, y, w_, h_ = centers[i]
        ax.annotate("", xy=(centers[i + 1][0], y + h_ / 2), xytext=(x + w_, y + h_ / 2), arrowprops=arrow)
    # tiling -> spatial blocking (elbow)
    x3, y3, _, _ = centers[3]; x4, y4, _, h4 = centers[4]
    ax.plot([x3 + w / 2, x3 + w / 2], [y3, 35], color="#555555", lw=1.0)
    ax.plot([x3 + w / 2, x4 + w / 2], [35, 35], color="#555555", lw=1.0)
    ax.annotate("", xy=(x4 + w / 2, y4 + h4), xytext=(x4 + w / 2, 35), arrowprops=arrow)
    for k, key in enumerate(["acq", "data", "bench", "map"]):
        edge, fill, label = stages[key]
        ax.add_patch(FancyBboxPatch((3 + 44 * k, 72.5), 4, 4, boxstyle="round,pad=0,rounding_size=0.8", fc=fill, ec=edge, lw=0.9))
        ax.text(9 + 44 * k, 74.5, label, fontsize=7, va="center")
    save(fig, "Figure2_workflow")


# ---------------------------------------------------------------- Figure 6
def figure6():
    lat = pd.read_csv(os.path.join(OUT, "cpu_latency.csv")).set_index("model")
    t3 = pd.read_csv(os.path.join(OUT, "table3_reproduced.csv"), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 74 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, mode in zip(axes, ["cpu", "gpu"]):
        for m in ORDER:
            y, ysd = t3.loc[m, "mAP50"], t3.loc[m, "mAP50_sd"]
            if mode == "cpu":
                x, lo, hi = lat.loc[m, "median"], lat.loc[m, "q25"], lat.loc[m, "q75"]
            else:
                x, lo, hi = t3.loc[m, "ms_per_img"], t3.loc[m, "ms_min"], t3.loc[m, "ms_max"]
            ax.errorbar(x, y, yerr=ysd, xerr=[[x - lo], [hi - x]], fmt="none", ecolor="#666666", elinewidth=0.7, capsize=0)
            ax.scatter(x, y, s=lat.loc[m, "GFLOPs"] * 14, color=COL[m], edgecolor="#222222", linewidth=0.6, zorder=3, alpha=0.9)
            dx = {"cpu": {"YOLOv9t": 4, "YOLO11n": -4, "YOLOv8n": 4, "YOLO11s": -4, "YOLOv10n": 4},
                  "gpu": {"YOLOv9t": 1.5, "YOLO11n": -1.5, "YOLOv8n": 1.5, "YOLO11s": 2.5, "YOLOv10n": 1.5}}[mode][m]
            dy = {"YOLOv9t": 0.0035, "YOLO11n": 0.0035, "YOLOv8n": -0.0045, "YOLO11s": 0.0055, "YOLOv10n": -0.0045}[m]
            ax.text(x + dx, y + dy, m, fontsize=7, ha="left" if dx > 0 else "right", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_xlabel("CPU forward-pass time per tile (ms)\nmedian and interquartile range")
    axes[1].set_xlabel("GPU time per image during validation (ms)\nmean and range over the five folds")
    axes[0].set_ylabel("mAP@0.5 (mean ± SD over five spatial folds)")
    axes[0].set_xlim(40, 165); axes[1].set_xlim(0, 58)
    axes[0].set_ylim(0.905, 0.985)
    axes[0].set_title("a  Controlled CPU benchmark", loc="left", fontweight="bold", fontsize=8.5)
    axes[1].set_title("b  Shared cloud GPU (Tesla T4)", loc="left", fontweight="bold", fontsize=8.5)
    for g in [6.3, 21.3]:
        axes[1].scatter([], [], s=g * 14, color="white", edgecolor="#222222", lw=0.6, label=f"{g} GFLOPs")
    axes[1].legend(loc="lower right", frameon=False, fontsize=6.5, labelspacing=1.2, borderpad=0.8)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.92)
    save(fig, "Figure6_tradeoff")


# ---------------------------------------------------------------- Figure 9
def figure9():
    f = pd.read_csv(os.path.join(ROOT, "data_input", "flowers_all.csv"))
    sa = pd.read_csv(os.path.join(ROOT, "data_input", "study_area_summary.csv")).iloc[0]
    xe = np.arange(sa.utm_left, sa.utm_right + 1, 1.0)
    ye = np.arange(sa.utm_bottom, sa.utm_top + 1, 1.0)
    H, _, _ = np.histogram2d(f.geo_x, f.geo_y, bins=[xe, ye])
    Hm = np.ma.masked_where(H.T == 0, H.T)
    fig, ax = plt.subplots(figsize=(174 * MM, 86 * MM))
    ax.set_facecolor("#f2f2f2")
    im = ax.pcolormesh(xe, ye, Hm, cmap="YlOrRd", vmin=0, vmax=50, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlim(sa.utm_left, sa.utm_right); ax.set_ylim(sa.utm_bottom, sa.utm_top)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_xlabel("Easting (m, WGS 84 / UTM zone 52N)"); ax.set_ylabel("Northing (m)")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="neither")
    cb.set_label("Capitula per 1-m cell (m$^{-2}$)")
    # scale bar and north arrow
    x0, y0 = sa.utm_left + 2, sa.utm_bottom + 2
    ax.plot([x0, x0 + 10], [y0, y0], color="black", lw=1.6)
    ax.text(x0 + 5, y0 + 1.0, "10 m", ha="center", fontsize=7)
    ax.annotate("N", xy=(sa.utm_right - 3, sa.utm_top - 3), xytext=(sa.utm_right - 3, sa.utm_top - 9),
                ha="center", fontsize=8, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
    fig.tight_layout()
    save(fig, "Figure9_density", vector=False)


# ---------------------------------------------------------------- Figure A1
def figureA1():
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 86 * MM), sharey=False, gridspec_kw={"wspace": 0.55})
    for ax, metric, title in zip(axes, ["mAP50", "mAP50_95"], ["a  mAP@0.5", "b  mAP@0.5:0.95"]):
        pw = pd.read_csv(os.path.join(OUT, f"pairwise_{metric}.csv"))
        pw["label"] = pw.model_1 + " – " + pw.model_2
        pw = pw.sort_values("diff").reset_index(drop=True)
        ax.axvspan(-0.02, 0.02, color="#dcefe2", zorder=0)
        for v in [-0.05, -0.01, 0.01, 0.05]:
            ax.axvline(v, color="#9a9a9a", lw=0.6, ls="--", zorder=1)
        ax.axvline(0, color="#444444", lw=0.6, zorder=1)
        for i, r in pw.iterrows():
            eq = r["equiv_0.02"]
            c = "#1b7a3e" if eq else "#333333"
            ax.plot([r.ci90nb_lo, r.ci90nb_hi], [i, i], color="#b5b5b5", lw=1.0, zorder=2)
            ax.plot([r.ci90_lo, r.ci90_hi], [i, i], color=c, lw=2.2, zorder=3, solid_capstyle="butt")
            ax.scatter(r["diff"], i, s=16, color="white", edgecolor=c, lw=1.0, zorder=4)
        ax.set_yticks(range(len(pw))); ax.set_yticklabels(pw.label, fontsize=6.8)
        ax.set_xlim(-0.08, 0.09)
        ax.set_xlabel("Difference (first − second detector)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, len(pw) - 0.2, "±0.02", ha="center", va="bottom", fontsize=6.3, color="#1b7a3e")
        ax.text(0.05, len(pw) - 0.2, "0.05", ha="center", va="bottom", fontsize=6.3, color="#777777")
        ax.text(-0.05, len(pw) - 0.2, "−0.05", ha="center", va="bottom", fontsize=6.3, color="#777777")
        ax.set_ylim(-0.7, len(pw) + 0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#1b7a3e", lw=2.2, label="90% CI, equivalent within ±0.02"),
               Line2D([0], [0], color="#333333", lw=2.2, label="90% CI, not equivalent within ±0.02"),
               Line2D([0], [0], color="#b5b5b5", lw=1.0, label="90% CI corrected for fold dependence")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.6)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.2, top=0.92)
    save(fig, "FigureA1_equivalence")


# ---------------------------------------------------------------- Figure A2
def figureA2():
    u = pd.read_csv(os.path.join(OUT, "v9t_unmatched_detections.csv")).head(40)
    tiles_ = []
    for _, r in u.iterrows():
        img = cv2.cvtColor(cv2.imread(os.path.join(TILES, "images", r.tile + ".jpg")), cv2.COLOR_BGR2RGB)
        pad = 48
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cx, cy = int((r.x0 + r.x1) / 2) + pad, int((r.y0 + r.y1) / 2) + pad
        crop = big[cy - 48:cy + 48, cx - 48:cx + 48].copy()
        x0, y0 = int(r.x0) + pad - (cx - 48), int(r.y0) + pad - (cy - 48)
        x1, y1 = int(r.x1) + pad - (cx - 48), int(r.y1) + pad - (cy - 48)
        tiles_.append((crop, (x0, y0, x1, y1), r.score, r.edge))
    fig, axes = plt.subplots(5, 8, figsize=(174 * MM, 112 * MM), gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for ax, (crop, (x0, y0, x1, y1), sc, edge) in zip(axes.ravel(), tiles_):
        ax.imshow(crop, interpolation="nearest")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1f6fff", lw=0.8))
        ax.text(3, 10, f"{sc:.2f}", color="white", fontsize=6, fontweight="bold",
                bbox=dict(fc="black", ec="none", alpha=0.55, pad=0.8))
        for s in ax.spines.values():
            s.set_edgecolor("#e07b00" if edge else "#2b6cb0"); s.set_linewidth(2.0)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.99)
    fig.text(0.02, 0.025, "■", color="#e07b00", fontsize=9, va="center")
    fig.text(0.04, 0.025, "centred within 32 px of a tile edge", fontsize=7, va="center")
    fig.text(0.36, 0.025, "■", color="#2b6cb0", fontsize=9, va="center")
    fig.text(0.38, 0.025, "tile interior", fontsize=7, va="center")
    save(fig, "FigureA2_unmatched", vector=False)


if __name__ == "__main__":
    figure2(); figure6(); figure9(); figureA1(); figureA2()
    print(sorted(os.listdir(FIG)))
