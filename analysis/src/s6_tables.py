"""Formats the manuscript tables (Tables 4-5, A1-A3) from the saved outputs.

Writes output/table4_mixed_model.csv, output/table5_cost.csv,
output/tableA1_pairwise.csv, output/tableA2_lofo.csv, output/tableA3_counts.csv.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
S = json.load(open(os.path.join(OUT, "stats_summary.json")))
ORDER = ["YOLOv9t", "YOLO11n", "YOLOv8n", "YOLO11s", "YOLOv10n"]
LABEL = {"mAP50": "mAP@0.5", "mAP50_95": "mAP@0.5:0.95", "precision": "Precision", "recall": "Recall"}


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# Table 4: detector and fold effects, shares of variation
rows = []
for m in ["mAP50", "mAP50_95", "precision", "recall"]:
    x = S[m]
    rows.append({
        "Metric": LABEL[m],
        "Detector F(4,16)": f"{x['F_det']:.2f}", "Detector p": fmt_p(x["p_det"]),
        "Fold F(4,16)": f"{x['F_fold']:.2f}", "Fold p": fmt_p(x["p_fold"]),
        "SS detector %": f"{100 * x['share_det']:.1f}", "SS fold %": f"{100 * x['share_fold']:.1f}",
        "SS residual %": f"{100 * x['share_res']:.1f}",
        "VC detector %": f"{100 * x['vshare_det']:.1f}", "VC fold %": f"{100 * x['vshare_fold']:.1f}",
        "VC residual %": f"{100 * x['vshare_res']:.1f}",
        "LRT chi2(4)": f"{x['lrt_chi2']:.2f}", "LRT p": fmt_p(x["lrt_p"]),
    })
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table4_mixed_model.csv"), index=False)

# Table 5: computational cost
lat = pd.read_csv(os.path.join(OUT, "cpu_latency.csv")).set_index("model")
t3 = pd.read_csv(os.path.join(OUT, "table3_reproduced.csv"), index_col=0)
rows = []
for m in ORDER:
    rows.append({"Model": m, "Parameters (M)": f"{lat.loc[m, 'params_M']:.2f}", "GFLOPs": f"{lat.loc[m, 'GFLOPs']:.1f}",
                 "CPU latency ms median (IQR)": f"{lat.loc[m, 'median']:.1f} ({lat.loc[m, 'q25']:.1f}–{lat.loc[m, 'q75']:.1f})",
                 "GPU ms per image mean (range)": f"{t3.loc[m, 'ms_per_img']:.1f} ({t3.loc[m, 'ms_min']:.1f}–{t3.loc[m, 'ms_max']:.1f})",
                 "Training time min": f"{t3.loc[m, 'train_min_total']:.1f}"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table5_cost.csv"), index=False)

# Table A1: pairwise comparisons (mAP@0.5 and mAP@0.5:0.95)
def smallest_margin(row, kind):
    key = {"pooled": "equiv_", "paired": "equiv_paired_", "nb": "equiv_nb_"}[kind]
    for d in [0.01, 0.02, 0.05]:
        if row.get(f"{key}{d}", False):
            return f"{d:.2f}"
    return "> 0.05"


rows = []
for m in ["mAP50", "mAP50_95"]:
    pw = pd.read_csv(os.path.join(OUT, f"pairwise_{m}.csv"))
    for _, r in pw.iterrows():
        rows.append({"Metric": LABEL[m], "Pair": f"{r.model_1} – {r.model_2}", "Difference": f"{r['diff']:.3f}",
                     "Tukey 95% CI": f"{r.tukey95_lo:.3f} to {r.tukey95_hi:.3f}", "Tukey p": fmt_p(r.p_tukey),
                     "90% CI": f"{r.ci90_lo:.3f} to {r.ci90_hi:.3f}", "TOST p (0.02)": fmt_p(r["p_tost_0.02"]),
                     "Smallest margin met": smallest_margin(r, "pooled"),
                     "90% CI corrected": f"{r.ci90nb_lo:.3f} to {r.ci90nb_hi:.3f}",
                     "Smallest margin met (corrected)": smallest_margin(r, "nb"),
                     "Smallest margin met (paired)": smallest_margin(r, "paired")})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tableA1_pairwise.csv"), index=False)

# Table A2: leave-one-fold-out
l50 = pd.read_csv(os.path.join(OUT, "lofo_mAP50.csv"))
l95 = pd.read_csv(os.path.join(OUT, "lofo_mAP50_95.csv"))
rows = []
for (_, a), (_, b) in zip(l50.iterrows(), l95.iterrows()):
    rows.append({"Omitted fold": int(a.dropped_fold), "mAP@0.5 detector p": fmt_p(a.p_det),
                 "mAP@0.5 SS detector %": f"{100 * a.share_det:.1f}", "mAP@0.5 SS fold %": f"{100 * a.share_fold:.1f}",
                 "mAP@0.5 pairs equivalent (0.02)": int(a["n_equiv_0.02"]),
                 "mAP@0.5:0.95 detector p": fmt_p(b.p_det), "mAP@0.5:0.95 SS detector %": f"{100 * b.share_det:.1f}",
                 "mAP@0.5:0.95 SS fold %": f"{100 * b.share_fold:.1f}"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tableA2_lofo.csv"), index=False)

# Table A3: counting accuracy of the YOLOv9t fold models at the mapping settings
c = pd.read_csv(os.path.join(OUT, "v9t_counts.csv"))
rows = []
for _, r in c.iterrows():
    rows.append({"Region": "Whole tile" if r.region == "whole_tile" else "Tile interior",
                 "Fold": "All" if str(r.fold) == "all" else str(r.fold), "Reference": int(r.n_ref),
                 "Detected": int(r["n_pred_0.25"]), "Matched": int(r["tp_0.25"]),
                 "Unmatched detections": int(r["fp_0.25"]), "Missed": int(r["fn_0.25"]),
                 "Precision": f"{r['precision_0.25']:.2f}", "Recall": f"{r['recall_0.25']:.2f}",
                 "Count ratio": f"{r['count_ratio_0.25']:.2f}",
                 "F1-optimal threshold": f"{r.f1opt_threshold:.2f}", "Count ratio at optimum": f"{r.f1opt_count_ratio:.2f}"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tableA3_counts.csv"), index=False)
for f in ["table4_mixed_model", "table5_cost", "tableA1_pairwise", "tableA2_lofo", "tableA3_counts"]:
    print("=====", f)
    print(pd.read_csv(os.path.join(OUT, f + ".csv")).to_string(index=False))
