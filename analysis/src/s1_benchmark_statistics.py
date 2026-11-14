"""Statistical comparison of the five detectors from the 25 per-fold results.

Reads the analysis settings in config/analysis.json and the per-fold
metrics saved by the cross-validation runs, and writes:
  output/table3_reproduced.csv        detector means and SDs (Table 3)
  output/anova_<metric>.csv           two-way additive ANOVA (detector, fold)
  output/varcomp_<metric>.csv         variance components and shares
  output/lmm_<metric>.txt             mixed model fit (detector fixed, fold random)
  output/pairwise_<metric>.csv        pairwise differences, Tukey CIs, TOST,
                                      paired t and dependence-corrected tests
  output/lofo_<metric>.csv            leave-one-fold-out influence check
  output/timing_checks.csv            GPU timing summaries and repeat-run check
  output/stats_summary.json           numbers quoted in the manuscript
"""
import json
import itertools
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "config", "analysis.json")))
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

NAMES = {"yolov8n": "YOLOv8n", "yolov9t": "YOLOv9t", "yolov10n": "YOLOv10n",
         "yolo11n": "YOLO11n", "yolo11s": "YOLO11s"}
PARAMS_M = {"yolov9t": 2.13, "yolo11n": 2.62, "yolov8n": 3.16, "yolo11s": 9.46, "yolov10n": 2.78}

d = pd.read_csv(os.path.join(ROOT, CFG["inputs"]["per_fold_metrics"]))
d["model"] = d["model"].astype(str)
assert set(d.model) == set(CFG["detectors"]) and set(d.fold) == set(CFG["folds"])
assert d.groupby(["model", "fold"]).size().eq(1).all() and len(d) == 25

tiles = pd.read_csv(os.path.join(ROOT, CFG["inputs"]["tiles"]))
n_val = tiles.groupby("fold").size().reindex(CFG["folds"])
n_train = len(tiles) - n_val
ratio = float((n_val / n_train).mean())

ORDER = d.groupby("model")["mAP50"].mean().sort_values(ascending=False).index.tolist()
METRICS = [CFG["primary_metric"]] + CFG["secondary_metrics"]
DELTAS = [CFG["equivalence"]["delta_primary"]] + CFG["equivalence"]["delta_sensitivity"]
ALPHA = CFG["equivalence"]["alpha"]


def wide(metric, data=d):
    return data.pivot(index="model", columns="fold", values=metric).loc[ORDER]


def anova(y):
    """Two-way additive ANOVA for a detector x fold table with one value per cell."""
    a, b = y.shape
    gm = y.values.mean()
    rm, cm = y.mean(axis=1), y.mean(axis=0)
    ss_det = b * ((rm - gm) ** 2).sum()
    ss_fold = a * ((cm - gm) ** 2).sum()
    ss_tot = ((y.values - gm) ** 2).sum()
    ss_res = ss_tot - ss_det - ss_fold
    df_det, df_fold, df_res = a - 1, b - 1, (a - 1) * (b - 1)
    ms_det, ms_fold, ms_res = ss_det / df_det, ss_fold / df_fold, ss_res / df_res
    f_det, f_fold = ms_det / ms_res, ms_fold / ms_res
    out = {
        "a": a, "b": b, "ss_det": ss_det, "ss_fold": ss_fold, "ss_res": ss_res, "ss_tot": ss_tot,
        "df_det": df_det, "df_fold": df_fold, "df_res": df_res,
        "ms_det": ms_det, "ms_fold": ms_fold, "ms_res": ms_res,
        "F_det": f_det, "p_det": stats.f.sf(f_det, df_det, df_res),
        "F_fold": f_fold, "p_fold": stats.f.sf(f_fold, df_fold, df_res),
        "share_det": ss_det / ss_tot, "share_fold": ss_fold / ss_tot, "share_res": ss_res / ss_tot,
        "eta2p_det": ss_det / (ss_det + ss_res), "eta2p_fold": ss_fold / (ss_fold + ss_res),
    }
    var_det = max(0.0, (ms_det - ms_res) / b)
    var_fold = max(0.0, (ms_fold - ms_res) / a)
    tot = var_det + var_fold + ms_res
    out.update({"var_det": var_det, "var_fold": var_fold, "var_res": ms_res,
                "vshare_det": var_det / tot, "vshare_fold": var_fold / tot, "vshare_res": ms_res / tot,
                "sd_det": np.sqrt(var_det), "sd_fold": np.sqrt(var_fold), "sd_res": np.sqrt(ms_res)})
    return out


def tost(diff, se, df, delta):
    p_low = stats.t.sf((diff + delta) / se, df)
    p_up = stats.t.cdf((diff - delta) / se, df)
    return max(p_low, p_up)


def pairwise(y, ms_res, df_res):
    a, b = y.shape
    se_pool = np.sqrt(2 * ms_res / b)
    q_crit = stats.studentized_range.ppf(0.95, a, df_res)
    t90_pool = stats.t.ppf(1 - ALPHA, df_res)
    t90_4 = stats.t.ppf(1 - ALPHA, b - 1)
    rows = []
    for m1, m2 in itertools.combinations(y.index, 2):
        dj = (y.loc[m1] - y.loc[m2]).values
        diff = dj.mean()
        t = diff / se_pool
        row = {"model_1": NAMES[m1], "model_2": NAMES[m2], "diff": diff,
               "se_pooled": se_pool, "t_pooled": t, "p_pooled": 2 * stats.t.sf(abs(t), df_res),
               "p_tukey": stats.studentized_range.sf(abs(diff) / np.sqrt(ms_res / b), a, df_res),
               "tukey95_lo": diff - q_crit * np.sqrt(ms_res / b),
               "tukey95_hi": diff + q_crit * np.sqrt(ms_res / b),
               "ci90_lo": diff - t90_pool * se_pool, "ci90_hi": diff + t90_pool * se_pool}
        for dl in DELTAS:
            p = tost(diff, se_pool, df_res, dl)
            row[f"p_tost_{dl}"] = p
            row[f"equiv_{dl}"] = bool(p < ALPHA)
        # paired t on the pair's own five differences (df = b - 1)
        sd_d = dj.std(ddof=1)
        se_p = sd_d / np.sqrt(b)
        row.update({"sd_diff": sd_d, "se_paired": se_p,
                    "p_paired": 2 * stats.t.sf(abs(diff / se_p), b - 1),
                    "ci90p_lo": diff - t90_4 * se_p, "ci90p_hi": diff + t90_4 * se_p})
        for dl in DELTAS:
            row[f"equiv_paired_{dl}"] = bool(tost(diff, se_p, b - 1, dl) < ALPHA)
        # Nadeau-Bengio corrected variance for dependent cross-validation folds
        se_nb = np.sqrt((1 / b + ratio) * sd_d ** 2)
        row.update({"se_nb": se_nb, "p_nb": 2 * stats.t.sf(abs(diff / se_nb), b - 1),
                    "ci90nb_lo": diff - t90_4 * se_nb, "ci90nb_hi": diff + t90_4 * se_nb})
        for dl in DELTAS:
            row[f"equiv_nb_{dl}"] = bool(tost(diff, se_nb, b - 1, dl) < ALPHA)
        rows.append(row)
    return pd.DataFrame(rows)


def joint_equiv(pw, members, col):
    names = {NAMES[m] for m in members}
    sub = pw[pw.model_1.isin(names) & pw.model_2.isin(names)]
    return bool(sub[col].all())


summary = {"n_runs": len(d), "val_train_ratio_mean": ratio, "order": [NAMES[m] for m in ORDER]}

# ---- Table 3 reproduced from the per-fold file -------------------------------------
t3 = d.groupby("model").agg(
    mAP50=("mAP50", "mean"), mAP50_sd=("mAP50", "std"),
    mAP50_95=("mAP50_95", "mean"), mAP50_95_sd=("mAP50_95", "std"),
    precision=("precision", "mean"), precision_sd=("precision", "std"),
    recall=("recall", "mean"), recall_sd=("recall", "std"),
    ms_per_img=("ms_per_img", "mean"), ms_sd=("ms_per_img", "std"),
    ms_min=("ms_per_img", "min"), ms_max=("ms_per_img", "max"),
    epochs_mean=("epochs_trained", "mean"), train_min_total=("train_min", "sum")).loc[ORDER]
t3["fps"] = 1000 / t3["ms_per_img"]
t3.insert(0, "params_M", [PARAMS_M[m] for m in t3.index])
t3.index = [NAMES[m] for m in t3.index]
t3.round(4).to_csv(os.path.join(OUT, "table3_reproduced.csv"))
master = pd.read_csv(os.path.join(ROOT, "data_input", "benchmark_master.csv")).set_index("model")
chk = []
for m in ORDER:
    for col in ["mAP50", "mAP50_sd", "mAP50_95", "mAP50_95_sd", "precision", "precision_sd",
                "recall", "recall_sd", "ms_per_img", "fps"]:
        mine = t3.loc[NAMES[m], col if col != "mAP50_95_sd" else "mAP50_95_sd"]
        chk.append({"model": NAMES[m], "column": col, "master": master.loc[m, col],
                    "recomputed": round(float(mine), 4),
                    "abs_diff": abs(master.loc[m, col] - float(mine))})
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(OUT, "table3_vs_master_check.csv"), index=False)
summary["table3_max_abs_diff_vs_master"] = float(chk.abs_diff.max())

# ---- old descriptive decomposition (mean SDs) -----------------------------------------
y50 = wide("mAP50")
summary["mean_sd_models_within_fold"] = float(y50.std(axis=0, ddof=1).mean())
summary["mean_sd_folds_within_model"] = float(y50.std(axis=1, ddof=1).mean())
summary["fold_means_mAP50"] = {int(k): float(v) for k, v in y50.mean(axis=0).items()}
summary["fold_range_mAP50"] = {int(k): [float(y50[k].min()), float(y50[k].max())] for k in y50.columns}

# ---- ANOVA, variance components, mixed model, pairwise ---------------------------------
for metric in METRICS:
    y = wide(metric)
    av = anova(y)
    pd.DataFrame([av]).to_csv(os.path.join(OUT, f"anova_{metric}.csv"), index=False)
    # mixed model with statsmodels (REML) and likelihood-ratio test on ML fits
    md = smf.mixedlm(f"{metric} ~ C(model)", d, groups=d["fold"])
    fit = md.fit(reml=True)
    full_ml = md.fit(reml=False)
    red_ml = smf.mixedlm(f"{metric} ~ 1", d, groups=d["fold"]).fit(reml=False)
    lr = 2 * (full_ml.llf - red_ml.llf)
    with open(os.path.join(OUT, f"lmm_{metric}.txt"), "w") as fh:
        fh.write(str(fit.summary()))
        fh.write(f"\nREML fold variance: {float(fit.cov_re.iloc[0, 0]):.6g}\n")
        fh.write(f"REML residual variance: {float(fit.scale):.6g}\n")
        fh.write(f"LRT detector (ML fits): chi2 = {lr:.4f}, df = 4, p = {stats.chi2.sf(lr, 4):.4g}\n")
    pw = pairwise(y, av["ms_res"], av["df_res"])
    pw.to_csv(os.path.join(OUT, f"pairwise_{metric}.csv"), index=False)
    top4 = [m for m in ORDER if m != "yolov10n"]
    s = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in av.items()}
    s.update({
        "reml_var_fold": float(fit.cov_re.iloc[0, 0]), "reml_var_res": float(fit.scale),
        "icc_fold": float(fit.cov_re.iloc[0, 0] / (fit.cov_re.iloc[0, 0] + fit.scale)),
        "lrt_chi2": float(lr), "lrt_p": float(stats.chi2.sf(lr, 4)),
        "means": {NAMES[m]: float(v) for m, v in y.mean(axis=1).items()},
        "sds": {NAMES[m]: float(v) for m, v in y.std(axis=1, ddof=1).items()},
        "ci90_halfwidth_pooled": float(stats.t.ppf(1 - ALPHA, av["df_res"]) * np.sqrt(2 * av["ms_res"] / av["b"])),
        "n_sig_tukey": int((pw.p_tukey < 0.05).sum()),
        "sig_pairs_tukey": pw.loc[pw.p_tukey < 0.05, ["model_1", "model_2", "diff", "p_tukey"]].round(4).values.tolist(),
        "max_abs_ci90_top4": float(pw[pw.model_1.isin([NAMES[m] for m in top4]) & pw.model_2.isin([NAMES[m] for m in top4])][["ci90_lo", "ci90_hi"]].abs().values.max()),
    })
    for dl in DELTAS:
        s[f"n_equiv_{dl}"] = int(pw[f"equiv_{dl}"].sum())
        s[f"top4_joint_equiv_{dl}"] = joint_equiv(pw, top4, f"equiv_{dl}")
        s[f"top4_joint_equiv_paired_{dl}"] = joint_equiv(pw, top4, f"equiv_paired_{dl}")
        s[f"top4_joint_equiv_nb_{dl}"] = joint_equiv(pw, top4, f"equiv_nb_{dl}")
        s[f"equiv_pairs_{dl}"] = pw.loc[pw[f"equiv_{dl}"], ["model_1", "model_2"]].values.tolist()
    summary[metric] = s

    # leave-one-fold-out influence check
    rows = []
    for f_drop in CFG["folds"]:
        yy = y.drop(columns=f_drop)
        av2 = anova(yy)
        pw2 = pairwise(yy, av2["ms_res"], av2["df_res"])
        rows.append({"dropped_fold": f_drop, "p_det": av2["p_det"], "p_fold": av2["p_fold"],
                     "share_det": av2["share_det"], "share_fold": av2["share_fold"],
                     "vshare_det": av2["vshare_det"], "vshare_fold": av2["vshare_fold"],
                     "n_equiv_0.02": int(pw2["equiv_0.02"].sum()),
                     "top4_joint_equiv_0.02": joint_equiv(pw2, top4, "equiv_0.02")})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"lofo_{metric}.csv"), index=False)
    summary[metric]["lofo"] = rows

# ---- GPU timing: spread and the repeated YOLOv9t runs -----------------------------------
ms = wide("ms_per_img")
rank = ms.rank(axis=0)
rep = pd.read_csv(os.path.join(ROOT, "data_input", "yolov9t_cv.csv")).set_index("fold").sort_index()
v9 = d[d.model == "yolov9t"].set_index("fold").sort_index()
timing = pd.DataFrame({
    "model": [NAMES[m] for m in ORDER],
    "ms_mean": ms.mean(axis=1).values, "ms_median": ms.median(axis=1).values,
    "ms_min": ms.min(axis=1).values, "ms_max": ms.max(axis=1).values,
    "times_fastest": (rank == 1).sum(axis=1).values, "times_slowest": (rank == 5).sum(axis=1).values})
timing.to_csv(os.path.join(OUT, "timing_checks.csv"), index=False)
summary["timing"] = timing.round(3).to_dict(orient="records")
summary["yolov9t_repeat"] = {
    "accuracy_identical": bool(np.allclose(v9[["mAP50", "mAP50_95", "precision", "recall"]].values,
                                           rep[["mAP50", "mAP50_95", "precision", "recall"]].values)),
    "ms_run1": v9.ms_per_img.tolist(), "ms_run2": rep.ms_per_img.tolist(),
    "max_ratio": float((np.maximum(v9.ms_per_img, rep.ms_per_img) / np.minimum(v9.ms_per_img, rep.ms_per_img)).max())}

# ---- training length -------------------------------------------------------------------
ep = wide("epochs_trained")
summary["epochs_mean"] = {NAMES[m]: float(v) for m, v in ep.mean(axis=1).items()}
summary["best_epoch_implied"] = {NAMES[m]: (ep.loc[m] - 50).tolist() for m in ORDER}
summary["runs_at_epoch_cap"] = int((ep == 150).values.sum())
summary["train_min_total"] = {NAMES[m]: float(v) for m, v in d.groupby("model")["train_min"].sum().items()}
summary["train_min_all"] = float(d.train_min.sum())


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(summary), open(os.path.join(OUT, "stats_summary.json"), "w"), indent=1)
print("done:", sys.version.split()[0])
