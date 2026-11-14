"""Re-evaluation of the saved YOLOv9t fold models and out-of-fold counting accuracy.

For each spatial fold k, the YOLOv9t model trained without fold k (saved as
runs/yolov9t/fold<k>/weights/best.pt) is applied to the tiles of fold k.

1. Framework validation (conf 0.001, NMS IoU 0.7, imgsz 640, CPU, FP32) to
   reproduce the per-fold metrics reported in the benchmark.
2. Counting accuracy with the settings of the wall-to-wall map: per-tile NMS
   IoU 0.7, then NMS IoU 0.5, confidence >= 0.25, detections whose centre lies on
   invalid (black or white) pixels removed; greedy one-to-one matching to the
   reference boxes at IoU >= 0.5.
3. The same at the F1-optimal confidence of each fold (reported only).

Outputs: output/v9t_fold_reval.csv, output/v9t_counts.csv,
output/v9t_threshold_scan.csv, output/v9t_fp_examples/ (crops for visual QA).
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
BENCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs", "yolov9t")
TILES = os.path.join(REPO, "data", "tiles")
OUT = os.path.join(ROOT, "output")
WORK = os.path.join(ROOT, "data_derived", "fold_sets")
CONF_MAP, NMS_MAP, IOU_MATCH = 0.25, 0.5, 0.5
torch.set_num_threads(2)

tiles = pd.read_csv(os.path.join(TILES, "tiles_summary.csv"))
reported = pd.read_csv(os.path.join(ROOT, "data_input", "benchmark_per_fold.csv"))
reported = reported[reported.model == "yolov9t"].set_index("fold")


def load_labels(tile):
    rows = [list(map(float, l.split())) for l in open(os.path.join(TILES, "labels", tile + ".txt")) if l.strip()]
    return np.array([[(x - w / 2) * 640, (y - h / 2) * 640, (x + w / 2) * 640, (y + h / 2) * 640]
                     for _, x, y, w, h in rows]).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def greedy_match(pred, score, ref, thr=IOU_MATCH):
    order = np.argsort(-score)
    iou = iou_matrix(pred[order], ref)
    used = np.zeros(len(ref), bool)
    tp = np.zeros(len(pred), bool)
    for i in range(len(order)):
        if len(ref) == 0:
            break
        cand = np.where(~used & (iou[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(iou[i, cand])]
            used[j] = True
            tp[order[i]] = True
    return tp, used


rows_val, rows_cnt, scan_rows = [], [], []
fp_dir = os.path.join(OUT, "v9t_fp_examples")
shutil.rmtree(fp_dir, ignore_errors=True); os.makedirs(fp_dir)
all_det = []
for k in range(5):
    w = os.path.join(BENCH, f"fold{k}", "weights", "best.pt")
    model = YOLO(w)
    sub = tiles[tiles.fold == k]
    # 1. framework validation on a data set holding only this fold's tiles
    d = os.path.join(WORK, f"fold{k}")
    shutil.rmtree(d, ignore_errors=True)
    for part in ["images/val", "labels/val"]:
        os.makedirs(os.path.join(d, part))
    for t in sub.tile:
        shutil.copy(os.path.join(TILES, "images", t + ".jpg"), os.path.join(d, "images/val"))
        shutil.copy(os.path.join(TILES, "labels", t + ".txt"), os.path.join(d, "labels/val"))
    yaml = os.path.join(d, "data.yaml")
    open(yaml, "w").write(f"path: {d}\ntrain: images/val\nval: images/val\nnc: 1\nnames: ['yellow_flower']\n")
    m = model.val(data=yaml, imgsz=640, batch=16, conf=0.001, iou=0.7, device="cpu", half=False,
                  plots=False, verbose=False, workers=0, project=os.path.join(WORK, "runs"), name=f"val{k}", exist_ok=True)
    rows_val.append({"fold": k, "mAP50": m.box.map50, "mAP50_95": m.box.map, "precision": m.box.mp,
                     "recall": m.box.mr, "reported_mAP50": reported.loc[k, "mAP50"],
                     "reported_mAP50_95": reported.loc[k, "mAP50_95"],
                     "reported_precision": reported.loc[k, "precision"], "reported_recall": reported.loc[k, "recall"]})
    # 2. detections with the mapping settings (all scores >= 0.001 kept for the threshold scan)
    for t in sub.tile:
        img = cv2.imread(os.path.join(TILES, "images", t + ".jpg"))
        r = model.predict(img, imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy()
        if len(xyxy):
            keep = nms(torch.tensor(xyxy), torch.tensor(sc), NMS_MAP).numpy()
            xyxy, sc = xyxy[keep], sc[keep]
            ccx = np.clip(((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int), 0, 639)
            ccy = np.clip(((xyxy[:, 1] + xyxy[:, 3]) / 2).astype(int), 0, 639)
            px = img[ccy, ccx]
            ok = ~(np.all(px == 0, axis=1) | np.all(px >= 254, axis=1))
            xyxy, sc = xyxy[ok], sc[ok]
        all_det.append({"fold": k, "tile": t, "xyxy": xyxy, "score": sc, "ref": load_labels(t)})

EDGE = 32  # px, about one capitulum diameter


def interior(b):
    c = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
    return np.all((c >= EDGE) & (c <= 640 - EDGE), axis=1)


thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
for region in ["whole_tile", "interior"]:
    for th in thresholds:
        for k in list(range(5)) + ["all"]:
            tp = fp = fn = 0
            for e in all_det:
                if k != "all" and e["fold"] != k:
                    continue
                s = e["score"] >= th
                xy, sc, ref = e["xyxy"][s], e["score"][s], e["ref"]
                if region == "interior":
                    # match against all references so that edge pairs are not counted as errors,
                    # then score only detections and references whose centres are interior
                    mt, used = greedy_match(xy, sc, ref)
                    pi, ri = interior(xy) if len(xy) else np.zeros(0, bool), interior(ref) if len(ref) else np.zeros(0, bool)
                    tp += int((mt & pi).sum()); fp += int((~mt & pi).sum()); fn += int((~used & ri).sum())
                else:
                    mt, used = greedy_match(xy, sc, ref)
                    tp += int(mt.sum()); fp += int((~mt).sum()); fn += int((~used).sum())
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            scan_rows.append({"region": region, "threshold": th, "fold": k, "tp": tp, "fp": fp, "fn": fn,
                              "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
                              "count_ratio": (tp + fp) / max(tp + fn, 1)})
scan = pd.DataFrame(scan_rows)
scan.to_csv(os.path.join(OUT, "v9t_threshold_scan.csv"), index=False)

for region, k in [(r_, k_) for r_ in ["whole_tile", "interior"] for k_ in list(range(5)) + ["all"]]:
    s = scan[(scan.fold == k) & (scan.region == region)]
    at = s[np.isclose(s.threshold, CONF_MAP)].iloc[0]
    best = s.loc[s.f1.idxmax()]
    rows_cnt.append({"region": region, "fold": k, "n_ref": int(at.tp + at.fn), "n_pred_0.25": int(at.tp + at.fp),
                     "tp_0.25": int(at.tp), "fp_0.25": int(at.fp), "fn_0.25": int(at.fn),
                     "precision_0.25": at.precision, "recall_0.25": at.recall, "f1_0.25": at.f1,
                     "count_ratio_0.25": at.count_ratio, "f1opt_threshold": best.threshold,
                     "f1opt_precision": best.precision, "f1opt_recall": best.recall, "f1opt_f1": best.f1,
                     "f1opt_count_ratio": best.count_ratio})
pd.DataFrame(rows_val).to_csv(os.path.join(OUT, "v9t_fold_reval.csv"), index=False)
cnt = pd.DataFrame(rows_cnt)
cnt.to_csv(os.path.join(OUT, "v9t_counts.csv"), index=False)

# crops of the highest-scoring false positives at the mapping threshold, for visual QA
fps = []
for e in all_det:
    s = e["score"] >= CONF_MAP
    mt, _ = greedy_match(e["xyxy"][s], e["score"][s], e["ref"])
    for b, sc in zip(e["xyxy"][s][~mt], e["score"][s][~mt]):
        fps.append((sc, e["tile"], b))
fps.sort(key=lambda z: -z[0])
edge_fp = [not interior(np.array([b]))[0] for _, _, b in fps]
pd.DataFrame([{"rank": i + 1, "score": round(float(sc), 4), "tile": t, "x0": float(b[0]), "y0": float(b[1]),
               "x1": float(b[2]), "y1": float(b[3]), "edge": bool(e)}
              for i, ((sc, t, b), e) in enumerate(zip(fps, edge_fp))]).to_csv(os.path.join(OUT, "v9t_unmatched_detections.csv"), index=False)
for i, (sc, t, b) in enumerate(fps[:40]):
    img = cv2.imread(os.path.join(TILES, "images", t + ".jpg"))
    x0, y0, x1, y1 = [int(v) for v in b]
    cxm, cym = (x0 + x1) // 2, (y0 + y1) // 2
    crop = img[max(cym - 48, 0):cym + 48, max(cxm - 48, 0):cxm + 48].copy()
    cv2.rectangle(crop, (x0 - max(cxm - 48, 0), y0 - max(cym - 48, 0)), (x1 - max(cxm - 48, 0), y1 - max(cym - 48, 0)), (255, 0, 0), 1)
    cv2.imwrite(os.path.join(fp_dir, f"fp{i:02d}_{sc:.2f}_{t}.png"), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
json.dump({"n_fp_0.25": len(fps), "n_fp_edge": int(sum(edge_fp)), "n_top40_edge": int(sum(edge_fp[:40]))}, open(os.path.join(OUT, "v9t_fp_summary.json"), "w"))
print(pd.DataFrame(rows_val).round(4).to_string())
print(cnt.round(4).to_string())
