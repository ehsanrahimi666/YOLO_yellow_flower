"""Independent re-derivation of the tiled training data set from the reference
polygons and the annotated orthomosaic subset, compared with the saved tiles and
labels used in the benchmark.

Inputs: yellow_flowers.shp (reference polygons), RGB_subset.tif (annotated
orthomosaic subset), data/tiles/ (saved tiles, YOLO labels and
tiles_summary.csv). Outputs: output/dataset_check.json, output/dataset_check_tiles.csv
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
# folder holding yellow_flowers.shp and the annotated orthomosaic subset RGB_subset.tif (available on request)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data", "reference_polygons")
TILES = os.path.join(REPO, "data", "tiles")
OUT = os.path.join(ROOT, "output")
TILE, MIN_VALID = 640, 0.30

poly = gpd.read_file(os.path.join(SRC, "yellow_flowers.shp"))
src = rasterio.open(os.path.join(SRC, "RGB_subset.tif"))
T = src.transform
assert poly.crs.to_epsg() == 32652 and src.crs.to_epsg() == 32652
px = T.a

# polygon bounds in pixel coordinates of the subset
b = poly.geometry.bounds
c0 = (b.minx - T.c) / px
c1 = (b.maxx - T.c) / px
r0 = (T.f - b.maxy) / px
r1 = (T.f - b.miny) / px
boxes = np.c_[c0, r0, c1, r1]
cx, cy = (c0 + c1) / 2, (r0 + r1) / 2
w_px, h_px = c1 - c0, r1 - r0
dia_px = (w_px + h_px) / 2
eq_dia_cm = 2 * np.sqrt(poly.geometry.area / np.pi) * 100

# valid fraction of every tile (valid = not all bands zero and not all bands >= 254, as in the original tiling)
ncol, nrow = int(np.ceil(src.width / TILE)), int(np.ceil(src.height / TILE))
valid = {}
for j in range(nrow):
    for i in range(ncol):
        a = src.read(window=Window(i * TILE, j * TILE, TILE, TILE), boundless=True, fill_value=0)
        valid[(i * TILE, j * TILE)] = float((~np.all(a == 0, axis=0) & ~np.all(a >= 254, axis=0)).mean())

# candidate tiles: tiles intersected by at least one reference box
tile_of = [(int(x // TILE) * TILE, int(y // TILE) * TILE) for x, y in zip(cx, cy)]
cand = set()
for x0, y0, x1, y1 in boxes:
    for tx in range(int(x0 // TILE) * TILE, int(np.ceil(x1 / TILE)) * TILE, TILE):
        for ty in range(int(y0 // TILE) * TILE, int(np.ceil(y1 / TILE)) * TILE, TILE):
            cand.add((tx, ty))
cand = sorted(c for c in cand if c in valid)
skipped = [t for t in cand if valid[t] < MIN_VALID]
kept_valid = [t for t in cand if valid[t] >= MIN_VALID]


def labels_for(t, rule):
    tx, ty = t
    out = []
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        ix0, iy0 = max(x0, tx), max(y0, ty)
        ix1, iy1 = min(x1, tx + TILE), min(y1, ty + TILE)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        frac = (ix1 - ix0) * (iy1 - iy0) / ((x1 - x0) * (y1 - y0))
        inside_centre = tx <= cx[k] < tx + TILE and ty <= cy[k] < ty + TILE
        ok = {"centre": inside_centre, "half_box": frac >= 0.5}[rule]
        if ok:
            out.append((k, ix0 - tx, iy0 - ty, ix1 - tx, iy1 - ty))
    return out


saved = pd.read_csv(os.path.join(TILES, "tiles_summary.csv"))
saved_set = {(int(r.tile_x), int(r.tile_y)) for r in saved.itertuples()}


def read_saved(tile):
    f = os.path.join(TILES, "labels", f"t_x{tile[0]}_y{tile[1]}.txt")
    rows = [list(map(float, l.split())) for l in open(f) if l.strip()]
    return np.array([[(x - w / 2) * TILE, (y - h / 2) * TILE, (x + w / 2) * TILE, (y + h / 2) * TILE]
                     for _, x, y, w, h in rows]) if rows else np.zeros((0, 4))


def match(pred, ref, thr=0.9):
    used, n = set(), 0
    for p in pred:
        best, bj = 0, -1
        for j, r in enumerate(ref):
            if j in used:
                continue
            iw = min(p[2], r[2]) - max(p[0], r[0])
            ih = min(p[3], r[3]) - max(p[1], r[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            iou = inter / ((p[2] - p[0]) * (p[3] - p[1]) + (r[2] - r[0]) * (r[3] - r[1]) - inter)
            if iou > best:
                best, bj = iou, j
        if best >= thr:
            used.add(bj)
            n += 1
    return n


kept = [t for t in kept_valid if labels_for(t, "half_box")]
empty_after_rule = [t for t in kept_valid if not labels_for(t, "half_box")]
res = {}
tile_rows = []
for rule in ["centre", "half_box"]:
    tot, exact_tiles, matched, n_saved_total = 0, 0, 0, 0
    for t in sorted(saved_set):
        mine = np.array([l[1:] for l in labels_for(t, rule)]) if labels_for(t, rule) else np.zeros((0, 4))
        ref = read_saved(t)
        m = match(mine, ref)
        tot += len(mine)
        n_saved_total += len(ref)
        matched += m
        exact_tiles += int(len(mine) == len(ref) == m)
        if rule == "half_box":
            tile_rows.append({"tile_x": t[0], "tile_y": t[1], "valid_recomputed": valid[t],
                              "n_rederived": len(mine), "n_saved": len(ref), "matched_iou90": m})
    res[rule] = {"boxes_rederived": tot, "boxes_saved": n_saved_total, "matched_iou90": matched,
                 "tiles_identical": exact_tiles}

tdf = pd.DataFrame(tile_rows).merge(saved[["tile_x", "tile_y", "valid", "fold"]], on=["tile_x", "tile_y"])
tdf["valid_diff"] = (tdf.valid_recomputed - tdf.valid).abs()
tdf.to_csv(os.path.join(OUT, "dataset_check_tiles.csv"), index=False)

# polygons that end up in no kept tile under the half-box rule
in_any = set()
for t in sorted(saved_set):
    for l in labels_for(t, "half_box"):
        in_any.add(l[0])
lost = sorted(set(range(len(poly))) - in_any)
lost_in_skipped = sum(1 for k in lost if tile_of[k] in skipped)

# adjacency of validation tiles to training tiles (4-neighbourhood)
fold_of = {(int(r.tile_x), int(r.tile_y)): int(r.fold) for r in saved.itertuples()}
adj = 0
for t, f in fold_of.items():
    nb = [(t[0] + dx, t[1] + dy) for dx, dy in [(-TILE, 0), (TILE, 0), (0, -TILE), (0, TILE)]]
    adj += any(fold_of.get(n, f) != f for n in nb)
strips = saved.groupby("fold").agg(x_min=("tile_x", "min"), x_max=("tile_x", "max"),
                                   tiles=("tile", "count"), flowers=("n", "sum"))
strips["columns"] = (strips.x_max - strips.x_min) // TILE + 1
strips["width_m"] = strips["columns"] * TILE * px

summary = {
    "subset_px": [src.width, src.height], "subset_m": [src.width * px, src.height * px],
    "pixel_size_m": px, "n_polygons": int(len(poly)),
    "candidate_tiles": len(cand), "kept_tiles_rederived": len(kept), "skipped_tiles_rederived": len(skipped), "tiles_empty_after_half_box_rule": len(empty_after_rule),
    "kept_set_equals_saved": set(kept) == saved_set,
    "max_valid_diff": float(tdf.valid_diff.max()),
    "rules": res, "polygons_lost_half_box": len(lost), "lost_in_skipped_tiles": lost_in_skipped,
    "polygon_dia_cm_bbox_mean": float(dia_px.mean() * px * 100),
    "polygon_dia_cm_bbox_median": float(np.median(dia_px) * px * 100),
    "polygon_dia_px_mean": float(dia_px.mean()), "polygon_dia_px_min": float(dia_px.min()),
    "polygon_dia_px_max": float(dia_px.max()),
    "polygon_eqcircle_dia_cm_mean": float(eq_dia_cm.mean()),
    "val_tiles_bordering_other_fold": int(adj), "n_tiles": len(fold_of),
    "strips": strips.reset_index().to_dict(orient="records"),
}
json.dump(summary, open(os.path.join(OUT, "dataset_check.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
