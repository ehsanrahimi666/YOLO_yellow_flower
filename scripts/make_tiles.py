"""Re-create the 640 x 640-pixel training tiles, YOLO labels and fold assignments
from the reference polygons and the annotated orthomosaic subset.

This script re-implements the tiling rules described in the manuscript (Sections 2.3, 2.4 and 2.7).
With yellow_flowers.shp and RGB_subset.tif it reproduces the 122 tiles, their 1,194 bounding boxes and
the five spatial folds of the published data set; tile images are re-encoded as JPEG and therefore
differ from the published tiles only by compression noise (see README).

Usage:
    python make_tiles.py <polygons.shp> <RGB_subset.tif> <output_dir> [--compare <published_tiles_dir>]
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

TILE = 640               # tile size in pixels (1.28 m at 0.2 cm per pixel)
MIN_VALID = 0.30         # minimum share of valid pixels for a tile to be kept
MIN_BOX_SHARE = 0.5      # a box is kept in a tile if at least half of its area lies inside the tile
FOLD_EDGES = [0, 2560, 5120, 7040, 8960, 10880]  # x pixel limits of the five strips (folds 0-4)


def read_tile(src, tx, ty):
    a = src.read(window=Window(tx, ty, TILE, TILE), boundless=True, fill_value=0)
    valid = ~np.all(a == 0, axis=0) & ~np.all(a >= 254, axis=0)
    img = np.clip(a, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return img, float(valid.mean())


def boxes_in_pixels(poly, transform):
    b = poly.geometry.bounds
    px = transform.a
    c0 = (b.minx - transform.c) / px
    c1 = (b.maxx - transform.c) / px
    r0 = (transform.f - b.maxy) / px
    r1 = (transform.f - b.miny) / px
    return np.c_[c0, r0, c1, r1]


def tile_labels(boxes, tx, ty):
    out = []
    for x0, y0, x1, y1 in boxes:
        ix0, iy0, ix1, iy1 = max(x0, tx), max(y0, ty), min(x1, tx + TILE), min(y1, ty + TILE)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) / ((x1 - x0) * (y1 - y0)) >= MIN_BOX_SHARE:
            out.append((ix0 - tx, iy0 - ty, ix1 - tx, iy1 - ty))
    return out


def fold_of(tx):
    for f in range(5):
        if FOLD_EDGES[f] <= tx < FOLD_EDGES[f + 1]:
            return f
    raise ValueError(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("polygons"); ap.add_argument("raster"); ap.add_argument("out")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    poly = gpd.read_file(a.polygons)
    src = rasterio.open(a.raster)
    assert poly.crs.to_epsg() == src.crs.to_epsg() == 32652
    boxes = boxes_in_pixels(poly, src.transform)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    cand = set()
    for x0, y0, x1, y1 in boxes:
        for tx in range(int(x0 // TILE) * TILE, int(np.ceil(x1 / TILE)) * TILE, TILE):
            for ty in range(int(y0 // TILE) * TILE, int(np.ceil(y1 / TILE)) * TILE, TILE):
                if 0 <= tx < src.width and 0 <= ty < src.height:
                    cand.add((tx, ty))
    rows = []
    for tx, ty in sorted(cand):
        img, valid = read_tile(src, tx, ty)
        if valid < MIN_VALID:
            continue
        labs = tile_labels(boxes, tx, ty)
        if not labs:
            continue
        name = f"t_x{tx}_y{ty}"
        Image.fromarray(img).save(os.path.join(a.out, "images", name + ".jpg"), quality=95)
        with open(os.path.join(a.out, "labels", name + ".txt"), "w") as f:
            for x0, y0, x1, y1 in labs:
                f.write(f"0 {(x0 + x1) / 2 / TILE:.6f} {(y0 + y1) / 2 / TILE:.6f} "
                        f"{(x1 - x0) / TILE:.6f} {(y1 - y0) / TILE:.6f}\n")
        rows.append({"tile": name, "tile_x": tx, "tile_y": ty, "n": len(labs), "valid": round(valid, 3),
                     "fold": fold_of(tx)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "tiles_summary.csv"), index=False)
    print(f"candidate tiles {len(cand)}, kept tiles {len(df)}, boxes {int(df.n.sum())}")
    print(df.groupby("fold").agg(tiles=("tile", "count"), flowers=("n", "sum")))
    if a.compare:
        pub = pd.read_csv(os.path.join(a.compare, "tiles_summary.csv"))
        same = set(pub.tile) == set(df.tile)
        m = df.merge(pub, on="tile", suffixes=("", "_pub"))
        print("same tile set:", same, "| same box counts:", bool((m.n == m.n_pub).all()),
              "| same folds:", bool((m.fold == m.fold_pub).all()))
        diffs = []
        for t in m.tile:
            x = np.asarray(Image.open(os.path.join(a.out, "images", t + ".jpg")), dtype=float)
            y = np.asarray(Image.open(os.path.join(a.compare, "images", t + ".jpg")), dtype=float)
            diffs.append(np.abs(x - y).mean())
        print(f"mean absolute pixel difference to the published tiles: {np.mean(diffs):.2f} "
              f"(max over tiles {np.max(diffs):.2f}) grey levels")


if __name__ == "__main__":
    main()
