"""Flight geometry from the image metadata and verification of the wall-to-wall
map statistics from the saved detections.

Inputs: data_input/exif_rgb.csv (position, relative altitude, yaw and gimbal pitch
of the 774 RGB frames, read from their XMP metadata), data_input/flowers_all.csv
(7,401 merged detections of the wall-to-wall run), data_input/study_area_summary.csv.
Outputs: output/flight_geometry.json, output/map_statistics.json
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
IMG_W, IMG_H = 5280, 3956          # RGB frame size (pixels)
GSD_ODM = 0.0021042124208549742    # m, average GSD reported by OpenDroneMap for the RGB reconstruction

# ---------------- flight geometry ----------------
e = pd.read_csv(os.path.join(ROOT, "data_input", "exif_rgb.csv"))
tr = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
e["x"], e["y"] = tr.transform(e.lon.values, e.lat.values)
e["time"] = pd.to_datetime(e.file.str.slice(4, 18), format="%Y%m%d%H%M%S")
dt = e.time.diff().dt.total_seconds()
step = np.hypot(e.x.diff(), e.y.diff())
heading = np.degrees(np.arctan2(e.x.diff(), e.y.diff())) % 180     # line orientation (undirected)
# a new line starts where the direction of travel turns by more than 45 degrees
dirn = np.degrees(np.arctan2(e.x.diff(), e.y.diff()))
turn = np.abs(((dirn.diff() + 180) % 360) - 180)
line = (turn > 45).cumsum().fillna(0).astype(int)
e["line"] = line
lines = e.groupby("line").filter(lambda g: len(g) >= 10).groupby("line")
main_orient = np.median(heading.dropna())
ux, uy = np.sin(np.radians(main_orient)), np.cos(np.radians(main_orient))     # along-track unit vector
px_, py_ = -uy, ux                                                            # across-track unit vector
cross = lines.apply(lambda g: np.median(g.x * px_ + g.y * py_)).sort_values()
spacing_lines = np.diff(cross.values)
lens = lines.apply(lambda g: np.ptp(g.x * ux + g.y * uy))
along = step[(dt > 0) & (dt < 3)]
foot_along, foot_across = IMG_H * GSD_ODM, IMG_W * GSD_ODM
geo = {
    "n_images": int(len(e)), "date": str(e.time.min().date()),
    "start": str(e.time.min().time()), "end": str(e.time.max().time()),
    "duration_min": float((e.time.max() - e.time.min()).total_seconds() / 60),
    "capture_interval_s_median": float(np.median(dt.dropna())),
    "rel_alt_min": float(e.rel_alt.min()), "rel_alt_median": float(e.rel_alt.median()), "rel_alt_max": float(e.rel_alt.max()),
    "pitch_min": float(e.pitch.min()), "pitch_max": float(e.pitch.max()),
    "n_lines": int(len(cross)), "line_length_m_median": float(np.median(lens)),
    "line_orientation_deg": float(main_orient),
    "along_track_spacing_m_median": float(np.median(along)),
    "line_spacing_m_median": float(np.median(spacing_lines)),
    "footprint_along_m": foot_along, "footprint_across_m": foot_across,
    "forward_overlap": float(1 - np.median(along) / foot_along),
    "side_overlap": float(1 - np.median(spacing_lines) / foot_across),
}
json.dump(geo, open(os.path.join(OUT, "flight_geometry.json"), "w"), indent=1)
print(json.dumps(geo, indent=1))

# ---------------- wall-to-wall statistics ----------------
f = pd.read_csv(os.path.join(ROOT, "data_input", "flowers_all.csv"))
sa = pd.read_csv(os.path.join(ROOT, "data_input", "study_area_summary.csv")).iloc[0]
valid_m2, bbox_m2 = float(sa.valid_m2), float(sa.bbox_m2)
w_cm = (f.px1 - f.px0) * sa.gsd_cm
h_cm = (f.py1 - f.py0) * sa.gsd_cm
disc_area_m2 = float((np.pi / 4 * f.dia_cm ** 2).sum() / 1e4)
box_area_m2 = float(f.area_cm2.sum() / 1e4)
# density grid exactly as in the mapping script (cell = extent / int(extent))
nx, ny = int(sa.width_m), int(sa.height_m)
H1, _, _ = np.histogram2d(f.geo_x, f.geo_y, bins=[nx, ny], range=[[sa.utm_left, sa.utm_right], [sa.utm_bottom, sa.utm_top]])
# exact 1-m cells anchored at the lower-left corner of the scene
xe = np.arange(sa.utm_left, sa.utm_right + 1, 1.0)
ye = np.arange(sa.utm_bottom, sa.utm_top + 1, 1.0)
H2, _, _ = np.histogram2d(f.geo_x, f.geo_y, bins=[xe, ye])
cell1 = (sa.width_m / nx) * (sa.height_m / ny)
m = {
    "n": int(len(f)), "density_valid": len(f) / valid_m2, "density_bbox": len(f) / bbox_m2,
    "dia_mean": float(f.dia_cm.mean()), "dia_median": float(f.dia_cm.median()), "dia_sd": float(f.dia_cm.std()),
    "dia_min": float(f.dia_cm.min()), "dia_max": float(f.dia_cm.max()),
    "dia_skew": float(f.dia_cm.skew()),
    "box_w_median_cm": float(w_cm.median()), "box_h_median_cm": float(h_cm.median()),
    "conf_mean": float(f.conf.mean()), "conf_median": float(f.conf.median()),
    "conf_min": float(f.conf.min()), "conf_max": float(f.conf.max()),
    "pct_conf_gt_0.5": float((f.conf > 0.5).mean() * 100), "pct_conf_gt_0.9": float((f.conf > 0.9).mean() * 100),
    "box_area_m2": box_area_m2, "box_cover_pct": box_area_m2 / valid_m2 * 100,
    "disc_area_m2": disc_area_m2, "disc_cover_pct": disc_area_m2 / valid_m2 * 100,
    "grid_script_cell_m2": float(cell1), "grid_script_max_count": float(H1.max()),
    "grid_script_mean_occupied": float(H1[H1 > 0].mean()), "grid_script_n_occupied": int((H1 > 0).sum()),
    "grid_exact_max": float(H2.max()), "grid_exact_mean_occupied": float(H2[H2 > 0].mean()),
    "grid_exact_n_occupied": int((H2 > 0).sum()),
    "valid_m2": valid_m2, "bbox_m2": bbox_m2,
}
json.dump(m, open(os.path.join(OUT, "map_statistics.json"), "w"), indent=1)
print(json.dumps(m, indent=1))
