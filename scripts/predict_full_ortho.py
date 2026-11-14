# =====================================================================
#  WALL-TO-WALL FLOWER DETECTION  —  full Riverside orthomosaic
#
#  1. reports STUDY-AREA details for Materials & Methods (extent, area, GSD)
#  2. slides the trained detector over the whole ortho (nodata-aware)
#  3. global NMS across tile seams
#  4. writes ONE merged georeferenced shapefile for the whole study area
#  5. produces the density map + size/confidence figures for the paper
#  6. writes a paste-back report (PREDICTION_REPORT.txt)
#
#  NOTE ON RADIOMETRY: the model was trained on RGB_subset.tif (uint16
#  holding 0-255 values, nodata = BLACK) and is applied here to RGB.tif
#  (uint8, nodata = WHITE/255). probe_scale() + the nodata rules below
#  put both rasters on the same 0-255 scale so the model sees what it
#  was trained on.
#
#  RUN:  python predict_full_ortho.py
#  pip install ultralytics rasterio geopandas shapely opencv-python matplotlib
# =====================================================================

import os, json, time, datetime, shutil
import numpy as np, pandas as pd

# ============================ CONFIG ============================
WEIGHTS = "best.pt"                      # final YOLOv9t model (repository root)
ORTHO   = "RGB.tif"                      # orthomosaic of the study plot (available on request)
OUT     = "YOLO_PREDICTION"

TILE       = 640
OVERLAP    = 64        # px overlap so flowers on seams are not cut
CONF       = 0.25      # confidence threshold
NMS_IOU    = 0.5       # global NMS across tile seams
MIN_VALID  = 0.20      # skip tiles with <20% real data
DEVICE     = "cpu"
PREVIEW_N  = 30        # tiles saved with boxes drawn
DENS_CELL_M = 1.0      # density grid cell size (m)
LIMIT_TILES = 0        # >0 = quick test on N tiles (set 0 for the real run)
# ===============================================================

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_xy
import geopandas as gpd
from shapely.geometry import box as shbox
import cv2, torch
from torchvision.ops import nms
from ultralytics import YOLO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

log("="*70); log("WALL-TO-WALL FLOWER DETECTION")
log("run:", datetime.datetime.now().isoformat(timespec="seconds")); log("="*70)

# =====================================================================
#  READER (identical logic to training: global scale + nodata rules)
# =====================================================================
def probe_scale(src):
    d = max(1, int(max(src.width, src.height) / 1500))
    a = src.read(indexes=(1,2,3), out_shape=(3, max(1,src.height//d), max(1,src.width//d)))
    v = a.astype(np.float64).ravel()
    if src.nodata is not None: v = v[v != src.nodata]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 99.9)) if v.size else 255.0

def read_tile(src, tx, ty, hi, size=TILE):
    raw = np.transpose(src.read(indexes=(1,2,3), window=Window(tx,ty,size,size),
                                boundless=True, fill_value=0), (1,2,0))
    if raw.dtype == np.uint8:                    img = raw
    elif hi <= 255:                              img = np.clip(raw,0,255).astype(np.uint8)
    else: img = np.clip(raw.astype(np.float32)*(255.0/hi),0,255).astype(np.uint8)
    valid = np.ones(img.shape[:2], bool)
    if src.nodata is not None:
        valid &= ~np.all(raw == src.nodata, axis=2)      # RGB.tif: nodata = 255 (white)
    valid &= ~np.all(img == 0,   axis=2)                 # black margin
    valid &= ~np.all(img >= 254, axis=2)                 # white margin
    # ultralytics expects BGR numpy arrays (as produced by cv2.imread).
    # rasterio gives RGB -> convert, or the model sees yellow flowers as blue.
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), valid

# =====================================================================
# 1. STUDY AREA  ->  paste this block into Materials & Methods
# =====================================================================
src = rasterio.open(ORTHO)
HI  = probe_scale(src)
GSD = abs(src.transform.a)
b   = src.bounds
W_m, H_m = b.right-b.left, b.top-b.bottom
bbox_ha  = W_m*H_m/1e4

d = max(1, int(max(src.width, src.height)/3000))
msk = src.read_masks(1, out_shape=(max(1,src.height//d), max(1,src.width//d)))
prev_img, _ = None, None
a_small = src.read(indexes=(1,2,3), out_shape=(3, max(1,src.height//d), max(1,src.width//d)))
a_small = np.transpose(a_small,(1,2,0))
valid_small = np.ones(a_small.shape[:2], bool)
if src.nodata is not None:
    valid_small &= ~np.all(a_small == src.nodata, axis=2)
valid_small &= ~np.all(a_small == 0, axis=2)
valid_small &= ~np.all(a_small >= 254, axis=2)
valid_frac = float(valid_small.mean())
valid_ha   = valid_frac*bbox_ha
valid_m2   = valid_ha*1e4

lon, lat   = warp_xy(src.crs, "EPSG:4326", [(b.left+b.right)/2], [(b.top+b.bottom)/2])
lons, lats = warp_xy(src.crs, "EPSG:4326", [b.left,b.right], [b.bottom,b.top])

log("\n[1] STUDY AREA  (for Materials & Methods)")
log(f"    orthomosaic       : {os.path.basename(ORTHO)}")
log(f"    size              : {src.width} x {src.height} px | {src.count} bands | {src.dtypes[0]}")
log(f"    CRS               : {src.crs}")
log(f"    GSD               : {GSD*100:.3f} cm/px  ({GSD*1000:.1f} mm/px)")
log(f"    bounding box (UTM): E {b.left:.2f}..{b.right:.2f}   N {b.bottom:.2f}..{b.top:.2f}")
log(f"    dimensions        : {W_m:.1f} m (E-W) x {H_m:.1f} m (N-S)  = {bbox_ha:.4f} ha ({W_m*H_m:.0f} m2)")
log(f"    imaged (valid) area: {valid_frac*100:.1f}% of bbox = {valid_ha:.4f} ha ({valid_m2:.0f} m2)")
log(f"    centre (lon, lat) : {lon[0]:.6f}, {lat[0]:.6f}")
log(f"    extent  lon       : {min(lons):.6f} .. {max(lons):.6f}")
log(f"    extent  lat       : {min(lats):.6f} .. {max(lats):.6f}")
log(f"    declared nodata   : {src.nodata}   | data range probe (99.9 pct): {HI:.0f}")
log(f"    tile footprint    : {TILE} px = {TILE*GSD:.2f} m")

pd.DataFrame([dict(ortho=os.path.basename(ORTHO), width_px=src.width, height_px=src.height,
    bands=src.count, dtype=str(src.dtypes[0]), crs=str(src.crs),
    gsd_cm=round(GSD*100,4), gsd_mm=round(GSD*1000,2),
    utm_left=round(b.left,2), utm_right=round(b.right,2), utm_bottom=round(b.bottom,2), utm_top=round(b.top,2),
    width_m=round(W_m,2), height_m=round(H_m,2), bbox_ha=round(bbox_ha,4), bbox_m2=round(W_m*H_m,1),
    valid_pct=round(valid_frac*100,2), valid_ha=round(valid_ha,4), valid_m2=round(valid_m2,1),
    centre_lon=round(lon[0],6), centre_lat=round(lat[0],6),
    lon_min=round(min(lons),6), lon_max=round(max(lons),6),
    lat_min=round(min(lats),6), lat_max=round(max(lats),6),
    tile_px=TILE, tile_m=round(TILE*GSD,3))]).to_csv(f"{OUT}/study_area_summary.csv", index=False)

# =====================================================================
# 2. MODEL
# =====================================================================
log("\n[2] MODEL")
model = YOLO(WEIGHTS)
try:
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    log(f"    weights   : {WEIGHTS}")
    log(f"    trained as: {ck.get('train_args',{}).get('model','?')} | "
        f"epoch {ck.get('epoch','?')} | imgsz {ck.get('train_args',{}).get('imgsz','?')}")
except Exception as e:
    log(f"    (metadata unreadable: {e})")
log(f"    parameters: {sum(p.numel() for p in model.model.parameters())/1e6:.2f} M | device {DEVICE}")
log(f"    classes   : {model.names}")

# --- SANITY: the model must detect on the tiles it was trained on ---
import glob as _g
_tt = sorted(_g.glob("data/tiles/images/t_x*.jpg"))[:5]
if _tt:
    _n = sum(len(model.predict(cv2.imread(t), imgsz=TILE, conf=CONF,
                               device=DEVICE, verbose=False)[0].boxes) for t in _tt)
    log(f"    sanity    : {_n} detections on {len(_tt)} training tiles "
        f"({'OK' if _n > 10 else 'FAIL'})")
    assert _n > 10, ("SANITY FAIL: wrong weights or wrong channel order — "
                     "aborting before a 40-minute run.")
else:
    log("    sanity    : training tiles not found, skipped")

# =====================================================================
# 3. SLIDING-WINDOW PREDICTION
# =====================================================================
log("\n[3] PREDICTION")
step = TILE - OVERLAP
xs = list(range(0, src.width,  step)); ys = list(range(0, src.height, step))
coords = [(x,y) for y in ys for x in xs]
if LIMIT_TILES: coords = coords[:LIMIT_TILES]
log(f"    tiles: {len(coords)} (stride {step} px, {OVERLAP} px overlap)")

PREV = os.path.join(OUT, "predicted_tiles"); shutil.rmtree(PREV, ignore_errors=True); os.makedirs(PREV)
boxes, scores, saved, skipped = [], [], 0, 0
t0 = time.time()
for i, (tx, ty) in enumerate(coords):
    img, valid = read_tile(src, tx, ty, HI)
    if valid.mean() < MIN_VALID:
        skipped += 1
    else:
        r = model.predict(img, imgsz=TILE, conf=CONF, device=DEVICE, verbose=False)[0]
        if len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for (a_,b_,c_,d_), s in zip(xyxy, cf):
                cy = int(min(max((b_+d_)/2,0),TILE-1)); cx = int(min(max((a_+c_)/2,0),TILE-1))
                if not valid[cy,cx]: continue            # detection on nodata -> drop
                boxes.append([a_+tx, b_+ty, c_+tx, d_+ty]); scores.append(float(s))
            if saved < PREVIEW_N:
                vis = img.copy()
                for (a_,b_,c_,d_), s in zip(xyxy, cf):
                    cv2.rectangle(vis,(int(a_),int(b_)),(int(c_),int(d_)),(255,0,0),2)
                    cv2.putText(vis,f"{s:.2f}",(int(a_),max(int(b_)-3,8)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
                cv2.imwrite(f"{PREV}/pred_x{tx}_y{ty}.jpg", vis)   # already BGR
                saved += 1
    if (i+1) % 100 == 0 or i == len(coords)-1:
        el = time.time()-t0; rate = (i+1)/max(el,1e-9)
        log(f"      {i+1}/{len(coords)} tiles | {len(boxes)} detections | "
            f"{el/60:.1f} min elapsed | ETA {(len(coords)-i-1)/max(rate,1e-9)/60:.1f} min")
log(f"    skipped as nodata : {skipped}")
log(f"    raw detections    : {len(boxes)}  in {(time.time()-t0)/60:.1f} min")

# =====================================================================
# 4. GLOBAL NMS  ->  ONE merged shapefile
# =====================================================================
log("\n[4] MERGING")
if not boxes:
    log("    !! no detections — check CONF, or that the model matches this ortho's radiometry")
    open(f"{OUT}/PREDICTION_REPORT.txt","w",encoding="utf-8").write("\n".join(LOG)); raise SystemExit
bt = torch.tensor(boxes, dtype=torch.float32); st = torch.tensor(scores)
keep = nms(bt, st, NMS_IOU).numpy()
n_raw = len(boxes)
boxes = bt.numpy()[keep].astype(np.float64)   # float32 northing -> 25 cm steps -> 0-height boxes
scores = st.numpy()[keep].astype(np.float64)
log(f"    global NMS (IoU {NMS_IOU}): {n_raw} -> {len(boxes)} flowers "
    f"({n_raw-len(boxes)} duplicates from tile seams removed)")

T = src.transform
def pix2geo(col, row):                      # force float64 (northing ~4.04e6)
    col = float(col); row = float(row)
    return T.a*col + T.b*row + T.c, T.d*col + T.e*row + T.f
geoms, recs = [], []
for i, ((a_,b_,c_,d_), s) in enumerate(zip(boxes, scores), 1):
    gx0,gy0 = pix2geo(a_,b_); gx1,gy1 = pix2geo(c_,d_)
    minx,maxx = sorted([gx0,gx1]); miny,maxy = sorted([gy0,gy1])
    geoms.append(shbox(minx,miny,maxx,maxy))
    recs.append(dict(flower_id=i, conf=round(float(s),3),
                     dia_cm=round(((c_-a_)+(d_-b_))/2*GSD*100, 2),
                     area_cm2=round((c_-a_)*(d_-b_)*(GSD*100)**2, 2),
                     geo_x=round((minx+maxx)/2,3), geo_y=round((miny+maxy)/2,3),
                     px0=round(float(a_),2), py0=round(float(b_),2),
                     px1=round(float(c_),2), py1=round(float(d_),2)))
gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=src.crs)
_bb = gdf.bounds; _w = (_bb.maxx-_bb.minx)*100; _h = (_bb.maxy-_bb.miny)*100
log(f"    box geometry: width {_w.median():.2f} cm | height {_h.median():.2f} cm | "
    f"aspect {(_h/_w).median():.2f}")
assert _h.median() > 1, "Zero-height boxes -> float32 precision loss in the geo transform"

gdf.to_file(f"{OUT}/flowers_all_boxes.shp")                       # <- ONE file, whole study area
gdf.to_file(f"{OUT}/flowers_all.gpkg", driver="GPKG")
pts = gdf.copy(); pts["geometry"] = pts.geometry.centroid
pts.to_file(f"{OUT}/flowers_all_points.shp")
pd.DataFrame(recs).to_csv(f"{OUT}/flowers_all.csv", index=False)
log(f"    >> flowers_all_boxes.shp  ({len(gdf)} boxes, single merged file)")
log(f"    >> flowers_all_points.shp | flowers_all.gpkg | flowers_all.csv")

# =====================================================================
# 5. STATISTICS  ->  Results section
# =====================================================================
log("\n[5] RESULTS STATISTICS")
df = pd.DataFrame(recs)
dens_bbox  = len(df)/(W_m*H_m)
dens_valid = len(df)/max(valid_m2,1e-9)
log(f"    flowers detected  : {len(df)}")
log(f"    density (imaged)  : {dens_valid:.3f} flowers/m2   over {valid_m2:.0f} m2")
log(f"    density (bbox)    : {dens_bbox:.3f} flowers/m2   over {W_m*H_m:.0f} m2")
log(f"    diameter (cm)     : mean {df.dia_cm.mean():.2f} | median {df.dia_cm.median():.2f} | "
    f"sd {df.dia_cm.std():.2f} | range {df.dia_cm.min():.2f}-{df.dia_cm.max():.2f}")
log(f"    confidence        : mean {df.conf.mean():.3f} | median {df.conf.median():.3f} | "
    f">0.5: {(df.conf>0.5).mean()*100:.1f}% | >0.9: {(df.conf>0.9).mean()*100:.1f}%")
tot_area_m2 = df.area_cm2.sum()/1e4
log(f"    total floral area : {tot_area_m2:.3f} m2 = {tot_area_m2/max(valid_m2,1e-9)*100:.3f}% of imaged area")

# ---- density grid ----
nx = max(1,int(W_m/DENS_CELL_M)); ny = max(1,int(H_m/DENS_CELL_M))
Hgrid, xe, ye = np.histogram2d(df.geo_x, df.geo_y, bins=[nx,ny],
                               range=[[b.left,b.right],[b.bottom,b.top]])
Hgrid = Hgrid/(DENS_CELL_M**2)
log(f"    density grid ({DENS_CELL_M} m cells): max {Hgrid.max():.1f} | "
    f"mean of occupied cells {Hgrid[Hgrid>0].mean():.2f} flowers/m2")

# =====================================================================
# 6. FIGURES  ->  paper
# =====================================================================
fig, ax = plt.subplots(figsize=(11,5.2))
im = ax.imshow(Hgrid.T, origin="lower", cmap="YlOrRd", aspect="equal",
               extent=[b.left,b.right,b.bottom,b.top])
ax.set_xlabel("Easting (m, UTM 52N)"); ax.set_ylabel("Northing (m)")
ax.set_title(f"Detected flower density ({DENS_CELL_M} m cells) — {len(df)} flowers", fontweight="bold")
plt.colorbar(im, ax=ax, label="flowers m$^{-2}$", fraction=.025)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_density_map.png", dpi=200, facecolor="white"); plt.close()

fig, axes = plt.subplots(1,3, figsize=(14,4))
axes[0].hist(df.dia_cm, bins=40, color="#C56A2C", edgecolor="white")
axes[0].set_xlabel("flower diameter (cm)"); axes[0].set_ylabel("count")
axes[0].set_title("(a) Detected flower size", fontweight="bold", loc="left")
axes[1].hist(df.conf, bins=40, color="#3E6DA6", edgecolor="white")
axes[1].set_xlabel("detection confidence"); axes[1].set_ylabel("count")
axes[1].set_title("(b) Confidence", fontweight="bold", loc="left")
axes[2].scatter(df.geo_x, df.geo_y, s=3, c=df.conf, cmap="viridis", alpha=.6)
axes[2].set_xlabel("Easting (m)"); axes[2].set_ylabel("Northing (m)")
axes[2].set_title("(c) Spatial distribution", fontweight="bold", loc="left"); axes[2].set_aspect("equal")
for a_ in axes[:2]: a_.spines[["top","right"]].set_visible(False); a_.grid(alpha=.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_detection_stats.png", dpi=200, facecolor="white"); plt.close()
log("\n    figures: fig_density_map.png | fig_detection_stats.png")

# =====================================================================
# 7. PASTE-BACK SUMMARY
# =====================================================================
summary = dict(n_flowers=int(len(df)), density_per_m2_imaged=float(round(dens_valid,4)),
    density_per_m2_bbox=float(round(dens_bbox,4)), imaged_area_m2=float(round(valid_m2,1)),
    bbox_area_m2=float(round(W_m*H_m,1)), gsd_mm=float(round(GSD*1000,2)),
    dia_cm_mean=float(round(df.dia_cm.mean(),2)), dia_cm_sd=float(round(df.dia_cm.std(),2)),
    dia_cm_median=float(round(df.dia_cm.median(),2)),
    conf_mean=float(round(df.conf.mean(),3)),
    pct_conf_gt_0_9=float(round((df.conf>0.9).mean()*100,1)),
    total_floral_area_m2=float(round(tot_area_m2,3)),
    pct_floral_cover=float(round(tot_area_m2/max(valid_m2,1e-9)*100,3)),
    tiles=int(len(coords)), tiles_skipped=int(skipped), raw_detections=int(n_raw),
    removed_by_nms=int(n_raw-len(df)), minutes=float(round((time.time()-t0)/60,1)))
json.dump(summary, open(f"{OUT}/prediction_summary.json","w"), indent=2)
pd.DataFrame([summary]).to_csv(f"{OUT}/prediction_summary.csv", index=False)

log("\n" + "="*70); log("PASTE THIS BACK FOR THE PAPER"); log("="*70)
for k,v in summary.items(): log(f"  {k:24}: {v}")
open(f"{OUT}/PREDICTION_REPORT.txt","w",encoding="utf-8").write("\n".join(LOG))
src.close()
print(f"\nDONE -> {OUT}")
print("  flowers_all_boxes.shp / _points.shp / .gpkg / .csv   (ONE merged file)")
print("  study_area_summary.csv | prediction_summary.csv/.json | PREDICTION_REPORT.txt")
print("  fig_density_map.png | fig_detection_stats.png | predicted_tiles/")
