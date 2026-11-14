# Yellow-flower detection in a riverside grassland with UAV imagery and YOLO detectors

Data, trained models, outputs and code for the study *Automated Yellow-Flower Detection in Grasslands Using UAV
Imagery and YOLO models* (E. Rahimi and C. Jung). Five one-stage detectors (YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n
and YOLO11s) were compared for detecting the flower heads (capitula) of *Coreopsis lanceolata* in a 2 mm RGB
orthomosaic of a riverside grassland near Andong, Republic of Korea, with spatially blocked five-fold
cross-validation, and the YOLOv9t model was applied wall-to-wall to the study plot.

All spatial data are in WGS 84 / UTM zone 52N (EPSG:32652); the orthomosaic pixel size is 0.2 cm.

## Contents

| Path | Contents |
|---|---|
| `data/reference_polygons/` | `yellow_flowers.shp` (+ `.shx`, `.dbf`, `.prj`, `.cpg`): 1,235 capitula digitised as polygons in the annotated subset of the orthomosaic. |
| `data/tiles/images/` | The 122 training tiles (640 × 640 pixels, 1.28 m on the ground, JPEG) used in the benchmark. File names give the tile origin in pixels of the annotated subset (`t_x<col>_y<row>`). |
| `data/tiles/labels/` | One YOLO label file per tile (`class x_centre y_centre width height`, normalised to the tile size; 1,194 boxes). |
| `data/tiles/tiles_summary.csv` | Tile name, origin (`tile_x`, `tile_y`), number of boxes (`n`), share of valid pixels (`valid`) and spatial fold (`fold`, 0–4). |
| `results/benchmark_per_fold.csv` | Per-fold results of all 25 training runs (5 detectors × 5 folds): epochs trained, mAP@0.5, mAP@0.5:0.95, precision, recall, training time and GPU time per image. |
| `results/benchmark_master.csv` | Means and standard deviations of the per-fold results. |
| `results/dataset_summary.csv` | Summary of the tiled data set. |
| `runs/yolov9t/fold0` … `fold4` | YOLOv9t cross-validation runs: training settings (`args.yaml`), per-epoch training log (`results.csv`) and weights (`weights/best.pt`) of the model trained without the respective fold. |
| `runs/yolov9t/final_all` | Settings and training log of the YOLOv9t model trained on all 122 tiles (58 epochs); its weights are `best.pt` in the repository root. |
| `best.pt` | Final YOLOv9t model used for the wall-to-wall map. |
| `flowers_all.csv`, `prediction_summary.*`, `study_area_summary.csv`, `PREDICTION_REPORT.txt`, `fig_*.png`, `pred_x*_y*.jpg` | Outputs of the wall-to-wall mapping: the 7,401 merged detections (confidence, box diameter, box area and coordinates), summaries, figures and example tiles with predicted boxes. |
| `yolov9t_cv.csv` | Per-fold results of a repeated YOLOv9t cross-validation run (identical accuracy, different GPU times). |
| `scripts/predict_full_ortho.py` | Script that produced the wall-to-wall map (tiling with 64-pixel overlap, confidence 0.25, global non-maximum suppression at IoU 0.5). Only the input and output paths were changed to repository-relative paths. |
| `scripts/make_tiles.py` | Re-creates the training tiles, labels and folds from the reference polygons and the annotated orthomosaic subset (see below). |
| `analysis/` | Statistical comparison of the detectors, counting-accuracy check, CPU latency benchmark, flight-geometry and map checks, manuscript tables and figures (see below). |

The RGB orthomosaic of the study plot (`RGB.tif`, 30,755 × 20,687 pixels, about 0.9 GB), the annotated subset
(`RGB_subset.tif`, 11,286 × 9,361 pixels) and the 774 original UAV images are too large for this repository and
are available from the corresponding author on request.

## Training

Each detector was trained with Ultralytics 8.4.100 at an input size of 640 pixels, batch size 16, at most 150
epochs, early stopping with a patience of 50 epochs, seed 0 in deterministic mode, automatic optimiser selection
(AdamW, initial learning rate 0.002, momentum 0.9, weight decay 0.0005; linear warm-up over at least 100 iterations and
linear decay to 1% of the initial rate), the default augmentation and mixed precision
on an NVIDIA Tesla T4 GPU, starting from COCO-pretrained weights. The complete settings of the YOLOv9t runs are in
their `args.yaml` files; for the other four detectors, the per-fold results are given in
`results/benchmark_per_fold.csv`.

## Re-creating the tiles

```
python scripts/make_tiles.py data/reference_polygons/yellow_flowers.shp RGB_subset.tif tiles_out --compare data/tiles
```

Tiles are 640 × 640-pixel windows of a regular grid over the annotated subset. A tile is kept if at least 30% of its
pixels are valid (not black and not white in all bands) and it contains at least one box; a box is assigned to a
tile if at least half of its area lies inside the tile. The five folds are strips of tile columns with x limits of
0, 2,560, 5,120, 7,040, 8,960 and 10,880 pixels. With `RGB_subset.tif`, the script reproduces the published set of
122 tiles, the 1,194 boxes and the fold assignments exactly; the re-encoded JPEG images differ from the published
tiles by a mean of 0.94 grey levels (compression noise).

## Analysis

The folder `analysis/` contains the analysis settings (`config/analysis.json`), the input tables
(`data_input/`), the scripts (`src/`), their outputs (`output/`) and the figures (`figures/`). Run the scripts from
the `analysis` folder:

| Script | Purpose |
|---|---|
| `src/s1_benchmark_statistics.py` | Linear mixed model (detector fixed, fold random), randomised complete block ANOVA, variance components, Tukey comparisons, two one-sided tests of equivalence (±0.01, ±0.02, ±0.05), paired and dependence-corrected tests, leave-one-fold-out refits. |
| `src/s2_dataset_check.py` | Re-derives the tile set and labels from the polygons and `RGB_subset.tif` and compares them with `data/tiles` (needs `RGB_subset.tif` in `data/reference_polygons/` or its folder as argument). |
| `src/s3_yolov9t_fold_evaluation.py` | Re-evaluates the YOLOv9t fold models on their held-out folds and computes counting accuracy at the mapping threshold. |
| `src/s4_cpu_latency.py` | Controlled CPU latency benchmark and GFLOPs of the five single-class architectures. |
| `src/s5_flight_and_map_checks.py` | Flight geometry from the image metadata (`data_input/exif_rgb.csv`) and statistics of the wall-to-wall map. |
| `src/s6_tables.py` | Tables 4, 5 and A1–A3 of the manuscript. |
| `src/s7_figures.py` | Figures 2, 6, 9, A1 and A2 of the manuscript. |

Software: Python 3.13 with numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, geopandas 1.2, rasterio 1.5,
shapely 2.2, matplotlib 3.11, Pillow 12.3, PyTorch 2.15 and Ultralytics 8.4.100 (see `requirements.txt`).

## Citation

If you use these data or code, please cite the article: Rahimi, E., & Jung, C. Automated Yellow-Flower Detection in
Grasslands Using UAV Imagery and YOLO models. Citation metadata are given in `CITATION.cff`.
