"""Controlled CPU forward-pass latency and theoretical cost of the five detectors.

The architectures are built from the framework configuration files with one
class (forward-pass cost does not depend on the weight values), fused as for
deployment, and timed on identical 640 x 640 inputs. Models are timed in
interleaved rounds so that any change in background load affects all of them.
Outputs: output/cpu_latency.csv, output/cpu_latency_raw.csv
"""
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
CFG = {"YOLOv9t": "yolov9t.yaml", "YOLO11n": "yolo11n.yaml", "YOLOv8n": "yolov8n.yaml",
       "YOLO11s": "yolo11s.yaml", "YOLOv10n": "yolov10n.yaml"}
THREADS, WARMUP, ROUNDS, PER_ROUND = 2, 30, 10, 20

torch.manual_seed(0)
torch.set_num_threads(THREADS)
x = torch.rand(1, 3, 640, 640)
models, info = {}, []
for name, cfg in CFG.items():
    m = DetectionModel(cfg, nc=1, verbose=False)
    n_par = sum(p.numel() for p in m.parameters())
    m = m.fuse(verbose=False).eval()
    gflops = get_flops(m, 640)
    with torch.inference_mode():
        for _ in range(WARMUP):
            m(x)
    models[name] = m
    info.append({"model": name, "params_M": n_par / 1e6,
                 "params_fused_M": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs": gflops})

raw = []
with torch.inference_mode():
    for r in range(ROUNDS):
        for name, m in models.items():
            for _ in range(PER_ROUND):
                t0 = time.perf_counter()
                m(x)
                raw.append({"round": r, "model": name, "ms": (time.perf_counter() - t0) * 1000})
raw = pd.DataFrame(raw)
raw.to_csv(os.path.join(OUT, "cpu_latency_raw.csv"), index=False)
summ = raw.groupby("model").ms.agg(median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75),
                                   mean="mean").reset_index()
round_med = raw.groupby(["round", "model"]).ms.median().unstack()
rank = round_med.rank(axis=1)
summ["fastest_rounds"] = summ.model.map((rank == 1).sum())
res = pd.DataFrame(info).merge(summ, on="model")
res["cpu"] = platform.processor() or platform.machine()
res["threads"] = THREADS
res["torch"] = torch.__version__
res.to_csv(os.path.join(OUT, "cpu_latency.csv"), index=False)
print(res.round(3).to_string())
print("round-median rank (1 = fastest):")
print(rank.astype(int).to_string())
