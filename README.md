# VinBigData Chest X-Ray Abnormalities Detection
## ML Research Project

### Project Structure

```
vinbigdata-chest-x-ray/
│
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── dataset-cited-works/               # Literature review of the VinDr-CXR dataset
│   └── VinBigData Dataset Cited Works.md
│
├── previous-implementations/          # Reference implementations for study
│   ├── awsaf49-vinbigdata-cxr-ad-yolov5-14-class-train-infer/
│   ├── nxhong93-yolov5-chest-512/
│   ├── sunghyunjun-kaggle-vinbigdata-chest-xray-abnormalities-detection-main/
│   └── tariqshaban-YOLOv7-VinBigData-Chest-X-Ray/
│
├── docs/                              # Documentation & analysis
│   ├── analysis/                      # Research notes, comparative studies
│   │   ├── dataset-research-notes.md
│   │   └── comparative-analysis.md
│   ├── literature-review/             # Paper reviews, bibliometric analyses
│   └── proposals/                     # Publication proposals, research plans
│
├── src/                               # Source code
│   ├── config.py                      # FROZEN experimental protocol — single source of truth
│   ├── data/
│   │   ├── fusion.py                  # Multi-rater bbox fusion (WBF/NMS) + visual check
│   │   └── prepare.py                 # YOLO format, stratified split, verification
│   ├── training/
│   │   └── train.py                   # 3 models, identical settings, resume-safe
│   ├── evaluation/
│   │   ├── metrics.py                 # mAP@0.4 — the competition metric
│   │   └── xai.py                     # CAMs + explanation faithfulness metrics
│   └── utils/
│       └── run_logger.py              # Hardware/param/result provenance
│
└── notebooks/                         # Thin drivers over src/
    ├── 01_data_prep.ipynb             # D1–D2
    ├── 02_train.ipynb                 # D3–D6
    └── 03_xai.ipynb                   # D7–D9
```

> Directories listed in earlier revisions (`experiments/`, `results/`, `docs/literature-review/`, `src/models/`) were aspirational and did not exist. Removed rather than left as fiction.

### Current Work — ICCIT 2026 Submission

**Paper**: YOLO26 on VinDr-CXR — modern YOLO comparison with *measured* explanation faithfulness.

The contribution is not the benchmark (≥5 YOLO-family works already exist on this dataset) but the explainability evaluation. VinDr ships ground-truth bounding boxes; most CXR-XAI work uses NIH ChestX-ray14, which has none and can therefore only show qualitative heatmaps. Boxes turn saliency into a metric — pointing game, energy-based pointing game, CAM–box IoU, deletion/insertion AUC.

- **[Execution plan](docs/proposals/iccit-2026-execution-plan.md)** — protocol, compute budget, day-by-day schedule, kill criteria
- **[Proposals](docs/proposals/publication-proposals.md)** — P0 (active) + P1–P7 portfolio
- **[Verification report](docs/analysis/verification-report.md)** — independent audit of prior-work claims

### Reproducibility

Protocol is frozen in [`src/config.py`](src/config.py) — 512px, `s`-scale, 40 epochs, batch 16, seed 0, positive-only subset, WBF@0.5 rater fusion, single GPU. Per-model tuning is deliberately not supported; the comparison's validity depends on identical conditions.

`src/utils/run_logger.py` records GPU model and compute capability, library versions, git commit, every hyperparameter and all metrics per run.

**Note on metrics**: the VinBigData competition used **mAP@0.4**. Ultralytics reports @0.5 and @0.5:0.95 only and cannot produce @0.4 — it is computed separately in `src/evaluation/metrics.py`. Both are reported.

### Dataset

The **VinDr-CXR** dataset contains 18,000 annotated PA chest radiographs from two Vietnamese hospitals, with 22 radiographic findings annotated as bounding boxes by 17 radiologists.

- **Training**: 15,000 images (3 raters/image)
- **Test**: 3,000 images (5-rater consensus)
- **Classes**: 14 abnormalities + No Finding
- **Original Kaggle competition**: [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

### Getting Started

*Instructions to be added as the project develops.*

### References

1. Nguyen, H.Q. et al. "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations." *Scientific Data*, 2022.
2. Competition page: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
