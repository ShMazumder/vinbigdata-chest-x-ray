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
├── src/                               # Source code for our implementation
│   ├── data/                          # Dataset loading, preprocessing, augmentation
│   ├── models/                        # Model architectures and configurations
│   ├── training/                      # Training loops, schedulers, loss functions
│   ├── evaluation/                    # Metrics, evaluation pipelines
│   └── utils/                         # Shared utilities (logging, visualization, etc.)
│
├── notebooks/                         # Jupyter notebooks for EDA & experiments
│
├── experiments/                       # Experiment management
│   ├── configs/                       # YAML/JSON experiment configurations
│   ├── logs/                          # Training logs, TensorBoard logs
│   └── checkpoints/                   # Saved model weights
│
└── results/                           # Output artifacts
    ├── figures/                       # Generated plots, visualizations
    └── tables/                        # Performance tables, comparison CSVs
```

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
