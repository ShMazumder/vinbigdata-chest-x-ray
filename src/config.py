"""
Frozen experimental protocol for the ICCIT 2026 submission.

Single source of truth. Every notebook imports from here. Do NOT edit values
mid-project -- protocol drift is the most common way a 12-day paper becomes
unpublishable, and the paper's entire claim is "identical conditions".

If a value must change, change it here, note it in CHANGELOG below, and re-run
every affected model. Never patch a value inside a notebook.

CHANGELOG
---------
2026-07-19  Initial freeze.
2026-07-20  FUSION_IOU 0.5 -> 0.4. Visual inspection of 12 fused images showed
            systematic under-merging: same-class boxes from different raters
            covering one finding survived as 2-4 separate ground-truth objects
            (Pulmonary fibrosis, Pleural thickening, Pleural effusion,
            Nodule/Mass). Measured example: two rater boxes on one pleural
            thickening had IoU 0.38, below the 0.5 cut. Per-class retention
            corroborated -- diffuse classes retained 72-84% vs 44-47% for the
            anatomically-anchored ones. Spatially distinct same-class lesions
            (e.g. ILD in both lungs, IoU ~0) are unaffected by the change.
            No retraining implied; nothing had been trained yet.
2026-07-20  Kaggle dataset paths confirmed against a live run.
2026-07-20  FUSION_IOU 0.4 -> 0.25, decided by measured sweep (not judgement):

                thr    boxes  retained  dup_pairs  dup_images
                0.25   21473     59.5%         58          56
                0.30   21840     60.5%        413         346
                0.40   22724     63.0%       1331         942
                0.50   23948     66.3%       2655        1576
                floor (all 3-rater groups merged): ~12032

            Duplicates roughly halve per step while retention moves only 6.8
            points across the entire range -- the threshold is almost purely
            trading duplicate labels for nothing. 0.4 still left 1331 duplicate
            pairs across 942 images (21% of the dataset), visually confirmed as
            8x-nested Pulmonary fibrosis boxes on one image and 3x-stacked
            Pleural effusion on another. 0.25 removes 96% of those for 3.5
            points of retention and stays 78% above the collapse floor.

            Residual risk: at 0.25, two genuinely distinct adjacent lesions
            (Nodule/Mass, Calcification) could merge. Watch those two classes
            in fusion_report on the next run -- if they crater relative to the
            0.4 baseline (70.9% and 75.5%), reconsider a per-class threshold.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (Kaggle defaults; override in the notebook if running locally)
# --------------------------------------------------------------------------

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# Pre-resized 1024px JPG dataset -- do NOT re-derive from the 192 GB DICOM set.
# Path confirmed against a live Kaggle run 2026-07-20.
RAW_IMAGES = KAGGLE_INPUT / "datasets" / "sunghyunjun" / "vinbigdata-1024-jpg-dataset"
TRAIN_CSV = RAW_IMAGES / "train.csv"
IMAGE_DIR = RAW_IMAGES / "train"

# train.csv carries BOTH coordinate systems:
#   x_min/y_min/x_max/y_max  -> already scaled to the 1024px JPGs  [USE THESE]
#   raw_x_min/... + raw_width/raw_height/scale_x/scale_y -> original DICOM space
# Using the raw_* columns against the resized JPGs would silently produce boxes
# 2-3x too large. Everything downstream assumes the unprefixed columns.

DATA_ROOT = WORK / "vindr_yolo"      # YOLO-format dataset gets built here
RUNS_DIR = WORK / "runs"             # RunLogger output
WEIGHTS_DIR = WORK / "weights"

# --------------------------------------------------------------------------
# Classes -- 14 abnormalities. Class 14 ("No finding") is NOT a detection
# class; images containing only class 14 are excluded by the positive-only
# subset. Order is the VinBigData competition order -- do not resort.
# --------------------------------------------------------------------------

CLASSES = [
    "Aortic enlargement",      # 0   large, high prevalence
    "Atelectasis",             # 1
    "Calcification",           # 2   17.67% small targets
    "Cardiomegaly",            # 3   large, high prevalence
    "Consolidation",           # 4   overlaps w/ Infiltration
    "ILD",                     # 5
    "Infiltration",            # 6   overlaps w/ Consolidation
    "Lung Opacity",            # 7
    "Nodule/Mass",             # 8   39.69% small targets
    "Other lesion",            # 9
    "Pleural effusion",        # 10
    "Pleural thickening",      # 11  high prevalence
    "Pneumothorax",            # 12
    "Pulmonary fibrosis",      # 13  high prevalence
]
NC = len(CLASSES)
NO_FINDING_ID = 14

# Used for the XAI Axis-A analysis (small-target vs large-target faithfulness).
SMALL_TARGET_CLASSES = [2, 8]        # Calcification, Nodule/Mass
LARGE_TARGET_CLASSES = [0, 3]        # Aortic enlargement, Cardiomegaly

# --------------------------------------------------------------------------
# Frozen protocol -- see docs/proposals/iccit-2026-execution-plan.md section 2
# --------------------------------------------------------------------------

SEED = 0
IMGSZ = 512
EPOCHS = 40
BATCH = 16                 # <4 destroys mAP on this dataset (documented in ref)
DEVICE = 0                 # single GPU. P100 preferred over single T4.
AMP = True                 # P100 has 2x FP16 but no tensor cores -> smaller gain
WORKERS = 2                # Kaggle CPU is limited; >2 can starve

MODELS = {
    "yolov8s": "yolov8s.pt",
    "yolo11s": "yolo11s.pt",
    "yolo26s": "yolo26s.pt",
}

# Rater bbox fusion. VinDr train has 3 independent radiologists per image;
# their boxes must be fused into single training labels.
#
# NOTE (state this in the paper): the vendored sunghyunjun reference's own
# ablation has WBF *winning* on private LB (0.185 vs 0.181) while *losing* on
# CV (0.4158 vs 0.4419). The author chose NMS by reasoning about test-set
# labeling, not measurement. This remains unsettled -- do not present either
# choice as established.
FUSION_METHOD = "wbf"      # "wbf" | "nms"
FUSION_IOU = 0.25          # measured sweep, not judgement -- see CHANGELOG
FUSION_SKIP_BOX_THR = 0.0  # keep all boxes; raters have no confidence scores

# Threshold for flagging surviving near-duplicates in the post-fusion audit.
# Two same-class boxes above this IoU after fusion almost certainly describe
# one finding and should have merged.
DUPLICATE_AUDIT_IOU = 0.25

# Split. Kaggle test labels are not public, so CV on the 15K train set is the
# only option. Stratified by the image's rarest present class.
SPLIT = (0.8, 0.1, 0.1)
POSITIVE_ONLY = True       # ~4.4K of 15K images. ~3.4x epoch-time reduction.

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

# The competition metric is mAP@0.4. Ultralytics reports @0.5 and @0.5:0.95
# natively and CANNOT give you @0.4 -- it is computed separately in
# src/evaluation/metrics.py. Report BOTH: @0.4 for the competition lineage
# (0.253 / 0.314), @0.5 for the modern YOLO literature (0.338 / 0.415 / 0.453).
IOU_COMPETITION = 0.4
IOU_LITERATURE = 0.5

# Published bar -- for the related-work table. DO NOT frame the paper as
# beating these; a 512px s-scale model at 40 epochs will land below all of them.
PUBLISHED_BASELINES = {
    "RT-DETR (IRBM 2025)":        {"map50": 0.453, "precision": 0.557, "recall": 0.430},
    "YOLOv11-MFF (PLOS ONE 2025)": {"map50": 0.415, "precision": 0.482, "recall": 0.425},
    "YOLO-CXR (IEEE Access)":     {"map50": 0.338, "map5095": 0.167, "recall": 0.365},
    "sunghyunjun (Kaggle 76th)":  {"map40_privateLB": 0.253},
    "Competition 1st place":      {"map40_privateLB": 0.314},
}

# --------------------------------------------------------------------------
# XAI
# --------------------------------------------------------------------------

XAI_METHODS = ["eigencam", "gradcam++"]   # D-RISE optional, only if ahead on D9
XAI_N_IMAGES = None          # None = whole test split; set an int to subsample
CAM_PERCENTILE = 90          # threshold for CAM-vs-box IoU
DELETION_STEPS = 20          # deletion/insertion AUC granularity


def find_data_yaml(search_root: Path | None = None) -> Path:
    """Locate the prepared dataset's data.yaml under /kaggle/input.

    Notebook outputs mount at a path derived from the notebook slug, which is
    not knowable in advance. Rather than hardcode a guess, find it.

    Also validates that the images are real files, not dangling symlinks -- a
    notebook-output mount can carry links that resolve in the producing session
    and break in the consuming one.
    """
    roots = [search_root] if search_root else [KAGGLE_INPUT, WORK]
    found = []
    for root in roots:
        if root and root.exists():
            found.extend(root.glob("*/data.yaml"))
            found.extend(root.glob("*/*/data.yaml"))

    if not found:
        raise FileNotFoundError(
            f"no data.yaml under {roots}. In notebook 02: Add Data -> "
            f"'Notebook Output Files' tab -> filter 'Your Work' -> notebook 01. "
            f"Note this requires notebook 01 to have been committed with "
            f"'Save & Run All', not Quick Save."
        )
    if len(found) > 1:
        print(f"[config] multiple data.yaml found, using first: {found}")

    path = found[0]
    imgs = list((path.parent / "images" / "train").glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{path.parent}/images/train is empty")
    n_broken = sum(1 for p in imgs[:200] if not p.exists())
    if n_broken:
        raise RuntimeError(
            f"{n_broken}/200 sampled images are broken links. Notebook 01 must "
            f"run to_yolo_labels with copy_images=True (now the default) and be "
            f"re-committed."
        )
    size_gb = sum(p.stat().st_size for p in imgs[:200]) / 200 * len(imgs) / 1e9
    print(f"[config] data.yaml -> {path}")
    print(f"[config] {len(imgs)} train images, ~{size_gb:.2f} GB")
    return path


def as_dict() -> dict:
    """Flat snapshot for RunLogger / the paper's reproducibility statement."""
    return {
        "seed": SEED, "imgsz": IMGSZ, "epochs": EPOCHS, "batch": BATCH,
        "device": DEVICE, "amp": AMP, "workers": WORKERS,
        "fusion": f"{FUSION_METHOD}@{FUSION_IOU}",
        "split": "/".join(str(int(s * 100)) for s in SPLIT),
        "subset": "positive_only" if POSITIVE_ONLY else "all",
        "n_classes": NC,
    }
