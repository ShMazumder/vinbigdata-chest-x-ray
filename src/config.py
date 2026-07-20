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
2026-07-20  Multi-seed protocol adopted: SEEDS = (0, 1, 2), report mean±std.
            Measured training cost on T4 is ~42 s/epoch true (39 s train + 3 s
            val) => ~28 min per model per seed, ~4.2 h for the full 3x3 matrix
            against a ~30 h weekly quota. The paper's central claim is a
            comparison between architectures; a single-seed gap of ~0.01 mAP
            is inside run-to-run noise and licenses no claim. seed_spread_
            warning() now refuses to endorse a ranking whose between-model gap
            is smaller than within-model std.
            (Earlier note that staging the dataset locally would cut epoch time
            was wrong -- I/O was already hidden behind GPU compute by the
            dataloader workers. Staging is retained only for the label cache.)
2026-07-20  data.yaml no longer writes an absolute `path:`. It baked in the
            producing session's location (/kaggle/working/vindr_yolo), which
            does not exist once the dataset is mounted read-only under
            /kaggle/input/notebooks/<user>/<slug>/. Ultralytics then failed
            with "images not found, missing path ...". Omitting `path:` makes
            it resolve the splits relative to the yaml itself, which is correct
            in both sessions. _validate_dataset() also repairs stale yamls in
            place (writing a corrected copy to /kaggle/working) so datasets
            committed before this change still work without re-running prep.
2026-07-20  GPU: use T4, NOT P100. P100 is sm_60; current PyTorch requires
            sm_70+. Confirmed on a live Kaggle session -- P100 raises
            "not compatible with the current PyTorch installation" and will
            fail or fall back to CPU. The earlier P100 recommendation (based on
            2.3x memory bandwidth and a 250W vs 70W power envelope) is void:
            the card simply does not run. T4 is sm_75 and has tensor cores, so
            AMP is genuinely faster there. Accelerator = "GPU T4 x2",
            device=0, no DDP.
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

SEED = 0                   # single-run default
SEEDS = (0, 1, 2)          # multi-seed protocol -- report mean±std over these.
# Why 3 seeds: the paper's claim is a COMPARISON between architectures. A
# single-seed gap of ~0.01 mAP is inside run-to-run noise for detection, so
# single-seed results license no claim at all. Measured cost is ~28 min per
# model per seed on T4 => ~4.2 h for 3 models x 3 seeds, against a ~30 h
# weekly Kaggle quota. Cheapest available upgrade to what the paper can assert.
IMGSZ = 512
EPOCHS = 40
BATCH = 16                 # <4 destroys mAP on this dataset (documented in ref)
DEVICE = 0                 # single GPU. Accelerator MUST be "GPU T4 x2" --
                           # P100 is sm_60 and unsupported by PyTorch (CHANGELOG).
                           # Select T4 x2, use one card; DDP in notebooks is a
                           # known time sink and saves ~2h at best.
AMP = True                 # T4 has tensor cores -> real FP16 speedup. No bf16.
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


def find_data_yaml(explicit: str | Path | None = None,
                   search_root: Path | None = None) -> Path:
    """Locate the prepared dataset's data.yaml. No hardcoded usernames or slugs.

    Kaggle mounts a notebook's output at
        /kaggle/input/notebooks/<username>/<notebook-slug>/<your-dirs>/
    so the real depth is 4+ levels and both the username and the slug vary.
    Searched depth-first from shallow to deep, with a recursive fallback.

    Pass `explicit` to skip the search entirely if you already know the path.

    Also validates that images are real files, not dangling symlinks -- a
    notebook-output mount can carry links that resolved in the producing
    session and break in the consuming one.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"explicit path does not exist: {path}")
        return _validate_dataset(path)

    roots = [search_root] if search_root else [KAGGLE_INPUT, WORK]
    # Shallow patterns first -- cheap. The recursive fallback walks every
    # attached dataset, which can mean tens of thousands of image files.
    patterns = ["data.yaml", "*/data.yaml", "*/*/data.yaml",
                "*/*/*/data.yaml", "*/*/*/*/data.yaml", "*/*/*/*/*/data.yaml"]

    found: list[Path] = []
    for root in roots:
        if not (root and root.exists()):
            continue
        for pat in patterns:
            found.extend(root.glob(pat))
            if found:
                break
        if found:
            break
    if not found:                                     # last resort
        for root in roots:
            if root and root.exists():
                found.extend(root.glob("**/data.yaml"))

    if not found:
        raise FileNotFoundError(
            f"no data.yaml under {roots}.\n"
            f"  1. Notebook 01 must be committed with 'Save & Run All' (not Quick Save)\n"
            f"  2. Here: Add Data -> 'Notebook Output Files' -> 'Your Work' -> notebook 01\n"
            f"  3. Or pass the path directly: C.find_data_yaml('/kaggle/input/.../data.yaml')"
        )

    # Prefer a candidate that actually has training images.
    usable = [p for p in found if (p.parent / "images" / "train").is_dir()]
    if len(found) > 1:
        print(f"[config] {len(found)} data.yaml candidates; "
              f"{len(usable)} with images/train")
    return _validate_dataset((usable or found)[0])


def _validate_dataset(path: Path) -> Path:
    """Confirm the dataset behind a data.yaml is real, and make it usable here.

    Older data.yaml files carry an absolute `path:` from the session that
    produced them (e.g. /kaggle/working/vindr_yolo). Once mounted read-only at
    /kaggle/input/..., that path is gone and Ultralytics fails with
    "images not found, missing path ...". Since /kaggle/input cannot be edited,
    a corrected copy is written to /kaggle/working and returned instead.
    """
    root = path.parent
    imgs = list((root / "images" / "train").glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{root}/images/train is empty")

    sample = imgs[:200]
    n_broken = sum(1 for p in sample if not p.exists())
    if n_broken:
        raise RuntimeError(
            f"{n_broken}/{len(sample)} sampled images are broken links. "
            f"Notebook 01 must run to_yolo_labels with copy_images=True "
            f"(now the default) and be re-committed."
        )

    size_gb = sum(p.stat().st_size for p in sample) / len(sample) * len(imgs) / 1e9
    print(f"[config] data.yaml -> {path}")
    print(f"[config] {len(imgs)} train images, ~{size_gb:.2f} GB")

    # Detect a stale absolute `path:` and repair it transparently.
    text = path.read_text()
    stale = None
    for line in text.splitlines():
        if line.strip().startswith("path:"):
            declared = Path(line.split("path:", 1)[1].strip())
            if declared.resolve() != root.resolve() and not declared.exists():
                stale = declared
            break

    if stale is None:
        return path

    fixed_lines = [ln for ln in text.splitlines()
                   if not ln.strip().startswith("path:")]
    fixed = "\n".join([f"path: {root}"] + fixed_lines) + "\n"

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "data.yaml"
    out.write_text(fixed)
    print(f"[config] stale 'path: {stale}' in the mounted yaml (does not exist "
          f"in this session)")
    print(f"[config] wrote corrected copy -> {out}")
    return out


def dataset_root(data_yaml: str | Path) -> Path:
    """Where the images/ and labels/ trees actually live for a given data.yaml.

    Not always the yaml's own parent. A yaml repaired by _validate_dataset()
    sits in /kaggle/working while the data stays on the read-only mount, and
    carries an explicit `path:` pointing back at it. Ultralytics uses the same
    rule -- `path:` if present, else the yaml's directory.
    """
    p = Path(data_yaml)
    for line in p.read_text().splitlines():
        if line.strip().startswith("path:"):
            declared = Path(line.split("path:", 1)[1].strip())
            if declared.is_dir():
                return declared
            break
    return p.parent


def stage_dataset_local(data_yaml: Path, dest: Path | None = None) -> Path:
    """Copy the mounted dataset to local disk and return the new data.yaml.

    Measured on Kaggle 2026-07-20: reading from /kaggle/input runs at ~35 MB/s.
    At 3,515 images x 216 KB = ~760 MB per epoch, that is ~22 s of a 54 s epoch
    spent on I/O. Ultralytics also cannot write its label cache to the
    read-only mount, so every run repeats a ~15 s label scan.

    Copy costs ~1 min once. Over 3 models x 40 epochs it pays for itself many
    times over.

    Resolves the source via dataset_root(), NOT the yaml's parent -- a repaired
    yaml lives in /kaggle/working while the data stays on the mount. Raises if
    nothing was copied rather than leaving an empty staging directory behind.
    """
    import shutil
    import time

    src_root = dataset_root(data_yaml)
    dest = Path(dest or WORK / "vindr_yolo_local")

    if src_root.resolve() == dest.resolve():
        print(f"[config] already local at {dest}, nothing to stage")
        return Path(data_yaml)

    n_existing = len(list((dest / "images" / "train").glob("*.jpg")))
    if n_existing:
        print(f"[config] already staged at {dest} ({n_existing} train images)")
    else:
        t0 = time.time()
        print(f"[config] staging {src_root} -> {dest} ...")
        copied = 0
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                s, d = src_root / kind / split, dest / kind / split
                if not s.is_dir():
                    continue
                shutil.copytree(s, d, dirs_exist_ok=True)
                copied += len(list(d.iterdir()))
        if copied == 0:
            raise FileNotFoundError(
                f"staged nothing from {src_root} -- expected images/<split> and "
                f"labels/<split> subdirectories there. Check what "
                f"dataset_root() resolved to."
            )
        print(f"[config] staged {copied} files in {time.time() - t0:.0f}s")

    dest.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    out = dest / "data.yaml"
    out.write_text(
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n"
    )
    print(f"[config] local data.yaml -> {out}")
    return out


def find_weights(search_root: Path | None = None) -> dict[str, str]:
    """Locate trained best.pt files, wherever notebook 02's output mounted.

    Returns {model_key: path}. Same rationale as find_data_yaml -- Kaggle
    mounts notebook output under /kaggle/input/notebooks/<username>/<slug>/,
    both of which vary. Recursive by design; no username or slug is assumed.
    """
    roots = [search_root] if search_root else [KAGGLE_INPUT, WORK]
    out: dict[str, str] = {}
    for root in roots:
        if not (root and root.exists()):
            continue
        for p in root.glob("**/weights/best.pt"):
            run = p.parent.parent.name          # e.g. yolo26s_512_e40_seed0
            key = run.split("_")[0]
            if key in MODELS and key not in out:
                out[key] = str(p)

    if not out:
        raise FileNotFoundError(
            f"no best.pt under {roots}.\n"
            f"  Add Data -> 'Notebook Output Files' -> 'Your Work' -> notebook 02.\n"
            f"  Or pass paths directly as a {{model_key: path}} dict."
        )
    for k in MODELS:
        print(f"[config] {k}: {out.get(k, 'MISSING')}")
    missing = [k for k in MODELS if k not in out]
    if missing:
        print(f"[config] WARNING: {missing} not found -- notebook 03 will run "
              f"on {len(out)} model(s). Two models still gives Axis B "
              f"(NMS vs NMS-free) if one of them is yolo26s.")
    return out


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
