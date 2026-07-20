"""
mAP@0.4 -- the VinBigData competition metric.

WHY THIS FILE EXISTS: Ultralytics reports mAP@0.5 and mAP@0.5:0.95 natively and
CANNOT give you mAP@0.4. The VinBigData competition used IoU > 0.4. If @0.5
silently lands in a column labelled @0.4, the entire related-work comparison
misaligns -- and that error survives to camera-ready.

Report BOTH:
  @0.4  -> comparable to the competition lineage (0.253 sunghyunjun, 0.314 1st)
  @0.5  -> comparable to modern YOLO literature (0.338, 0.415, 0.453)

Implementation is standard VOC-style AP: greedy IoU matching per class,
all-point interpolation of the precision-recall curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. Boxes are [x1,y1,x2,y2]. Returns (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """All-point interpolated AP (VOC 2010+ / COCO style)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):          # monotonic envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    preds: pd.DataFrame, gts: pd.DataFrame, class_id: int, iou_thr: float
) -> tuple[float, int]:
    """AP for one class. Returns (ap, n_ground_truth).

    preds: [image_id, class_id, score, x1, y1, x2, y2]
    gts:   [image_id, class_id, x1, y1, x2, y2]
    """
    p = preds[preds.class_id == class_id].sort_values("score", ascending=False)
    g = gts[gts.class_id == class_id]
    n_gt = len(g)
    if n_gt == 0:
        return float("nan"), 0          # undefined, not zero -- do not average this in
    if len(p) == 0:
        return 0.0, n_gt

    gt_by_img = {k: v[["x1", "y1", "x2", "y2"]].to_numpy()
                 for k, v in g.groupby("image_id")}
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}

    tp = np.zeros(len(p))
    fp = np.zeros(len(p))

    for i, (_, row) in enumerate(p.iterrows()):
        gt_boxes = gt_by_img.get(row.image_id)
        if gt_boxes is None:
            fp[i] = 1
            continue
        box = np.array([[row.x1, row.y1, row.x2, row.y2]])
        ious = iou_matrix(box, gt_boxes)[0]
        best = int(np.argmax(ious))
        # Greedy: each GT box can be matched once. Duplicate detections on an
        # already-matched GT count as false positives.
        if ious[best] >= iou_thr and not matched[row.image_id][best]:
            tp[i] = 1
            matched[row.image_id][best] = True
        else:
            fp[i] = 1

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.clip(ctp + cfp, 1e-9, None)
    return _ap_from_pr(rec, prec), n_gt


def mean_average_precision(
    preds: pd.DataFrame, gts: pd.DataFrame, iou_thr: float = 0.4,
    n_classes: int | None = None,
) -> dict:
    """mAP at a single IoU threshold + per-class breakdown.

    Classes with zero ground truth are EXCLUDED from the mean rather than
    counted as 0.0 -- counting them would silently deflate mAP and make your
    numbers non-comparable to published results.
    """
    from ..config import CLASSES

    n_classes = n_classes or len(CLASSES)
    per_class, n_gts = {}, {}
    for c in range(n_classes):
        ap, n = average_precision(preds, gts, c, iou_thr)
        per_class[c] = ap
        n_gts[c] = n

    valid = [v for v in per_class.values() if not np.isnan(v)]
    return {
        "map": float(np.mean(valid)) if valid else float("nan"),
        "iou_thr": iou_thr,
        "n_classes_scored": len(valid),
        "per_class": {CLASSES[c]: per_class[c] for c in per_class},
        "n_gt": {CLASSES[c]: n_gts[c] for c in n_gts},
    }


def predictions_from_model(model, image_paths, imgsz=512, conf=0.001, max_det=100):
    """Run inference -> tidy prediction DataFrame.

    conf=0.001 (not 0.25): AP integrates the full precision-recall curve, so
    low-confidence detections matter. A high conf threshold truncates the curve
    and understates AP. This is a common and silent error.
    """
    from pathlib import Path

    rows = []
    for i, path in enumerate(image_paths):
        if i % 200 == 0:
            print(f"  predict {i}/{len(image_paths)}")
        res = model.predict(str(path), imgsz=imgsz, conf=conf,
                            max_det=max_det, verbose=False)[0]
        img_id = Path(path).stem
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scr = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scr):
            rows.append({"image_id": img_id, "class_id": int(c), "score": float(s),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows, columns=["image_id", "class_id", "score",
                                       "x1", "y1", "x2", "y2"])


def ground_truth_from_yolo(labels_dir, images_dir) -> pd.DataFrame:
    """Read YOLO txt labels back to absolute-pixel GT boxes."""
    from pathlib import Path

    from PIL import Image

    rows = []
    for lbl in Path(labels_dir).glob("*.txt"):
        img_id = lbl.stem
        img_path = Path(images_dir) / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            rows.append({"image_id": img_id, "class_id": int(c),
                         "x1": cx - bw / 2, "y1": cy - bh / 2,
                         "x2": cx + bw / 2, "y2": cy + bh / 2})
    return pd.DataFrame(rows, columns=["image_id", "class_id", "x1", "y1", "x2", "y2"])


def aggregate_seeds(results_by_run: dict) -> "pd.DataFrame":
    """Collapse {(model, seed): eval_dict} into mean±std per model.

    Input values must each carry 'map40' and 'map50'.

    Returns one row per model with mean, std and n for both metrics, plus a
    preformatted 'mAP@0.4' string for the paper table.
    """
    rows = []
    for (model_key, seed), ev in results_by_run.items():
        rows.append({"model": model_key, "seed": seed,
                     "map40": ev["map40"], "map50": ev["map50"]})
    df = pd.DataFrame(rows)

    agg = df.groupby("model").agg(
        map40_mean=("map40", "mean"), map40_std=("map40", "std"),
        map50_mean=("map50", "mean"), map50_std=("map50", "std"),
        n_seeds=("seed", "count"),
    ).round(4)

    for m in ("map40", "map50"):
        label = f"mAP@{m[3]}.{m[4]}"
        agg[label] = (agg[f"{m}_mean"].map("{:.3f}".format) + " ± "
                      + agg[f"{m}_std"].fillna(0).map("{:.3f}".format))
    return agg


def seed_spread_warning(agg: "pd.DataFrame", metric: str = "map40_") -> None:
    """Flag when between-model gaps are smaller than within-model noise.

    This is the whole reason for running multiple seeds. If the best-to-worst
    model gap does not exceed the typical seed-to-seed std, the ranking is not
    supported by the data and must not be presented as a finding -- report the
    models as indistinguishable on this metric instead.
    """
    means, stds = agg[f"{metric}mean"], agg[f"{metric}std"].fillna(0)
    gap = means.max() - means.min()
    noise = stds.mean()
    print(f"\nbetween-model gap: {gap:.4f}   mean within-model std: {noise:.4f}")
    if gap < noise:
        print("⚠ Gap is INSIDE seed noise. The model ranking is not supported.")
        print("  Report as indistinguishable; do not claim an ordering.")
    elif gap < 2 * noise:
        print("⚠ Gap is under 2x seed noise. Weak evidence -- hedge the claim.")
    else:
        print("✓ Gap exceeds 2x seed noise. Ranking is defensible.")


def evaluate_full(model, images_dir, labels_dir, imgsz=512) -> dict:
    """Both thresholds in one pass. This is what feeds the results table."""
    from pathlib import Path

    paths = sorted(Path(images_dir).glob("*.jpg"))
    preds = predictions_from_model(model, paths, imgsz=imgsz)
    gts = ground_truth_from_yolo(labels_dir, images_dir)
    print(f"  {len(preds)} predictions vs {len(gts)} ground-truth boxes")

    m40 = mean_average_precision(preds, gts, iou_thr=0.4)
    m50 = mean_average_precision(preds, gts, iou_thr=0.5)
    print(f"  mAP@0.4 = {m40['map']:.4f}   mAP@0.5 = {m50['map']:.4f}")
    return {"map40": m40["map"], "map50": m50["map"],
            "detail_40": m40, "detail_50": m50,
            "preds": preds, "gts": gts}
