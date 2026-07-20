"""
Multi-rater bounding box fusion for VinDr-CXR.

VinDr's 15,000 training images were each annotated independently by 3 of 17
radiologists. Overlapping boxes for the same finding must be fused into single
training labels.

THIS MODULE IS WHERE THE PAPER QUIETLY DIES.

If fusion is wrong, every downstream number is wrong, and the D4 sanity gate
will NOT catch it -- broken-but-plausible labels produce a broken-but-plausible
0.25 mAP that looks perfectly reasonable. The visual check in
`plot_fusion_examples()` is not optional. Look at 20 images before training.

Known rater pathology in this dataset (documented in dataset-research-notes.md):
  - R8, R9, R10 annotated the vast majority of positive findings, with smaller boxes
  - R1-R7 marked almost zero abnormalities, defaulting to "No finding"
So fused boxes are dominated by a handful of raters. Worth one sentence in the
paper's limitations.

Dependency: `pip install ensemble-boxes` (ZFTurbo). Same lineage as the
vendored sunghyunjun reference, which used ZFTurbo's NMS/WBF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_normalized(df: pd.DataFrame, w: float, h: float) -> np.ndarray:
    """[x_min,y_min,x_max,y_max] px -> normalized [0,1]. ensemble_boxes requires this."""
    boxes = df[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    return np.clip(boxes, 0.0, 1.0)


def fuse_image_boxes(
    df_img: pd.DataFrame,
    width: float,
    height: float,
    method: str = "wbf",
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0,
) -> pd.DataFrame:
    """Fuse one image's multi-rater boxes.

    Parameters
    ----------
    df_img : rows for a single image_id, columns
             [class_id, x_min, y_min, x_max, y_max, rad_id]
    method : "wbf" (weighted box fusion) or "nms"

    Returns
    -------
    DataFrame [class_id, x_min, y_min, x_max, y_max] in ORIGINAL PIXEL coords.

    Note
    ----
    Raters provide no confidence scores, so all boxes enter with score=1.0 and
    equal weight. WBF therefore reduces to an unweighted average of clustered
    boxes -- which is the intended behaviour for consensus annotation, but is
    NOT what WBF does in its usual ensemble-of-models setting. Worth knowing if
    a reviewer asks.
    """
    from ensemble_boxes import nms, weighted_boxes_fusion

    df_img = df_img[df_img["class_id"] != 14]     # drop "No finding" rows
    if len(df_img) == 0:
        return pd.DataFrame(columns=["class_id", "x_min", "y_min", "x_max", "y_max"])

    boxes = _to_normalized(df_img, width, height)
    labels = df_img["class_id"].to_numpy(dtype=int)
    scores = np.ones(len(boxes), dtype=float)

    # ensemble_boxes expects list-of-lists (one inner list per "model"/rater set).
    # We pass a single combined list: clustering is by IoU + label, which is
    # exactly the multi-rater consensus we want.
    if method == "wbf":
        f_boxes, _, f_labels = weighted_boxes_fusion(
            [boxes.tolist()], [scores.tolist()], [labels.tolist()],
            iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )
    elif method == "nms":
        f_boxes, _, f_labels = nms(
            [boxes.tolist()], [scores.tolist()], [labels.tolist()], iou_thr=iou_thr
        )
    else:
        raise ValueError(f"unknown fusion method: {method!r}")

    f_boxes = np.asarray(f_boxes, dtype=float).reshape(-1, 4)
    f_boxes[:, [0, 2]] *= width
    f_boxes[:, [1, 3]] *= height

    return pd.DataFrame({
        "class_id": np.asarray(f_labels, dtype=int),
        "x_min": f_boxes[:, 0], "y_min": f_boxes[:, 1],
        "x_max": f_boxes[:, 2], "y_max": f_boxes[:, 3],
    })


def fuse_dataset(
    df: pd.DataFrame,
    dims: dict[str, tuple[int, int]],
    method: str = "wbf",
    iou_thr: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fuse every image. `dims` maps image_id -> (width, height)."""
    # Pre-group once. Filtering with df[df.image_id == id] inside the loop is a
    # full scan per image -- O(n_rows * n_images), ~160M comparisons here.
    grouped = dict(tuple(df.groupby("image_id")))

    out = []
    ids = df["image_id"].unique()
    for i, img_id in enumerate(ids):
        if verbose and i % 1000 == 0:
            print(f"  fusing {i}/{len(ids)}")
        w, h = dims[img_id]
        fused = fuse_image_boxes(
            grouped[img_id], w, h, method=method, iou_thr=iou_thr
        )
        fused["image_id"] = img_id
        out.append(fused)

    result = pd.concat(out, ignore_index=True)
    if verbose:
        raw = len(df[df["class_id"] != 14])
        print(f"\nfusion: {raw} raw boxes -> {len(result)} fused "
              f"({len(result)/max(raw,1):.1%} retained, method={method}@{iou_thr})")
    return result


def fusion_report(df_raw: pd.DataFrame, df_fused: pd.DataFrame) -> pd.DataFrame:
    """Per-class before/after. Sanity check #1 -- read this before training.

    Red flags:
      - retention near 100% -> fusion did nothing, iou_thr likely too high
      - retention near 33%  -> collapsing all 3 raters always, thr too low
      - any class dropping to zero -> bug
    Expect roughly 40-70% retention overall.
    """
    from ..config import CLASSES

    raw = df_raw[df_raw["class_id"] != 14].groupby("class_id").size()
    fused = df_fused.groupby("class_id").size()
    rep = pd.DataFrame({"raw": raw, "fused": fused}).fillna(0).astype(int)
    rep["retained_%"] = (rep["fused"] / rep["raw"].replace(0, np.nan) * 100).round(1)
    rep["class"] = [CLASSES[i] if i < len(CLASSES) else "?" for i in rep.index]
    return rep[["class", "raw", "fused", "retained_%"]]


def count_near_duplicates(df_fused: pd.DataFrame, iou_thr: float = 0.25) -> dict:
    """Count same-class boxes that still overlap after fusion.

    A same-class pair above ~0.25 IoU post-fusion almost certainly describes
    ONE finding that two raters bounded differently. Training on it teaches the
    model to emit duplicate detections, which costs precision at every IoU.

    Spatially distinct same-class lesions -- ILD in both lungs, nodules in
    different zones -- have IoU ~0 and are correctly NOT counted here. That is
    what makes this a usable signal rather than a proxy for "boxes per image".
    """
    from ..evaluation.metrics import iou_matrix

    dup_pairs, dup_images, per_class = 0, set(), {}
    for (img_id, cls), g in df_fused.groupby(["image_id", "class_id"]):
        if len(g) < 2:
            continue
        b = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
        m = iou_matrix(b, b)
        np.fill_diagonal(m, 0.0)
        n = int((m > iou_thr).sum() // 2)
        if n:
            dup_pairs += n
            dup_images.add(img_id)
            per_class[int(cls)] = per_class.get(int(cls), 0) + n

    return {
        "n_boxes": len(df_fused),
        "dup_pairs": dup_pairs,
        "dup_images": len(dup_images),
        "dup_per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
    }


def sweep_fusion_iou(
    df_pos: pd.DataFrame, dims: dict, thresholds=(0.3, 0.4, 0.5),
    method: str = "wbf", audit_iou: float = 0.25,
) -> pd.DataFrame:
    """Fuse at several thresholds and count surviving duplicates at each.

    How to read the output -- you are looking for the knee:
      - `dup_pairs` should fall sharply as the threshold drops
      - `n_boxes` must NOT collapse toward n_raw/3 (that means every trio is
        merging unconditionally, including genuinely distinct lesions)
    Pick the lowest threshold that kills duplicates while retention stays
    comfortably above ~40%.
    """
    rows = []
    n_raw = len(df_pos[df_pos.class_id != 14])
    for thr in thresholds:
        fused = fuse_dataset(df_pos, dims, method=method, iou_thr=thr, verbose=False)
        audit = count_near_duplicates(fused, iou_thr=audit_iou)
        rows.append({
            "iou_thr": thr,
            "n_boxes": audit["n_boxes"],
            "retained_%": round(audit["n_boxes"] / n_raw * 100, 1),
            "dup_pairs": audit["dup_pairs"],
            "dup_images": audit["dup_images"],
        })
        print(f"  thr={thr}: {audit['n_boxes']} boxes "
              f"({rows[-1]['retained_%']}% retained), "
              f"{audit['dup_pairs']} dup pairs across {audit['dup_images']} images")
    print(f"\n  floor if every 3-rater group merged: ~{n_raw // 3} boxes")
    return pd.DataFrame(rows)


def find_duplicate_images(df_fused: pd.DataFrame, iou_thr: float = 0.25,
                          limit: int = 6) -> list[str]:
    """Worst-offending image ids, for targeted re-plotting after a threshold change."""
    from ..evaluation.metrics import iou_matrix

    scored = []
    for img_id, g_img in df_fused.groupby("image_id"):
        n = 0
        for _, g in g_img.groupby("class_id"):
            if len(g) < 2:
                continue
            b = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
            m = iou_matrix(b, b)
            np.fill_diagonal(m, 0.0)
            n += int((m > iou_thr).sum() // 2)
        if n:
            scored.append((n, img_id))
    scored.sort(reverse=True)
    return [img_id for _, img_id in scored[:limit]]


def plot_fusion_examples(
    df_raw: pd.DataFrame,
    df_fused: pd.DataFrame,
    image_dir,
    image_ids: list[str],
    n_cols: int = 2,
):
    """Draw raw rater boxes (thin, red) vs fused (thick, green). MANDATORY CHECK.

    What you are looking for:
      - fused boxes sit sensibly inside the cluster of rater boxes
      - no fused box spanning the whole image (sign of over-merging)
      - no duplicate near-identical fused boxes (sign of under-merging)
      - anatomically plausible placement (cardiomegaly over the heart, etc.)
    """
    from pathlib import Path

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    from ..config import CLASSES

    n_rows = int(np.ceil(len(image_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, img_id in zip(axes, image_ids):
        img = Image.open(Path(image_dir) / f"{img_id}.jpg")
        ax.imshow(img, cmap="gray")

        for _, r in df_raw[(df_raw.image_id == img_id) & (df_raw.class_id != 14)].iterrows():
            ax.add_patch(patches.Rectangle(
                (r.x_min, r.y_min), r.x_max - r.x_min, r.y_max - r.y_min,
                fill=False, edgecolor="red", linewidth=1, linestyle=":"))

        for _, r in df_fused[df_fused.image_id == img_id].iterrows():
            ax.add_patch(patches.Rectangle(
                (r.x_min, r.y_min), r.x_max - r.x_min, r.y_max - r.y_min,
                fill=False, edgecolor="lime", linewidth=2.5))
            ax.text(r.x_min, max(r.y_min - 8, 0), CLASSES[int(r.class_id)],
                    color="lime", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.6, pad=1))

        ax.set_title(f"{img_id}  (red dotted = raters, green = fused)", fontsize=9)
        ax.axis("off")

    for ax in axes[len(image_ids):]:
        ax.axis("off")
    plt.tight_layout()
    return fig
