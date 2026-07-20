"""
Explanation faithfulness, measured against radiologist bounding boxes.

THIS IS THE PAPER'S CONTRIBUTION. Detection accuracy is table stakes -- ≥5
YOLO-family works already exist on VinDr-CXR. What no verified prior work does
is *quantify* whether the model looks where the radiologist looked.

The asset that makes it possible: VinDr ships ground-truth boxes. Most CXR-XAI
work uses NIH ChestX-ray14, which has none, so it can only show qualitative
heatmaps. Boxes turn saliency into a metric.

Metrics
-------
  pointing_game      : does CAM argmax fall inside the GT box?          [0,1]
  ebpg               : fraction of CAM energy inside the GT box         [0,1]
  cam_box_iou        : IoU of thresholded CAM vs GT box                 [0,1]
  deletion/insertion : does removing salient pixels degrade detection?   AUC

Three analysis axes (these are the findings, not the numbers):
  A  per-class x object size -- do explanations degrade faster than mAP on
     small targets? A "yes" means the model looks right while explaining wrong.
  B  NMS-free (YOLO26) vs NMS (v8/v11) -- ties the two halves of the paper
     together. Without this you have two unrelated experiments stapled.
  C  accuracy/faithfulness decoupling -- does the best-mAP model give the best
     explanations? A "no" is the most publishable outcome available.

Dependency: `pip install grad-cam` (pytorch-grad-cam).

⚠ UNTESTED. Target-layer selection for Ultralytics models is the fragile part
and differs across v8/v11/YOLO26 -- `pick_target_layer()` resolves it
dynamically and prints what it chose. VERIFY that print on D7 before trusting
any number downstream.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# CAM generation
# --------------------------------------------------------------------------


def pick_target_layer(yolo_model, scale: str = "p3", verbose: bool = True):
    """Neck feature map(s) feeding the detection head.

    DO NOT take the last Conv2d in the network -- that was the original
    implementation and it was wrong on every architecture, in different ways:

        YOLOv8s / YOLO11s -> in_ch=16 out_ch=1   (the DFL conv: a FIXED,
                             non-learned projection over 16 distribution bins,
                             buried inside the head)
        YOLO26s           -> in_ch=128 out_ch=14 (the classification output
                             conv -- YOLO26 removed DFL, so [-1] lands
                             somewhere structurally different again)

    Each would have produced a plausible-looking heatmap from the wrong tensor,
    and because they differ per architecture the cross-model comparison would
    have been meaningless without ever erroring.

    Correct target: the multi-scale feature maps the Detect head consumes.
    Ultralytics stores these indices on the head itself as `Detect.f`
    (e.g. [15, 18, 21] for YOLOv8s, [16, 19, 22] for YOLO11s/YOLO26s), so this
    resolves architecture-agnostically.

    scale
    -----
    "p3"  stride 8,  64x64 at 512px input -- highest resolution. DEFAULT,
          because Axis A concerns small lesions (Nodule/Mass 39.69% small,
          Calcification 17.67%) and coarse maps cannot localise them.
    "p4"  stride 16, 32x32
    "p5"  stride 32, 16x16 -- too coarse for small-lesion attribution
    "all" all three; pytorch-grad-cam aggregates across layers
    """
    net = yolo_model.model.model          # nn.Sequential

    detect_idx = None
    for i, m in enumerate(net):
        if type(m).__name__ in ("Detect", "v10Detect", "YOLOEDetect"):
            detect_idx = i
    if detect_idx is None:
        raise RuntimeError(
            "no Detect head found -- inspect model.model.model manually"
        )

    feeds = list(getattr(net[detect_idx], "f", []))
    if not feeds:
        raise RuntimeError(f"Detect at index {detect_idx} exposes no .f indices")

    order = {"p3": [0], "p4": [1], "p5": [2], "all": list(range(len(feeds)))}
    if scale not in order:
        raise ValueError(f"scale must be one of {list(order)}, got {scale!r}")
    chosen = [feeds[i] for i in order[scale]]
    layers = [net[i] for i in chosen]

    if verbose:
        print(f"[xai] Detect head at index {detect_idx}, consumes {feeds}")
        print(f"[xai] scale={scale} -> module indices {chosen}")
        for i, layer in zip(chosen, layers):
            print(f"[xai]   net[{i}] = {type(layer).__name__}")
    return layers if len(layers) > 1 else layers[0]


def verify_cam_resolution(yolo_model, imgsz: int = 512, scale: str = "p3") -> None:
    """Print the CAM grid size, so coarseness is visible before a batch run.

    A stride-32 (P5) map is 16x16 at 512px input -- each cell covers 32x32
    input pixels. Many Calcification boxes are smaller than one cell, which
    makes box-overlap metrics meaningless at that scale.
    """
    import numpy as np

    dummy = np.zeros((imgsz, imgsz), dtype=np.uint8)
    layer = pick_target_layer(yolo_model, scale=scale, verbose=False)
    cam = compute_cam(yolo_model, dummy, method="eigencam",
                      target_layer=layer, imgsz=imgsz)
    stride = imgsz / max(cam.shape)
    print(f"[xai] scale={scale}: CAM grid {cam.shape}, ~{stride:.0f}px per cell")
    if stride > 16:
        print(f"[xai] ⚠ {stride:.0f}px cells are coarse for small-lesion "
              f"attribution -- prefer scale='p3'")


def _wrap_for_cam(detection_model):
    """Adapt an Ultralytics DetectionModel to what pytorch-grad-cam expects.

    pytorch-grad-cam assumes a classifier: forward() returns a (B, C) tensor,
    and it calls `outputs.cpu()` plus `np.argmax(..., axis=-1)`. Ultralytics
    returns a tuple (predictions, feature maps), so the raw model raises
    `'tuple' object has no attribute 'cpu'`.

    This unwraps to the prediction tensor and reduces over anchors/spatial
    positions to give (B, C).

    For EigenCAM the reduction is irrelevant -- it is non-gradient, computing
    the first principal component of the activations, so no target is used.
    For GradCAM++ it defines the scalar being differentiated: the peak
    response of the strongest channel.

    Module identity is unchanged by wrapping, so hooks registered on
    target_layers still fire.
    """
    import torch.nn as nn

    class _CamAdapter(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            while isinstance(out, (list, tuple)):     # (preds, feats) -> preds
                out = out[0]
            if out.ndim == 3:        # (B, 4+nc, anchors) -> (B, 4+nc)
                return out.amax(dim=-1)
            if out.ndim == 4:        # (B, C, H, W)     -> (B, C)
                return out.amax(dim=(-2, -1))
            return out

    return _CamAdapter(detection_model)


def compute_cam(yolo_model, image: np.ndarray, method: str = "eigencam",
                target_layer=None, imgsz: int = 512) -> np.ndarray:
    """CAM for one image, normalized to [0,1] at (imgsz, imgsz).

    EigenCAM is preferred: it is non-gradient (first principal component of the
    activations), so it needs no backward pass and no differentiable target --
    which sidesteps the awkwardness of defining a scalar target for a detection
    head with variable output count.
    """
    import cv2
    import torch
    from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus

    if target_layer is None:
        target_layer = pick_target_layer(yolo_model, verbose=False)
    layers = target_layer if isinstance(target_layer, list) else [target_layer]
    cls = {"eigencam": EigenCAM, "gradcam++": GradCAMPlusPlus}[method]

    img = cv2.resize(image, (imgsz, imgsz))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0

    net = yolo_model.model.eval()
    device = next(net.parameters()).device
    tensor = tensor.to(device)

    cam = cls(model=_wrap_for_cam(net), target_layers=layers)
    grayscale = cam(input_tensor=tensor, targets=None)[0]

    rng = grayscale.max() - grayscale.min()
    return (grayscale - grayscale.min()) / (rng + 1e-9)


# --------------------------------------------------------------------------
# Faithfulness metrics
# --------------------------------------------------------------------------


def _scale_boxes(boxes, orig_wh, imgsz):
    """GT boxes in original pixels -> CAM grid coords."""
    w, h = orig_wh
    out = np.asarray(boxes, dtype=float).copy().reshape(-1, 4)
    out[:, [0, 2]] *= imgsz / w
    out[:, [1, 3]] *= imgsz / h
    return np.clip(out, 0, imgsz - 1)


def pointing_game(cam: np.ndarray, boxes, orig_wh, imgsz: int = 512) -> float:
    """1.0 if the CAM's peak lands inside any GT box. Coarse but standard."""
    if len(boxes) == 0:
        return float("nan")
    b = _scale_boxes(boxes, orig_wh, imgsz)
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    inside = ((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3]))
    return float(inside.any())


def energy_pointing_game(cam: np.ndarray, boxes, orig_wh, imgsz: int = 512) -> float:
    """Fraction of total CAM energy inside GT boxes.

    More robust than pointing_game -- it uses the whole map, not one pixel, so
    it distinguishes "peaked correctly by luck" from "consistently attends to
    the lesion". This is the metric to lead with.
    """
    if len(boxes) == 0:
        return float("nan")
    b = _scale_boxes(boxes, orig_wh, imgsz).astype(int)
    mask = np.zeros_like(cam, dtype=bool)
    for x1, y1, x2, y2 in b:
        mask[y1:y2 + 1, x1:x2 + 1] = True
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else float("nan")


def cam_box_iou(cam: np.ndarray, boxes, orig_wh, imgsz: int = 512,
                percentile: int = 90) -> float:
    """IoU between the thresholded CAM and the GT box union."""
    if len(boxes) == 0:
        return float("nan")
    b = _scale_boxes(boxes, orig_wh, imgsz).astype(int)
    gt = np.zeros_like(cam, dtype=bool)
    for x1, y1, x2, y2 in b:
        gt[y1:y2 + 1, x1:x2 + 1] = True
    pred = cam >= np.percentile(cam, percentile)
    union = (gt | pred).sum()
    return float((gt & pred).sum() / union) if union > 0 else float("nan")


def deletion_insertion_auc(yolo_model, image: np.ndarray, cam: np.ndarray,
                           imgsz: int = 512, steps: int = 20) -> dict:
    """Causal faithfulness: does removing salient pixels actually hurt?

    deletion  -- progressively blank the most-salient pixels. Confidence should
                 FALL fast. Lower AUC = more faithful.
    insertion -- reveal most-salient pixels onto a blurred base. Confidence
                 should RISE fast. Higher AUC = more faithful.

    A saliency map that looks convincing but has flat deletion AUC is not
    explaining the model -- this is precisely the failure MedFocus /
    MedGround-Bench reported for standard attribution on VinDr.
    """
    import cv2

    img = cv2.resize(image, (imgsz, imgsz))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)

    order = np.argsort(cam.ravel())[::-1]
    n_px = len(order)
    blurred = cv2.GaussianBlur(img, (51, 51), 0)

    def confidence(arr) -> float:
        r = yolo_model.predict(arr, imgsz=imgsz, conf=0.001, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0
        return float(r.boxes.conf.max())

    del_scores, ins_scores = [], []
    for i in range(steps + 1):
        k = int(n_px * i / steps)
        idx = np.unravel_index(order[:k], cam.shape)

        d = img.copy()
        d[idx] = 0
        del_scores.append(confidence(d))

        ins = blurred.copy()
        ins[idx] = img[idx]
        ins_scores.append(confidence(ins))

    xs = np.linspace(0, 1, steps + 1)
    return {
        "deletion_auc": float(np.trapz(del_scores, xs)),
        "insertion_auc": float(np.trapz(ins_scores, xs)),
        "deletion_curve": del_scores,
        "insertion_curve": ins_scores,
    }


# --------------------------------------------------------------------------
# Batch evaluation
# --------------------------------------------------------------------------


def evaluate_xai(yolo_model, images_dir, labels_dir, method: str = "eigencam",
                 imgsz: int = 512, n_images=None, with_causal: bool = False,
                 causal_subsample: int = 50, scale: str = "p3"):
    """Run every metric across a split. Returns a tidy per-image DataFrame.

    with_causal=False by default: deletion/insertion needs (steps+1)*2 forward
    passes per image, so it is ~40x the cost. Run it on a subsample only, and
    only if D9 is on schedule.
    """
    from pathlib import Path

    import cv2
    import pandas as pd

    from ..config import CLASSES
    from .metrics import ground_truth_from_yolo

    gts = ground_truth_from_yolo(labels_dir, images_dir)
    paths = sorted(Path(images_dir).glob("*.jpg"))
    if n_images:
        paths = paths[:n_images]

    target_layer = pick_target_layer(yolo_model, scale=scale)
    rows = []

    for i, p in enumerate(paths):
        if i % 50 == 0:
            print(f"  xai {i}/{len(paths)}")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        g = gts[gts.image_id == p.stem]
        if len(g) == 0:
            continue

        try:
            cam = compute_cam(yolo_model, img, method=method,
                              target_layer=target_layer, imgsz=imgsz)
        except Exception as e:
            print(f"  ! CAM failed on {p.stem}: {e}")
            continue

        # Per-class, so Axis A (small vs large target) is separable.
        for cls_id, grp in g.groupby("class_id"):
            boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy()
            row = {
                "image_id": p.stem,
                "class_id": int(cls_id),
                "class": CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else "?",
                "method": method,
                "n_boxes": len(boxes),
                "box_area_frac": float(
                    ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum()
                    / (w * h)
                ),
                "pointing_game": pointing_game(cam, boxes, (w, h), imgsz),
                "ebpg": energy_pointing_game(cam, boxes, (w, h), imgsz),
                "cam_box_iou": cam_box_iou(cam, boxes, (w, h), imgsz),
            }
            rows.append(row)

        if with_causal and i < causal_subsample:
            causal = deletion_insertion_auc(yolo_model, img, cam, imgsz=imgsz)
            rows[-1].update({k: v for k, v in causal.items() if k.endswith("auc")})

    return pd.DataFrame(rows)


def summarize_xai(df, small_classes=None, large_classes=None):
    """Axis A summary: small-target vs large-target faithfulness."""
    from ..config import LARGE_TARGET_CLASSES, SMALL_TARGET_CLASSES

    small = small_classes or SMALL_TARGET_CLASSES
    large = large_classes or LARGE_TARGET_CLASSES
    metrics = ["pointing_game", "ebpg", "cam_box_iou"]

    per_class = df.groupby("class")[metrics].mean().round(4)
    group = df.assign(
        group=df.class_id.map(
            lambda c: "small" if c in small else ("large" if c in large else "other")
        )
    ).groupby("group")[metrics].mean().round(4)

    return {"per_class": per_class, "by_target_size": group,
            "overall": df[metrics].mean().round(4).to_dict()}
