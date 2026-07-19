"""
Build the YOLO-format VinDr-CXR detection dataset.

Pipeline:  train.csv (raw, 3 raters/image)
             -> drop "No finding"-only images  [positive-only subset]
             -> fuse rater boxes                [fusion.py]
             -> stratified 80/10/10 split
             -> YOLO txt labels + symlinked images + data.yaml

Positive-only is the compute enabler: ~4.4K of 15K images, ~3.4x epoch-time
reduction. It is also the established protocol on this dataset -- the vendored
sunghyunjun 76th-place solution trained its detector on abnormal images only
and reported CV on positives only. State this explicitly in the paper; it is a
protocol choice, not a shortcut, but it does mean the model never sees normal
images and cannot be evaluated on normal-vs-abnormal discrimination.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    required = {"image_id", "class_id", "x_min", "y_min", "x_max", "y_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    print(f"raw: {len(df)} rows, {df.image_id.nunique()} images")
    return df


def positive_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep images with >=1 real finding. Drops the 'No finding' majority."""
    pos_ids = df.loc[df.class_id != 14, "image_id"].unique()
    out = df[df.image_id.isin(pos_ids)].copy()
    print(f"positive-only: {len(pos_ids)} images "
          f"({len(pos_ids)/df.image_id.nunique():.1%} of total)")
    return out


def get_image_dims(image_dir: Path, image_ids) -> dict[str, tuple[int, int]]:
    """Read (w,h) per image. Needed to normalize boxes for fusion + YOLO."""
    from PIL import Image

    dims = {}
    for i, img_id in enumerate(image_ids):
        if i % 1000 == 0:
            print(f"  dims {i}/{len(image_ids)}")
        with Image.open(image_dir / f"{img_id}.jpg") as im:
            dims[img_id] = im.size          # (width, height)
    return dims


def stratified_split(
    df_fused: pd.DataFrame, split=(0.8, 0.1, 0.1), seed: int = 0
) -> dict[str, list[str]]:
    """Split by image, stratified on each image's RAREST present class.

    Stratifying on the rarest class is what protects the tail classes
    (Pneumothorax, Other lesion) from vanishing entirely out of val/test --
    which would make their per-class AP undefined and break the Axis-A
    small-vs-large analysis.
    """
    from sklearn.model_selection import train_test_split

    freq = df_fused.class_id.value_counts()
    strata = (
        df_fused.groupby("image_id")["class_id"]
        .apply(lambda s: min(s, key=lambda c: freq[c]))
    )
    ids = strata.index.to_numpy()
    y = strata.to_numpy()

    # Collapse strata with <3 members -- train_test_split cannot stratify them.
    counts = Counter(y)
    y = np.array([c if counts[c] >= 3 else -1 for c in y])

    tr, tmp = train_test_split(ids, train_size=split[0], random_state=seed, stratify=y)
    y_tmp = pd.Series(y, index=ids).loc[tmp].to_numpy()
    rel = split[1] / (split[1] + split[2])
    counts_tmp = Counter(y_tmp)
    y_tmp = np.array([c if counts_tmp[c] >= 2 else -1 for c in y_tmp])
    va, te = train_test_split(tmp, train_size=rel, random_state=seed, stratify=y_tmp)

    out = {"train": list(tr), "val": list(va), "test": list(te)}
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")
    return out


def to_yolo_labels(
    df_fused: pd.DataFrame, dims: dict, splits: dict, out_root: Path,
    image_dir: Path, copy_images: bool = False,
) -> None:
    """Write YOLO txt labels + place images. Format: cls cx cy w h (normalized)."""
    for split_name, ids in splits.items():
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    grouped = {k: v for k, v in df_fused.groupby("image_id")}

    for split_name, ids in splits.items():
        for img_id in ids:
            w, h = dims[img_id]
            rows = grouped.get(img_id)
            lines = []
            if rows is not None:
                for _, r in rows.iterrows():
                    cx = ((r.x_min + r.x_max) / 2) / w
                    cy = ((r.y_min + r.y_max) / 2) / h
                    bw = (r.x_max - r.x_min) / w
                    bh = (r.y_max - r.y_min) / h
                    # Degenerate boxes would silently poison training.
                    if bw <= 0 or bh <= 0:
                        continue
                    cx, cy = np.clip([cx, cy], 0, 1)
                    bw, bh = np.clip([bw, bh], 0, 1)
                    lines.append(f"{int(r.class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (out_root / "labels" / split_name / f"{img_id}.txt").write_text("\n".join(lines))

            src = image_dir / f"{img_id}.jpg"
            dst = out_root / "images" / split_name / f"{img_id}.jpg"
            if dst.exists():
                continue
            if copy_images:
                shutil.copy(src, dst)
            else:
                try:
                    dst.symlink_to(src)      # saves ~4 GB on Kaggle
                except OSError:
                    shutil.copy(src, dst)
        print(f"  wrote {split_name}: {len(ids)} images")


def write_data_yaml(out_root: Path, classes: list[str]) -> Path:
    path = out_root / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    path.write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(classes)}\nnames:\n{names}\n"
    )
    print(f"wrote {path}")
    return path


def verify(out_root: Path, splits: dict) -> bool:
    """Post-build assertions. Run before D3. Cheap; catches silent breakage."""
    ok = True
    for split_name, ids in splits.items():
        imgs = list((out_root / "images" / split_name).glob("*.jpg"))
        lbls = list((out_root / "labels" / split_name).glob("*.txt"))
        if len(imgs) != len(ids) or len(lbls) != len(ids):
            print(f"  ✗ {split_name}: {len(imgs)} imgs / {len(lbls)} lbls, expected {len(ids)}")
            ok = False
        empty = sum(1 for p in lbls if not p.read_text().strip())
        if empty:
            print(f"  ⚠ {split_name}: {empty} empty label files "
                  f"(should be ~0 under positive-only)")
        # bounds check
        for p in lbls[:200]:
            for line in p.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    print(f"  ✗ malformed line in {p.name}: {line!r}")
                    ok = False
                    break
                if not all(0 <= float(v) <= 1 for v in parts[1:]):
                    print(f"  ✗ out-of-range coords in {p.name}: {line!r}")
                    ok = False
                    break
    print("  ✓ verify passed" if ok else "  ✗ verify FAILED")
    return ok


def class_distribution(df_fused: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Per-split class counts. Confirms no tail class vanished from val/test."""
    from ..config import CLASSES

    rows = {}
    for name, ids in splits.items():
        rows[name] = df_fused[df_fused.image_id.isin(ids)].class_id.value_counts()
    out = pd.DataFrame(rows).fillna(0).astype(int).sort_index()
    out.insert(0, "class", [CLASSES[i] if i < len(CLASSES) else "?" for i in out.index])
    return out
