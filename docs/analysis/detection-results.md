# Detection Results — YOLOv8s / YOLO11s / YOLO26s on VinDr-CXR

*Measured 2026-07-20. 3 models × 3 seeds = 9 complete runs. Tesla T4, 512px, `s` scale, 40 epochs, batch 16, positive-only subset, WBF@0.25 fusion.*

> [!IMPORTANT]
> **All headline numbers here are TEST split.** Val-split numbers appear only where the val/test divergence is itself the point — see §2. Reporting val would have produced a different and wrong ranking.

---

## 1. Headline

### Test split, mean ± std over 3 seeds

| Model | mAP@0.4 | mAP@0.5 | s/epoch | Latency (bs=1) | Params |
|---|---|---|---|---|---|
| **YOLOv8s** | **0.413 ± 0.008** | **0.371 ± 0.003** | 45.6 | **8.01 ms** | 11.13 M |
| YOLO11s | 0.402 ± 0.003 | 0.360 ± 0.004 | 46.5 | 9.22 ms | 9.42 M |
| YOLO26s | 0.400 ± 0.008 | 0.361 ± 0.007 | 56.0 | 10.65 ms | 9.47 M |

**Seed-noise test (mAP@0.4)**: between-model gap **0.0124**, mean within-model std **0.0062** → ratio **2.0×**. Clears the 2× bar, but only just. On mAP@0.5 the gap-to-noise ratio is ~2.7×, which is firmer.

### What can be claimed

✅ **YOLOv8s outperforms both newer architectures.** Consistent across both IoU thresholds and all three seeds.

✅ **YOLO11s and YOLO26s are statistically indistinguishable.** 0.402 vs 0.400 at mAP@0.4; 0.360 vs 0.361 at mAP@0.5. Report as tied — do not rank them.

✅ **Newer is not better here.** Architectures released in 2024 (YOLO11) and 2026 (YOLO26) do not beat the 2023 baseline on this task at this scale and resolution.

❌ **Do not claim** YOLO26 is "significantly worse" — that conclusion came from val and does not survive on test.

---

## 2. The val/test divergence — report test only

| | val mAP@0.5 | test mAP@0.5 |
|---|---|---|
| YOLOv8s | 0.386 ± 0.005 | 0.371 ± 0.003 |
| YOLO11s | 0.384 ± 0.005 | 0.360 ± 0.004 |
| YOLO26s | **0.366 ± 0.009** | **0.361 ± 0.007** |

On val, YOLO26 trails YOLO11 by 0.018 — nearly 4× its own seed std, and it looks like a clear result. On test the two are **tied**.

The val split is 439 images against the test split's 440, both drawn by the same stratified procedure, so this is not a size or sampling-design effect. It is ordinary split-to-split variance at this scale, and it is larger than the between-model differences being measured.

> **Methodological note worth stating in the paper.** Single-split model selection on datasets of this size can invert a ranking. This is an argument for the multi-seed protocol *and* for holding out a genuine test split rather than reporting the split used for checkpoint selection (`best.pt` is chosen on val).

---

## 3. Small-target performance — YOLO26's STAL works

Per-class AP@0.4, mean over 3 seeds. `SMALL_TARGET_CLASSES` in `config.py` are Calcification and Nodule/Mass.

| Class | YOLOv8s | YOLO11s | **YOLO26s** | n_gt |
|---|---|---|---|---|
| **Calcification** | 0.079 | 0.073 | **0.126** | 55 |
| **Nodule/Mass** | 0.281 | 0.328 | **0.331** | 202 |

**Calcification is decisive.** YOLO26 is 60% above the next model; the 0.047 gap is ~5× the per-class std (0.009–0.017). On Nodule/Mass it ties YOLO11 while YOLOv8s trails by 0.05.

This supports YOLO26's **Small-Target-Aware Label Assignment** doing what it claims — on the dataset property (39.69% small targets for Nodule/Mass, 17.67% for Calcification) that motivated choosing it.

### Where YOLO26 gives it back

| Class | YOLOv8s | YOLO26s | Δ |
|---|---|---|---|
| Atelectasis | 0.310 | 0.219 | **−0.091** |
| Pleural thickening | 0.320 | 0.273 | −0.047 |
| Pulmonary fibrosis | 0.371 | 0.341 | −0.030 |
| Consolidation | 0.444 | 0.431 | −0.013 |

Larger, diffuse, poorly-bounded findings — the opposite end of the size distribution from where it gains.

> **This is the paper's most interesting detection finding.** Not "newest model is worse," but: **YOLO26 trades large-lesion performance for small-lesion performance, netting slightly negative overall on this dataset.** It also sets up Axis A directly — does explanation faithfulness split the same way?

---

## 4. Anchored classes dominate the headline number

| Model | all 14 classes | excluding 2 anchored |
|---|---|---|
| YOLOv8s | 0.413 | **0.322** |
| YOLO11s | 0.402 | **0.310** |
| YOLO26s | 0.400 | **0.310** |

| | AP@0.4 |
|---|---|
| Aortic enlargement | 0.943 – 0.956 |
| Cardiomegaly | 0.946 – 0.953 |
| *remaining 12, mean* | ~0.31 |

Both leaders are **anatomically anchored** — the aorta and heart occupy the same image region in every posteroanterior radiograph, so a detector can score highly by learning position rather than pathology. They are ~3× the remaining twelve and contribute disproportionately to every model's mAP.

Their seed std is also remarkably low (0.0015–0.0071), consistent with an easy, position-determined task.

**Report per-class throughout.** A single mAP conceals that ~25% of the headline number comes from two classes that may not require pathological reasoning at all. Whether they do is a question only saliency measurement can answer — see notebook 03 §2b.

---

## 5. Latency — the NMS-free advantage does not transfer

Measured back-to-back, one session, one T4, batch size 1, 512px, 100 iterations after 10 warmup.

| Model | ms/image | FPS |
|---|---|---|
| **YOLOv8s** | **8.01** | **124.8** |
| YOLO11s | 9.22 | 108.5 |
| YOLO26s | 10.65 | 93.9 |

YOLO26 is **slowest**, despite the NMS-free head cutting postprocess dramatically (0.4 ms vs 3.5 ms in the Ultralytics validation loop). Its heavier backbone — 260 layers against YOLOv8s's 130 — more than offsets the saving. It is also 22% slower to train (56.0 vs 45.6 s/epoch).

> [!CAUTION]
> **Scope this claim narrowly.** YOLO26's marketed advantage is **CPU** inference, and batch-1 GPU latency at 512px is partly kernel-launch bound rather than compute bound. These measurements do **not** refute the vendor claim; they show it does not transfer to single-image GPU inference at this resolution. State it that way.

---

## 6. Protocol caveats — must appear in the paper

**Not comparable to published numbers.** Training and evaluation use the **positive-only subset** (4,394 of 15,000 images; every image contains ≥1 finding). The model never has an opportunity to false-positive on a healthy chest, which inflates precision and mAP relative to full-set evaluation. Whether YOLO-CXR (0.338), YOLOv11-MFF (0.415) or RT-DETR (0.453) used positive-only is **not established**. Keep them in a related-work table with an explicit protocol column; never in the same column as ours. **State the protocol in the abstract.**

**Label provenance.** 95.5% of positive boxes come from three radiologists (R9/R10/R8); six of seventeen contributed none. Labels are effectively a three-rater consensus, not seventeen.

**Fusion.** WBF @ IoU 0.25, selected by measured duplicate sweep (see `dataset-research-notes.md` §2b). 36,096 raw boxes → 21,473 fused.

**Thin classes.** Pneumothorax n=10 and Atelectasis n=22 in test. Per-class AP on those is unstable — seed std reaches 0.083 for YOLO26 on Atelectasis. Reporting rule fixed before any score was seen: keep all 14 in mAP, show `n`, caveat in limitations.

**Configuration.** 512px, `s` scale, single fold, 40 epochs, batch 16, AdamW (Ultralytics `optimizer=auto`, lr0=0.000556), AMP, Tesla T4, seeds 0/1/2. `deterministic=True` — repeated runs at a fixed seed reproduced bit-identically.

---

## 7. Open item

**Detection density.** 36,000–37,800 predictions against 2,112 ground-truth boxes (~82 per image) at `conf=0.001`. Low confidence is correct for AP — it integrates the full precision-recall curve — but `max_det=100 × 440 = 44,000` is the ceiling and we sit at ~82% of it. Confirm truncation is not clipping dense images:

```python
counts = preds.groupby("image_id").size()
print((counts >= 100).sum(), "images at the max_det ceiling")
```

If more than a handful hit exactly 100, raise `max_det` and re-evaluate — AP would be understated for those images.
