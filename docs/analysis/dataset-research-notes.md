# VinBigData Chest X-Ray: Dataset & Implementation Research Notes

## Dataset Overview (VinDr-CXR)

| Attribute | Training | Test |
|---|---|---|
| Total Radiographs | 15,000 | 3,000 |
| Raters per Scan | 3 (blinded) | 5 (consensus panel) |
| Median Resolution | 2788×2446 | 2748×2394 |
| Median Age | 43.77 yrs | 31.80 yrs |
| Male % | 52.21% | 55.90% |
| Female % | 47.79% | 44.10% |
| Data Volume | 161 GB | 31.3 GB |

- **Source**: Two tertiary hospitals in Vietnam (Hanoi Medical University Hospital, 108 Military Central Hospital)
- **Full dataset**: 100,000+ raw PA chest radiographs; 18,000 curated and annotated
- **Annotation tool**: VinLab (open-source, OHIF-based)
- **Competition metric**: mAP @ IoU > 0.4
- **1st place score**: 0.314 mAP

### 14 Abnormality Classes + No Finding

| ID | Class Name | Notes |
|---|---|---|
| 0 | Aortic enlargement | High prevalence |
| 1 | Atelectasis | |
| 2 | Calcification | 17.67% small targets |
| 3 | Cardiomegaly | High prevalence |
| 4 | Consolidation | Overlaps with infiltration |
| 5 | ILD (Interstitial Lung Disease) | |
| 6 | Infiltration | Overlaps with consolidation |
| 7 | Lung Opacity | |
| 8 | Nodule/Mass | 39.69% small targets |
| 9 | Other lesion | |
| 10 | Pleural effusion | |
| 11 | Pleural thickening | High prevalence |
| 12 | Pneumothorax | |
| 13 | Pulmonary fibrosis | High prevalence |
| 14 | No finding | Majority class |

---

## Known Dataset Issues

### 1. Metadata Leakage (CRITICAL)
Simple gradient-boosting on DICOM header metadata (age, sex, Source Application Entity Title / SAET) achieves high validation scores without analyzing pixels. SAET correlates strongly with normal vs. abnormal labels because different scanners were used at different institutions.

### 2. Rater Imbalance (CRITICAL) — ✅ *measured 2026-07-20, worse than previously stated*

Counted directly from `train.csv` (positive boxes only, `class_id != 14`):

| Rater | Boxes | Rater | Boxes |
|---|---|---|---|
| **R9** | 13,729 | R16 | 198 |
| **R10** | 10,971 | R12 | 149 |
| **R8** | 9,762 | R17 | 69 |
| R14 | 324 | R2 | 3 |
| R13 | 319 | R1, R3–R7 | **0** |
| R15 | 315 | | |
| R11 | 257 | **Total** | **36,096** |

- **R8 + R9 + R10 = 34,462 / 36,096 = 95.5%** of all positive boxes
- Only **11 of 17 raters** appear at all; R1 and R3–R7 contribute nothing, R2 contributes 3 boxes
- The earlier phrasing ("R1–R7 marked almost zero") understated this — they are effectively absent

**Implication for any multi-rater work (P5):** the "3 raters per image" framing is nominal, not effective. Fused labels are substantively three radiologists' opinions, not a 17-rater consensus. State in limitations.

### 2b. Rater Disagreement on Box Extent — ✅ *measured 2026-07-20*

Raters agree on *whether* a finding exists far more than on *where it ends*. Post-fusion retention at IoU 0.5 splits cleanly by finding type:

| Retention | Classes | Interpretation |
|---|---|---|
| 44–47% | Cardiomegaly, Aortic enlargement | Anatomically anchored — raters agree where the heart is, boxes merge |
| 72–84% | Pleural thickening, Other lesion, Atelectasis, Lung Opacity, Pulmonary fibrosis, ILD | Diffuse extent — raters disagree, boxes fail to merge |

At IoU 0.5 this produced **systematic under-merging**: one finding surviving as 2–4 separate ground-truth boxes. Measured example — two rater boxes on a single pleural thickening had IoU 0.38, just below the cut.

**Threshold selected by measured sweep, not judgement:**

| IoU thr | fused boxes | retained | duplicate pairs | affected images |
|---|---|---|---|---|
| **0.25** ← chosen | **21,473** | **59.5%** | **58** | 56 |
| 0.30 | 21,840 | 60.5% | 413 | 346 |
| 0.40 | 22,724 | 63.0% | 1,331 | 942 |
| 0.50 | 23,948 | 66.3% | 2,655 | 1,576 |
| *floor if all 3-rater groups merged* | *~12,032* | *33.3%* | — | — |

Duplicate pairs roughly **halve per step while retention moves only 6.8 points across the entire range** — the threshold trades duplicate labels for almost nothing. Two intermediate values were rejected on evidence: 0.5 left 2,655 duplicate pairs, and 0.4 still left 1,331 across 21% of images (visually confirmed as 8×-nested Pulmonary fibrosis on one image, 3×-stacked Pleural effusion on another).

Final: **WBF @ IoU 0.25** (`config.py` CHANGELOG 2026-07-20). Spatially distinct same-class lesions (ILD in both lungs, IoU ≈ 0) are unaffected at any threshold.

**Residual risk**: at 0.25, two genuinely distinct *adjacent* lesions could merge. Nodule/Mass and Calcification are the exposed classes — baselines at 0.4 were 70.9% and 75.5% retention. If they crater, the fix is a per-class threshold rather than a lower global one.

> The principled fix is a **consensus rule** — require ≥2 of 3 raters to agree before a box becomes ground truth. That merges *and* filters single-rater spurious findings, which threshold tuning does not address: currently any single rater's box becomes ground truth. Out of scope for the ICCIT deadline; natural bridge to P5.

### 3. Overlapping Pathology Definitions
Consolidation and infiltration have overlapping radiological definitions, creating label ambiguity.

### 4. Small Object Prevalence
- Pulmonary nodules/masses: 39.69% are small targets
- Calcifications: 17.67% are small targets
- Requires multi-scale feature fusion architectures

### 5. Class Imbalance
Heavy skew toward "No Finding"; rare pathologies severely underrepresented.

---

## Previous Implementations Summary

### 1. awsaf49 — YOLOv5 14-Class (Kaggle Notebooks)
- Format: 2 Kaggle .ipynb notebooks (train + inference)
- Architecture: YOLOv5 core engine
- Task: 14-class abnormality detection
- Foundational reference cited by others

### 2. nxhong93 — YOLOv5 Chest 512
- Format: 1 Kaggle .ipynb notebook
- Architecture: YOLOv5 at 512px resolution
- Lightweight baseline approach

### 3. sunghyunjun — Kaggle 76th Place (Full Python Project)
- Format: Complete Python project (7 .py files)
- Architecture: Two-stage (EfficientNet classifier + EfficientDet detector) + One-stage (15-class EfficientDet)
- 15 classifiers + 18 detectors + 2 one-stage detectors, blended via NMS
- Framework: PyTorch Lightning
- Best result: Private LB 0.253 (blended)
- Key insight: ZFTurbo NMS > torchvision NMS > WBF for this dataset

### 4. tariqshaban — YOLOv7 VinBigData (Course Project)
- Format: Notebook + README + assets
- Architecture: YOLOv7 single-stage detection
- 1024×1024 resized PNG, 90/10 split, batch size 8
- Pretrained YOLOv7 weights, halted before convergence
- Well-documented methodology and EDA

---

## Key Design Insights

| Decision | Recommendation |
|---|---|
| BBox fusion | ⚠️ **Unsettled.** The `sunghyunjun` ablation has ZFTurbo NMS winning on CV (0.4419 vs 0.4158) but WBF winning on private LB (0.185 vs 0.181); the author chose NMS by reasoning about test labeling, not measurement. Do not treat either as established. |
| Fusion IoU threshold | **0.25** — selected by measured duplicate sweep, not judgement (see §2b). 0.5 and 0.4 both under-merge diffuse findings badly. |
| Tail classes | Pneumothorax (~11) and Atelectasis (~24) test boxes — too few for stable per-class AP. Report with explicit *n* or aggregate. |
| Image resolution | Larger = better mAP (limited by VRAM) |
| Normal handling | Two-stage approach effective if threshold tuned |
| Small objects | Multi-scale fusion critical |
| Augmentation | Resize+Scale+Crop+Brightness+Blur works; CLAHE/Equalize/HSV don't help |
| Batch size | < 4 yields poor mAP |
| Metadata | Strip DICOM headers; don't use age/sex/SAET as features |
