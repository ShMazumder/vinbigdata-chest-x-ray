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

### 2. Rater Imbalance (CRITICAL)
- Radiologists R8, R9, R10: annotated vast majority of positive findings, smaller bounding boxes
- Radiologists R1–R7: marked almost zero abnormalities, defaulting to "No Finding"
- Models must handle rater-specific variance and label noise

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
| BBox fusion | ZFTurbo NMS > torchvision NMS > WBF |
| Image resolution | Larger = better mAP (limited by VRAM) |
| Normal handling | Two-stage approach effective if threshold tuned |
| Small objects | Multi-scale fusion critical |
| Augmentation | Resize+Scale+Crop+Brightness+Blur works; CLAHE/Equalize/HSV don't help |
| Batch size | < 4 yields poor mAP |
| Metadata | Strip DICOM headers; don't use age/sex/SAET as features |
