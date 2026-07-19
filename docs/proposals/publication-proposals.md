# Publication-Oriented Research Proposals for VinBigData Chest X-Ray

*Prepared: July 2026 — Updated with deep novelty search results*

---

## Overview: Gap Analysis

Based on the comparative analysis of 25+ existing works **and deep search verification (July 2026)**, the following table shows the updated status of each proposed gap:

| Gap ID | Research Gap | Current State (Verified) | Opportunity Status |
|---|---|---|---|
| G1 | Modern YOLO systematic benchmark on VinDr-CXR | **PARTIALLY FILLED** — YOLO-CXR (YOLOv8+RefConv+ELCA) published; YOLOv11-MFF published. But no multi-YOLO comparison (v8 vs v9 vs v10 vs v11) exists. | ⚠️ Narrower gap — needs novel angle |
| G2 | Rater-aware training | **PARTIALLY FILLED** — TwinTrack (MIDL 2026) models inter-rater disagreement. But not applied to VinDr-CXR specifically, and not for object detection. | ✅ Gap exists for detection task |
| G3 | Transformer-based detector on VinDr | **PARTIALLY FILLED** — CD-DETR (2025, NIH dataset), CGF-DETR (2025, RSNA Pneumonia). Neither evaluated on VinDr-CXR specifically. | ✅ Gap exists for VinDr-CXR |
| G4 | GroundingDINO + SAM/SAM2 for CXR | **PARTIALLY FILLED** — LuGSAM (ICU CXR) exists; GroundingDINO-Med fine-tuned on VinDr exists. But no systematic GroundingDINO+SAM2 few-shot study on VinDr-CXR. | ⚠️ Narrower gap — needs differentiation |
| G5 | Lightweight/edge deployment | **PARTIALLY FILLED** — General KD for CXR classification exists widely. But no KD specifically for CXR object detection with edge benchmarks. | ✅ Gap exists for detection KD |
| G6 | Active learning for CXR detection | **PARTIALLY OPEN** — AL for CXR classification exists. AL for CXR object detection with bounding boxes is underexplored. No work on VinDr-CXR. | ✅ Strong gap |
| G7 | Metadata leakage mitigation | **PARTIALLY FILLED** — Domain adversarial CXR work exists (scanner bias, artifact robustness). But no work specifically addressing VinDr-CXR SAET leakage with DANN. | ✅ Gap exists for VinDr-specific |

---

## Revised Proposals with Novelty Risk Assessment

### Proposal 1: Modern YOLO Benchmark + Rater-Aware Training *(REVISED — merged with rater-awareness)*
**🎯 Target Venue**: Computers in Biology and Medicine / IEEE Access

**Novelty Risk**: ⚠️ **MEDIUM** — YOLO-CXR (YOLOv8+RefConv+ELCA, mAP@0.5=0.338) already published. YOLOv11-MFF also exists.

**Existing Prior Work Found**:
- **YOLO-CXR** (2025/2026): YOLOv8s + RefConv + ELCA → mAP@0.5=0.338 on VinDr-CXR
- **YOLOv11-MFF** (2025, PLOS ONE): Multi-scale frequency fusion → mAP@0.5=0.415
- No multi-YOLO comparison paper exists (v8 vs v9 vs v10 vs v11 under identical conditions)

**How to Differentiate**:
1. ~~Simple benchmark~~ → Add **rater-reliability-weighted bbox consensus** as novel contribution
2. Include **rater-conditioned training** where model awareness of annotator identity improves robustness
3. Compare 5+ architectures (YOLOv8s, v9, v10, v11, YOLO-CXR) under identical settings
4. Ablation: reliability-weighted consensus vs. majority-vote vs. NMS vs. WBF

**Updated Assessment**: Publishable if rater-aware training is the primary contribution, with multi-YOLO benchmark as supporting empirical evidence.

---

### Proposal 2: Transformer Detection (RT-DETR / Co-DETR) on VinDr-CXR
**🎯 Target Venue**: Medical Image Analysis / MICCAI Workshop

**Novelty Risk**: ✅ **LOW** — No DETR-family model has been evaluated on VinDr-CXR.

**Existing Prior Work Found**:
- **CD-DETR** (2025, PLOS ONE): Custom DETR for CXR → evaluated on **NIH ChestXray14** only (P=88.3%, R=86.6%)
- **CGF-DETR** (2025, arXiv): RT-DETR for pneumonia → evaluated on **RSNA Pneumonia** only (mAP@0.5=82.2%)
- **Neither** has been applied to VinDr-CXR's 14-class multi-label detection with multi-rater annotations

**How to Position**:
1. First RT-DETR/Co-DETR evaluation on VinDr-CXR (14-class bounding box detection)
2. NMS-free detection paradigm's advantage with multi-rater label noise
3. Head-to-head comparison: DETR variants vs. YOLO variants vs. EfficientDet on same dataset
4. Attention map analysis as inherent explainability

**Updated Assessment**: **Strong candidate** — clear novelty in dataset application + NMS-free advantage for noisy labels.

---

### Proposal 3: Foundation Model — GroundingDINO + SAM2 for CXR Detection *(REVISED)*
**🎯 Target Venue**: Nature Scientific Reports / Medical Image Analysis

**Novelty Risk**: ⚠️ **MEDIUM** — Grounded SAM pipeline exists for medical imaging. GroundingDINO-Med fine-tuned on VinDr exists.

**Existing Prior Work Found**:
- **GroundingDINO-Med** (2024, GitHub): Fine-tuned GroundingDINO on VinBigData → open-vocabulary 14-class detection
- **LuGSAM** (2025): Lung Grounded-SAM for ICU CXR segmentation
- **Grounded-SAM-2** (IDEA-Research): General pipeline; not applied to CXR abnormality detection+segmentation
- No systematic **few-shot study** (1%/5%/10%/25% labels) comparing foundation models vs supervised baselines on VinDr-CXR

**How to Differentiate**:
1. ~~Basic pipeline~~ → Focus on **label-efficiency study**: how much labeled data can foundation models save?
2. **Clinical prompt taxonomy**: systematic comparison of prompt strategies (class name vs. anatomical description vs. radiological finding)
3. **GroundingDINO 1.5/DINO-X + SAM2** (latest models, not yet applied to CXR)
4. Compare against fully-supervised YOLO/EfficientDet baselines at each label budget

**Updated Assessment**: Publishable with focus on few-shot label efficiency + prompt engineering. Pure pipeline paper would be weak.

---

### Proposal 4: Lightweight Edge-Deployable CXR Detection with KD *(REVISED)*
**🎯 Target Venue**: IEEE JBHI / Artificial Intelligence in Medicine

**Novelty Risk**: ⚠️ **MEDIUM** — KD for CXR classification is well-studied. KD for CXR **object detection** is underexplored.

**Existing Prior Work Found**:
- KD for CXR classification: MobileNet students, 95% parameter reduction, TFLite deployment — all exist
- **DISTL framework** (2025): Self-evolving distillation for CXR
- **No KD work specifically for CXR object detection** (bounding box regression + classification distillation)
- No edge deployment benchmarks (Jetson, TFLite) for CXR detection models

**How to Differentiate**:
1. Focus exclusively on **detection task** (bbox KD), not classification
2. Feature-level + logit-level + bbox-regression distillation
3. Actual hardware benchmarks (NVIDIA Jetson, mobile TFLite, ONNX Runtime)
4. Teacher: large RT-DETR or EfficientDet-d5 → Student: YOLOv8n or NanoDet

**Updated Assessment**: Publishable — detection KD is a clear gap. Combine with real hardware benchmarks.

---

### Proposal 5: Multi-Rater Disagreement as Uncertainty
**🎯 Target Venue**: Medical Image Analysis / MICCAI / ICLR Health Track

**Novelty Risk**: ⚠️ **MEDIUM** — Multi-rater uncertainty modeling for CXR classification exists (TwinTrack, MIDL 2026). Not applied to object detection.

**Existing Prior Work Found**:
- **TwinTrack** (MIDL 2026): Distinguishes inter-rater disagreement from noise for CXR **classification**
- Individual rater annotation training (Wellcome Open Research, 2024): Trains on per-reader labels for CXR **classification**
- Heteroscedastic noise modeling (GMM-based, 2024): For CXR **classification**
- **No work modeling multi-rater disagreement for CXR object detection** (bbox-level uncertainty)

**How to Differentiate**:
1. **Bounding-box-level uncertainty**: Model spatial disagreement between raters' bbox annotations, not just label disagreement
2. Per-detection calibrated confidence intervals reflecting inter-rater bbox IoU
3. Specific to VinDr-CXR's 3-rater training setup (unique multi-rater detection dataset)
4. Clinical triage simulation: does bbox-level uncertainty improve false negative reduction?

**Updated Assessment**: **Strong candidate** — classification uncertainty exists, but detection-level (bbox) uncertainty is genuinely novel.

---

### Proposal 6: Active Learning for Annotation-Efficient CXR Detection
**🎯 Target Venue**: IEEE TMI / MIDL / Medical Image Analysis

**Novelty Risk**: ✅ **LOW** — AL for CXR object detection is largely unexplored.

**Existing Prior Work Found**:
- AL for CXR **classification**: Well-studied (uncertainty, diversity, coreset strategies)
- AL for CXR **bounding box detection**: Minimal literature; NLP-generated "silver" annotations exist but not AL-driven
- **No AL study on VinDr-CXR** for detection
- Few-shot object detection in CXR mentioned but not formalized as AL

**How to Position**:
1. First AL study for CXR abnormality **detection** (bounding box task)
2. VinDr-CXR as benchmark: 15K training images → what's the minimum for 90% performance?
3. Class-balanced AL strategy for long-tailed medical detection
4. Compare AL strategies: uncertainty, diversity, badge, learning loss, coreset

**Updated Assessment**: **Very strong candidate** — genuinely underexplored area with practical clinical impact.

---

### Proposal 7: SAET Metadata Leakage Mitigation via Domain-Adversarial Training
**🎯 Target Venue**: Nature Scientific Reports / Radiology: AI

**Novelty Risk**: ⚠️ **MEDIUM** — Domain adversarial CXR work exists, but not specifically for VinDr-CXR SAET leakage.

**Existing Prior Work Found**:
- Domain adversarial CXR: Scanner-invariant features via DANN — exists for general CXR domain adaptation
- Adversarial CAM Guidance (2026): Forces models to ignore framing artifacts — exists
- Bias mitigation frameworks (CNN + XGBoost, demographic debiasing) — exist
- **No paper specifically reproduces and mitigates VinDr-CXR SAET leakage** with DANN
- The SAET leakage was documented in Kaggle discussion (2021) but never formally studied/mitigated in a publication

**How to Differentiate**:
1. **VinDr-CXR specific**: Formally document the SAET leakage as a case study
2. Compare: DANN vs. metadata stripping vs. adversarial CAM guidance vs. batch norm strategies
3. Cross-scanner generalization: Train Hospital A → Test Hospital B within VinDr
4. Framework generalizable to other multi-center medical imaging datasets

**Updated Assessment**: Publishable — the VinDr-specific SAET leakage has never been formally addressed in a publication. Strong case study narrative.

---

## Updated Priority Ranking

| Rank | Proposal | Novelty Risk | Why This Rank |
|---|---|---|---|
| 🥇 | **P6: Active Learning for CXR Detection** | ✅ LOW | Genuinely underexplored; no prior AL work for CXR bbox detection; high clinical value |
| 🥈 | **P2: RT-DETR/Co-DETR on VinDr-CXR** | ✅ LOW | No DETR on VinDr-CXR; clear first-mover advantage; NMS-free advantage for noisy labels |
| 🥉 | **P5: Multi-Rater Bbox Uncertainty** | ⚠️ MED | Classification uncertainty exists; bbox-level uncertainty is novel; high-impact venue potential |
| 4 | **P7: SAET Leakage Mitigation** | ⚠️ MED | VinDr-specific case study; never formally published; quick paper with strong narrative |
| 5 | **P4: Detection KD for Edge** | ⚠️ MED | Classification KD exists; detection KD is gap; needs real hardware benchmarks |
| 6 | **P3: Foundation Model Few-Shot** | ⚠️ MED | Pipeline exists; needs label-efficiency + prompt engineering angle to differentiate |
| 7 | **P1: YOLO Benchmark + Rater-Aware** | ⚠️ MED | YOLO-CXR already published; rater-aware angle is novel but incremental |

---

## Recommended Execution Strategy (Revised)

```
Phase 1 (Weeks 1–8): Strongest Novelty First
├── P6: Active Learning for CXR Detection → HIGHEST novelty, builds evaluation framework
├── P2: RT-DETR on VinDr-CXR → First DETR on this dataset, reusable pipeline
└── P7: SAET Leakage Case Study → Quick standalone paper

Phase 2 (Weeks 6–14): Novel Methodology
├── P5: Multi-Rater Bbox Uncertainty → Novel research contribution
├── P4: Detection KD → Edge deployment paper
└── Leverage models from Phase 1 as teachers/baselines

Phase 3 (Weeks 12–20): Extended Studies
├── P3: Foundation Model Few-Shot → Uses Phase 1 baselines for comparison
└── P1: Rater-Aware YOLO → Combines with P5 uncertainty framework
```

> [!TIP]
> **Strongest publication candidates (lowest risk, highest novelty)**:
> - **P6** (Active Learning for Detection) — genuinely unexplored
> - **P2** (DETR on VinDr-CXR) — clear first-mover advantage
> 
> **Combined paper opportunity**: P2 + P5 → "Uncertainty-Aware Transformer Detection with Multi-Rater Calibration on VinDr-CXR" — this would be a very strong Medical Image Analysis submission.
