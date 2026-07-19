# Publication-Oriented Research Proposals for VinBigData Chest X-Ray

*Prepared: July 2026 — Actionable research directions with clear publication targets*

---

## Overview: Gap Analysis

Based on the comparative analysis of 25+ existing works, the following **unaddressed or underexplored gaps** present viable publication opportunities:

| Gap ID | Research Gap | Current State | Opportunity |
|---|---|---|---|
| G1 | No YOLO v8/v10/v11 + modern attention systematic benchmark on VinDr-CXR | Only YOLOv5, v7, v11-MFF exist; no unified comparison | First comprehensive modern YOLO benchmark |
| G2 | Rater-aware training is unexplored | All works fuse raters naively (NMS/WBF) or ignore rater IDs | Novel rater-conditioned learning framework |
| G3 | No transformer-based detector evaluated | Only CNN-based detectors (EfficientDet, YOLO) | DETR/DINO/Co-DETR on medical detection |
| G4 | Foundation model fine-tuning gap | GroundingDINO-Med exists but shallow; no SAM/DINO+SAM | Systematic foundation model adaptation study |
| G5 | No lightweight/edge-deployable model study | tariqshaban mentions mobile but no actual work done | Efficient architecture search for edge |
| G6 | Active learning / annotation efficiency unexplored | All works use full 15K training set | Reduce annotation cost via AL strategies |
| G7 | Explainability beyond saliency maps | MedFocus shows standard attribution fails causal tests | Concept-based or prototype-based explanations |
| G8 | Multi-rater disagreement as training signal | Rater disagreement treated as noise to suppress | Leverage disagreement as uncertainty signal |

---

## Proposed Research Directions

### Proposal 1: Modern YOLO Architecture Benchmark with Rater-Aware Training
**🎯 Target Venue**: PLOS ONE / IEEE Access / Computers in Biology and Medicine

**Novelty**: First systematic evaluation of YOLOv8, YOLOv9, YOLOv10, and YOLO11 on VinDr-CXR with a novel rater-aware bbox consensus mechanism that weights annotations by individual radiologist reliability scores.

**Methodology**:
1. Implement all modern YOLO architectures under identical conditions
2. Compute per-radiologist reliability scores based on inter-rater agreement
3. Replace naive NMS fusion with reliability-weighted soft consensus
4. Evaluate at multiple resolutions (512, 640, 768, 1024, 1280)
5. Ablate: rater-weighted vs. uniform vs. majority-vote vs. ZFTurbo NMS

**Metrics**: mAP@0.4, mAP@0.5, mAP@0.5:0.95, per-class AP, precision, recall, F1, inference FPS

**Expected Contribution**:
- Benchmark table comparing 5+ YOLO architectures on medical detection
- Novel rater-aware training protocol that outperforms naive fusion
- Open-source codebase for reproducibility

**Feasibility**: ⭐⭐⭐⭐⭐ (High — uses existing dataset and off-the-shelf architectures)
**Impact**: ⭐⭐⭐ (Moderate — incremental but solid engineering contribution)
**Timeline**: 6–8 weeks

---

### Proposal 2: Transformer-Based Detection (RT-DETR / Co-DETR) for Chest X-Ray Abnormalities
**🎯 Target Venue**: Medical Image Analysis / IEEE TMI / MICCAI Workshop

**Novelty**: First application of end-to-end transformer detectors (RT-DETR, Co-DETR, DINO-DETR) to the VinDr-CXR benchmark, eliminating NMS post-processing and leveraging global attention for multi-scale pathology detection.

**Methodology**:
1. Adapt RT-DETR, Co-DETR, and Deformable-DETR for the 14-class VinDr task
2. Compare against EfficientDet-d4/d5 and YOLOv11-MFF baselines
3. Leverage multi-scale deformable attention for small lesion detection (nodules, calcifications)
4. Analyze attention maps as inherent explainability (no separate saliency method needed)
5. Evaluate NMS-free paradigm's impact on multi-rater label noise

**Metrics**: mAP@0.4, mAP@0.5, per-class AP (especially small objects: class 2, 8), attention visualizations

**Expected Contribution**:
- First DETR-family results on VinDr-CXR
- Demonstrate that NMS-free detection handles multi-rater noise better
- Attention-based explainability as a clinical side-benefit

**Feasibility**: ⭐⭐⭐⭐ (Good — RT-DETR has efficient implementations)
**Impact**: ⭐⭐⭐⭐ (High — strong narrative for medical AI)
**Timeline**: 8–12 weeks

---

### Proposal 3: Foundation Model Adaptation — GroundingDINO + SAM2 for Zero-Shot and Few-Shot CXR Detection
**🎯 Target Venue**: Nature Scientific Reports / MICCAI 2027 / Medical Image Analysis

**Novelty**: A unified pipeline combining text-prompted detection (GroundingDINO) with automatic segmentation (SAM2) for zero-shot and few-shot chest X-ray abnormality detection, including prompt engineering strategies for clinical vocabulary.

**Methodology**:
1. Evaluate GroundingDINO zero-shot with clinical prompts vs. simple class names
2. Chain GroundingDINO → SAM2 for detection + segmentation pipeline
3. Few-shot adaptation: fine-tune with 1%, 5%, 10%, 25% labeled data
4. Compare against fully-supervised EfficientDet and YOLO baselines
5. Clinical prompt taxonomy: anatomical descriptors vs. class names vs. radiological descriptions

**Metrics**: mAP@0.4, mAP@0.5, Dice (segmentation), detection rate per class, prompt sensitivity analysis

**Expected Contribution**:
- First GroundingDINO+SAM2 pipeline for CXR abnormality detection and segmentation
- Clinical prompt engineering guidelines for medical object detection
- Demonstrates that foundation models can match supervised baselines with <25% labels

**Feasibility**: ⭐⭐⭐⭐ (Good — pretrained models available)
**Impact**: ⭐⭐⭐⭐⭐ (Very High — foundation models are trending topic)
**Timeline**: 10–14 weeks

---

### Proposal 4: Lightweight Edge-Deployable CXR Detection with Knowledge Distillation
**🎯 Target Venue**: IEEE JBHI / Computers in Biology and Medicine / Artificial Intelligence in Medicine

**Novelty**: Design an efficient, mobile-deployable chest X-ray detector using knowledge distillation from a large teacher ensemble to a compact student model, targeting real-time inference on edge devices.

**Methodology**:
1. Teacher: Ensemble of EfficientDet-d5 + YOLOv11 + RT-DETR (from Proposals 1–2)
2. Student architectures: YOLOv8n, MobileNetV3-SSD, EfficientDet-d0, NanoDet-Plus
3. Distillation: Feature-level + logit-level + bbox-level KD
4. Quantization: INT8 and FP16 for mobile deployment
5. Benchmark on CPU, mobile GPU (TFLite), and NVIDIA Jetson

**Metrics**: mAP@0.4, inference latency (ms), model size (MB), FLOPs, power consumption

**Expected Contribution**:
- Production-ready mobile CXR screening model
- KD recipe for medical object detection (teacher → student transfer)
- Edge deployment benchmarks (Jetson, TFLite, ONNX Runtime)

**Feasibility**: ⭐⭐⭐⭐ (Good — standard ML pipeline)
**Impact**: ⭐⭐⭐⭐ (High — clinical deployment narrative)
**Timeline**: 10–12 weeks

---

### Proposal 5: Multi-Rater Disagreement as Uncertainty — Probabilistic Detection with Calibrated Confidence
**🎯 Target Venue**: Medical Image Analysis / MICCAI / ICLR (Health Track)

**Novelty**: Instead of suppressing multi-rater disagreement, model it as aleatoric uncertainty. Train detectors that output calibrated confidence intervals reflecting inter-rater variability.

**Methodology**:
1. Compute per-image rater agreement statistics (Fleiss' κ, IoU between raters)
2. Design uncertainty-aware loss: higher loss weight where raters agree, softer targets where they disagree
3. Output: per-detection confidence + calibrated uncertainty estimate
4. Evaluate: Expected Calibration Error (ECE), reliability diagrams, uncertainty-quality correlation
5. Clinical simulation: Does uncertainty-aware triage reduce false negatives?

**Metrics**: mAP@0.4, ECE, Brier score, uncertainty-mAP correlation, triage simulation

**Expected Contribution**:
- First work to explicitly model multi-rater disagreement as useful signal (not noise)
- Calibrated uncertainty estimates for clinical decision support
- Framework applicable to any multi-rater medical imaging dataset

**Feasibility**: ⭐⭐⭐ (Moderate — requires careful probability modeling)
**Impact**: ⭐⭐⭐⭐⭐ (Very High — novel research direction)
**Timeline**: 12–16 weeks

---

### Proposal 6: Active Learning for Annotation-Efficient CXR Detection
**🎯 Target Venue**: IEEE TMI / MIDL / Medical Image Analysis

**Novelty**: Determine the minimum annotation budget needed to achieve competitive detection performance on VinDr-CXR using active learning strategies tailored for multi-label object detection with class imbalance.

**Methodology**:
1. Simulate annotation scenarios: 5%, 10%, 25%, 50%, 100% of training data
2. AL strategies: uncertainty sampling, diversity sampling, badge, learning loss, coreset
3. Class-balanced AL: prioritize underrepresented classes in selection
4. Compare against random sampling and full-data baselines
5. Evaluate annotation cost vs. performance trade-off curves

**Metrics**: mAP@0.4 at each budget level, annotation efficiency curves, per-class AP evolution

**Expected Contribution**:
- First active learning study for CXR abnormality detection
- Practical guidelines: "X% of labels achieves Y% of full performance"
- Class-balanced AL strategy for long-tailed medical detection

**Feasibility**: ⭐⭐⭐⭐ (Good — well-studied AL methods)
**Impact**: ⭐⭐⭐⭐ (High — practical clinical annotation cost reduction)
**Timeline**: 10–14 weeks

---

### Proposal 7: Metadata Leakage Mitigation — Domain-Adversarial Training for Scanner-Invariant CXR Detection
**🎯 Target Venue**: Nature Scientific Reports / Medical Image Analysis / Radiology: AI

**Novelty**: Explicitly address the documented SAET/scanner metadata leakage in VinDr-CXR using domain-adversarial neural networks (DANN) to learn scanner-invariant features.

**Methodology**:
1. Quantify leakage: reproduce gradient-boosting on DICOM metadata (age, sex, SAET)
2. Domain-adversarial training: adversary predicts SAET; detector learns scanner-invariant features
3. Evaluate on cross-scanner generalization (train on Hospital A, test on Hospital B)
4. Compare: Standard training vs. DANN vs. metadata-stripped training vs. batch normalization strategies
5. External validation on NIH ChestXray14 and CheXpert

**Metrics**: mAP@0.4 (same-scanner, cross-scanner), AUROC, scanner prediction accuracy (should decrease)

**Expected Contribution**:
- First systematic study of metadata leakage mitigation in VinDr-CXR
- Scanner-invariant model that generalizes across institutions
- Generalizable framework for medical imaging dataset debiasing

**Feasibility**: ⭐⭐⭐⭐ (Good — DANN is well-established)
**Impact**: ⭐⭐⭐⭐⭐ (Very High — addresses critical clinical deployment concern)
**Timeline**: 8–12 weeks

---

## Priority Ranking for Publication

| Rank | Proposal | Venue Tier | Novelty | Feasibility | Time | Recommended Priority |
|---|---|---|---|---|---|---|
| 🥇 | **P3: Foundation Model (GroundingDINO+SAM2)** | Q1 / Top Conference | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 10–14 wk | **Start first** — trending topic, high impact |
| 🥈 | **P5: Multi-Rater Uncertainty** | Q1 / Top Conference | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 12–16 wk | **Start in parallel** — novel research contribution |
| 🥉 | **P7: Metadata Leakage Mitigation** | Q1 / Q2 Journal | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8–12 wk | **Quick win** — addresses documented critical issue |
| 4 | **P2: Transformer Detection (RT-DETR)** | Q1 Journal / Workshop | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8–12 wk | Strong standalone or combined with P1 |
| 5 | **P1: Modern YOLO Benchmark** | Q2/Q3 Journal | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 6–8 wk | Easiest first paper; build codebase for other proposals |
| 6 | **P6: Active Learning** | Q1/Q2 Journal | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 10–14 wk | Practical clinical contribution |
| 7 | **P4: Edge Deployment + KD** | Q2 Journal | ⭐⭐⭐ | ⭐⭐⭐⭐ | 10–12 wk | Depends on having strong teacher models (from P1/P2) |

---

## Recommended Execution Strategy

```
Phase 1 (Weeks 1–8): Foundation Building
├── P1: Modern YOLO Benchmark → builds reusable codebase
├── P7: Metadata Leakage study → quick independent paper
└── Setup: data pipelines, evaluation framework

Phase 2 (Weeks 6–14): Core Research
├── P3: GroundingDINO + SAM2 → highest impact paper
├── P2: RT-DETR adaptation → reuses Phase 1 infrastructure
└── P5: Multi-Rater Uncertainty → parallel novel research

Phase 3 (Weeks 12–20): Advanced Topics
├── P6: Active Learning → builds on Phase 1–2 models
└── P4: Edge Deployment → distills best models from above
```

> **Combined Paper Strategy**: Proposals P1 + P2 can be merged into a single comprehensive benchmark paper comparing CNN-based (YOLO family) vs. Transformer-based (DETR family) detectors, significantly strengthening the contribution for a Q1 journal.
