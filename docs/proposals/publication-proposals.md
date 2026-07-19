# Publication-Oriented Research Proposals for VinBigData Chest X-Ray

*Prepared: July 2026 — Updated with deep novelty search results*
*Revised: July 2026 — annotated with independent verification. See `docs/analysis/verification-report.md` for full audit.*

> [!IMPORTANT]
> **Nothing below has been deleted.** Original proposals, ratings and rankings are preserved as written. Verification findings are added inline as `🔍 VERIFICATION` blocks. Where a verified finding contradicts the original text, the original is struck through or marked, not removed — the record of what was believed and why it changed is itself useful.
>
> Two proposals (P2, P1) had their core novelty claims **refuted** by independent search. One citation (TwinTrack) could not be located at all. Read the verification blocks before committing effort.

---

# 🚀 P0 (ACTIVE): YOLO26 + Measured Explanation Faithfulness on VinDr-CXR

> [!IMPORTANT]
> **This is the proposal currently being executed.** Everything below P0 is the original journal-oriented portfolio and remains valid for later cycles. P0 is a conference-scoped paper written against a hard 12-day deadline; it is deliberately smaller than P1–P7.
>
> **Full schedule, compute budget, protocol and kill criteria: [`iccit-2026-execution-plan.md`](./iccit-2026-execution-plan.md)**

**🎯 Target Venue**: **ICCIT 2026** (29th Intl. Conference on Computer and Information Technology, IEEE Bangladesh Section)
**📅 Deadline**: **July 31, 2026** · Notification Oct 15 · Conference Dec 18–20, Cox's Bazar
**💻 Compute**: Kaggle 2× T4
**Novelty Risk**: ⚠️ **MEDIUM** — benchmark half is crowded; XAI half is the contribution

### Why this exists as a separate proposal

The venue analysis changed the problem. P1–P7 were scoped for journals (Medical Image Analysis, IEEE TMI, JBHI) with months of compute. ICCIT is a 12-day window on 2× T4. **Tier must match tier**: P6 and P5 are journal-grade and should not be burned on a regional conference; a well-scoped conference paper is the right instrument here.

P0 is the defensible remnant of **P1** — which verification rated ❌ HIGH risk because ≥5 YOLO-family works now exist on VinDr. P1 is not resurrected; its benchmark framing is still dead. What survives is the combination below.

### The claim

> Train YOLOv8s / YOLO11s / YOLO26s on VinDr-CXR 14-class detection under identical conditions, then evaluate **explanation faithfulness measured against radiologist bounding boxes** — not just detection accuracy.

### Why it is defensible

1. **VinDr has ground-truth boxes.** Most CXR-XAI work uses NIH ChestX-ray14, which has none, so it can only show qualitative heatmaps. Boxes turn saliency into a *metric*: pointing game, energy-based pointing game, CAM–box IoU, deletion/insertion AUC.
2. **YOLO26 is genuinely new** (Ultralytics, released Jan 14 2026). Its **STAL** (Small-Target-Aware Label Assignment) maps onto VinDr's small-target profile — nodule/mass 39.69% small, calcification 17.67%. Its **native NMS-free** head maps onto this dataset's unsettled bbox-fusion problem. Hypothesis, not just "we ran the new model."
3. **Axis B ties the halves together**: does NMS-free produce *more faithful* explanations than NMS-based? Without this the paper is two experiments stapled together.

**Ranked claims** — lead with #2, it has the longest shelf life:

| | Claim | Strength |
|---|---|---|
| 1 | First YOLO26 evaluation on VinDr-CXR | Real, but shelf life ~months |
| 2 | First **quantified** explanation-faithfulness study on VinDr detection | **Lead with this** |
| 3 | First controlled v8/v11/YOLO26 comparison, identical settings | Weakest — ≥5 YOLO works already exist here |

### Hard constraints

> [!WARNING]
> **Do not claim SOTA.** Published bar: RT-DETR **45.3** mAP@0.5, YOLOv11-MFF **41.5**, YOLO-CXR **0.338**. A 512px `s`-scale model at 40 epochs lands below all of them. Frame as controlled comparison + explainability audit. A SOTA claim landing at 0.25 kills the paper.

> [!CAUTION]
> **The failure mode**: "we ran three YOLOs and made heatmaps" is a course project — precisely what the vendored `tariqshaban` repo already is. The measured saliency evaluation is the whole difference. **If schedule slips, cut a model, never cut the XAI metrics.**

**Compute enabler**: train on positive images only (~4.4K of 15K), the established protocol on this dataset (`sunghyunjun` did the same). ~3.4× epoch-time reduction; 3 models ≈ 6 h on one T4.

### Closest prior work

| Work | Overlap | Gap exploited |
|---|---|---|
| **MASR-DNet** (SSRN preprint, Apr 2026) | YOLO-style detector, VinDr | **Cardiomegaly only — 1 class. No XAI.** |
| **YOLO-CXR** (IEEE Access) | YOLOv8s variant, 0.338 | No version comparison, no XAI |
| **YOLOv11-MFF** (PLOS ONE 2025) | YOLOv11 variant, 0.415 | Single architecture, no XAI |
| **RT-DETR on VinBigData** (IRBM 2025) | 45.3 mAP | Transformer, no XAI |
| **Sensitivity-Oriented YOLOv11** (2026) | YOLOv11 variant | No YOLO26, no XAI |

### Relationship to the rest of the portfolio

- **Replaces P1** as the executable form of the YOLO line. P1's rater-aware half still belongs in **P5**.
- **Produces reusable assets**: a working VinDr detection pipeline, fused labels, a frozen evaluation protocol. That is the missing **Phase 0** every other proposal depends on. Executing P0 de-risks P5, P6 and P4 regardless of whether it is accepted.
- **Does not consume** P5 or P6 novelty.

### Venue note

**OMLET 2026 is not an option** — it was held June 12–14, 2026, already past. **COMPAS 2026** shares the same July 31 deadline (Oct 9–10, Dhaka University, 3rd edition) and is the fallback, but ICCIT's multiple-submission policy forbids parallel submission. Pick one; ICCIT is the stronger venue (29th edition, longest-running IEEE-sponsored conference in Bangladesh).

---

## Overview: Gap Analysis

Based on the comparative analysis of 25+ existing works **and deep search verification (July 2026)**, the following table shows the updated status of each proposed gap:

| Gap ID | Research Gap | Current State (as originally assessed) | Opportunity Status (original) | 🔍 Independent Verification (July 2026) |
|---|---|---|---|---|
| G1 | Modern YOLO systematic benchmark on VinDr-CXR | **PARTIALLY FILLED** — YOLO-CXR (YOLOv8+RefConv+ELCA) published; YOLOv11-MFF published. But no multi-YOLO comparison (v8 vs v9 vs v10 vs v11) exists. | ⚠️ Narrower gap — needs novel angle | ❌ **Worse than assessed.** Both cited works confirmed real (YOLO-CXR mAP@0.5 0.338 ✅ exact; YOLOv11-MFF all four metrics ✅ exact). But **two more missed**: Sensitivity-Oriented YOLOv11 (Applied Computer Systems 2026) and Mamba-YOLOvX (Expert Systems w/ Applications 2025). Plus a Diagnostics 14(23):2636 survey likely covering part of the benchmark. Gap largely consumed. |
| G2 | Rater-aware training | **PARTIALLY FILLED** — TwinTrack (MIDL 2026) models inter-rater disagreement. But not applied to VinDr-CXR specifically, and not for object detection. | ✅ Gap exists for detection task | ⚠️ **TwinTrack could not be located** in web or arXiv search. Gap may still exist, but the stated justification rests on an unverifiable citation. Locate or remove before citing. |
| G3 | Transformer-based detector on VinDr | **PARTIALLY FILLED** — CD-DETR (2025, NIH dataset), CGF-DETR (2025, RSNA Pneumonia). Neither evaluated on VinDr-CXR specifically. | ✅ Gap exists for VinDr-CXR | ❌ **REFUTED.** CD-DETR = PLOS ONE `pone.0323239`, which contains **Table 4: "Comparison of different methods in VinBigData dataset"** and states it used "NIH … **and a subset of the VinBigData dataset**." Separately, **RT-DETR already published on VinBigData** (IRBM 2025): 55.7% P / 43% R / **45.3% mAP**. Gap does not exist. |
| G4 | GroundingDINO + SAM/SAM2 for CXR | **PARTIALLY FILLED** — LuGSAM (ICU CXR) exists; GroundingDINO-Med fine-tuned on VinDr exists. But no systematic GroundingDINO+SAM2 few-shot study on VinDr-CXR. | ⚠️ Narrower gap — needs differentiation | ⚠️ **Narrower still.** SP-Det (arXiv 2512.04875, self-prompted dual-text fusion for multi-label lesion detection) overlaps the prompt-engineering contribution. Label-efficiency angle remains open. |
| G5 | Lightweight/edge deployment | **PARTIALLY FILLED** — General KD for CXR classification exists widely. But no KD specifically for CXR object detection with edge benchmarks. | ✅ Gap exists for detection KD | ⚪ **Unchallenged, unconfirmed.** No contradicting work surfaced, but no dedicated search pass was run either. Treat as plausible, not established. |
| G6 | Active learning for CXR detection | **PARTIALLY OPEN** — AL for CXR classification exists. AL for CXR object detection with bounding boxes is underexplored. No work on VinDr-CXR. | ✅ Strong gap | ✅ **Holds.** Independent search returned AL-for-CXR-*classification* only; no AL-for-detection on VinDr surfaced. Strongest surviving gap in the set. |
| G7 | Metadata leakage mitigation | **PARTIALLY FILLED** — Domain adversarial CXR work exists (scanner bias, artifact robustness). But no work specifically addressing VinDr-CXR SAET leakage with DANN. | ✅ Gap exists for VinDr-specific | ⚠️ **Unverified premise.** SAET leakage sourced only to a 2021 Kaggle discussion; not independently checked. Since the whole proposal hinges on "never formally studied," this needs a real search pass first. |

> [!CAUTION]
> **Method warning on the gap analysis as a whole.** The original analysis was built on 25 works. Verification surfaced **at least 7 relevant works it missed** (listed in `verification-report.md` §3). The search underpinning these gaps was not exhaustive enough to support unqualified "no work exists" claims. Every remaining ✅ should be read as "not found by our search," not "does not exist."

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

> [!WARNING]
> **🔍 VERIFICATION — Risk raised ⚠️ MEDIUM → ❌ HIGH. Do not proceed as scoped.**
>
> **Confirmed correct**: YOLO-CXR mAP@0.5 = 0.338 exact ✅ (also mAP@[.5:.95] 0.167, recall 0.365; IEEE Access). YOLOv11-MFF all four metrics exact ✅ (PLOS ONE, 24 Oct 2025, Guan/Zhang/Zhao).
>
> **Issue — the "no multi-YOLO comparison exists" premise is eroding fast.** Two further published YOLO variants on VinDr were missed:
> - **Sensitivity-Oriented YOLOv11 for Robust Multi-Label Lesion Detection in Chest X-rays** (Applied Computer Systems, 2026) — directly overlaps
> - **Mamba-YOLOvX** (Expert Systems with Applications, 2025) — an architecture family this proposal does not even consider
> - **Deep Learning-Based Object Detection Strategies for CXR** (Diagnostics 14(23):2636) — survey likely already doing part of the benchmark job
>
> That is ≥4 published YOLO variants on this dataset. The benchmark contribution is largely consumed, and a 5-architecture comparison would now be catching up rather than leading.
>
> **Second issue**: the rater-aware half is P5's contribution. Running it here as a secondary angle splits the same idea across two papers and weakens both.
>
> **Recommendation**: do not run standalone. Fold the rater-reliability-weighted consensus work into **P5**, where it is the primary contribution. If a YOLO baseline is needed, produce it as an ablation inside P5/P6 rather than as a paper.

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

> [!CAUTION]
> **🔍 VERIFICATION — Risk raised ✅ LOW → ❌ HIGH. Both core premises REFUTED. This was ranked #2; it should not be.**
>
> **Refutation 1 — "CD-DETR evaluated on NIH ChestXray14 only" is false.**
> CD-DETR is PLOS ONE `pone.0323239` ("An optimized transformer model for efficient detection of thoracic diseases in chest X-rays with multi-scale feature fusion"). It contains a dedicated **Table 4: "Comparison of different methods in VinBigData dataset"**, and its own limitations section reads: *"The current study primarily utilized the NIH Chest X-ray dataset **and a subset of the VinBigData dataset**."* A DETR-family model has already been evaluated on this data.
>
> **Refutation 2 — "No DETR-family model has been evaluated on VinDr-CXR" is false.**
> **"Transformer-Based RT-DETR Framework for Accurate Chest X-Ray Disease Detection"** (ScienceDirect S1959031825000375, IRBM 2025) reports **55.7% precision, 43% recall, 45.3% mAP** on VinBigData multi-label detection.
>
> **Consequence beyond novelty**: that 45.3 mAP **exceeds YOLOv11-MFF's 41.5**. Contribution #3 of this proposal ("head-to-head DETR vs YOLO on the same dataset") has therefore partly been run already — and the transformer already won. The intended finding is published.
>
> **Also missed**: Representation Learning with a Transformer-Based Detection Model for Localized CXR Disease (MICCAI 2024); Align Your Query (arXiv 2510.02789).
>
> **Salvage path — the proposal is not dead, but the framing is.** Drop every first-mover claim. Reframe as: *"Co-DETR / DINO-DETR on VinDr-CXR: beating the published RT-DETR baseline under multi-rater label noise."* That is a legitimate paper — beating a known number (45.3 mAP) with an explicit noise analysis — but it is incremental, not first-mover, and belongs in the lower half of the ranking. Reduced-scope version is also a good source of baselines for P5/P6.

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

> [!WARNING]
> **🔍 VERIFICATION — Risk raised ⚠️ MEDIUM → ⚠️ MEDIUM-HIGH. Half the differentiation is gone.**
>
> **Issue**: differentiator #2 (clinical prompt taxonomy / systematic prompt-strategy comparison) overlaps **SP-Det: Self-Prompted Dual-Text Fusion for Generalized Multi-Label Lesion Detection** (arXiv 2512.04875), which was not in the original analysis. Prompting as a contribution is largely consumed.
>
> **What survives**: differentiator #1, the **label-efficiency study** (1%/5%/10%/25% budgets vs fully-supervised baselines). No verified work does this on VinDr-CXR. That, not prompting, is now the whole paper.
>
> **Structural note**: the label-efficiency framing makes this a close sibling of **P6** — both ask "how little labeled data suffices for VinDr detection?" P6 answers via acquisition strategy, P3 via foundation-model priors. Running both separately risks self-overlap. Consider merging P3 as the zero-shot/foundation-model arm of P6's label-budget curve — one stronger paper instead of two thin ones, and it amortizes the expensive budget-sweep training runs across both contributions.

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

> [!NOTE]
> **🔍 VERIFICATION — Risk unchanged ⚠️ MEDIUM. Unchallenged, but also unconfirmed.**
>
> **Status**: no contradicting prior work surfaced during verification. But no dedicated search pass was run on detection-KD either — this proposal was not the focus of the audit. The ✅ on gap G5 means "not found," not "does not exist." **Run a proper search before committing.**
>
> **Unaddressed blocker — hardware.** The proposal requires NVIDIA Jetson, mobile TFLite and ONNX Runtime benchmarks. The repository establishes no access to any of this hardware. Real device numbers are the entire differentiator versus generic KD work; without them this collapses into "we distilled a detector," which is weak. Confirm hardware access before this can be ranked at all.
>
> **Dependency**: teacher model (large RT-DETR or EfficientDet-d5) must exist first. That makes P4 downstream of a working baseline pipeline — see Phase 0 below. Note the now-published RT-DETR VinBigData result (45.3 mAP, IRBM 2025) gives a concrete teacher-quality target to match before distilling.

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

> [!WARNING]
> **🔍 VERIFICATION — Risk unchanged ⚠️ MEDIUM, but positioning is unstable. Promoted to #2 on merit; fix the citation first.**
>
> **Issue — TwinTrack (MIDL 2026) could not be located.** No trace in web or arXiv search. This citation is load-bearing twice over: it justifies gap **G2**, and it supplies the entire "classification uncertainty exists, bbox uncertainty does not" contrast that this proposal's novelty claim rests on. If TwinTrack does not exist, the related-work section is arguing against a phantom baseline; if it does, it needs a real reference. **Do not cite until located.** The other two cited works (individual-rater training, Wellcome Open Research 2024; heteroscedastic GMM noise modeling 2024) were not verified either.
>
> **What survives regardless**: the underlying asset is real and rare. VinDr-CXR's 15,000-image training set with **3 independent radiologists per image and bounding boxes** is an unusual resource, and no verified work exploits inter-rater *spatial* disagreement at bbox level. The idea does not depend on TwinTrack — only the framing does.
>
> **Absorbs P1**: the rater-reliability-weighted consensus and rater-conditioned training from Proposal 1 belong here as primary contributions, not scattered into a YOLO benchmark paper.
>
> **Caveat on the test set**: train is 3-rater, test is 5-rater *consensus* — the disagreement signal this proposal models is present in train but collapsed in test. Evaluation design needs to address this explicitly; it is not currently mentioned.
>
> **Action**: rewrite related work without TwinTrack, then commit. Feasibility is good — no special hardware, and it reuses whatever detector Phase 0 produces.

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

> [!TIP]
> **🔍 VERIFICATION — Risk ✅ LOW confirmed. The only proposal whose novelty survived verification intact. Remains #1 — with one serious caveat.**
>
> **Confirmed**: independent search returned AL-for-CXR-**classification** work only (uncertainty, diversity, coreset). No AL-for-CXR-**detection** study surfaced, and none on VinDr-CXR. The gap holds as stated.
>
> **⚠️ Caveat — this is the most compute-hungry proposal in the entire set, and that was never assessed.** The proposed design is 5 acquisition strategies × ~5 label budgets = **≥25 full detector training runs**, plus baselines. For calibration, the vendored `sunghyunjun` reference needed **35 trained models on a V100** to reach 0.253 mAP, and reported that batch size < 4 destroys mAP while larger images help but exhaust VRAM. Detection training on this dataset is not cheap.
>
> Ranking by novelty alone put the least-executable proposal first. This is not an argument to demote it — it is the best idea here — but **rank 1 is conditional on a stated GPU budget.** If compute is a single consumer GPU, either scope down (3 strategies × 3 budgets, single fold, reduced resolution) or promote **P7** (cheapest paper in the set) to run first while compute is arranged.
>
> **Cost-sharing opportunity**: the label-budget sweep this requires is the same sweep **P3** needs. Merging P3 in as the foundation-model arm amortizes the expensive part across two contributions.
>
> **Dependency**: absolutely requires a working, validated detection pipeline before any AL loop can run. See Phase 0.

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

> [!NOTE]
> **🔍 VERIFICATION — Risk unchanged ⚠️ MEDIUM. Premise unverified, but cheapest paper in the set.**
>
> **Issue — the founding premise was never independently checked.** "The SAET leakage was documented in Kaggle discussion (2021) but never formally studied/mitigated in a publication" traces to a single forum thread. This audit did not verify it, and given that verification found **7 works the original analysis missed**, an unchecked "never published" claim is exactly the kind of statement that has already failed twice here (see P2). **Run a systematic search with a logged query set before investing.**
>
> **If the premise holds, this should probably run first.** It is the cheapest proposal here — the leakage demonstration is gradient boosting on DICOM headers (minutes, not GPU-days), and DANN vs. metadata-stripping vs. adversarial-CAM comparison is far lighter than a detection budget sweep. Strong narrative, low compute, fast turnaround. Good candidate to run *while* GPU capacity for P6 is being arranged.
>
> **Data check needed**: contribution #3 (train Hospital A → test Hospital B within VinDr) assumes hospital-of-origin is recoverable per image. VinDr is sourced from Hanoi Medical University Hospital and 108 Military Central Hospital, but whether the public release exposes a per-image site label — or whether SAET is the only proxy for it — is unconfirmed. If SAET *is* the only site proxy, then training a site-invariant model and evaluating cross-site are circular. Verify before designing the experiment.

---

## Priority Ranking

### ~~Original Ranking~~ *(superseded — retained for record)*

*This ranking was produced before independent verification. **P2's rank of #2 rests on a claim now refuted** (see P2 verification block). Retained so the reasoning change is traceable; do not execute against it.*

| Rank | Proposal | Novelty Risk | Why This Rank |
|---|---|---|---|
| 🥇 | **P6: Active Learning for CXR Detection** | ✅ LOW | Genuinely underexplored; no prior AL work for CXR bbox detection; high clinical value |
| 🥈 | **P2: RT-DETR/Co-DETR on VinDr-CXR** | ✅ LOW | ❌ *Refuted — no DETR on VinDr-CXR; clear first-mover advantage; NMS-free advantage for noisy labels* |
| 🥉 | **P5: Multi-Rater Bbox Uncertainty** | ⚠️ MED | Classification uncertainty exists; bbox-level uncertainty is novel; high-impact venue potential |
| 4 | **P7: SAET Leakage Mitigation** | ⚠️ MED | VinDr-specific case study; never formally published; quick paper with strong narrative |
| 5 | **P4: Detection KD for Edge** | ⚠️ MED | Classification KD exists; detection KD is gap; needs real hardware benchmarks |
| 6 | **P3: Foundation Model Few-Shot** | ⚠️ MED | Pipeline exists; needs label-efficiency + prompt engineering angle to differentiate |
| 7 | **P1: YOLO Benchmark + Rater-Aware** | ⚠️ MED | YOLO-CXR already published; rater-aware angle is novel but incremental |

### ✅ Revised Ranking (post-verification, July 2026)

Ranked by **verified novelty × feasibility** rather than novelty alone. No proposal removed.

| Rank | Proposal | Verified Risk | Δ | Reason for placement / why not to proceed as-is |
|---|---|---|---|---|
| 1 | **P6: Active Learning for CXR Detection** | ✅ LOW | = | Only proposal whose novelty survived verification intact. No AL-for-detection work found on VinDr. **Conditional**: most compute-hungry item here (~25 full training runs); rank 1 assumes a stated GPU budget. Scope down or reorder if compute is a single consumer GPU. |
| 2 | **P5: Multi-Rater Bbox Uncertainty** | ⚠️ MED | ▲1 | Core asset is real and rare — 3-rater bbox annotations, no verified work exploits spatial disagreement. Promoted by P2's fall. **Blocker**: TwinTrack citation unlocatable; rewrite related work without it first. Also must address 3-rater-train vs 5-rater-consensus-test asymmetry. Absorbs P1's rater-aware contribution. |
| 3 | **P7: SAET Leakage Mitigation** | ⚠️ MED | ▲1 | Cheapest paper in the set, strong narrative, minimal GPU. **Blockers**: "never formally studied" premise unverified; and cross-hospital split may be circular if SAET is the only site proxy. Verify both, then this could jump to #1 on feasibility — good work to run while P6 compute is arranged. |
| 4 | **P4: Detection KD for Edge** | ⚠️ MED | ▲1 | Unchallenged by verification, but never positively searched either — gap status is "not found," not "established." **Blocker**: requires Jetson/mobile hardware the repo has not established access to; without real device numbers the contribution collapses to generic KD. Downstream of a working teacher model. |
| 5 | **P3: Foundation Model Few-Shot** | ⚠️ MED-HIGH | ▲1 | Prompt-taxonomy differentiator consumed by SP-Det (arXiv 2512.04875). Only the label-efficiency study survives — which overlaps P6's core question. **Recommendation**: merge in as the foundation-model arm of P6's label-budget curve rather than running standalone. |
| 6 | **P2: RT-DETR/Co-DETR on VinDr-CXR** | ❌ HIGH | ▼4 | **Both premises refuted.** CD-DETR has a VinBigData comparison table; RT-DETR already published on VinBigData at 45.3 mAP — exceeding YOLOv11-MFF's 41.5, so the DETR-vs-YOLO finding is already in print. **Do not run as scoped.** Salvage only as "Co-DETR/DINO-DETR vs. the published 45.3 baseline + multi-rater noise analysis" — beating a number, not claiming a first. Useful as a baseline generator for P5/P6. |
| 7 | **P1: YOLO Benchmark + Rater-Aware** | ❌ HIGH | ▼6 | ≥4 published YOLO variants now on VinDr (YOLO-CXR, YOLOv11-MFF, Sensitivity-Oriented YOLOv11 2026, Mamba-YOLOvX 2025) plus a Diagnostics survey. Benchmark value consumed; a 5-way comparison would be catching up, not leading. Rater-aware half belongs to P5. **Do not run standalone — fold into P5, keep YOLO runs as ablations.** |

**Two blockers gate this ordering.** State the available compute budget (decides P6 vs P7 for rank 1), and verify the SAET premise (could move P7 to #1 on feasibility).

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
> - ~~**P2** (DETR on VinDr-CXR) — clear first-mover advantage~~ ❌ *refuted, see P2 block*
> 
> **Combined paper opportunity**: P2 + P5 → "Uncertainty-Aware Transformer Detection with Multi-Rater Calibration on VinDr-CXR" — this would be a very strong Medical Image Analysis submission.
>
> 🔍 *The P2+P5 combination still works, but the framing must change: with RT-DETR already published on VinBigData at 45.3 mAP, the transformer half is no longer a first-mover claim. The paper's contribution becomes the **multi-rater calibration**, with the transformer as the vehicle — i.e. P5 leading, not P2.*

---

## 🔍 Missing Phase 0: Baseline Reproduction & Protocol *(added post-verification)*

The execution plan above jumps straight to novel contributions. **Every proposal in this document depends on a working VinDr detection pipeline that does not exist yet** — the repository currently contains four vendored reference implementations and three analysis documents, no original code, and no reproduced baseline. This shared dependency is absent from Phases 1–3 and is realistically **2–4 weeks** on its own.

```
Phase 0 (Weeks 1–4): Foundation — BLOCKS EVERYTHING ELSE
├── Reproduce a VinDr detection baseline (target: sunghyunjun's 0.253 mAP@0.4 range)
├── Fix the evaluation protocol (see below) — decide once, apply everywhere
├── Verify unverified premises: SAET leakage (P7), detection-KD gap (P4), TwinTrack (P5)
├── State compute budget → re-rank P6 vs P7 for first slot
└── Confirm hardware access for P4 (Jetson / mobile) or drop its device-benchmark claims
```

**Evaluation protocol must be decided before any result is generated.** Currently unspecified, and it determines whether any number produced here is comparable to the literature:

| Ambiguity | Competing options in play |
|---|---|
| Metric | Competition **mAP@0.4** (0.253, 0.314) vs. literature **mAP@0.5** (0.338, 0.415, 0.453) vs. **mAP@0.5:0.95** (0.167, 0.226) |
| Split | Kaggle public/private LB vs. local CV vs. a held-out split of the 15K train set |
| Label handling | 3-rater train (fusion required) vs. 5-rater consensus test — different labeling processes, not interchangeable |
| BBox fusion | NMS vs. WBF — **note**: the vendored reference's own table shows WBF *winning* on private LB (0.185 vs 0.181) while losing on CV; that choice was made by reasoning, not measurement. Do not inherit it as settled. |

Reporting mAP@0.5 and comparing against a 0.253 mAP@0.4 number — or vice versa — will invalidate the comparison. Pick one, state it, keep it.
