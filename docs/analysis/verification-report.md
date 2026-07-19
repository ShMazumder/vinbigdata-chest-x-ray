# Verification Report: Repository, Prior Works, and Proposal Novelty

*Audit date: July 2026. Method: direct reading of vendored implementation code/READMEs; web + arXiv search against every load-bearing literature claim in `comparative-analysis.md` and `publication-proposals.md`.*

> [!WARNING]
> **Headline finding:** Proposal 2 (RT-DETR/Co-DETR on VinDr-CXR), currently ranked #2 with "✅ LOW novelty risk," rests on a claim that is **factually false**. Two transformer detectors — including RT-DETR itself — have already been published on VinBigData. P2 should be downgraded to HIGH risk or substantially reframed.

---

## 1. Repository State

| Claim in `README.md` | Reality | Status |
|---|---|---|
| `src/` with data/models/training/evaluation/utils | Does not exist | ❌ |
| `notebooks/` | Does not exist | ❌ |
| `experiments/` (configs/logs/checkpoints) | Does not exist | ❌ |
| `results/` (figures/tables) | Does not exist | ❌ |
| `docs/literature-review/` | Does not exist | ❌ |
| `docs/walkthrough.md` | Exists, undocumented in README | ⚠️ |

The README presents an aspirational layout in the indicative mood. Anyone cloning this expects code and finds none. Either mark the tree as planned, or scaffold the directories with `.gitkeep`.

**No original code exists yet.** The repo is currently four vendored reference implementations plus three analysis documents. Every performance number in the analysis docs is second-hand; nothing has been reproduced locally.

---

## 2. Verification of Previous Implementations (code-level)

Checked `comparative-analysis.md` Tables 1A/2/3 against the actual vendored `sunghyunjun` README and source tree.

### Confirmed ✅
- Two-stage (EfficientNet CLF + EfficientDet DET) + one-stage 15-class blend
- 15 classifiers + 18 detectors + 2 one-stage detectors
- AdamW, CosineAnnealingLR, 50 epochs, StratifiedKFold-5, FP16, V100 16GB
- Checkpoint selection: max mAP among top-3 min val_loss
- CLF image sizes 456–1024; DET image sizes 768–1024
- Augmentation list (Resize/RandomScale/Crop/Brightness/ChannelDropout/Blur/HFlip) matches exactly
- Failed experiments (Freeze BN, GroupNorm, Downconv, NIH concat, aspect ratios) match exactly
- Private LB 0.253 final blended submission

### Errors found ❌

| # | Location | Claim | Actual | Severity |
|---|---|---|---|---|
| E1 | Table 1A | "0.259 (two-stage best)" | 0.259 is the **two-stage + one-stage blend** at thr 0.65–0.30. Two-stage-only best was **0.256**. | High — misattributes the best score to the wrong system |
| E2 | Table 1A | "bs=3–12" | Actual range **2–16** (CLF 4–16, DET 2–4). Contradicts this doc's own Section 2. | Medium |
| E3 | Table 1A | "lr=3e-4–7.5e-5" | Garbled merge of two disjoint ranges: CLF **2.5e-5–1e-4**, DET **2e-4–4e-4** | Medium |
| E4 | Table 1A | "1024px JPG … DET: 768–1024" | Correct, but the headline NMS ablation was run at **d4/896px/30 epochs**, not 50 | Low |
| E5 | Title | "76th place" | README says 76th; the linked submission notebook is titled "91th place" | Low |

### Interpretation error ⚠️ (affects design decisions)

`dataset-research-notes.md` states as flat fact: **"ZFTurbo NMS > torchvision NMS > WBF."**

The source table does not support this:

| Fusion | CV (mAP@0.4) | Public LB | Private LB |
|---|---|---|---|
| torchvision batched_nms | 0.4317 | 0.155 | 0.168 |
| ZFTurbo nms | 0.4419 | 0.164 | 0.181 |
| ZFTurbo wbf | 0.4158 | 0.157 | **0.185** |

**WBF won on private LB.** NMS won on CV. The author explicitly wrote that "direct comparison of local cv was not possible, and LB was also somewhat difficult to use as a criterion," and chose NMS on a *reasoning* argument about test-set labeling, not a measurement. Propagating this as settled fact risks baking a wrong default into new work. It should read: *NMS wins on CV; WBF wins on private LB; n=1, undecided.*

### Metric incomparability ⚠️

`comparative-analysis.md` §3 "Results Comparison" places competition **mAP@0.4** scores (0.253, 0.314) in the same table as **mAP@0.5** scores (YOLOv11-MFF 41.5%) and **AUROC** (ConvNeXtV2-ViT 0.9525). These are three different quantities on different task formulations (detection vs. classification). As printed, the table invites the false read that YOLOv11-MFF beat the competition winner by 10 points. Split by metric or add explicit non-comparability notes.

Also: the "Citations (est.)" column throughout all five tables is unsourced estimation presented in a data column. Either pull real counts from Semantic Scholar/Scopus or delete the column — it cannot be cited in a paper as-is.

---

## 3. Literature Verification

### Verified ✅

| Work | Verification |
|---|---|
| **YOLOv11-MFF** | PLOS ONE, 24 Oct 2025, Guan/Zhang/Zhao. All four metrics exact (P 48.2, R 42.5, mAP@0.5 41.5, mAP@0.5:0.95 22.6) ✅ |
| **YOLO-CXR** | IEEE Access. mAP@0.5 = 0.338 exact ✅ (also mAP@[.5:.95] 0.167, recall 0.365) |
| **Who Gets Missed in the Tail?** | arXiv 2607.07717, 4 July 2026 ✅ |
| **VinDr-CXR-VQA** | arXiv 2511.00504; 17,597 Q-A pairs confirmed ✅ |
| **CGF-DETR** | arXiv 2511.01730 ✅ |
| **PaCX-MAE** | arXiv 2606.01537 — exists ⚠️ but see below |

### Refuted ❌ — these change the conclusions

**R1. "CD-DETR … evaluated on NIH ChestXray14 only" — FALSE.**

CD-DETR is PLOS ONE `pone.0323239` ("An optimized transformer model for efficient detection of thoracic diseases in chest X-rays with multi-scale feature fusion"). It evaluates on **NIH *and* VinBigData**, and contains a dedicated **Table 4: "Comparison of different methods in VinBigData dataset."** The paper's own limitations section states it "utilized the NIH Chest X-ray dataset **and a subset of the VinBigData dataset**."

This is the single load-bearing sentence under P2's "✅ LOW" novelty rating.

**R2. RT-DETR has already been run on VinBigData — FALSE that "no DETR-family model has been evaluated on VinDr-CXR."**

"Transformer-Based RT-DETR Framework for Accurate Chest X-Ray Disease Detection" (ScienceDirect S1959031825000375, IRBM 2025) reports **55.7% precision, 43% recall, 45.3% mAP** on VinBigData multi-label detection. Note this mAP **exceeds YOLOv11-MFF's 41.5%** — meaning the transformer-vs-YOLO comparison P2 proposes as its contribution has partly been run, and the transformer already won.

### Unverifiable ⚠️

**U1. "TwinTrack (MIDL 2026)"** — no trace in web or arXiv search. This citation is load-bearing for gap **G2** and for **P5**'s novelty framing ("classification uncertainty exists, but bbox-level is novel"). If TwinTrack does not exist, P5's positioning is built on a phantom baseline; if it does, it needs a real reference. **Do not cite until located.**

**U2. PaCX-MAE details mismatch.** Paper is real, but `comparative-analysis.md` lists datasets as "VinDr-CXR, MedMod, MIMIC-CXR" while the paper's abstract lists TB, VinDr-CXR, NIH-14, ChestX6, CheXchoNet. The claimed "+6.5 F1 on VinDr; +2.7 AUROC on MedMod" could not be confirmed. Also, the doc describes VinDr as 14 classes here but the paper cites 28 labels — the 22-finding/14-class distinction is being blurred (the repo README says 22 findings, `dataset-research-notes.md` says 14+1; both are true at different granularities but the docs never disambiguate).

**U3. SAET metadata leakage (P7 foundation).** Sourced only to a 2021 Kaggle discussion. Not independently verified in this audit. Since P7's entire premise is "never formally studied," this needs a proper systematic search before committing.

### Prior work the analysis missed 🔍

Surfaced incidentally during verification; none appear in `comparative-analysis.md`:

- **Sensitivity-Oriented YOLOv11 for Robust Multi-Label Lesion Detection in Chest X-rays** (Applied Computer Systems, 2026) — directly overlaps P1
- **Mamba-YOLOvX** for CXR abnormality localization (Expert Systems with Applications, 2025) — a non-YOLO/non-DETR architecture family P1 doesn't consider
- **BarlowTwins-CXR** (arXiv 2402.06499) — SSL for CXR abnormality *localization*, cross-domain
- **SP-Det: Self-Prompted Dual-Text Fusion for Generalized Multi-Label Lesion Detection** (arXiv 2512.04875) — overlaps P3's prompting angle
- **Align Your Query: Representation Alignment for Multimodality Medical Object Detection** (arXiv 2510.02789)
- **Representation Learning with a Transformer-Based Detection Model for Localized Chest X-Ray Disease** (MICCAI 2024)
- **Deep Learning-Based Object Detection Strategies for Disease Detection and Localization in Chest X-Ray** (Diagnostics 14(23):2636) — a survey that likely already does part of P1's benchmark job

The gap analysis was built on 25 works; at least 7 relevant ones were not found. The search was not exhaustive enough to support "no work exists" claims.

---

## 4. Proposal Validation — Revised Risk Table

| # | Proposal | Doc's rating | **Verified rating** | Basis |
|---|---|---|---|---|
| P6 | Active Learning for CXR detection | ✅ LOW | **✅ LOW (holds)** | Independent search found AL-for-CXR-*classification* only; no AL-for-detection on VinDr surfaced. Strongest surviving claim. |
| P2 | RT-DETR/Co-DETR on VinDr-CXR | ✅ LOW | **❌ HIGH** | R1 + R2. Both premises refuted. RT-DETR already published on VinBigData at 45.3 mAP. |
| P5 | Multi-rater bbox uncertainty | ⚠️ MED | **⚠️ MED (unstable)** | Core idea plausible and genuinely underexplored, but positioning depends on unverifiable TwinTrack (U1). |
| P7 | SAET leakage mitigation | ⚠️ MED | **⚠️ MED (unverified)** | Premise never independently checked (U3). |
| P4 | Detection KD for edge | ⚠️ MED | **⚠️ MED (holds)** | No contradicting evidence found; also not positively verified. |
| P3 | Foundation model few-shot | ⚠️ MED | **⚠️ MED→HIGH** | SP-Det (2512.04875) overlaps the prompting contribution. |
| P1 | YOLO benchmark + rater-aware | ⚠️ MED | **❌ HIGH** | Now ≥4 published YOLO variants on VinDr (YOLO-CXR, YOLOv11-MFF, Sensitivity-Oriented YOLOv11, Mamba-YOLOvX). Benchmark value largely consumed. |

### Recommended re-ranking

1. **P6 — Active Learning for CXR detection.** The only proposal whose novelty survived verification intact. Promote and commit.
2. **P5 — Multi-rater bbox uncertainty.** Re-ground the related-work section without TwinTrack first, then commit. VinDr's 3-rater training set is a genuinely rare asset and no verified work exploits it at bbox level.
3. **P4 — Detection KD.** Unchallenged; needs its own verification pass before committing.
4. **P7** — verify the SAET premise before investing.
5. **P2** — do not run as scoped. Salvageable only as *"Co-DETR / DINO-DETR vs. the published RT-DETR baseline, with multi-rater label-noise analysis"* — i.e. beating a known number, not claiming first-mover. Note the existing 45.3 mAP is now the bar.
6. **P1, P3** — shelve or fold into P5/P6 as ablations.

---

## 5. Cross-Cutting Gaps in the Proposals

**No feasibility or compute analysis anywhere.** The sunghyunjun reference needed 35 trained models on a V100 to reach 0.253, and reported that batch size < 4 destroys mAP while larger images help but exceed VRAM. Against that baseline:

- **P6 (Active Learning) is the most compute-hungry proposal in the set** — it requires retraining a detector at every label budget × every acquisition strategy (5 strategies × 5 budgets ≈ 25 full training runs minimum). It is ranked #1 on novelty with no acknowledgment of this. If compute is a single consumer GPU, P6 is the *least* executable proposal despite being the most novel. This tension needs resolving before Phase 1 starts.
- **P4 requires physical hardware** (Jetson, mobile device) that the repo does not establish access to.

**No baseline reproduction step.** The execution plan jumps straight to novel contributions. Nothing in the repo has reproduced even the 0.253 baseline. Every proposal needs a working, validated VinDr detection pipeline first — that shared dependency is missing from the Phase 1/2/3 plan and is realistically 2–4 weeks on its own.

**No evaluation protocol decision.** Competition mAP@0.4, literature mAP@0.5, and mAP@0.5:0.95 are all in play, and the test set (5-rater consensus) differs in labeling process from train (3-rater). Which split and which metric will be reported is unspecified — yet it determines whether any result is comparable to the 0.253 / 0.338 / 0.415 / 0.453 numbers in the literature.

**Dated-forward citations.** Multiple entries are dated 2026 with venues that may not have occurred yet (CXR-LT 2026 / ISBI 2026, MIDL 2026, ICML FM4LS 2026). Each needs a date-of-record check before appearing in a submission.

---

## 6. Recommended Next Actions

1. Fix E1–E5 and the NMS/WBF interpretation in `comparative-analysis.md` and `dataset-research-notes.md`.
2. Rewrite the P2 section; move it out of the top-2.
3. Locate or remove the TwinTrack citation.
4. Run one systematic search pass on the SAET leakage premise (P7) and on AL-for-detection (P6) using a reproducible query log, so "no prior work" becomes a documented claim rather than an assertion.
5. Add the 7 missed works to the comparison tables.
6. Split §3 results table by metric; delete or source the citation-estimate column.
7. Add a Phase 0 to the execution plan: reproduce a VinDr detection baseline and fix the evaluation protocol.
8. State the available compute budget, then re-rank proposals by novelty **× feasibility** rather than novelty alone.

---

## Sources

- [YOLOv11-MFF, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0334283)
- [YOLO-CXR (IEEE Access)](https://pure.ul.ie/en/publications/yolo-cxr-a-novel-detection-network-for-locating-multiple-small-le/)
- [CD-DETR — optimized transformer model, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0323239)
- [Transformer-Based RT-DETR Framework for CXR Disease Detection](https://www.sciencedirect.com/science/article/abs/pii/S1959031825000375)
- [Sensitivity-Oriented YOLOv11, Applied Computer Systems](https://dl.acm.org/doi/10.2478/acss-2026-0004)
- [Mamba-YOLOvX](https://www.sciencedirect.com/science/article/pii/S0957417425015519)
- [BarlowTwins-CXR](https://arxiv.org/pdf/2402.06499)
- [SP-Det](https://arxiv.org/pdf/2512.04875)
- [Align Your Query](https://arxiv.org/pdf/2510.02789)
- [CGF-DETR](https://arxiv.org/pdf/2511.01730)
- [VinDr-CXR-VQA](https://arxiv.org/html/2511.00504)
- [Who Gets Missed in the Tail?](https://www.alphaxiv.org/abs/2607.07717)
- [PaCX-MAE](https://arxiv.org/pdf/2606.01537)
- [Deep Learning-Based Object Detection Strategies for CXR, Diagnostics](https://www.mdpi.com/2075-4418/14/23/2636)
