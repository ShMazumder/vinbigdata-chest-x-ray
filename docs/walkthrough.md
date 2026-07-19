# VinBigData CXR Project — Walkthrough

*Last updated: July 19, 2026 — Deep search novelty verification complete*

> [!IMPORTANT]
> **This document is now partly stale.** Independent verification (July 2026) refuted the core novelty claims of **P2** and **P1**, and the project has since committed to a new conference-scoped proposal, **P0**.
>
> Current sources of truth:
> - **[verification-report.md](./analysis/verification-report.md)** — what was checked, what failed
> - **[publication-proposals.md](./proposals/publication-proposals.md)** — P0 (active) + revised P1–P7 ranking
> - **[iccit-2026-execution-plan.md](./proposals/iccit-2026-execution-plan.md)** — active 12-day plan, ICCIT 2026 deadline **July 31, 2026**
>
> The P1 row in the table below ("Add rater-aware training as primary contribution") is superseded: P1 is now rated ❌ HIGH and its rater-aware half moved to P5.

---

## ✅ Completed Tasks

### 1. Analysis Saved to Project
- [dataset-research-notes.md](file:///Users/rohansmac/Documents/ml-research/vinbigdata-chest-x-ray/docs/analysis/dataset-research-notes.md) — Dataset overview, class definitions, known issues, implementation summaries

### 2. Organized Project Structure
- [README.md](file:///Users/rohansmac/Documents/ml-research/vinbigdata-chest-x-ray/README.md) — Project readme with full directory map
- [.gitignore](file:///Users/rohansmac/Documents/ml-research/vinbigdata-chest-x-ray/.gitignore) — ML-appropriate ignore rules

```
vinbigdata-chest-x-ray/
├── docs/analysis/            ← Research notes, comparative analysis
├── docs/literature-review/   ← Paper reviews
├── docs/proposals/           ← Publication proposals
├── docs/walkthrough.md       ← This file (synced copy)
├── src/{data,models,training,evaluation,utils}/
├── notebooks/
├── experiments/{configs,logs,checkpoints}/
└── results/{figures,tables}/
```

### 3. Comprehensive Comparative Analysis
- [comparative-analysis.md](file:///Users/rohansmac/Documents/ml-research/vinbigdata-chest-x-ray/docs/analysis/comparative-analysis.md) — 25 works across 5 tables

### 4. Publication Proposals (Deep-Search Verified)
- [publication-proposals.md](file:///Users/rohansmac/Documents/ml-research/vinbigdata-chest-x-ray/docs/proposals/publication-proposals.md) — 7 proposals with novelty verification

---

## 🔍 Deep Search Novelty Verification Results

| Proposal | Novelty Risk | Key Competing Work Found | Our Differentiation |
|---|---|---|---|
| **P1: YOLO Benchmark** | ⚠️ MEDIUM | YOLO-CXR (YOLOv8+RefConv, mAP=0.338), YOLOv11-MFF | Add rater-aware training as primary contribution |
| **P2: RT-DETR on VinDr** | ✅ LOW | CD-DETR (NIH only), CGF-DETR (RSNA only) | First DETR on VinDr-CXR; NMS-free for noisy labels |
| **P3: GroundingDINO+SAM2** | ⚠️ MEDIUM | GroundingDINO-Med (VinDr), LuGSAM (ICU) | Few-shot label efficiency + prompt taxonomy study |
| **P4: Edge KD** | ⚠️ MEDIUM | Classification KD widely studied | Focus on detection KD (bbox); real hardware benchmarks |
| **P5: Multi-Rater Uncertainty** | ⚠️ MEDIUM | TwinTrack (MIDL 2026, classification only) | Bbox-level spatial uncertainty (novel for detection) |
| **P6: Active Learning** | ✅ LOW | AL for CXR classification exists; detection AL unexplored | First AL for CXR bounding box detection |
| **P7: SAET Leakage** | ⚠️ MEDIUM | General CXR domain adversarial exists | VinDr-specific SAET case study (never published) |

### Revised Priority Ranking

| Rank | Proposal | Why |
|---|---|---|
| 🥇 | **P6: Active Learning for CXR Detection** | Genuinely underexplored; no prior work |
| 🥈 | **P2: RT-DETR on VinDr-CXR** | Clear first-mover; DETR never tested on VinDr |
| 🥉 | **P5: Multi-Rater Bbox Uncertainty** | Classification exists; bbox uncertainty is novel |
| 4 | **P7: SAET Leakage Mitigation** | Quick standalone paper; strong narrative |
| 5 | **P4: Detection KD for Edge** | Detection KD is gap; needs hardware benchmarks |
| 6 | **P3: Foundation Model Few-Shot** | Needs strong angle to differentiate |
| 7 | **P1: YOLO + Rater-Aware** | YOLO-CXR already published; incremental |

> **Best combined paper**: P2 + P5 → "Uncertainty-Aware Transformer Detection with Multi-Rater Calibration on VinDr-CXR"

---

## Change Log

| Date | Change |
|---|---|
| 2026-07-19 | Initial project setup, structure, comparative analysis, 7 proposals created |
| 2026-07-19 | Deep search novelty verification; proposals revised with competing work; priority reranked |
