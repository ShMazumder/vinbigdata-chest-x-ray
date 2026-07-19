# ICCIT 2026 — 12-Day Execution Plan

**Paper**: YOLO26 on VinDr-CXR — Modern YOLO Comparison with Measured Explanation Faithfulness
**Venue**: ICCIT 2026, 29th International Conference on Computer and Information Technology
**Deadline**: **July 31, 2026** (full paper) · Notification Oct 15 · Conference Dec 18–20, Cox's Bazar
**Today**: July 19, 2026 — **12 days**
**Compute**: Kaggle, 2× NVIDIA T4

---

## 0. The One-Paragraph Framing

> We train YOLOv8s, YOLO11s and YOLO26s on VinDr-CXR 14-class abnormality detection under identical conditions, then evaluate not only detection accuracy but **explanation faithfulness measured against radiologist bounding boxes**. Because VinDr provides ground-truth boxes — unlike the classification datasets most CXR-XAI work uses — saliency quality becomes a *metric* rather than a figure. We further ask whether YOLO26's NMS-free architecture yields more faithful explanations than its NMS-based predecessors.

**Not claiming SOTA.** Published bar is RT-DETR 45.3 mAP@0.5, YOLOv11-MFF 41.5, YOLO-CXR 0.338. A 512px `s`-scale model at 40 epochs will land below these. Claim: controlled comparison + first YOLO26 evaluation on this dataset + quantified explainability. If a reviewer reads a SOTA claim anywhere in the draft, the paper dies.

---

## 1. Compute Budget — the binding constraint

| Item | Figure |
|---|---|
| Kaggle GPU session cap | 12 h (hard kill) |
| Weekly GPU quota | ~30 h (**verify on your account — this changes**) |
| Quota weeks in window | ~1.85 → **~50 h total, minus debugging** |

**The decision that makes this survivable: train on positive images only.**

VinDr's 15,000 training images are dominated by "No Finding." Only ~4,400 carry annotated abnormalities. Training the detector on abnormal images only is the established protocol on this dataset — the vendored `sunghyunjun` 76th-place solution did exactly this (classifier on all images, detector on abnormal only), and reports CV on positive images only.

Effect: ~3.4× reduction in epoch time. 15K → 4.4K images.

**Estimated cost per model** (4.4K images, 512px, `s` scale, batch 16, single T4, FP16):

| | Estimate |
|---|---|
| Per epoch | ~2–3 min |
| 40 epochs | **~1.5–2 h** |
| 3 models | **~6 h** |
| With reruns/debugging (×3 contingency) | **~18 h** |

Comfortably inside budget. The surplus goes to XAI, which is where the contribution lives.

### T4 gotchas — read before writing any training code

- **No bf16.** T4 has FP16 tensor cores only. Set `amp=True`, never `bf16`.
- **DDP breaks in Kaggle notebooks.** Ultralytics `device=[0,1]` frequently fails on spawn inside notebook kernels. **Use `device=0`, single T4.** The second GPU is not worth the debugging time under this clock. If you want it, write a `.py` and invoke via `!python train.py`.
- **12 h session cap will interrupt you.** Save checkpoints every epoch; use `resume=True`. Persist weights to Kaggle Datasets, not `/kaggle/working` alone.
- T4 ≈ ⅓ of V100 throughput here. The `sunghyunjun` timings were V100 — do not extrapolate them.
- Ultralytics is **AGPL-3.0**. Fine for academic publication; state it.

---

## 2. Frozen Experimental Protocol

Decide once, now, and do not revisit mid-run. Protocol drift is the most common way a 12-day paper becomes unpublishable.

| Decision | Value | Rationale |
|---|---|---|
| Task | 14-class detection | The actual benchmark; MASR-DNet only did 1 class |
| Training subset | Positive images only (~4.4K) | Compute; matches established protocol |
| Split | 80/10/10 from the 15K train set, **stratified by class** | Kaggle test labels are not public — CV on train is the only option |
| Image size | 512 × 512 | T4 budget |
| Model scale | `s` for all three | Fair comparison; `n` too weak, `m`+ too slow |
| Epochs | 40, fixed, all models | Identical conditions is the whole point |
| Batch | 16 | <4 destroys mAP on this dataset (documented in reference) |
| Optimizer | Ultralytics default per model | Do **not** tune per-model — that breaks the controlled comparison |
| Seed | Fixed, logged; 3 seeds if time permits | Single-seed detection results are noisy |
| Rater bbox fusion | **WBF @ IoU 0.5** | 3 raters per image must be fused into training labels |
| Primary metric | **mAP@0.4** (competition) | Comparable to the 0.253 / 0.314 line |
| Secondary metric | **mAP@0.5, mAP@0.5:0.95** | Comparable to YOLO-CXR 0.338 / YOLOv11-MFF 0.415 |

> [!WARNING]
> **Report both mAP@0.4 and mAP@0.5.** The competition literature uses @0.4, the modern YOLO literature uses @0.5. Reporting only one makes half your related-work table non-comparable. This is the single most common reviewer complaint on this dataset.

> [!NOTE]
> **Fusion caveat worth one sentence in the paper.** The vendored reference's own ablation shows WBF *winning* on private LB (0.185) while *losing* on CV (0.4158 vs 0.4419) — and the author chose NMS by reasoning about test labeling, not measurement. Note that this remains unsettled rather than inheriting it as fact. Cheap credibility.

### 2.1 GPU Choice: P100 over a single T4

Kaggle bills GPU sessions by **wall-clock, not per-GPU** — selecting `T4 x2` and using `device=0` burns identical quota while giving you strictly less card. So the real choice is single P100 vs single T4.

| | P100 (Pascal GP100) | T4 (Turing TU104) |
|---|---|---|
| Memory bandwidth | **732 GB/s** | 320 GB/s |
| FP32 | 9.5 TFLOPS | 8.1 TFLOPS |
| FP16 | 19 TFLOPS (2× rate, **no tensor cores**) | 65 TFLOPS (tensor cores) |
| Power envelope | 250 W | **70 W** (throttles under sustained load) |
| VRAM | 16 GB HBM2 | 16 GB GDDR6 |

T4 wins on FP16 paper specs. Reality is closer: T4 is bandwidth-starved and power-capped, so it does not hold peak throughput across a 40-epoch run. **Expect P100 ≈ 1.0–1.3× a single T4 here.** Modest, but free, and one less thing to configure.

**Decision: P100, `device=0`.** Do not spend deadline time on 2×T4 DDP — notebook spawn failures are a known sink, best case saves ~2 h across three models, worst case eats a day.

**AMP caveat**: P100 has 2× FP16 math but no tensor cores, so `amp=True` gains less than on Turing. If you hit NaN losses or gradient-scaler thrash, drop to FP32 — 9.5 TFLOPS is fine for `s`-scale at 512px and you have VRAM headroom.

> [!TIP]
> This decision is worth ~20–40 min across the whole project. The positive-only subset decision is worth ~20 h. Do not over-optimize here.

### 2.2 Run Logging — mandatory from run 1

**One notebook, not two.** The paper's entire claim is *identical conditions across three models*. Two training scripts drift — someone patches a bug in one, tweaks augmentation in the other, and by D8 the controlled comparison is unprovable. Keep config in a `.py`, keep the notebook thin. The config file becomes a paper artifact.

Use **[`src/utils/run_logger.py`](../../src/utils/run_logger.py)**. It captures hardware, software, hyperparameters, data shape and results per run → one JSON each plus an append-only `master_results.csv`.

```python
from run_logger import RunLogger, benchmark_inference

log = RunLogger(run_name="yolo26s_512_e40_seed0", out_dir="/kaggle/working/runs")
log.set_params(model="yolo26s", imgsz=512, epochs=40, batch=16, seed=0,
               optimizer="auto", amp=True, device=0,
               subset="positive_only", fusion="wbf@0.5", split="80/10/10")
log.set_data(n_train=3520, n_val=440, n_test=440, n_classes=14)
# ... train ...
log.from_ultralytics(results).set_results(map40=0.271, per_class={...})
log.finish()
```

**Log params BEFORE training, not after.** Retroactively reconstructing which config produced which number on D10 is misery, and it is exactly when you have no time for it.

> [!CAUTION]
> **The FPS confound — mAP is hardware-independent, latency is not.**
>
> If run 1 lands on P100 and run 3 on T4, detection metrics stay valid but **every FPS / inference-time / throughput number is invalid**.
>
> This matters here specifically: YOLO26 is marketed as edge-first with a claimed 43% CPU inference-speed gain over YOLO11. A latency table is a natural inclusion and reviewers will expect one.
>
> **Rule**: train wherever, then run `benchmark_inference()` over all three checkpoints **back to back, one session, one card**. Costs minutes. Call `RunLogger.check_hardware_consistency()` before writing the results section — it fails loudly on mixed hardware.

**What gets logged** (→ feeds the paper's reproducibility statement directly): GPU name + compute capability + VRAM + tensor-core flag, torch/CUDA/cuDNN/Ultralytics versions, git commit, seed, every hyperparameter, dataset split sizes, wall-clock and sec/epoch, all metrics.

> [!NOTE]
> **`from_ultralytics()` does not give you mAP@0.4.** Ultralytics reports @0.5 and @0.5:0.95 natively; the VinBigData competition metric @0.4 must be computed separately. Do not let @0.5 silently land in a column labelled @0.4 — that single error would misalign your entire related-work comparison.

---

## 3. XAI Design — the actual contribution

### Methods (2 primary, 1 optional)

| Method | Type | Cost | Status |
|---|---|---|---|
| **EigenCAM** | Non-gradient | Cheap | Primary — works on YOLO without layer surgery |
| **Grad-CAM++** | Gradient | Cheap | Primary — on detection head |
| D-RISE | Perturbation, detection-specific | **Expensive** | Optional, subsample ~200 images only if D9 is on schedule |

### Metrics — saliency vs. ground-truth boxes

This is what converts XAI from a figure into a result. Most CXR-XAI work uses NIH ChestX-ray14, which has no boxes, so it can only show qualitative heatmaps. You have boxes.

1. **Pointing Game accuracy** — does CAM argmax fall inside the GT box?
2. **Energy-Based Pointing Game (EBPG)** — fraction of total CAM energy inside the GT box. More robust than pointing game.
3. **CAM–box IoU** — threshold CAM at the 90th percentile, IoU against GT box
4. **Deletion / Insertion AUC** — faithfulness: does removing high-saliency pixels actually degrade the detection?

### Three analysis axes — these are the findings

- **Axis A — per-class × object size.** Small-target classes (nodule/mass 39.69% small; calcification 17.67%) vs. large (cardiomegaly, aortic enlargement). Hypothesis: explanation faithfulness degrades on small targets faster than mAP does. If true, that is a clinically meaningful result — the model looks right while explaining wrong.
- **Axis B — NMS-free vs NMS.** Does YOLO26's native NMS-free head produce more faithful explanations than v8/v11? This is what ties the two halves of the paper together. Without it you have two unrelated experiments stapled together.
- **Axis C — accuracy/faithfulness decoupling.** Does the highest-mAP model also give the best explanations? A "no" here is the most publishable outcome available to you.

> [!IMPORTANT]
> **The trap.** "We ran three YOLOs and made heatmaps" is a course project — that is literally what the vendored `tariqshaban` repo already is. The measured saliency-vs-box evaluation is the entire difference between a paper and a class assignment. If the schedule slips, **cut a model, never cut the XAI metrics.**

---

## 4. Day-by-Day

Writing runs **in parallel** from D6. Do not leave it to the end.

| Day | Date | Work | Gate |
|---|---|---|---|
| **D1** | Jul 19–20 | Fix disk space → sandbox boots. Pull VinDr 1024px JPG (Kaggle public dataset, 3.59 GB — do not re-derive from 192 GB DICOM). Inspect annotations. | Data loads |
| **D2** | Jul 21 | Rater bbox fusion (WBF@0.5). Convert to YOLO format. Stratified 80/10/10 split. Sanity-visualize 20 fused boxes over images. | **Labels verified visually** |
| **D3** | Jul 22 | YOLOv8s train run 1, **wired to `RunLogger` from the first run**. Validate pipeline end-to-end on 5 epochs first, *then* launch 40. | Pipeline works + `master_results.csv` has a row |
| **D4** | Jul 23 | YOLOv8s completes. Compute mAP@0.4/@0.5/@0.5:0.95. **Baseline sanity check.** | ⛔ **Gate 1** — see kill criteria |
| **D5** | Jul 24 | YOLO11s + YOLO26s launched. YOLO26 API differences may need fixing — budget for it. | Both training |
| **D6** | Jul 25 | Both complete. Results table v1. **Start writing: Intro + Related Work.** | 3 models done |
| **D7** | Jul 26 | EigenCAM + Grad-CAM++ implemented on all 3. Qualitative check on 10 images. | CAMs render |
| **D8** | Jul 27 | XAI metrics coded: pointing game, EBPG, CAM-box IoU. Run across test split. | ⛔ **Gate 2** |
| **D9** | Jul 28 | Deletion/insertion AUC. Axis A/B/C analysis. Figures. D-RISE only if ahead. | Findings identified |
| **D10** | Jul 29 | `RunLogger.check_hardware_consistency()` → then `benchmark_inference()` over all 3 checkpoints, one session one card. Tables + figures final (`master_table()` → `latex-results-table` skill). **Write Methods + Results.** | Draft complete |
| **D11** | Jul 30 | Discussion, limitations, abstract. Full read-through. Reference check. | Full draft |
| **D12** | Jul 31 | Format to IEEE template, proofread, **submit via Microsoft CMT**. Do not submit in the final hour. | ✅ Submitted |

---

## 5. Kill Criteria — decide fast, do not sink cost

**⛔ Gate 1 (D4) — baseline sanity.** If YOLOv8s mAP@0.4 on positive-only CV is **< 0.20**, something is wrong with labels or fusion, not the model. The reference hit ~0.45 CV positive-only with EfficientDet. Stop and debug the data pipeline; do not launch two more models on broken labels.

**⛔ Gate 2 (D8) — XAI viability.** If saliency metrics cannot be computed reliably across all 3 models by end of D8, **drop to 2 models** (YOLOv8s vs YOLO26s — the NMS vs NMS-free contrast, which is Axis B) and keep the full XAI analysis. Two models with measured explanations beats three with pretty pictures.

**⛔ Hard abort (D9).** If no draft section exists by end of D9, submit to **COMPAS 2026** instead — same July 31 deadline, Oct 9–10 conference, newer venue, likely softer acceptance bar. Decide on D9, not D12.

---

## 6. Paper Skeleton

1. **Introduction** — CXR detection burden; black-box barrier to clinical adoption; VinDr's under-exploited ground-truth boxes enable *measured* explainability
2. **Related Work** — VinDr detection lineage (YOLO-CXR 0.338; YOLOv11-MFF 0.415; RT-DETR 45.3; **MASR-DNet** — YOLO-style, but cardiomegaly-only); CXR XAI (mostly NIH, classification, qualitative only)
3. **Method** — the three detectors, identical protocol; YOLO26's STAL + NMS-free relevance to VinDr's small-target profile; XAI methods; the four saliency metrics
4. **Experiments** — protocol table, detection results, saliency results, Axes A/B/C
5. **Discussion** — accuracy/faithfulness decoupling; small-target explanation failure; clinical implication
6. **Limitations** — 512px, `s`-scale, single fold, positive-only subset, 40 epochs, not SOTA-competitive. **State all of these plainly.** Reviewers forgive acknowledged limits and punish hidden ones.

---

## 7. Positioning Against the Closest Neighbours

| Work | What it did | Your delta |
|---|---|---|
| **MASR-DNet** (SSRN preprint, Apr 2026) | Custom YOLO-style net; VinDr for **cardiomegaly only, 1 class**; no XAI | 14-class; XAI is the contribution |
| **YOLO-CXR** (IEEE Access) | YOLOv8s + RefConv + ELCA; mAP@0.5 0.338 | No version comparison, no XAI |
| **YOLOv11-MFF** (PLOS ONE 2025) | YOLOv11 + frequency fusion; mAP@0.5 0.415 | Single architecture, no XAI |
| **RT-DETR on VinBigData** (IRBM 2025) | 45.3 mAP@0.5 | Transformer, no XAI, no YOLO comparison |
| **Sensitivity-Oriented YOLOv11** (2026) | YOLOv11 variant | No YOLO26, no XAI |

**Defensible claims, in order of strength:**

1. First evaluation of **YOLO26** on VinDr-CXR (released Jan 2026 — genuinely new, and timing-defensible unlike the P2 "first DETR" claim that got refuted)
2. First **quantified** explanation-faithfulness study on VinDr-CXR detection, validated against radiologist boxes
3. First controlled v8/v11/YOLO26 comparison on this dataset under identical settings

Lead with 2. Claim 1 has a shelf life measured in months. Claim 3 is the weakest — ≥5 YOLO-family works already exist on this dataset.

---

## 8. Pre-Flight Checklist

- [ ] Free disk space → sandbox boots (blocks *all* data work)
- [ ] Verify Kaggle weekly GPU quota on your account
- [ ] Confirm YOLO26 weights download + train on a 100-image toy set **before D3**
- [ ] Download ICCIT IEEE template + confirm page limit
- [ ] Create Microsoft CMT account early (ICCIT uses CMT)
- [ ] Check ICCIT plagiarism / multiple-submission policy — you may not parallel-submit to COMPAS
- [ ] Set notebook accelerator to **GPU P100**
- [ ] Smoke-test `python src/utils/run_logger.py` on Kaggle — confirm it prints the right card before any real run
- [ ] Copy `run_logger.py` into the Kaggle notebook (or attach the repo as a Kaggle Dataset)
- [ ] Persist `runs/` to a Kaggle Dataset, not just `/kaggle/working` — the 12 h session kill takes `/kaggle/working` with it
- [ ] Log every hyperparameter from run 1, not retroactively
