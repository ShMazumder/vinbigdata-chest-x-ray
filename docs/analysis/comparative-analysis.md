# Comparative Analysis of Existing Works on VinBigData Chest X-Ray Dataset

*Prepared: July 2026*

---

## 1. Master Comparison Table — All Cited Works & Implementations

### Table 1A: Supervised Object Detection Works

| # | Work / Paper Title | Authors | Year | Venue | Algorithm / Architecture | Dataset(s) Used | Image Resolution | BBox Fusion | Key Hyperparameters | Primary Metrics | Best Results | Code Available | Citations (est.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **awsaf49 YOLOv5 14-Class** | Awsaf (Kaggle) | 2021 | Kaggle Notebook | YOLOv5 | VinDr-CXR (train 15K) | Variable | Standard NMS | Default YOLOv5 | mAP@0.4 | Competitive Kaggle submission | ✅ Kaggle | N/A (notebook) |
| 2 | **nxhong93 YOLOv5-512** | nxhong93 (Kaggle) | 2021 | Kaggle Notebook | YOLOv5 | VinDr-CXR | 512×512 | Standard NMS | 512px input | mAP@0.4 | Baseline performance | ✅ Kaggle | N/A (notebook) |
| 3 | **sunghyunjun 76th Place** | Sunghyunjun (Kaggle) | 2021 | Kaggle / GitHub | EfficientNet-b5/b6 + EfficientDet-d3/d4/d5 (Two-stage + One-stage ensemble) | VinDr-CXR 1024px JPG | CLF: 456–1024px; DET: 768–1024px | ZFTurbo NMS | AdamW, CosineAnnealingLR, lr=3e-4–7.5e-5, wd=1e-4–1e-3, bs=3–12, 50 epochs, StratifiedKFold-5 | mAP@0.4 (IoU>0.4), val_loss, AUC | Private LB: **0.253** (blended), **0.259** (two-stage best) | ✅ GitHub | N/A (competition) |
| 4 | **tariqshaban YOLOv7** | Tariq Shaban | 2023 | GitHub (CIS735 Course) | YOLOv7 | VinDr-CXR 1024×1024 PNG | 1024×1024 | Implicit YOLOv7 NMS | Pretrained YOLOv7, bs=8, conf_thr=0.25, 90/10 split | mAP@0.4, Precision, Recall, F1, Confusion Matrix | Comparable to mid-range competition (not converged) | ✅ GitHub | N/A (course project) |
| 5 | **YOLOv5-based Thoracic Abnormalities** | Nawaz, M. et al. | 2024 | IJACSA | YOLOv5 | VinDr-CXR | Standardized resize | Standard anchor-box | Standard YOLOv5 config | mAP@0.4 | Prominent mAP on Kaggle split | ❌ | ~5–10 |
| 6 | **YOLOv11-MFF** | Guan, L. et al. | 2025 | PLOS ONE | YOLOv11 + Frequency-Adaptive Hybrid Gate (FAHG) + Multi-Scale Parallel Large Conv (MSPLC) + Feature Fusion (FF) | VinDr-CXR | Multi-scale | Class-weighted resampling | FAHG, MSPLC, FF modules | Precision, Recall, mAP@0.5, mAP@0.5:0.95 | P: **48.2%**, R: **42.5%**, mAP@0.5: **41.5%**, mAP@0.5:0.95: **22.6%** | ❌ | ~3–5 |
| 7 | **GroundingDINO-Med** | Musgrove, Kyle | 2024 | GitHub | GroundingDINO + Swin-T Backbone | VinDr-CXR (ODVG format) | Variable | Custom thresholding | Open-vocabulary prompts | Custom bbox thresholds | Fine-tuned on 14 thoracic classes | ✅ GitHub | N/A |
| 8 | **Stanford CS230 Benchmarks** | Stanford Research Team | 2021 | CS230 Report | YOLO Framework | VinDr-CXR | High-res | — | Oversampling + heavy color-shift augmentation | mAP@0.4 | Highly competitive mAP | ❌ | N/A |
| 26 | **MASR-DNet** ✅*verified* | Doan, V.T.; Hsu, H-C.; Jonnagaddala, J.; Pham, T.T.T.; Nguyen, D.H.; Dai, H-J. | 2026 | SSRN preprint (19pp, posted 26 Apr 2026) — **not peer reviewed** | YOLO-style one-stage + GADC (Geometric Adaptive Deformable Conv) + EASPP + MSRF (Multi-Scale Refinement) + HSA (Hierarchical Spatial Attention); ASCL loss | AI CUP 2025 (aortic valve CT) + **VinDr-CXR — cardiomegaly only, 1 class** | Not stated in abstract | Not stated | ASCL for foreground/background discriminability under class imbalance | Not stated in abstract | "Competitive performance" + low parameter count (no figures in abstract) | ✅ [GitHub](https://github.com/thinhdoanvu/MASR-DNet) | 0 (46 downloads, 93 abstract views as of Jul 2026) |

> [!NOTE]
> **On MASR-DNet (#26).** Added July 2026 during verification — it was published while the original gap analysis was being written. Closest neighbour to the active **P0** proposal, so worth stating the boundary precisely:
>
> - **It uses VinDr for cardiomegaly localization only — a single class**, not the 14-class detection benchmark. Cardiomegaly is among the easiest classes on this dataset (large, high-prevalence); the paper's own framing cites "diffuse cardiac boundaries," not small-target difficulty. It therefore says nothing about nodule/mass (39.69% small targets) or calcification (17.67%).
> - **No XAI/explainability component.**
> - **No YOLO version comparison** — a single custom architecture, not a benchmark.
> - Primary dataset is AI CUP 2025 (aortic valve CT); VinDr is secondary validation.
>
> **Weight appropriately**: real prior art, low authority — SSRN preprint, not peer reviewed, 0 citations. Must be cited in related work; should not be over-deferred to. Notable that the authors had a 14-class dataset and reported one class.

---

### Table 1B: Unsupervised / Self-Supervised / Vision-Language Works

| # | Work / Paper Title | Authors | Year | Venue | Algorithm / Approach | Dataset(s) Used | Key Innovation | Primary Metrics | Best Results on VinDr | Code Available | Citations (est.) | Refs in Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | **DDAD: Dual-Distribution Discrepancy** | Cai, Y. et al. | 2022 | MICCAI 2022 | Normative Distribution Module (NDM) + Unknown Distribution Module (UDM) | VinDr-CXR, RSNA, SIIM-ACR | Dual-distribution reconstruction modeling; anomaly scoring via intra/inter-discrepancy | AUROC, AP | SOTA on multiple CXR benchmarks using only normal data | ✅ GitHub | ~80–100 | ~35 |
| 10 | **Deep kNN Anomaly Detection** | Liu, X. et al. | 2024 | MLMI / MICCAI Workshop | Lightweight adaptors + Siamese learning + k-NN distances in frozen latent space | VinDr-CXR, RSNA | Geometric mean of k-NN distances; no fine-tuning needed | AUROC | Strong anomaly detection using frozen features | ✅ GitHub | ~10–15 | ~25 |
| 11 | **PPAD: Position-guided Prompt Learning** | PPAD Research Group | 2024 | arXiv | Learnable position-guided text/image prompts + anomaly synthesis over frozen CLIP | VinDr-CXR | Spatial priors for chest X-ray CLIP adaptation | AUROC, AP | Enhanced zero-shot and few-shot classification | ❌ | ~5–10 | ~30 |
| 12 | **OFF-CLIP** | Park, J. et al. | 2025 | MICCAI 2025 | Off-diagonal contrastive loss + sentence text filtering over CLIP | VinDr-CXR | Tight normal-case feature clustering to suppress false positives | AUROC | Significant AUROC improvement over baseline CLIP | ❌ | ~3–5 | ~20 |
| 13 | **PaCX-MAE** | Liu, Y. et al. | 2026 | ICML FM4LS Workshop | Cross-modal distillation: ECG+lab priors → visual MAE encoder via LoRA | VinDr-CXR, MedMod, MIMIC-CXR | Physiology-augmented masked autoencoding; unimodal at inference | AUROC, F1 | **+6.5 F1** on VinDr; **+2.7 AUROC** on MedMod | ❌ | ~1–3 (new) | ~40 |

---

### Table 1C: Explainability, VQA & Causal Attribution Works

| # | Work / Paper Title | Authors | Year | Venue | Approach | Dataset(s) Used | Key Contribution | Primary Metrics | Key Results | Code Available | Citations (est.) | Refs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | **VinDr-CXR-VQA** | Nguyen, D.H. et al. | 2025 | IEEE ISBI 2025 | Multi-task VQA with spatial grounding | VinDr-CXR (4,394 images, 17,597 Q-A pairs) | First CXR VQA dataset with bounding box rationales; 41.7%/58.3% positive/negative balance | F1, Accuracy | F1: **0.624** (+11.8% over baselines) | ✅ HuggingFace | ~5–10 | ~25 |
| 15 | **Indo-CXR-VQA** | Softcase | 2025 | HuggingFace Dataset | Cross-lingual VQA translation | VinDr-CXR-VQA → Indonesian | Evaluates cross-lingual multimodal performance | F1 | Indonesian VQA benchmark | ✅ HuggingFace | ~1–3 | ~10 |
| 16 | **Anti-Aliased B-cos Networks** | Behrendt, F. et al. | 2025 | ResearchGate | B-cos networks + FLCPooling + BlurPool | VinDr-CXR | Clinically clean, faithful explanation maps | IoU, Faithfulness metrics | Improved saliency map quality | ❌ | ~3–5 | ~20 |
| 17 | **MedFocus / MedGround-Bench** | Xiong, G. et al. | 2025 | arXiv | Causal counterfactual editing + attribution validation | VinDr-CXR | Validates that saliency maps reflect true causal diagnostic features | Causal output shift, IoU | Showed standard attribution methods often fail causal tests | ✅ GitHub | ~3–5 | ~30 |

---

### Table 1D: Fairness, Long-Tail & Multi-Center Works

| # | Work / Paper Title | Authors | Year | Venue | Focus | Dataset(s) Used | Key Finding | Primary Metrics | Key Results | Code Available | Citations (est.) | Refs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18 | **Who Gets Missed in the Tail?** | — | 2026 | arXiv | Subgroup underdiagnosis after thresholding in long-tailed CXR classification | VinDr-CXR | Group-tail weighting reduces age-conditioned FNR from 0.822 to 0.133 | FNR, AP, Fairness metrics | Tail-aware thresholds critical for demographic equity | ❌ | ~1–3 (new) | ~25 |
| 19 | **CXR-LT 2026 Challenge** | VISHC team (winners) | 2026 | ISBI 2026 | Multi-center long-tailed + zero-shot CXR classification (30 seen + 6 unseen classes) | PadChest + NIH + VinDr-CXR | Vision-language pre-training improves but domain shift remains primary challenge | mAP Task1, mAP Task2 | Task1: **0.5854**, Task2: **0.4315** | ❌ | ~5–10 | ~35 |
| 20 | **ConvNeXtV2–ViT Hybrid CDSS** | — | 2026 | MDPI Diagnostics | Multi-label classification + OOD detection + symbolic decision support | VinDr-CXR, NIH ChestXray14, CheXpert | Grad-CAM + CheXmask → fuzzy ontologies; OOD via confidence/energy/Mahalanobis | Macro-AUROC, Micro-AUROC | Macro: **0.9525**, Micro: **0.9777** (VinDr); Cross-center: **0.9106** (NIH), **0.8487** (CheXpert) | ❌ | ~3–5 | ~40 |

---

### Table 1E: Structural Extensions & Derivative Datasets

| # | Work / Dataset | Authors | Year | Venue | Method | Contribution | Performance Impact | Citations (est.) |
|---|---|---|---|---|---|---|---|---|
| 21 | **CheXmask Database** | Gaggion, N. et al. | 2024 | Scientific Data | HybridGNet (encoder-decoder + graph neural networks) | 657,566 anatomical masks (lungs, heart) across 5 datasets; RCA-based quality control | Enables saliency verification against true anatomical regions | ~30–50 |
| 22 | **VinDr-RibCXR** | Nguyen, H.C. et al. | 2021 | PhysioNet | Pixel-level segmentation + sequential rib numbering | Individual rib masks for filtering bony occlusions | Direct rib segmentation for parenchymal evaluation | ~15–20 |
| 23 | **Topological Rib Segmentation** | Wei, X. et al. | 2025 | MICCAI | Connectivity + interactivity priors (plug-and-play modules) | Resolves rib overlap/occlusion for anatomically consistent outlines | Improved rib segmentation accuracy | ~3–5 |
| 24 | **OrthoFrac-XR** | — | 2025 | figshare | Multi-center bone fracture detection | Adopted VinDr-CXR's multi-stage validation workflow | Cross-domain annotation methodology transfer | ~1–3 |
| 25 | **CHD-CXR** | — | 2024 | PMC | Pediatric congenital heart disease classification | Uses VinDr-CXR pre-training weights | Bootstrap classification for subtle cardiac variations | ~5–10 |

---

## 2. Hyperparameter Comparison (Implementations Only)

| Parameter | awsaf49 (YOLOv5) | nxhong93 (YOLOv5) | sunghyunjun (EfficientDet) | tariqshaban (YOLOv7) |
|---|---|---|---|---|
| **Framework** | YOLOv5 (Ultralytics) | YOLOv5 (Ultralytics) | PyTorch Lightning + effdet + timm | YOLOv7 |
| **Image Size** | Variable | 512×512 | CLF: 456–1024, DET: 768–1024 | 1024×1024 |
| **Batch Size** | Default | Default | CLF: 4–16, DET: 2–4 | 8 |
| **Optimizer** | SGD (default) | SGD (default) | AdamW | SGD (default) |
| **Learning Rate** | Default YOLOv5 | Default YOLOv5 | CLF: 2.5e-5–1e-4, DET: 2e-4–4e-4 | Default YOLOv7 |
| **Weight Decay** | Default | Default | 1e-4–1e-3 | Default |
| **LR Schedule** | Default | Default | CosineAnnealingLR | Default |
| **Epochs** | Default | Default | 50 | Limited (not converged) |
| **Precision** | FP32 | FP32 | FP16 (mixed) | FP32 |
| **Validation** | Kaggle split | Kaggle split | StratifiedKFold 5-fold | 90/10 random |
| **Augmentation** | Standard YOLOv5 | Standard YOLOv5 | Resize+Scale(-0.9,1.0)+Crop+Brightness+ChannelDropout+Blur+HFlip | Standard YOLOv7 |
| **BBox Handling** | Standard NMS | Standard NMS | ZFTurbo NMS (multi-rater fusion) | Implicit YOLOv7 NMS |
| **Anchor Config** | Default | Default | anchor_scale=4, max_det=100 | Default |
| **Checkpoint** | — | — | max mAP among top-3 min val_loss | — |
| **GPU** | Kaggle P100/T4 | Kaggle P100/T4 | Colab Pro V100 16GB | Limited (GPU memory constrained) |

---

## 3. Results Comparison

| Implementation | mAP@0.4 (Private LB) | mAP@0.4 (Public LB) | Local CV | Other Metrics |
|---|---|---|---|---|
| **1st Place (Competition)** | **0.314** | — | — | — |
| **sunghyunjun (two-stage)** | 0.246–0.256 | 0.215–0.219 | 0.454–0.461 (pos-only) | CLF AUC: 0.993, CLF Acc: 0.96 |
| **sunghyunjun (one-stage)** | 0.245 | 0.198 | 0.455 (pos-only) | — |
| **sunghyunjun (blended)** | **0.253–0.259** | 0.220–0.224 | — | — |
| **tariqshaban (YOLOv7)** | Not submitted | Not submitted | Generally inadequate (not converged) | Confusion matrix shows high background FN |
| **YOLOv11-MFF** | — | — | — | P: 48.2%, R: 42.5%, mAP@0.5: 41.5% |
| **ConvNeXtV2-ViT** | — | — | — | Macro-AUROC: 0.9525, Micro-AUROC: 0.9777 |

---

## 4. Limitations & Conclusions Summary

| Work | Key Limitations | Main Conclusion | Suggested Future Work |
|---|---|---|---|
| **sunghyunjun** | Single V100 GPU; Freeze BN, GroupNorm, Downconv, NIH concat all failed | Two-stage + one-stage blending is effective; NMS > WBF for this dataset | Larger models, more epochs, better normal threshold tuning |
| **tariqshaban** | Halted before convergence; single model, no ensemble; batch size limited | YOLOv7 is feasible but needs more compute and ensemble methods | Per-class models, mobile deployment optimization |
| **YOLOv11-MFF** | Performance still modest; class imbalance not fully resolved | Multi-scale frequency fusion captures small/subtle lesions missed by vanilla YOLO | Better frequency-domain features, improved long-tail handling |
| **DDAD** | Requires normal-only training split; may miss nuanced anomalies | Dual-distribution captures anomalous patterns invisible to single-distribution methods | Integration with supervised signals for hybrid approaches |
| **PaCX-MAE** | Requires paired ECG data for training (not always available) | Cross-modal physiology distillation improves label efficiency dramatically | Extend to more modalities; reduce paired data requirement |
| **VinDr-CXR-VQA** | Limited to 4,394 images; question template diversity constrained | Spatial grounding + clinical reasoning VQA reduces hallucinations | Scale to larger CXR datasets; multi-language support |
| **ConvNeXtV2-ViT** | Cross-center generalization gap remains; OOD detection is lightweight | Hybrid architectures with ontology-driven explainability achieve clinical-grade performance | Deeper ontological reasoning; prospective clinical validation |
| **CXR-LT 2026** | Zero-shot rare disease detection still challenging; multi-center domain shift | Vision-language pre-training helps but doesn't solve domain shift | Better domain adaptation; continual learning for emerging diseases |
| **MASR-DNet** | Not peer reviewed (SSRN preprint); VinDr used for **1 class only** (cardiomegaly), so 14-class generalization unshown; no explainability; abstract reports no numeric metrics | Deformable + multi-scale attention modules on a YOLO-style backbone give competitive detection at low parameter count across CT and CXR | Extend to full 14-class VinDr detection; validate on small-target classes; add interpretability |

---

## 5. Dataset & Reference Paper Statistics

| Paper / Resource | Venue | Year | Citations (est. July 2026) | References in Paper |
|---|---|---|---|---|
| VinDr-CXR (Dataset Paper) | Scientific Data (Nature) | 2022 | **~350–400** | 45 |
| VinBigData Kaggle Competition | Kaggle | 2021 | N/A (competition) | — |
| CheXmask Database | Scientific Data | 2024 | ~30–50 | 35 |
| VinDr-RibCXR | PhysioNet | 2021 | ~15–20 | 20 |
| DDAD (MICCAI 2022) | MICCAI | 2022 | ~80–100 | 35 |
| VinDr-CXR-VQA | IEEE ISBI 2025 | 2025 | ~5–10 | 25 |
| YOLOv11-MFF | PLOS ONE | 2025 | ~3–5 | 30 |
| ConvNeXtV2-ViT Hybrid | MDPI Diagnostics | 2026 | ~3–5 | 40 |
| PaCX-MAE | ICML FM4LS Workshop | 2026 | ~1–3 | 40 |
| CXR-LT 2026 Challenge | ISBI 2026 | 2026 | ~5–10 | 35 |
| MedFocus / MedGround-Bench | arXiv | 2025 | ~3–5 | 30 |
| Who Gets Missed in the Tail? | arXiv | 2026 | ~1–3 | 25 |
