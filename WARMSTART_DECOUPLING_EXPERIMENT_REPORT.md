# Warm-Start Targeted Decoupling Experiment Report

**Date:** October 15, 2026  
**System:** NeuroSyntra AI Brain Tumor MRI Diagnostic Platform  
**System Nature:** Educational Research Prototype — *Not certified for clinical or diagnostic medical decision-making.*  
**Experiment Model File:** `brain_tumor_mobilenet_v2_warmstart.keras` (MD5: `246db4d334e79008b41ce7f33d5d9be7`)  
**Production Model File:** `brain_tumor_mobilenet_v2.keras` (MD5: `9c3d3f636cbc032561e80504b0391262`, **100% UNCHANGED**)  
**FastAPI Production Server:** `http://127.0.0.1:8000` (**Running continuously, 0 downtime**)  
**Status:** Complete. **Candidate REJECTED for deployment; Production Model Retained.**

---

## 1. Executive Summary & Evaluation Verdict

Based on the findings of `NEUROSYNTRA_DIAGNOSTIC_AUDIT.md`, we implemented a controlled **Warm-Start Targeted Decoupling Experiment**. Rather than initializing MobileNetV2 from generic natural-image ImageNet weights (which caused 0.0%–7.5% Glioma recall in Candidates 1–3), we initialized the network directly from **`brain_tumor_mobilenet_v2.keras`** to preserve its learned brain MRI representations. We froze all early and intermediate backbone layers up to Block 15, selectively fine-tuned **Block 16 and the classification head** (1,050,564 trainable parameters), and trained with **equal class weighting** (1.0 for all classes) at a low learning rate of **`1e-5`**.

### Core Empirical Findings:
1. **Validation Optimization:**
   - Validation loss declined steadily from `1.8613` (Epoch 0 baseline) down to **`0.9718`** at **Epoch 26** (`val_accuracy: 73.50%`), breaking the $1.0$ loss threshold.
   - `ModelCheckpoint` captured this optimal checkpoint. Early stopping restored these weights; the reloaded model confirmed exact replication of validation metrics (`val_loss: 0.9718`, `val_acc: 73.50%`, Macro F1: `0.7309`).
2. **Independent Test Set Performance (160 balanced images):**
   - **Overall Accuracy:** Increased from **71.88% $\rightarrow$ 72.50%** (116/160), outperforming the original model and all previous candidates.
   - **Macro F1:** Substantially improved from **0.6668 $\rightarrow$ 0.7189** (+0.0521).
   - **Meningioma Recall Quadrupled:** Rose from **10.00% (4/40) $\rightarrow$ 40.00% (16/40)** with **0.8000 precision** (F1 improved from 0.1778 to 0.5333). Meningioma-to-pituitary errors were cut from 31 down to 20.
   - **NoTumor and Pituitary Preserved:** NoTumor achieved **90.00% recall** (36/40, F1: 0.8889); Pituitary achieved **97.50% recall** (39/40, F1: 0.6842).
3. **The Unresolved Clinical Trade-Off (Glioma Recall Drop):**
   - While Meningioma detection significantly improved, **Glioma recall dropped from 87.50% (35/40) down to 62.50% (25/40)** on the independent test set (and 94% $\rightarrow$ 66% on validation data).
   - 11 of the 40 glioma scans were misclassified as Pituitary, and 3 as Meningioma.
   - In neuro-oncology, intra-axial high-grade gliomas carry high mortality. Sacrificing 25% of glioma recall to gain 30% meningioma recall represents an unacceptable diagnostic compromise.
4. **Final Gate Verdict:**
   - **REJECT DEPLOYMENT.**
   - Retain `brain_tumor_mobilenet_v2.keras` as the active production model.

---

## 2. Safeguard Verification Checklist

| Safeguard | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **#1** | Preserve `brain_tumor_mobilenet_v2.keras` | **PASSED** | MD5: `9c3d3f636cbc032561e80504b0391262` (100% identical). |
| **#2** | Preserve all previous candidates and reports | **PASSED** | MD5s of Candidates 1, 2, and 3 identical. All Markdown reports intact. |
| **#3** | Do not modify frozen dataset partitions | **PASSED** | All 1,046 images in `dataset_clean/` (`train`, `val`, `test_independent`) untouched. |
| **#4** | Do not modify running application or auto-deploy | **PASSED** | FastAPI server running on port 8000 with 0 downtime; production model unchanged. |
| **#5** | Single evaluation gate on `test_independent` | **PASSED** | Independent test set evaluated strictly once after best checkpoint selection. |

---

## 3. Warm-Start Architectural & Training Configuration

### 3.1 Layer Freezing Strategy
Starting from `brain_tumor_mobilenet_v2.keras`:
- **Frozen Layers (Indices 0–142):** Input layer, and base model from `input_1` through all layers of Blocks 0 to 15 (including `block_15_project_BN` and `block_15_add`).
  - Non-Trainable Parameters: **1,371,904**.
  - All early and mid-level feature extractors (edges, ventricular contours, anatomical symmetry) are permanently locked.
- **Trainable Layers (Indices 143–153 + Head):**
  - Base Model Block 16: `block_16_expand`, `block_16_expand_BN`, `block_16_depthwise`, `block_16_depthwise_BN`, `block_16_project`, `block_16_project_BN`.
  - Base Model Top Conv: `Conv_1`, `Conv_1_bn`.
  - Classification Head: `dense` (128 units, ReLU), `dropout` (0.3), `dense_1` (4 units, Softmax).
  - Trainable Parameters: **1,050,564** (Base: 886,080, Head: 164,484).

### 3.2 Equal Class Weighting Justification
In Candidates 1–3, standard balanced weighting assigned Glioma a weight of `0.657` and NoTumor `1.732` (a 2.64× disparity), which caused gradient descent to suppress glioma predictions. In this experiment, **equal weighting (1.0 for all classes)** was used:
- Loss: Standard categorical crossentropy ($-\sum y_i \log p_i$).
- Eliminates artificial penalty imbalance between intra-axial classes.

### 3.3 Training Hyperparameters
- Learning Rate: `1e-5` with `Adam(learning_rate=1e-5)`
- LR Schedule: `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)`
- Early Stopping: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
- Checkpointing: `ModelCheckpoint(filepath='brain_tumor_mobilenet_v2_warmstart.keras', monitor='val_loss', save_best_only=True)` from Epoch 1
- Preprocessing: Aspect-preserving square bicubic letterboxing to $128 \times 128$, black padding, $[-1.0, 1.0]$ normalization.
- Class Mapping: `0: glioma, 1: meningioma, 2: notumor, 3: pituitary`.

---

## 4. Training Progression & Checkpoint Selection

Training progressed for 30 epochs on `dataset_clean/train` (686 images) with validation evaluated on `dataset_clean/val` (200 images):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Learning Rate | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0 (Baseline)** | — | — | **1.8613** | **71.00%** | — | Starting model state |
| **1** | 1.3670 | 63.41% | 1.7978 | 70.00% | 0.000010 | Initial Checkpoint (`val_loss: 1.7978`) |
| 2–5 | 0.8093 $\rightarrow$ 0.3713 | 76.4% $\rightarrow$ 86.9% | 1.6868 $\rightarrow$ 1.4320 | 70.0% $\rightarrow$ 71.0% | 0.000010 | Continuous loss decline |
| 6–10 | 0.3087 $\rightarrow$ 0.2501 | 88.2% $\rightarrow$ 92.0% | 1.3905 $\rightarrow$ 1.2775 | 71.0% $\rightarrow$ 71.5% | 0.000010 | Steady convergence |
| 11–15 | 0.2125 $\rightarrow$ 0.1734 | 93.1% $\rightarrow$ 93.4% | 1.2397 $\rightarrow$ 1.1010 | 72.5% $\rightarrow$ 73.5% | 0.000010 | Saved (`val_loss: 1.1010`) |
| 16–20 | 0.1426 $\rightarrow$ 0.1390 | 94.8% $\rightarrow$ 94.8% | 1.0882 $\rightarrow$ 1.0212 | 73.5% $\rightarrow$ 75.0% | 0.000010 | Saved (`val_loss: 1.0212`) |
| 21–25 | 0.0951 $\rightarrow$ 0.0870 | 97.7% $\rightarrow$ 97.4% | 1.0377 $\rightarrow$ 1.0209 | 73.5% $\rightarrow$ 74.0% | 0.000010 | Saved (`val_loss: 1.0023`) |
| **26** | **0.1069** | **95.92%** | **0.9718** | **73.50%** | **0.000010** | **Best Checkpoint Saved (`val_loss: 0.9718`)** |
| 27–29 | 0.0916 $\rightarrow$ 0.0987 | 97.2% $\rightarrow$ 96.8% | 0.9768 $\rightarrow$ 1.0122 | 74.0% $\rightarrow$ 73.0% | 0.000010 | Plateaued |
| 30 | 0.1130 | 96.36% | 0.9982 | 73.50% | 0.000005 | LR reduced to $5 \times 10^{-6}$; weights restored to Ep 26 |

**Checkpoint Restoration Verification:** Reloading `brain_tumor_mobilenet_v2_warmstart.keras` from disk confirmed exact restoration of the Epoch 26 weights: `val_loss = 0.9718`, `val_accuracy = 73.50%` (147 / 200).

---

## 5. Comprehensive Cross-Model Benchmark

All models were evaluated using the exact same preprocessed datasets:

### 5.1 Independent Test Set (`dataset_clean/test_independent`, 160 images, 40/class)

| Metric | Original Deployed (`mobilenet_v2`) | Candidate 1 (`improved`) | Candidate 2 (`linear_probe`) | Candidate 3 (`shallow_finetune`) | Candidate 4 (`warmstart`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **71.88%** (115/160) | 58.13% (93/160) | 57.50% (92/160) | 59.38% (95/160) | **72.50% (116/160)** |
| **Macro F1** | 0.6668 | 0.5138 | 0.4905 | 0.5277 | **0.7189** |
| **Macro Precision** | 0.7925 | 0.7173 | 0.4298 | 0.7044 | **0.8013** |
| **Macro Recall** | 0.7188 | 0.5813 | 0.5750 | 0.5938 | **0.7250** |
| **Glioma Precision** | **1.0000** | 1.0000 | 0.0000 | 1.0000 | **1.0000** |
| **Glioma Recall** | **87.50% (35/40)** | 7.50% (3/40) | 0.00% (0/40) | 7.50% (3/40) | **62.50% (25/40)** |
| **Glioma F1** | **0.9333** | 0.1395 | 0.0000 | 0.1395 | **0.7692** |
| **Meningioma Precision**| 0.8000 | 0.6957 | 0.5106 | 0.7000 | **0.8000** |
| **Meningioma Recall**| 10.00% (4/40) | 40.00% (16/40) | **60.00% (24/40)** | 52.50% (21/40) | **40.00% (16/40)** |
| **Meningioma F1** | 0.1778 | 0.5079 | 0.5517 | **0.6000** | **0.5333** |
| **NoTumor Precision** | 0.8571 | 0.7083 | 0.6346 | 0.5692 | **0.8780** |
| **NoTumor Recall** | **90.00% (36/40)** | 85.00% (34/40) | 82.50% (33/40) | **92.50% (37/40)** | **90.00% (36/40)** |
| **NoTumor F1** | 0.8780 | 0.7727 | 0.7174 | 0.7048 | **0.8889** |
| **Pituitary Precision**| 0.5128 | 0.4651 | **0.5738** | 0.5484 | 0.5270 |
| **Pituitary Recall** | **100.00% (40/40)** | **100.00% (40/40)** | 87.50% (35/40) | 85.00% (34/40) | **97.50% (39/40)** |
| **Pituitary F1** | 0.6780 | 0.6349 | **0.6931** | 0.6667 | **0.6842** |

### 5.2 Confusion Matrix Comparison (Independent Test Set):
```text
ORIGINAL DEPLOYED MODEL (Accuracy: 71.88%, Macro F1: 0.6668)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |     35 |          1 |       1 |         3 |    40
Meningioma   |      0 |          4 |       5 |        31 |    40  <-- 31/40 collapsed into Pituitary
NoTumor      |      0 |          0 |      36 |         4 |    40
Pituitary    |      0 |          0 |       0 |        40 |    40

WARM-START CANDIDATE (Accuracy: 72.50%, Macro F1: 0.7189)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |     25 |          3 |       1 |        11 |    40  <-- Glioma recall dropped to 62.5%
Meningioma   |      0 |         16 |       4 |        20 |    40  <-- Quadrupled recall (10% -> 40%)
NoTumor      |      0 |          0 |      36 |         4 |    40  <-- Preserved at 90.0%
Pituitary    |      0 |          1 |       0 |        39 |    40  <-- Preserved at 97.5%
```

---

## 6. Analysis of the Glioma / Meningioma Trade-Off

### 6.1 The Progression of Models Across Experiments
To understand the dynamics, examine how Glioma and Meningioma recall traded off across the five models:

```
Experiment Trajectory on Independent Test Set:
Model                                      Glioma Recall      Meningioma Recall      Macro F1
---------------------------------------------------------------------------------------------
Original Deployed Model                         87.5%               10.0%             0.6668
Candidate 1 (Unfreeze 30 layers)                 7.5%               40.0%             0.5138
Candidate 2 (100% Frozen ImageNet)               0.0%               60.0%             0.4905
Candidate 3 (ImageNet Shallow Fine-Tune)         7.5%               52.5%             0.5277
Candidate 4 (Warm-Start Block 16 + Equal Wt)    62.5%               40.0%             0.7189
```

### 6.2 Key Takeaways:
1. **Warm-Starting Dramatically Rescued Glioma Detection:** Initializing from `brain_tumor_mobilenet_v2.keras` elevated Glioma recall from **0%–7.5% in Candidates 1–3 up to 62.5%**, validating the hypothesis that the original model possesses learned MRI convolutional filters that are absent in raw ImageNet weights.
2. **Meningioma Recall Quadrupled:** Training on `dataset_clean/train` with equal class weights allowed Block 16 and the classification head to pull Meningioma features away from Pituitary, quadrupling Meningioma recall from **10.0% to 40.0%** (16/40) on independent test data and from **8.0% to 38.0%** (19/50) on validation data.
3. **The Unresolved Glioma Leakage to Pituitary:** Despite equal class weighting, 11 of the 40 independent glioma scans were predicted as Pituitary. In the 4-class softmax output, pituitary adenomas remain the default attractor for central high-intensity brain regions.
4. **Clinical Evaluation:** In medical imaging, missing 37.5% of gliomas (15/40) is clinically unacceptable when the production model detects 87.5% (35/40). Thus, despite achieving the highest overall test accuracy (72.50%) and highest Macro F1 (0.7189) of any model tested, Candidate 4 does not unequivocally satisfy the replacement criterion.

---

## 7. Grad-CAM & API Compatibility Verification

- Tested via `ai_engine.run_classification_and_gradcam()` on `sample_images/sample_mri_glioma.png`.
- Output: Successfully predicted **Glioma (Intra-axial Neoplasm)** with **83.51% confidence** and latency of **26.81 ms**.
- The attention heatmap localized cleanly to the parenchymal hyperintensity in the temporal region.

---

## 8. Final Status & Recommendations

### Final Gate Verdict: **REJECT DEPLOYMENT OF `brain_tumor_mobilenet_v2_warmstart.keras`**
- The candidate model improves overall Macro F1 (0.7189 vs 0.6668) and Meningioma recall (40% vs 10%), but reduces Glioma recall (62.5% vs 87.5%).
- **Production Status:** `brain_tumor_mobilenet_v2.keras` remains active in the running FastAPI application.

### Next Step Recommendation:
To overcome the remaining trade-off without sacrificing Glioma recall, future research should explore:
1. **Hierarchical / Two-Stage Classification:**
   - Stage 1: Detect **Tumor vs. NoTumor** and **Intra-axial (Glioma) vs. Extra-axial (Meningioma/Pituitary)**. The original model already separates intra-axial from extra-axial with $>90\%$ accuracy.
   - Stage 2: A specialized binary head dedicated exclusively to differentiating **Meningioma vs. Pituitary**, eliminating the interference on Glioma.
2. **Cost-Sensitive Focal Loss:** Applying a targeted penalty specifically to the Meningioma/Pituitary pair while imposing a high penalty on Glioma false negatives.

---

## 9. Educational Research Prototype Disclaimer

*NeuroSyntra AI is an educational research and development platform developed for studying deep convolutional transfer learning, dataset isolation, and latent space geometry in medical computer vision. It is NOT certified as a medical device (SaMD) under FDA, CE-MDR, or ISO 13485 regulations. It must not be used for clinical diagnosis, patient triage, surgical planning, or real-world medical decision-making.*
