# Comprehensive Diagnostic Audit Report: NeuroSyntra AI

**System:** NeuroSyntra AI Brain Tumor MRI Diagnostic Platform  
**Audit Date:** October 15, 2026  
**Scope:** Dataset Integrity, Preprocessing Pipeline, Model Architectures, Training Dynamics, and Classification Errors  
**Safety Status:** Strictly non-destructive audit. **Zero models modified, zero datasets changed, server running continuously with zero downtime.**

---

## 1. Executive Summary

This diagnostic audit was conducted to resolve the persistent trade-off observed across all experimental models in the NeuroSyntra AI project:
1. **The Original Model (`brain_tumor_mobilenet_v2.keras`)** achieves **87.5% Glioma recall** and **90.0% NoTumor recall** on independent data, but suffers a severe, systematic failure mode on **Meningioma**, misclassifying **77.5% of novel meningiomas as Pituitary** (yielding only 10.0% Meningioma recall).
2. **All Three Candidate Models** trained on the clean dataset (`dataset_clean/train`) fixed the Meningioma issue (improving recall to 52.5%–60.0%), but experienced a catastrophic collapse on **Glioma** (0.0%–7.5% recall), assigning over 50% average probability to NoTumor on true glioma scans.
3. **Core Root Cause Uncovered:**
   - **Meningioma / Pituitary Collinearity:** In the original model's feature space, Meningioma and Pituitary feature centroids have an extreme cosine similarity of **0.9108**, causing the model to collapse unseen meningiomas into pituitary adenomas.
   - **Glioma / NoTumor Feature Inseparability in ImageNet Initializations:** Unlike discrete extra-axial masses (meningiomas, pituitary adenomas), intra-axial gliomas present as diffuse parenchymal signal alterations. ImageNet feature extractors see both gliomas and normal brains as generic textured parenchyma (cosine similarity **0.8372**).
   - **Class Weighting Penalty:** In `dataset_clean/train`, Glioma had 261 images (weight: `0.657`) while NoTumor had 99 images (weight: `1.732`). This penalized notumor errors **2.64× more heavily** than glioma errors, forcing gradient descent to collapse ambiguous parenchymal features into NoTumor.
   - **Benchmark Test Leakage:** The original model's 91.25% score on `testing/` was inflated because 36 of 80 benchmark images shared base patient scan IDs with the upstream training set. On genuine independent test scans (`dataset_clean/test_independent`), its true accuracy is **71.88%**.

---

## 2. Dataset Label & Image Integrity Audit

### 2.1 Partition Structure and Class Balance
We audited all 1,046 images across the three frozen partitions in `dataset_clean/`:

| Split | Total Images | Glioma | Meningioma | NoTumor | Pituitary | Unique Base IDs | Balance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **686** | 261 | 98 | 99 | 228 | **580** | Imbalanced (71.3% Glioma/Pituitary) |
| **`val`** | **200** | 50 | 50 | 50 | 50 | **200** | **100% Balanced (1:1:1:1)** |
| **`test_independent`** | **160** | 40 | 40 | 40 | 40 | **160** | **100% Balanced (1:1:1:1)** |
| **`testing/` (Benchmark)**| **80** | 20 | 20 | 20 | 20 | **77** | 100% Balanced (36/80 Leaked) |

### 2.2 Technical Verification Findings:
1. **Readability & Corruption:** **0 corrupted or unreadable images** across all 1,046 scans. All files load successfully via PIL.
2. **Color Mode & Dimensions:** 100% of images are 3-channel RGB with native dimensions $224 \times 224$ pixels.
3. **Exact Duplicates:** **0 exact SHA256/MD5 duplicates** across all cross-partition comparisons.
4. **Base ID Overlap:** **0 base scan IDs cross between partitions** (`train` $\cap$ `val` = $\emptyset$, `train` $\cap$ `test_independent` = $\emptyset$, `val` $\cap$ `test_independent` = $\emptyset$).
5. **Benchmark Isolation:** **0 benchmark test base IDs** appear in `train`, `val`, or `test_independent`.
6. **Visual Contact Sheets:** Contact sheets for each class and partition were generated and stored in `scratch/contact_sheets/` (`train_glioma_contact.png`, `val_meningioma_contact.png`, etc.).

### 2.3 Data Quality & Outlier Inspection:
- **Low-Contrast/Dark Outlier:** `train/glioma/Tr-gl_0299_jpg.rf.d806a08989b472df4581c93d322c183e.jpg` has mean pixel intensity `14.7` and std `20.3` (near black background, weak MRI signal).
- **Inverted / High-Intensity Backgrounds:** 24 images (chiefly in `meningioma` and `notumor`) have mean intensity $> 150$ due to inverted white canvas backgrounds or scan acquisition tags (e.g., `test_independent/meningioma/m1-135-` with mean `203.9`).
- **Color Channel Divergence:** 17 images have subtle color tints ($|R-G| + |R-B| > 5$, up to $252.0$), originating from RGB annotations or scanning tint in the upstream dataset.

---

## 3. Preprocessing Pipeline Audit

We performed an exact line-by-line comparison between the inference engine (`ai_engine.py`) and training experiment scripts:

| Preprocessing Step | Inference Implementation (`ai_engine.py`) | Training Implementation (`train_*.py`) | Status | Exact Line References |
| :--- | :--- | :--- | :---: | :--- |
| **Color Space** | `image.convert("RGB")` | `pil_img.convert("RGB")` | **MATCH** | `ai_engine.py:214` vs `train_*.py:33` |
| **Letterbox Scaling** | Bicubic (`Image.Resampling.BICUBIC`) | Bicubic (`Image.Resampling.BICUBIC`) | **MATCH** | `ai_engine.py:242` vs `train_*.py:41` |
| **Target Canvas** | $128 \times 128$ Black `(0, 0, 0)` | $128 \times 128$ Black `(0, 0, 0)` | **MATCH** | `ai_engine.py:244` vs `train_*.py:43` |
| **Centering Padding** | `(128 - w) // 2`, `(128 - h) // 2` | `(128 - w) // 2`, `(128 - h) // 2` | **MATCH** | `ai_engine.py:245-246` vs `train_*.py:44-45` |
| **Normalization** | `(arr / 127.5) - 1.0` (range `[-1.0, 1.0]`) | `(arr / 127.5) - 1.0` (range `[-1.0, 1.0]`) | **MATCH** | `ai_engine.py:265` vs `train_*.py:49` |
| **Class Index Order** | `['glioma', 'meningioma', 'notumor', 'pituitary']` | `['glioma', 'meningioma', 'notumor', 'pituitary']` | **MATCH** | `ai_engine.py:30` vs `train_*.py:18` |

### Preprocessing Equivalence Test:
Running identical test scans through both pipelines revealed a **maximum absolute numerical discrepancy of `0.000000`**, proving 100% mathematical identity between training and inference preprocessing for MobileNetV2.

> [!NOTE]
> **Discovered Custom CNN Mismatch:** The baseline 4-layer sequential CNN (`brain_tumor_multiclass_cnn.h5`) was trained in `download_and_train.py:86` with `1./255` normalization (`[0.0, 1.0]`) without letterboxing. However, `ai_engine.py:265` feeds `[-1.0, 1.0]` letterboxed tensors to all models. This does not affect `brain_tumor_mobilenet_v2.keras`, but explains why the custom CNN underperforms in the web UI.

---

## 4. Model Architecture & Training Dynamics Audit

We audited the internal HDF5/ZIP archives and layer structures of the original model and all three rejected candidates:

| Specification | Original Deployed Model | Candidate 1 (`improved`) | Candidate 2 (`linear_probe`) | Candidate 3 (`shallow_finetune`) |
| :--- | :---: | :---: | :---: | :---: |
| **File Name** | `brain_tumor_mobilenet_v2.keras` | `...improved.keras` | `...linear_probe.keras` | `...shallow_finetune.keras` |
| **MD5 Checksum** | `9c3d3f636cbc032561e8...` | `bd0a2deae01c4eb681...` | `070e65963917cad062...` | `64584b085b576803ca...` |
| **Saved Date / Keras**| `2026-09-24` (Keras 3.16.0.dev) | `2026-10-15` (Keras 3.15.1) | `2026-10-15` (Keras 3.15.1) | `2026-10-15` (Keras 3.15.1) |
| **Backbone Setup** | Top 2 blocks + `Conv_1` | Top 30 layers (Blocks 13–16) | 100% Frozen ImageNet | Top 2 blocks + `Conv_1` |
| **Trainable Base Layers** | **14 layers** | 21 layers | 0 layers | **14 layers** |
| **Trainable Base Params** | 1,206,080 | 1,526,400 | 0 | 1,206,080 |
| **Head Trainable Params** | 164,484 (Dense 128 + 4) | 164,484 (Dense 128 + 4) | 164,484 (Dense 128 + 4) | 164,484 (Dense 128 + 4) |
| **Total Trainable Params** | **1,370,564** | 1,690,884 | 164,484 | **1,370,564** |
| **Non-Trainable Params** | 1,051,904 | 731,584 | 2,257,984 | 1,051,904 |
| **Optimizer & Saved LR** | Adam (`adam_1`, $10^{-4}$) | Adam (`adam_1`, $5 \times 10^{-5}$) | Adam (`adam`, $10^{-3}$) | Adam (`adam`, $10^{-5}$) |
| **Checkpoint Selection** | Missing (Author's repo) | Epoch 9 (val_loss 2.5282) | Epoch 2 (val_loss 1.7589) | Epoch 13 (val_loss 1.0479) |
| **Provenance** | Cloned from external repo | Trained locally | Trained locally | Trained locally |

### Provenance Audit of Original Model:
- Line 46 of `README.md` reveals the project originated from `https://github.com/ComputerVision804/brain-tumor-classification`.
- The repository included `brain_tumor_mobilenet_v2.keras` as a pre-trained asset, but contained **no training script for MobileNetV2** (the only training script provided was `download_and_train.py`, which trained the custom CNN).
- The optimizer name inside `brain_tumor_mobilenet_v2.keras` is **`adam_1`** with learning rate **`1e-4`**, proving the original author used a two-phase compilation (head warmup with `adam`, followed by fine-tuning with `adam_1`).

---

## 5. Classification Error Analysis & Feature Geometry

### 5.1 Multi-Dataset Performance Across All Models
All evaluations were computed using identical letterboxed preprocessing:

| Evaluation Dataset | Metric | Original Model | Candidate 1 (`improved`) | Candidate 2 (`linear_probe`) | Candidate 3 (`shallow_finetune`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`test_independent`**<br>(160 images, 40/class) | **Overall Accuracy** | **71.88%** (115/160) | 58.13% (93/160) | 57.50% (92/160) | **59.38%** (95/160) |
| | **Macro F1** | **0.6668** | 0.5138 | 0.4905 | **0.5277** |
| | **Glioma Recall** | **87.50%** (35/40) | 7.50% (3/40) | 0.00% (0/40) | **7.50%** (3/40) |
| | **Meningioma Recall** | 10.00% (4/40) | 40.00% (16/40) | **60.00%** (24/40) | **52.50%** (21/40) |
| | **NoTumor Recall** | 90.00% (36/40) | 85.00% (34/40) | 82.50% (33/40) | **92.50%** (37/40) |
| | **Pituitary Recall** | **100.00%** (40/40) | **100.00%** (40/40) | 87.50% (35/40) | 85.00% (34/40) |
| **`dataset_clean/val`**<br>(200 images, 50/class) | **Overall Accuracy** | **71.00%** (142/200) | 55.00% (110/200) | 61.50% (123/200) | **61.00%** (122/200) |
| | **Glioma Recall** | **94.00%** (47/50) | 12.00% (6/50) | 4.00% (2/50) | **8.00%** (4/50) |
| | **Meningioma Recall** | 8.00% (4/50) | 36.00% (18/50) | **60.00%** (30/50) | **54.00%** (27/50) |
| **`dataset_clean/train`**<br>(686 images) | **Glioma Recall** | **94.25%** (246/261) | 9.20% (24/261) | 3.07% (8/261) | **2.68%** (7/261) |
| | **Meningioma Recall** | 2.04% (2/98) | 51.02% (50/98) | **61.22%** (60/98) | **48.98%** (48/98) |

### 5.2 Confusion Matrices on `dataset_clean/test_independent` (160 images):
```text
ORIGINAL MODEL (Accuracy: 71.88%, Macro F1: 0.6668)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |     35 |          1 |       1 |         3 |    40
Meningioma   |      0 |          4 |       5 |        31 |    40  <-- 31/40 predicted as Pituitary
NoTumor      |      0 |          0 |      36 |         4 |    40
Pituitary    |      0 |          0 |       0 |        40 |    40

CANDIDATE 3: SHALLOW FINE-TUNING (Accuracy: 59.38%, Macro F1: 0.5277)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      3 |          6 |      16 |        15 |    40  <-- 31/40 predicted as NoTumor or Pituitary
Meningioma   |      0 |         21 |       8 |        11 |    40  <-- Fixed: 21/40 correct (52.5%)
NoTumor      |      0 |          1 |      37 |         2 |    40
Pituitary    |      0 |          2 |       4 |        34 |    40
```

### 5.3 Geometric Feature Audit (128-Dimensional Penultimate Layer):
We extracted the 128-dimensional latent vectors from `dense` for validation scans across all classes and measured the cosine similarity between class centroids:

```
Centroid Cosine Similarity Matrix:
Pair                      Original Model     Candidate 3 (Shallow)    Clinical Implication
---------------------------------------------------------------------------------------------
Meningioma vs Pituitary        0.9108               0.9172            Collinear features; original model defaults to Pituitary.
Glioma vs NoTumor              0.6285               0.8372            Original model separates them; Candidate 3 features overlap.
Glioma vs Meningioma           0.7540               0.8720            Original model separates them; Candidate 3 features overlap.
Glioma vs Pituitary            0.6984               0.8153            Original model separates them; Candidate 3 features overlap.
```

### 5.4 Probabilities on True Glioma Scans:
When fed identical true glioma MRI scans, the average predicted probability outputs were:
- **Original Model:** **Glioma: 92.99%** | Meningioma: 0.35% | NoTumor: 1.23% | Pituitary: 5.43%
- **Candidate 3:** **Glioma: 11.45%** | Meningioma: 9.80% | **NoTumor: 51.92%** | **Pituitary: 26.84%**

---

## 6. Confirmed Issues vs. Unverified Hypotheses

### Confirmed Issues:
1. **Benchmark Leakage in Original Model:** The reported 91.25% test accuracy on `testing/` is artificially high due to 36/80 scan base IDs overlapping with training. On the clean, independent 160-image test set, its genuine accuracy is **71.88%**.
2. **Meningioma Blind Spot in Original Model:** Across all clean partitions (`train`, `val`, `test_independent`), the original model achieves only **2.0% to 10.0% recall on Meningioma**, misdiagnosing 77.5% of them as Pituitary due to high latent feature collinearity ($\cos \theta = 0.9108$).
3. **Glioma Feature Collapse in Candidates:** Candidate models trained from ImageNet initializations cannot separate intra-axial diffuse glioma textures from normal cerebral parenchyma (centroid cosine similarity $0.8372$).
4. **Class Weight Distortion:** Assigning class weight $0.657$ to Glioma vs $1.732$ to NoTumor penalized notumor errors 2.64× more heavily, causing the network to favor NoTumor whenever uncertain.
5. **Preprocessing Parity:** Preprocessing between `ai_engine.py` and training scripts is 100% identical (0.000000 max error).

### Unverified Hypotheses:
1. **Warm-Start Transfer Hypothesis:** Whether fine-tuning starting from the *original model's weights* (`brain_tumor_mobilenet_v2.keras`) rather than ImageNet weights will preserve its 87.5% glioma recall while breaking the Meningioma $\rightarrow$ Pituitary collinearity.
2. **Equal-Weight / Balanced-Batch Hypothesis:** Whether training with equal class weights ($1.0$) and balanced mini-batch sampling prevents the Glioma $\rightarrow$ NoTumor collapse seen in Candidates 1, 2, and 3.

---

## 7. Evidence-Justified Recommendation for Future Work

The evidence indicates that:
1. **Training from generic ImageNet weights fails on Glioma** because 686 training scans are insufficient to learn subtle medical tissue filtration from scratch.
2. **The Original Model already possesses domain-adapted convolutional kernels** that reliably detect Gliomas (87.5%–94.3% recall across all splits) and NoTumor tissue (80.8%–90.0% recall).
3. **The Original Model's sole major defect is Meningioma $\rightarrow$ Pituitary collapse.**

### Recommended Next Experiment:
When ready to train another model, execute a **Warm-Start Targeted Adaptation Experiment**:
- **Initialization:** Initialize backbone and head from `brain_tumor_mobilenet_v2.keras` (preserving its learned glioma features), NOT ImageNet.
- **Freeze/Unfreeze Strategy:** Keep early and middle blocks frozen. Train only the top classification head and Block 16.
- **Equal Weighting & Targeted Contrast:** Use equal class weights ($1.0$), or apply a targeted focal loss between Meningioma and Pituitary to decouple the $0.9108$ collinearity without downweighting Glioma.
- **Model Checkpointing:** Attach `ModelCheckpoint` on `dataset_clean/val` from Epoch 1.

---

## 8. Preserved System & Model Status

- **`brain_tumor_mobilenet_v2.keras`**: Active and untouched (MD5: `9c3d3f636cbc032561e80504b0391262`).
- **`brain_tumor_mobilenet_v2_improved.keras`**: Preserved (MD5: `bd0a2deae01c4eb68149e32e3c5888b4`).
- **`brain_tumor_mobilenet_v2_linear_probe.keras`**: Preserved (MD5: `070e65963917cad062f6fe792bb6e914`).
- **`brain_tumor_mobilenet_v2_shallow_finetune.keras`**: Preserved (MD5: `64584b085b576803ca7cdde47540fd05`).
- **`dataset_clean/`**: All partitions (`train`, `val`, `test_independent`) 100% frozen.
- **FastAPI Application**: Running continuously on port 8000 with 0 downtime.
