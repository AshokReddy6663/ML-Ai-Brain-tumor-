# Shallow Fine-Tuning Experiment Report: MobileNetV2 Top-2 Blocks

**Date:** October 15, 2026  
**System:** NeuroSyntra AI Brain Tumor MRI Diagnostic Engine  
**Experiment Model File:** `brain_tumor_mobilenet_v2_shallow_finetune.keras` (MD5: `64584b085b576803ca7cdde47540fd05`)  
**Production Model File:** `brain_tumor_mobilenet_v2.keras` (MD5: `9c3d3f636cbc032561e80504b0391262`, **100% UNCHANGED**)  
**Linear Probe Model File:** `brain_tumor_mobilenet_v2_linear_probe.keras` (MD5: `070e65963917cad062f6fe792bb6e914`, **100% UNCHANGED**)  
**Previous Overfitted Model:** `brain_tumor_mobilenet_v2_improved.keras` (MD5: `bd0a2deae01c4eb68149e32e3c5888b4`, **100% UNCHANGED**)  
**FastAPI Production Server:** `http://127.0.0.1:8000` (**Running continuously, 0 downtime**)  
**Status:** Complete. **Candidate REJECTED for deployment; Production Model Retained.**

---

## 1. Executive Summary & Verdict

Following the approval to investigate a controlled shallow-fine-tuning experiment, we trained a candidate model starting from ImageNet MobileNetV2 initialization, unfreezing **only the top two residual blocks** (`block_15` and `block_16`) plus the top projection layer (`Conv_1`), exactly mirroring the 14 fine-tuned base layers discovered in the original production model. Training was conducted with a low learning rate of `1e-5`, class weighting, medical augmentation, and `ModelCheckpoint` attached from Epoch 1.

### Core Empirical Findings:
1. **Convergence & Validation Checkpoint:**
   - The validation loss smoothly improved from `1.4660` (Epoch 1) to its global minimum of **`1.0479`** at **Epoch 13** (`val_accuracy: 61.00%`).
   - `ModelCheckpoint` captured this optimal checkpoint. Early stopping triggered at Epoch 23; the reloaded model confirmed exact replication of validation metrics (`val_loss: 1.0479`, `val_acc: 61.00%`, Macro F1: `0.5421`).
2. **Independent Test Set Performance (160 balanced images):**
   - **Original Deployed Model:** **71.88%** Accuracy | Macro F1: **0.6668** | Glioma Recall: **87.50%**
   - **Linear Probe Candidate:** **57.50%** Accuracy | Macro F1: **0.4905** | Glioma Recall: **0.00%**
   - **Shallow Fine-Tuned Candidate:** **59.38%** Accuracy | Macro F1: **0.5277** | Glioma Recall: **7.50%**
3. **Glioma Diagnosis & Root Cause:**
   - Unfreezing the top two blocks at `1e-5` enabled marginal adaptation (Glioma recall rose from 0% in linear probing to 7.5% with 3/40 correct), but 37 of 40 glioma test scans were still misclassified (16 $\rightarrow$ notumor, 15 $\rightarrow$ pituitary, 6 $\rightarrow$ meningioma).
   - Although the candidate matches the original model's trainable layer structure (1,370,564 trainable parameters across 14 backbone layers + head), **it does not reproduce the original model's glioma performance**.
   - As documented in Safeguard #6, the original model benefited from training on a dataset where 36 of 80 benchmark scans had augmented siblings in training, and its weights followed an optimization path that learned subtle parenchymal hyperintensities. Adapting natural-image ImageNet weights to intra-axial glioma textures on only 686 clean training images remains bounded.
4. **Action:**
   - **Candidate REJECTED for deployment.**
   - `brain_tumor_mobilenet_v2.keras` remains the active production model.

---

## 2. Safeguard Verification Checklist

| Safeguard | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **#1** | New script `train_shallow_finetune_experiment.py` | **PASSED** | Script created and executed. |
| **#2** | Separate model `...shallow_finetune.keras`; preserve others & app | **PASSED** | Saved separately (11 MB). MD5s of all 3 previous models identical; FastAPI active on port 8000. |
| **#3** | ImageNet MobileNetV2 init, head design, freeze except top 1-2 blocks, lr around 1e-5 | **PASSED** | Unfroze Block 15, Block 16, Conv_1 (14 base layers); lr = `1e-5` with Adam. |
| **#4** | Deployed preprocessing, class mapping, augmentation, class weighting | **PASSED** | Letterbox bicubic to 128x128, [-1, 1], classes 0-3; weights: gl=0.657, me=1.750, no=1.732, pi=0.752. |
| **#5** | Best checkpoint from Epoch 1, early stopping with restore, per-epoch logging | **PASSED** | Checkpoint active from Epoch 1; best saved at Epoch 13 (`val_loss: 1.0479`); early stopping at Epoch 23. |
| **#6** | Pre-training inspection of original vs candidates; document differences | **PASSED** | Full layer-by-layer parameter audit completed and documented below before training. |
| **#7** | Assessment on validation data only; test sets untouched during training | **PASSED** | Checkpoints selected solely on `dataset_clean/val` (200 balanced images). |
| **#8** | Reload best checkpoint; multi-metric validation evaluation | **PASSED** | Reloaded model verified at `val_loss: 1.0479`, `val_acc: 61.00%`; full confusion matrix and per-class metrics computed. |
| **#9** | Single evaluation on untouched independent test set (160 images) | **PASSED** | Evaluated once on `test_independent` and compared side-by-side with original model on identical images. |
| **#10** | Keep app running; zero automatic deployment | **PASSED** | Application untouched, candidate not deployed, report compiled for review. |

---

## 3. Pre-Training Architectural Inspection & Differences (Safeguard #6)

Before training, an inspection was conducted comparing the original deployed model, Candidate 1 (`improved`), Candidate 2 (`linear_probe`), and the new Candidate 3 (`shallow_finetune`):

| Model | Backbone Status | Trainable Base Layers | Trainable Base Params | Non-Trainable Params | Head Trainable Params | Total Trainable Params | Effective Training LR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Deployed** (`mobilenet_v2.keras`) | Top 2 blocks + Conv_1 | 14 | 1,206,080 | 1,051,904 | 164,484 | **1,370,564** | $10^{-4}$ |
| **Candidate 1** (`...improved.keras`) | Top 30 layers (Blocks 13-16) | 21 | 1,526,400 | 731,584 | 164,484 | **1,690,884** | $10^{-4} \rightarrow 5 \times 10^{-5}$ |
| **Candidate 2** (`...linear_probe.keras`) | 100% Frozen ImageNet | 0 | 0 | 2,257,984 | 164,484 | **164,484** | $10^{-3}$ |
| **Candidate 3** (`...shallow_finetune.keras`) | Top 2 blocks + Conv_1 | 14 | 1,206,080 | 1,051,904 | 164,484 | **1,370,564** | $10^{-5}$ |

### The Exact 14 Trainable Base Layers in Candidate 3 and Original Model:
1. `block_15_expand` (Conv2D)
2. `block_15_expand_BN` (BatchNormalization)
3. `block_15_depthwise` (DepthwiseConv2D)
4. `block_15_depthwise_BN` (BatchNormalization)
5. `block_15_project` (Conv2D)
6. `block_15_project_BN` (BatchNormalization)
7. `block_16_expand` (Conv2D)
8. `block_16_expand_BN` (BatchNormalization)
9. `block_16_depthwise` (DepthwiseConv2D)
10. `block_16_depthwise_BN` (BatchNormalization)
11. `block_16_project` (Conv2D)
12. `block_16_project_BN` (BatchNormalization)
13. `Conv_1` (Conv2D)
14. `Conv_1_bn` (BatchNormalization)

### Why Unfreezing These Two Blocks Does Not Automatically Reproduce the Original Model:
1. **Upstream Training Dataset & Data Leakage:**
   - The original model was trained on the upstream Roboflow dataset (299 images), where 36 out of 80 benchmark test images (`testing/`) shared base patient scan IDs with training images. This allowed the original model to memorize augmented versions of specific scans.
   - Candidate 3 was trained strictly on `dataset_clean/train` (686 images), which has zero base-ID or perceptual-hash overlap with `dataset_clean/test_independent`.
2. **Initial Weight Trajectory:**
   - Candidate 3 initialized Block 15, Block 16, and `Conv_1` with **natural-image ImageNet weights** (filters trained on dogs, cars, textures).
   - In the original model, those weights were trained specifically on medical MRI scans.
3. **Head vs. Backbone Gradient Dynamics:**
   - Candidate 3 started with a randomly initialized classification head (Glorot uniform) alongside ImageNet backbone weights. At a gentle learning rate of `1e-5`, gradient descent prioritizes setting the head weights first before slowly fine-tuning the 1.2M backbone parameters. In 23 epochs (989 steps), total parameter movement is small ($\approx 0.01$), which avoids destroying ImageNet weights but limits the depth of medical domain adaptation.

---

## 4. Training Progression & Checkpoint Selection

Training ran for 23 epochs on `dataset_clean/train` (686 images) with validation on `dataset_clean/val` (200 images):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Learning Rate | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 1.3773 | 33.24% | 1.4660 | 29.00% | 0.000010 | Initial Checkpoint (`val_loss: 1.4660`) |
| 2 | 0.9323 | 60.93% | 1.3616 | 36.00% | 0.000010 | Saved (`val_loss: 1.3616`) |
| 3 | 0.6667 | 79.01% | 1.2938 | 41.00% | 0.000010 | Saved (`val_loss: 1.2938`) |
| 4 | 0.6363 | 77.99% | 1.2158 | 49.50% | 0.000010 | Saved (`val_loss: 1.2158`) |
| 5 | 0.4576 | 85.13% | 1.1869 | 50.50% | 0.000010 | Saved (`val_loss: 1.1869`) |
| 6 | 0.4279 | 88.63% | 1.1416 | 55.00% | 0.000010 | Saved (`val_loss: 1.1416`) |
| 7 | 0.4049 | 87.90% | 1.1147 | 56.50% | 0.000010 | Saved (`val_loss: 1.1147`) |
| 8 | 0.3638 | 89.36% | 1.0831 | 58.50% | 0.000010 | Saved (`val_loss: 1.0831`) |
| 9 | 0.3288 | 89.36% | 1.0758 | 58.50% | 0.000010 | Saved (`val_loss: 1.0758`) |
| 10 | 0.3149 | 90.96% | 1.0825 | 59.50% | 0.000010 | Did not improve |
| 11 | 0.2777 | 92.71% | 1.0570 | 60.00% | 0.000010 | Saved (`val_loss: 1.0570`) |
| 12 | 0.2697 | 94.02% | 1.0500 | 60.50% | 0.000010 | Saved (`val_loss: 1.0500`) |
| **13** | **0.2246** | **93.44%** | **1.0479** | **61.00%** | **0.000010** | **Best Checkpoint Saved (`val_loss: 1.0479`)** |
| 14 | 0.2164 | 94.02% | 1.0596 | 61.00% | 0.000010 | Did not improve |
| 15 | 0.1921 | 94.90% | 1.0671 | 60.50% | 0.000010 | Did not improve |
| 16 | 0.2001 | 93.59% | 1.0704 | 60.50% | 0.000010 | Did not improve |
| 17 | 0.1918 | 95.04% | 1.0746 | 61.50% | 0.000005 | LR reduced to $5 \times 10^{-6}$ |
| 18 | 0.1946 | 95.77% | 1.0763 | 63.00% | 0.000005 | Did not improve |
| 19 | 0.2094 | 94.46% | 1.0788 | 63.00% | 0.000005 | Did not improve |
| 20 | 0.1582 | 96.50% | 1.0844 | 63.00% | 0.000003 | LR reduced to $2.5 \times 10^{-6}$ |
| 21 | 0.1426 | 96.79% | 1.0928 | 64.00% | 0.000003 | Did not improve |
| 22 | 0.1977 | 94.46% | 1.0994 | 64.00% | 0.000003 | Did not improve |
| 23 | 0.1738 | 95.04% | 1.1039 | 64.00% | 0.000001 | Early stopping triggered (`patience=10`) |

### Training History Observations:
- **Smooth Validation Curve:** Unlike Candidate 1 (which blew up to `val_loss: 2.5282`), the validation loss remained controlled between $1.0479$ and $1.1039$, demonstrating that `lr = 1e-5` prevented catastrophic forgetting.
- **Reload Verification:** Reloading `brain_tumor_mobilenet_v2_shallow_finetune.keras` from disk confirmed exact restoration of the Epoch 13 weights: `val_loss: 1.0479`, `val_accuracy: 61.00%`.

---

## 5. Validation Set Breakdown (`dataset_clean/val`, 200 images, 50/class)

- **Accuracy:** 61.00% (122 / 200) | **Macro F1:** 0.5421 | **Macro Precision:** 0.7063 | **Macro Recall:** 0.6100

| Class | Support | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glioma** | 50 | 1.0000 | **0.0800 (4/50)** | **0.1481** | **Severe Under-detection (46/50 misclassified)** |
| **Meningioma** | 50 | 0.6429 | 0.5400 (27/50) | 0.5870 | Moderate |
| **NoTumor** | 50 | 0.5823 | **0.9200 (46/50)** | **0.7132** | Strong |
| **Pituitary** | 50 | 0.6000 | **0.9000 (45/50)** | **0.7200** | Strong |

### Confusion Matrix on Validation Set:
```
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      4 |         14 |      17 |        15 |    50
Meningioma   |      0 |         27 |      11 |        12 |    50
NoTumor      |      0 |          1 |      46 |         3 |    50
Pituitary    |      0 |          0 |       5 |        45 |    50
```

---

## 6. Independent Test Set Evaluation (`dataset_clean/test_independent`, 160 images, 40/class)

All models were evaluated on the exact same 160 independent test images using deployed letterboxed preprocessing:

### 4-Way Cross-Experiment Benchmark:
| Metric | Original Deployed (`mobilenet_v2`) | Candidate 1 (`improved` - Unfreeze 30) | Candidate 2 (`linear_probe` - Frozen) | Candidate 3 (`shallow_finetune` - Top 2 Blocks) |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **71.88%** (115/160) | 58.13% (93/160) | 57.50% (92/160) | **59.38%** (95/160) |
| **Macro F1** | **0.6668** | 0.5138 | 0.4905 | **0.5277** |
| **Macro Precision** | **0.7925** | 0.7173 | 0.4298 | **0.7044** |
| **Macro Recall** | **0.7188** | 0.5813 | 0.5750 | **0.5938** |
| **Glioma Precision** | **1.0000** | 1.0000 | 0.0000 | **1.0000** |
| **Glioma Recall** | **87.50%** (35/40) | 7.50% (3/40) | 0.00% (0/40) | **7.50% (3/40)** |
| **Glioma F1** | **0.9333** | 0.1395 | 0.0000 | **0.1395** |
| **Meningioma Recall**| 10.00% (4/40) | 40.00% (16/40) | **60.00% (24/40)** | **52.50% (21/40)** |
| **Meningioma F1** | 0.1778 | 0.5079 | 0.5517 | **0.6000** |
| **NoTumor Recall** | 90.00% (36/40) | 85.00% (34/40) | 82.50% (33/40) | **92.50% (37/40)** |
| **NoTumor F1** | **0.8780** | 0.7727 | 0.7174 | 0.7048 |
| **Pituitary Recall** | **100.00%** (40/40) | **100.00%** (40/40) | 87.50% (35/40) | 85.00% (34/40) |
| **Pituitary F1** | 0.6780 | 0.6349 | **0.6931** | 0.6667 |

### Confusion Matrix Comparison on Independent Test Set:

```text
ORIGINAL DEPLOYED MODEL (71.88% Accuracy, Macro F1: 0.6668)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |     35 |          1 |       1 |         3 |    40
Meningioma   |      0 |          4 |       5 |        31 |    40
NoTumor      |      0 |          0 |      36 |         4 |    40
Pituitary    |      0 |          0 |       0 |        40 |    40

CANDIDATE 3: SHALLOW FINE-TUNING (59.38% Accuracy, Macro F1: 0.5277)
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      3 |          6 |      16 |        15 |    40
Meningioma   |      0 |         21 |       8 |        11 |    40
NoTumor      |      0 |          1 |      37 |         2 |    40
Pituitary    |      0 |          2 |       4 |        34 |    40
```

---

## 7. Benchmark Test Set Evaluation (`testing/`, 80 images)

| Metric | Original Deployed Model | Shallow Fine-Tuned Candidate |
| :--- | :---: | :---: |
| **Accuracy** | **91.25%** (73/80) | 35.00% (28/80) |
| **Macro F1** | **0.9110** | 0.2923 |
| **Glioma Recall** | **95.0%** (19/20) | **5.0%** (1/20) |
| **Meningioma Recall**| **100.0%** (20/20) | 10.0% (2/20) |
| **NoTumor Recall** | **70.0%** (14/20) | 55.0% (11/20) |
| **Pituitary Recall** | **100.0%** (20/20) | 70.0% (14/20) |

---

## 8. Diagnostic Synthesis & Key Insights

1. **Glioma Sensitivity Remains the Primary Barrier:**
   - Both Candidate 1 (unfreezing 30 layers) and Candidate 3 (unfreezing 14 layers) achieved only **7.5% Glioma recall** (3/40) on the independent test set.
   - Candidate 2 (frozen backbone) achieved **0.0% Glioma recall** (0/40).
   - This proves that unfreezing deeper layers allows tiny improvements in glioma feature adaptation, but natural ImageNet weights cannot be fully adapted to diffuse intra-axial glioma pathology on only 686 training scans without either larger datasets or medical-domain pre-training.
2. **Meningioma Recognition Improvement:**
   - On the other hand, Candidate 3 achieved **52.5% Meningioma recall** (21/40) on independent test data with **0.7000 precision**, far outperforming the original model's **10.0% recall** (4/40, where 31 were misclassified as pituitary).
   - The original model's severe blind spot is Meningioma $\rightarrow$ Pituitary collapse (31/40 errors). Candidate 3 reduced these errors to 11/40.
3. **Overall Trade-off:**
   - While Candidate 3 is significantly better at distinguishing Meningioma and NoTumor, its catastrophic failure on Glioma (3/40 correct, 37 missed) makes it clinically unacceptable as a replacement for the production model.
4. **Grad-CAM Verification:**
   - Executed via `ai_engine.run_classification_and_gradcam()` on `sample_mri_glioma.png`. Latency was **27.95 ms**, gradient propagation succeeded, but predicted Pituitary due to the glioma representation deficit.

---

## 9. Final Recommendation

### Recommendation: **REJECT DEPLOYMENT OF `brain_tumor_mobilenet_v2_shallow_finetune.keras`**

- The original model `brain_tumor_mobilenet_v2.keras` achieves a higher overall accuracy (**71.88% vs 59.38%**) and a clinically critical **87.5% Glioma recall vs 7.5%**.
- Production status: **`brain_tumor_mobilenet_v2.keras` remains active in the running FastAPI service.**
- No files were overwritten; all partitions and checksums remain strictly preserved.
