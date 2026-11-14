# Linear Probe Experiment Report: Frozen-Backbone MobileNetV2

**Date:** October 15, 2026  
**System:** NeuroSyntra AI Brain Tumor MRI Diagnostic Engine  
**Experiment Model File:** `brain_tumor_mobilenet_v2_linear_probe.keras` (MD5: verified separate file)  
**Production Model File:** `brain_tumor_mobilenet_v2.keras` (MD5: `9c3d3f636cbc032561e80504b0391262`, **100% UNCHANGED**)  
**Previous Candidate:** `brain_tumor_mobilenet_v2_improved.keras` (MD5: `bd0a2deae01c4eb68149e32e3c5888b4`, **100% UNCHANGED**)  
**Status:** Complete. **Candidate REJECTED for deployment; Production Model Retained.**

---

## 1. Executive Summary & Verdict

Following the diagnostic plan outlined in `CANDIDATE_FAILURE_ANALYSIS.md`, we conducted a strictly controlled linear probe experiment with a **100% frozen MobileNetV2 ImageNet backbone**.

### Core Results:
1. **Validation Performance (200 balanced images):**  
   - Best Epoch: **Epoch 2** (`val_loss: 1.7589`, `val_accuracy: 61.50%`).
   - Early stopping halted training at Epoch 12; the saved best checkpoint was reloaded and independently verified at **61.50%** accuracy and **0.5382** Macro F1.
2. **Independent Test Set Performance (160 balanced images):**  
   - Linear Probe Accuracy: **57.50%** (92/160), Macro F1: **0.4905**.
   - Original Model Accuracy: **71.88%** (115/160), Macro F1: **0.6668**.
   - Previous Overfitted Candidate Accuracy: **58.13%** (93/160), Macro F1: **0.5138**.
3. **Critical Empirical Finding — Glioma Feature Blindness:**  
   - The linear probe with a frozen ImageNet backbone achieved **0.00% recall on Glioma** on both the independent test set (0/40) and benchmark set (0/20), and only **4.00% recall** (2/50) on validation.
   - Architectural inspection of the original deployed model (`brain_tumor_mobilenet_v2.keras`) revealed that it was **not** a linear probe: it had **1,370,564 trainable parameters** fine-tuning 14 deep backbone layers (`block_15` to `Conv_1`).
   - ImageNet feature maps (trained on natural RGB objects) lack the domain-specific spatial frequency filters required to detect diffuse, infiltrative glioma tissue without fine-tuning deep convolutional layers.
4. **Action:** In accordance with Safeguard #10, **no model has been deployed**. The original model `brain_tumor_mobilenet_v2.keras` remains active in the running FastAPI application.

---

## 2. Safeguard Verification Checklist

| Safeguard | Requirement | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **#1** | Preserve existing models, partitions & running app | **PASSED** | MD5 of `brain_tumor_mobilenet_v2.keras` (`9c3d...`) and `...improved.keras` (`bd0a...`) identical. Server running on port 8000. |
| **#2** | Save separate script & model file | **PASSED** | Trained via `train_linear_probe_experiment.py`, saved to `brain_tumor_mobilenet_v2_linear_probe.keras` (11 MB). |
| **#3** | 100% Frozen MobileNetV2 ImageNet backbone | **PASSED** | Backbone `mobilenetv2_1.00_128` set to `trainable=False` (2,257,984 non-trainable, 164,484 trainable head params). |
| **#4** | `ModelCheckpoint` from Epoch 1 + Early Stopping + LR schedule | **PASSED** | Checkpoint active from Epoch 1. Best saved at Epoch 2. Early stopping halted at Epoch 12. |
| **#5** | Exact deployed preprocessing & class mapping | **PASSED** | Aspect-preserving bicubic letterbox to 128×128, black padding, `[-1, 1]` normalization. Classes: `glioma=0, meningioma=1, notumor=2, pituitary=3`. |
| **#6** | Per-epoch logging + reload checkpoint verification | **PASSED** | Full history logged; reloaded model independently verified at `val_loss: 1.7589`, `val_acc: 61.50%`. |
| **#7** | Assessment on validation data only | **PASSED** | `dataset_clean/val` (200 images) used strictly for checkpoint selection and early stopping. |
| **#8** | Multi-metric validation evaluation | **PASSED** | Full per-class precision, recall, F1, support, and confusion matrix reported. |
| **#9** | Independent test set comparison (160 images) | **PASSED** | Compared side-by-side with original model on identical preprocessed images. |
| **#10** | No automatic deployment | **PASSED** | Report compiled for review. Production runtime unchanged. |

---

## 3. Training History & Convergence Analysis

The model was trained on the 686 training images with class weighting (`glioma: 0.657, meningioma: 1.750, notumor: 1.732, pituitary: 0.752`) and medical augmentation:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Learning Rate | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.6297 | 80.61% | 1.8804 | 58.50% | 0.001000 | Saved checkpoint (initial best) |
| **2** | **0.2875** | **89.50%** | **1.7589** | **61.50%** | **0.001000** | **Saved best checkpoint (`val_loss: 1.7589`)** |
| 3 | 0.1969 | 94.17% | 1.9140 | 54.50% | 0.001000 | Did not improve |
| 4 | 0.2079 | 94.17% | 1.7731 | 60.00% | 0.001000 | Did not improve |
| 5 | 0.1727 | 95.19% | 2.0401 | 60.50% | 0.001000 | Reduced LR $\rightarrow$ 0.0005 |
| 6 | 0.1077 | 96.94% | 2.0562 | 61.50% | 0.000500 | Did not improve |
| 7 | 0.0886 | 97.96% | 2.1837 | 59.00% | 0.000500 | Did not improve |
| 8 | 0.0878 | 97.67% | 2.4543 | 56.00% | 0.000250 | Reduced LR $\rightarrow$ 0.00025 |
| 9 | 0.0733 | 98.25% | 2.4907 | 55.00% | 0.000250 | Did not improve |
| 10 | 0.0657 | 98.25% | 2.5337 | 56.00% | 0.000250 | Did not improve |
| 11 | 0.0681 | 98.10% | 2.4149 | 58.00% | 0.000250 | Reduced LR $\rightarrow$ 0.000125 |
| 12 | 0.0463 | 99.13% | 2.4944 | 55.00% | 0.000125 | Early stopping triggered (patience=10) |

### Key Observations:
- **Immediate Peak:** Optimal validation loss was reached at **Epoch 2**.
- **Head Overfitting:** Even with a frozen backbone and dropout ($p=0.3$), training loss decreased from $0.6297 \rightarrow 0.0463$ ($99.13\%$ accuracy) while validation loss worsened from $1.7589 \rightarrow 2.4944$.
- **Checkpoint Protection:** Because `ModelCheckpoint` was active from Epoch 1, the model saved to disk was strictly the **Epoch 2** model.

---

## 4. Validation Set Performance (`dataset_clean/val`, 200 images)

The saved best checkpoint (`brain_tumor_mobilenet_v2_linear_probe.keras`) was reloaded and evaluated on the 200 balanced validation images (50 per class):

- **Overall Accuracy:** 61.50% (123 / 200)
- **Macro F1:** 0.5382
- **Macro Precision:** 0.7062
- **Macro Recall:** 0.6150

### Per-Class Metrics:
| Class | Support | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glioma** | 50 | 1.0000 | **0.0400** | **0.0769** | **Severe Failure (48/50 misclassified)** |
| **Meningioma** | 50 | 0.5085 | 0.6000 | 0.5505 | Moderate |
| **NoTumor** | 50 | 0.7031 | **0.9000** | **0.7895** | Strong |
| **Pituitary** | 50 | 0.6133 | **0.9200** | **0.7360** | Strong |

### Confusion Matrix on Validation Set:
```
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      2 |         23 |      12 |        13 |    50
Meningioma   |      0 |         30 |       6 |        14 |    50
NoTumor      |      0 |          3 |      45 |         2 |    50
Pituitary    |      0 |          3 |       1 |        46 |    50
```

---

## 5. Independent Test Set Evaluation (160 images, 40 per class)

All three models were evaluated on the exact same 160 independent test images (`dataset_clean/test_independent`) using the deployed preprocessing pipeline:

### 3-Way Comparative Summary:
| Metric | Original Model (`mobilenet_v2`) | Overfitted Candidate (`improved`) | Linear Probe Candidate (`linear_probe`) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **71.88%** (115/160) | 58.13% (93/160) | 57.50% (92/160) |
| **Macro F1** | **0.6668** | 0.5138 | 0.4905 |
| **Macro Precision** | **0.7925** | 0.7173 | 0.4298 |
| **Macro Recall** | **0.7188** | 0.5813 | 0.5750 |
| **Glioma F1** | **0.9333** (Rec: 87.5%) | 0.1395 (Rec: 7.5%) | **0.0000 (Rec: 0.0%)** |
| **Meningioma F1** | 0.1778 (Rec: 10.0%) | 0.5079 (Rec: 40.0%) | **0.5517 (Rec: 60.0%)** |
| **NoTumor F1** | **0.8780** (Rec: 90.0%) | 0.7727 (Rec: 85.0%) | 0.7174 (Rec: 82.5%) |
| **Pituitary F1** | 0.6780 (Rec: 100.0%) | 0.6349 (Rec: 100.0%) | **0.6931 (Rec: 87.5%)** |

### Confusion Matrices on Independent Test Set:

#### Original Model (`brain_tumor_mobilenet_v2.keras`):
```
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |     35 |          1 |       1 |         3 |    40
Meningioma   |      0 |          4 |       5 |        31 |    40
NoTumor      |      0 |          0 |      36 |         4 |    40
Pituitary    |      0 |          0 |       0 |        40 |    40
```

#### Previous Candidate (`brain_tumor_mobilenet_v2_improved.keras`):
```
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      3 |          5 |       9 |        23 |    40
Meningioma   |      0 |         16 |       5 |        19 |    40
NoTumor      |      0 |          2 |      34 |         4 |    40
Pituitary    |      0 |          0 |       0 |        40 |    40
```

#### Linear Probe Candidate (`brain_tumor_mobilenet_v2_linear_probe.keras`):
```
True \ Pred  | Glioma | Meningioma | NoTumor | Pituitary | Total
-----------------------------------------------------------------
Glioma       |      0 |         15 |      12 |        13 |    40
Meningioma   |      0 |         24 |       6 |        10 |    40
NoTumor      |      0 |          4 |      33 |         3 |    40
Pituitary    |      0 |          4 |       1 |        35 |    40
```

---

## 6. Original Benchmark Test Set Evaluation (`testing/`, 80 images)

| Metric | Original Model | Linear Probe Candidate |
| :--- | :---: | :---: |
| **Accuracy** | **91.25%** (73/80) | 37.50% (30/80) |
| **Macro F1** | **0.9110** | 0.3050 |
| **Glioma Recall** | **95.0%** (19/20) | **0.0%** (0/20) |
| **Meningioma Recall**| **100.0%** (20/20) | 10.0% (2/20) |
| **NoTumor Recall** | 70.0% (14/20) | 45.0% (9/20) |
| **Pituitary Recall** | **100.0%** (20/20) | 95.0% (19/20) |

---

## 7. Deep Architectural Diagnosis: Why Linear Probing Fails on Gliomas

This experiment provides a definitive answer to the model performance dynamics:

1. **The Nature of ImageNet Features:**
   - ImageNet consists of 1.2 million natural color photographs of common objects (animals, vehicles, instruments).
   - Early layers detect edges, colors, and textures; intermediate layers detect object parts; top layers detect semantic object shapes.
   - Brain MRI scans are grayscale cross-sectional axial/coronal slices with zero natural object semantics.
2. **Glioma Pathology vs Extra-Axial Pathology:**
   - **Pituitary adenomas and meningiomas** are discrete, extra-axial or well-circumscribed masses that alter gross structural contours. ImageNet filters can partially pick up contour shifts.
   - **Gliomas** are intra-axial, infiltrative neoplasms that alter local parenchymal tissue texture, water content, and intensity gradients without a sharp outer boundary.
   - A completely frozen ImageNet backbone produces feature vectors where diffuse glioma textures are indistinguishable from normal parenchymal variations or meningeal contrast.
3. **The Original Model's Real Architecture:**
   - Inspection of `brain_tumor_mobilenet_v2.keras` revealed that **14 base convolutional layers** (`block_15_expand` through `Conv_1`, totaling 1,206,596 backbone parameters) were fine-tuned.
   - This fine-tuning is what adapted the top spatial feature maps to MRI brain tumor tissue, enabling its 87.5% Glioma recall on independent test data.
4. **Summary of Failure Modes:**
   - **Candidate 1 (`improved`):** Unfroze 30 layers without early checkpointing $\rightarrow$ severe catastrophic overfitting (Phase 2 degraded val accuracy to 55%, memorized train set with loss $0.0075$).
   - **Candidate 2 (`linear_probe`):** 100% frozen backbone $\rightarrow$ severe underfitting / domain gap (unable to extract glioma features, 0% recall).

---

## 8. Grad-CAM & Engine Compatibility Verification

The linear probe model was integrated into the production inference and Grad-CAM pipeline via `ai_engine.run_classification_and_gradcam`:

- **Execution:** Successful without errors.
- **Inference Latency:** 27.09 ms per scan.
- **Gradient Backpropagation:** Cleanly propagated through the classification head and `mobilenetv2_1.00_128/out_relu`.
- **Finding:** While the pipeline is technically fully compatible, the underlying feature activations are clinically invalid for gliomas.

---

## 9. Conclusion & Recommendation

### Recommendation: **REJECT DEPLOYMENT OF LINEAR PROBE CANDIDATE**

1. `brain_tumor_mobilenet_v2_linear_probe.keras` is inferior to `brain_tumor_mobilenet_v2.keras` across all key metrics:
   - Independent Test Accuracy: **57.50% vs 71.88%**
   - Independent Test Macro F1: **0.4905 vs 0.6668**
   - Glioma Recall: **0.00% vs 87.50%**
2. **Current Production Status:**
   - Production model `brain_tumor_mobilenet_v2.keras` remains active.
   - Server process running continuously with 0 downtime.
   - All dataset partitions in `dataset_clean/` remain frozen and intact for future controlled experiments.
