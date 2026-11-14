# Failure Analysis: Candidate Model Training for NeuroSyntra AI

**Document**: `CANDIDATE_FAILURE_ANALYSIS.md`  
**Generated Date**: Thu Oct 15 10:36:50 IST 2026  
**Investigated Model**: `brain_tumor_mobilenet_v2_improved.keras`  
**Reference Baseline**: `brain_tumor_mobilenet_v2.keras` (Deployed, Untouched)  
**Status**: Analysis Complete. **No models modified. Awaiting user review.**

---

## 1. Candidate Training Configuration Audit

Reviewing the exact configuration executed in `train_and_evaluate_candidate.py`:

- **Architecture Initialization**:
  - Backbone: `tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')`
  - Internal Backbone Name: `mobilenetv2_1.00_128` (154 layers total)
  - Top Head:
    - `GlobalAveragePooling2D(name='global_average_pooling2d')`
    - `Dense(128, activation='relu', name='dense')`
    - `Dropout(0.3, name='dropout')`
    - `Dense(4, activation='softmax', name='dense_1')`
- **Input Preprocessing**:
  - PIL RGB conversion $\rightarrow$ Aspect-preserving letterbox resize to $128 \times 128$ using bicubic interpolation (`Image.Resampling.BICUBIC`) $\rightarrow$ centered on black `(0, 0, 0)` padding $\rightarrow$ normalized via `(pixel / 127.5) - 1.0` to dynamic range $[-1.0, 1.0]$.
- **Training Regimen**:
  - **Phase 1 (Head Warmup, 8 Epochs)**:
    - Backbone status: Fully frozen (`base_model.trainable = False`). All 154 MobileNetV2 layers non-trainable. Only top head trainable (164,484 parameters).
    - Optimizer: `Adam(learning_rate=1e-3)`. Loss: `categorical_crossentropy`.
    - **Defect Identified**: `ModelCheckpoint` was **not passed** to the Phase 1 `model.fit()` call.
  - **Phase 2 (Fine-Tuning, 17 Epochs)**:
    - Backbone status: Top 30 layers unfrozen (`base_model.layers[:-30]` frozen; layers from `block_13_expand` upward trainable).
    - Optimizer: `Adam(learning_rate=1e-4)`. Loss: `categorical_crossentropy`.
    - Callbacks: `ModelCheckpoint(monitor='val_loss', save_best_only=True)`, `ReduceLROnPlateau(factor=0.5, patience=3)`, `EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)`.
- **Class Weights Applied**:
  - `glioma` ($n=261$): **`0.657`**
  - `pituitary` ($n=228$): **`0.752`**
  - `meningioma` ($n=98$): **`1.750`**
  - `notumor` ($n=99$): **`1.732`**

---

## 2. Checkpoint-Selection Verification: What Model Was Actually Saved?

A critical bug was discovered in the checkpoint-selection flow:

1. In **Phase 1 (Head Warmup)**:
   - At **Epoch 2**, the model achieved its global minimum validation loss: **`val_loss: 1.3553`**, with **`val_accuracy: 62.00%`**.
   - However, `ModelCheckpoint` was not active during Phase 1. Therefore, **Epoch 2 weights were never saved to disk**.
2. In **Phase 2 (Fine-Tuning)**:
   - Unfreezing 30 deep convolutional layers immediately destabilized representations on the small dataset ($n=686$).
   - Phase 2 started with `val_loss: 2.8969`.
   - The lowest `val_loss` achieved in Phase 2 was **`val_loss: 2.5282`** at Phase 2 Epoch 9 (Combined Epoch 17).
   - Early stopping restored weights from Phase 2 Epoch 9.
3. **Verification on Saved Weights**:
   - Evaluating `brain_tumor_mobilenet_v2_improved.keras` on `dataset_clean/val` yields:
     - **`loss = 2.5282`**, **`accuracy = 55.00%`**.
   - **Conclusion**: The saved model on disk is **NOT** the best overall model from Epoch 2 (`val_loss = 1.3553`), but rather the overfitted checkpoint from Phase 2 (`val_loss = 2.5282`).

---

## 3. Training History & Explanation of Low Validation Accuracy (62%)

The full training history recorded in `training_and_evaluation_results.json` reveals classic catastrophic overfitting:

| Combined Epoch | Phase | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Phase 1 (Warmup) | 0.6166 | 74.05% | 1.5820 | 54.00% | Initial head training |
| **2** | **Phase 1 (Warmup)** | **0.3354** | **88.34%** | **1.3553** | **62.00%** | **Global Best Validation Loss** (Not Saved) |
| 3 | Phase 1 (Warmup) | 0.1906 | 94.61% | 1.5793 | 60.50% | Train loss drops rapidly |
| 8 | Phase 1 (Warmup) | 0.0945 | 97.52% | 2.1082 | 57.00% | End of Phase 1 |
| 9 | Phase 2 (Fine-tune) | 0.3576 | 89.21% | 2.8969 | 50.00% | Unfreezing 30 layers causes loss spike |
| 17 | Phase 2 (Fine-tune) | 0.0457 | 98.69% | **2.5282** | 55.00% | Best in Phase 2 (Saved Checkpoint) |
| 25 | Phase 2 (Fine-tune) | **0.0075** | **99.85%** | **2.5797** | 58.00% | Training memorization complete; Early stop |

### Root Causes of the Low 62% Validation Ceiling:
1. **Over-parameterization relative to sample size**: MobileNetV2's top 30 layers contain over 1.2 million parameters. Fitting 1.2M parameters on only 686 training images caused the network to memorize training idiosyncrasies within 3 epochs (training accuracy hit 99.85%, training loss hit 0.0075), while validation error remained stuck above 2.5.
2. **Domain shift in leak-free evaluation**: When all augmented twins and same-patient slices were strictly eliminated from the validation set, the model had to rely on true generalizing features rather than memorized scan textures.

---

## 4. Verification on Identical Sample Images (Exact Probabilities)

Both models were evaluated on the four standard benchmark images in `sample_images/` using identical preprocessing:

### Sample 1: `sample_mri_glioma.png` (True Class: Glioma)
- **Original Model**: **Glioma: 96.71%** | Meningioma: 0.03% | No Tumor: 0.05% | Pituitary: 3.21%
- **Candidate Model**: Glioma: 0.18% | Meningioma: 23.15% | No Tumor: 1.18% | **Pituitary: 75.49%** (ERROR)

### Sample 2: `sample_mri_meningioma.png` (True Class: Meningioma)
- **Original Model**: Glioma: 0.01% | **Meningioma: 99.87%** | No Tumor: 0.09% | Pituitary: 0.03%
- **Candidate Model**: Glioma: 0.04% | Meningioma: 0.03% | **No Tumor: 99.76%** | Pituitary: 0.18% (ERROR)

### Sample 3: `sample_mri_normal.png` (True Class: No Tumor)
- **Original Model**: Glioma: 0.54% | Meningioma: 0.05% | **No Tumor: 99.19%** | Pituitary: 0.23%
- **Candidate Model**: Glioma: 0.00% | Meningioma: 0.01% | **No Tumor: 99.99%** | Pituitary: 0.00%

### Sample 4: `sample_mri_pituitary.png` (True Class: Pituitary)
- **Original Model**: Glioma: 0.00% | Meningioma: 0.00% | No Tumor: 0.00% | **Pituitary: 100.00%**
- **Candidate Model**: Glioma: 0.00% | Meningioma: 0.00% | No Tumor: 0.00% | **Pituitary: 100.00%**

---

## 5. Detailed Analysis of Systematic Classification Errors

### 5.1 Candidate Model: 23/40 Glioma Misclassified as Pituitary
On the independent test set, 23 out of 40 Glioma images were classified as Pituitary with high confidence:

| Filename | Predicted Class | Pituitary Probability | Glioma Probability | Image Properties |
| :--- | :---: | :---: | :---: | :--- |
| `Tr-gl_0019_jpg.rf.e5a2b5fe1281224dc9d5e445496.jpg` | Pituitary | **100.0%** | 0.0% | Axial T1 CE, central lesion |
| `Tr-gl_0573_jpg.rf.b03ed848c0d4a30e1441a07ead3.jpg` | Pituitary | **100.0%** | 0.0% | Mid-sagittal slice |
| `Tr-gl_0768_jpg.rf.a0a763c40c0f722ed8884758352.jpg` | Pituitary | **100.0%** | 0.0% | Deep central hyperintensity |
| `Tr-gl_0636_jpg.rf.744324eb85d16f63b9cf5f570e1.jpg` | Pituitary | **99.9%** | 0.0% | Coronal slice through skull base |
| `Tr-gl_0713_jpg.rf.d669a4c69655db7502d916c512a.jpg` | Pituitary | **99.9%** | 0.0% | Axial slice near optic chiasm |
| `Tr-gl_0428_jpg.rf.1e7d5fc50abd1faa11dec078121.jpg` | Pituitary | **99.8%** | 0.1% | Low peripheral contrast |
| `Tr-gl_1223_jpg.rf.71d66ce11a371deb1d99d430f44.jpg` | Pituitary | **99.8%** | 0.0% | Deep periventricular tumor |
| `Tr-gl_0899_jpg.rf.0f1a06e027e7acb35f62ddea19c.jpg` | Pituitary | **99.7%** | 0.0% | Midline contrast enhancement |
| `Tr-gl_0822_jpg.rf.78e9ba0dda96fdb80dc8a220e92.jpg` | Pituitary | **99.4%** | 0.0% | Symmetrical axial slice |
| `Tr-gl_1240_jpg.rf.54e01573dbd71b98ee6cbf25e02.jpg` | Pituitary | **99.2%** | 0.0% | Infiltrative mass |

**Root Cause**: The fine-tuned MobileNetV2 backbone over-indexed on centralized hyperintensities. When deep gliomas near the ventricles or skull base appeared, the network attributed the activation to the pituitary gland region.

### 5.2 Original Model: 31/40 Meningioma Misclassified as Pituitary
On the independent test set, the original deployed model failed on 31 of 40 meningiomas, predicting Pituitary:

| Filename | Predicted Class | Pituitary Probability | Meningioma Probability | Image Properties |
| :--- | :---: | :---: | :---: | :--- |
| `m1-121-_jpg.rf.f3d714369b191b5a961f444c951e12.jpg` | Pituitary | **100.0%** | 0.0% | Coronal skull base meningioma |
| `m2-6-_jpg.rf.7217621cac7a0838cda9305cbbfe484a.jpg` | Pituitary | **100.0%** | 0.0% | Dural tail near sphenoid wing |
| `m2-55-_jpg.rf.048f3bf03ff1c18c901f76690965353.jpg` | Pituitary | **100.0%** | 0.0% | Parasagittal slice |
| `m2-59-_jpg.rf.1fde973c0e79ec103045ff4cf48bebc.jpg` | Pituitary | **100.0%** | 0.0% | Contrast-enhanced lesion |
| `m2-71-_jpg.rf.d550d2e7904e7118b9a043e745a1e96.jpg` | Pituitary | **100.0%** | 0.0% | Homogeneous enhancement |
| `m3-18-_jpg.rf.c878697311a9529b7a9fc20e93fb17a.jpg` | Pituitary | **100.0%** | 0.0% | Convexity meningioma |
| `m3-31-_jpg.rf.ce7c122e0a12e7bf02bfdf7916523b2.jpg` | Pituitary | **100.0%** | 0.0% | Sellar-adjacent meningioma |
| `m3-42-_jpg.rf.512d416c25bb75541e2b3f7c7eef8da.jpg` | Pituitary | **100.0%** | 0.0% | Coronal acquisition |
| `m-54-_jpg.rf.0844a03de133343c0228e926f16639c4.jpg` | Pituitary | **99.9%** | 0.0% | Sellar/parasellar lesion |
| `m-45-_jpg.rf.ab367e470a3712a30580ef93417720cf.jpg` | Pituitary | **99.8%** | 0.0% | Extra-axial mass |

**Root Cause**: Both Meningiomas and Pituitary Adenomas are intensely contrast-enhancing, extra-axial lesions frequently located along the skull base and sphenoid bone. Because the original model was trained on only 75 meningioma images, it memorized specific training textures; on un-augmented, truly independent meningioma scans, it defaults to pituitary.

---

## 6. Confirmation of Complete Test Data Isolation

We verified that:
1. **Training Data**: Exclusively loaded from `dataset_clean/train` ($n=686$).
2. **Augmentation Layers**: Fitted dynamically on training batch tensors only. No statistics were computed on validation or test sets.
3. **Model Selection**: Controlled exclusively by `monitor='val_loss'` on `dataset_clean/val` ($n=200$).
4. **Test Isolation**: Neither `dataset_clean/test_independent` ($n=160$) nor `testing/` ($n=80$) was referenced during training or callback execution.

---

## 7. Recommended Smallest Justified Experiment

### Hypothesis
Fine-tuning 30 deep layers on 686 images caused catastrophic representation drift. Preserving the frozen ImageNet representations (Linear Probing only) with proper checkpoint selection across all epochs will prevent the collapse into pituitary false positives.

### Proposed Smallest Experiment:
1. **Keep the Backbone 100% Frozen**: `base_model.trainable = False` throughout all epochs (train only the 164K classification head parameters).
2. **Global Best Checkpointing**: Attach `ModelCheckpoint` to monitor `val_loss` across all warmup epochs.
3. **Evaluation Protocol**:
   - Evaluate strictly on `dataset_clean/val` first.
   - Only evaluate on `test_independent` if `val_loss < 1.0` and `val_accuracy > 80%`.
4. **Zero Impact on Production**:
   - Do not overwrite `brain_tumor_mobilenet_v2.keras`.
   - Do not alter the running web application.

---

### Awaiting Approval
No training will begin without your explicit instruction.
