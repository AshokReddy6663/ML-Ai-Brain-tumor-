# Final Dataset Audit Report: NeuroSyntra AI

**Generated Date**: Thu Oct 15 10:25:37 IST 2026  
**Target Candidate Model**: `brain_tumor_mobilenet_v2_improved.keras`  
**Current Preserved Model**: `brain_tumor_mobilenet_v2.keras` (Active & Untouched)  
**Status**: Dataset Frozen & Audited. **Awaiting User Review Prior to Model Training.**

---

## 1. Executive Summary & Audit Conclusions

- **Preservation of Existing Assets**: 
  - The deployed model (`brain_tumor_mobilenet_v2.keras`) has been preserved and was **not** overwritten.
  - The original benchmark evaluation folder (`testing/`, 80 images) remains unaltered.
  - The running FastAPI backend server (`python3 main.py`) remains online and fully operational.
- **Similarity Metric Specification**:
  - `skimage.metrics.structural_similarity` was checked and is **not installed** in the virtual environment.
  - Standard local-window Structural Similarity (Wang et al., 2004) was computed via **`tf.image.ssim`** using an $11 \times 11$ Gaussian filter with $\sigma = 1.5$ over $128 \times 128$ grayscale representations.
- **Leakage & Independence Status**:
  - **Exact MD5 duplicates**: **0** across all partition comparisons.
  - **Filename Base ID overlaps**: **0** across all partition comparisons.
  - **Benchmark Test Contamination**: **0** benchmark test base IDs appear in `train`, `val`, or `test_independent`.
  - **Perceptual dHash Collisions with Benchmark Test**: **0** identical perceptual dHash matches between `train` and `benchmark_test`.

> [!WARNING]
> **Provenance & Patient-Level Independence Disclosure**:  
> In public Kaggle/Roboflow MRI datasets (`PranomVignesh/MRI-Images-of-Brain-Tumor`), original DICOM headers and patient tracking metadata are stripped. Multiple slices from the same scan acquisition share similar anatomical contours. While exact base IDs, identical files, and visual twins were systematically screened and partitioned using local-window SSIM and dHash, **patient-level separation cannot be guaranteed with 100% mathematical certainty** in the absence of authentic clinical accession numbers. Similarity thresholds serve as rigorous screening tools rather than clinical ground truth.

---

## 2. Partition & Class Distribution Audit

### 2.1 Distribution Table

| Partition | Total Images | Unique Base IDs | Glioma | Meningioma | No Tumor | Pituitary | Partition Balance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **686** | **580** | 261 | 98 | 99 | 228 | Unbalanced (See Section 2.2) |
| **`val`** | **200** | **200** | 50 | 50 | 50 | 50 | **100% Balanced (1:1:1:1)** |
| **`test_independent`** | **160** | **160** | 40 | 40 | 40 | 40 | **100% Balanced (1:1:1:1)** |
| **`testing/` (Original Benchmark)** | **80** | **77** | 20 | 20 | 20 | 20 | **100% Balanced (1:1:1:1)** |

### 2.2 Class Balance Disclosure
- **Validation & Test Sets**: Both `val` (200 images, 50/class) and `test_independent` (160 images, 40/class) are **perfectly balanced** with a 1:1:1:1 ratio.
- **Training Set Imbalance**:
  - `glioma`: 261 images
  - `pituitary`: 228 images
  - `meningioma`: 98 images
  - `notumor`: 99 images
  - *Analysis*: In the training split, `meningioma` (98) and `notumor` (99) have fewer images than `glioma` (261) and `pituitary` (228) because a significant number of candidate images were strictly purged due to structural similarity to the benchmark test set.
  - *Mitigation for Training*: Class weights (`compute_class_weight`) and data augmentation will be applied during candidate model training to prevent majority-class bias.

---

## 3. Duplication & Overlap Matrix

Every image in each partition was checked against all images in all other partitions:

| Comparison Pair | Exact MD5 Matches | Base ID Overlaps | Identical dHash (Dist=0) | Pairs with Local SSIM $\ge 0.80$ | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train` vs `val`** | **0** | **0** | **0** | **0** | **PASS** |
| **`train` vs `test_independent`** | **0** | **0** | **0** | **0** | **PASS** |
| **`val` vs `test_independent`** | **0** | **0** | **0** | **0** | **PASS** |
| **`train` vs `benchmark_test`** | **0** | **0** | **0** | **0** | **PASS (Zero Leakage)** |
| **`val` vs `benchmark_test`** | **0** | **0** | **0** | **0** | **PASS (Zero Leakage)** |
| **`test_independent` vs `benchmark_test`** | **0** | **0** | **0** | **0** | **PASS (Zero Leakage)** |

---

## 4. Cross-Class Conflict & Mismatch Analysis

We checked all cross-partition and intra-partition pairs **across differing classes** (e.g., comparing `glioma` images against `notumor` images):
- **Cross-Class MD5 Matches**: **0**. No image file is labeled under two different classes.
- **Cross-Class Base ID Matches**: **0**. No base scan identifier belongs to more than one class.
- **Cross-Class High Local SSIM ($\ge 0.80$)**: **0**. No image labeled as one class exhibits high structural similarity to an image labeled as another class.

---

## 5. File Integrity & Technical Specifications

- **Corrupt / Unreadable Images**: **`0`** across all partitions (verified via PIL header and byte parsing).
- **Format & Color Space**: 100% of images are 3-channel RGB JPEG format.
- **Dimensions**: All native image files in the clean dataset are $224 \times 224$ pixels.
- **Directory Paths**:
  - Clean Dataset Root: `/Users/tharun/Downloads/NeuroScan-main 2/dataset_clean`
  - Subdirectories: `train/`, `val/`, `test_independent/`
  - Each split strictly contains four subfolders: `glioma`, `meningioma`, `notumor`, `pituitary`.
  - Benchmark Test Directory: `/Users/tharun/Downloads/NeuroScan-main 2/testing`

---

## 6. Training Preprocessing & Pipeline Alignment Plan

When approved to train `brain_tumor_mobilenet_v2_improved.keras`:
1. **Preprocessing Pipeline (Exact Match to Deployment)**:
   - Input Tensor Size: $(128, 128, 3)$
   - Letterboxing: Aspect-ratio preserving square letterbox with black padding
   - Interpolation: Bicubic interpolation (`Image.Resampling.BICUBIC`)
   - Normalization: Exact $(x / 127.5) - 1.0$ (mapping pixels to $[-1.0, 1.0]$)
2. **Model Selection Protocol**:
   - Training will optimize on `train` and evaluate checkpoints strictly against `val`.
   - Learning rate reduction on plateau and early stopping will be guided exclusively by validation loss / validation macro F1.
   - `test_independent` and `testing/` (benchmark) will remain **completely untouched** throughout training, used only for the final evaluation.
3. **Artifact Protection**:
   - The candidate model will be saved separately as `brain_tumor_mobilenet_v2_improved.keras`.
   - The original model `brain_tumor_mobilenet_v2.keras` will **not** be modified or overwritten.

---

## 7. Dataset Suitability Conclusion

The dataset has undergone multi-stage auditing:
- Filename decoupling and base scan identification
- Exact byte-level duplicate removal (MD5)
- Standard local-window SSIM (`tf.image.ssim`) cross-partition screening
- Benchmark test set purge (zero overlap with original 80 test images)

**Recommendation**: The dataset is structurally sound, leak-screened, and **ready for candidate model training** upon user approval.
