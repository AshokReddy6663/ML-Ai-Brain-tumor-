import os
import time
import json
import ssl
import certifi
import numpy as np

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# macOS SSL certifi patch for Keras downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 4
IMG_SIZE = (128, 128)

TRAIN_DIR = "dataset_clean/train"
VAL_DIR = "dataset_clean/val"
TEST_INDEP_DIR = "dataset_clean/test_independent"
BENCHMARK_DIR = "testing"

CANDIDATE_MODEL_PATH = "brain_tumor_mobilenet_v2_improved.keras"
ORIGINAL_MODEL_PATH = "brain_tumor_mobilenet_v2.keras"

def letterbox_preprocess(pil_img: Image.Image, target_size=(128, 128)) -> np.ndarray:
    """Exact replica of deployment preprocessing."""
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)
    
    scale = target_size[0] / max_dim
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    
    img_scaled = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    pad_left = (target_size[0] - new_w) // 2
    pad_top = (target_size[1] - new_h) // 2
    canvas.paste(img_scaled, (pad_left, pad_top))
    
    arr = np.array(canvas, dtype=np.float32)
    normalized = arr / 127.5 - 1.0
    return normalized

def load_split_data(split_dir: str):
    images = []
    labels = []
    filepaths = []
    for cls in CLASS_NAMES:
        cdir = os.path.join(split_dir, cls)
        for fname in sorted(os.listdir(cdir)):
            if fname.startswith('.'): continue
            fpath = os.path.join(cdir, fname)
            try:
                with Image.open(fpath) as img:
                    arr = letterbox_preprocess(img, IMG_SIZE)
                images.append(arr)
                labels.append(CLASS_MAP[cls])
                filepaths.append(fpath)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), filepaths

def build_candidate_architecture():
    """Builds MobileNetV2 architecture identical to deployed model for 100% Grad-CAM compatibility."""
    inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer_1')
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    # Match functional name
    base_model._name = 'mobilenetv2_1.00_128'
    
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax', name='dense_1')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='functional')
    return model, base_model

def evaluate_on_dataset(model, images, labels, filepaths, title="Evaluation"):
    preds = model.predict(images, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(labels, pred_indices):
        cm[t][p] += 1
        
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class[name] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': int(np.sum(cm[i, :]))
        }
        
    acc = np.sum(np.diag(cm)) / len(labels)
    macro_prec = np.mean([per_class[n]['precision'] for n in CLASS_NAMES])
    macro_rec = np.mean([per_class[n]['recall'] for n in CLASS_NAMES])
    macro_f1 = np.mean([per_class[n]['f1'] for n in CLASS_NAMES])
    
    misclassified = []
    for idx, (t, p) in enumerate(zip(labels, pred_indices)):
        if t != p:
            misclassified.append({
                "filepath": filepaths[idx],
                "true": CLASS_NAMES[t],
                "pred": CLASS_NAMES[p],
                "confidence": float(preds[idx][p]),
                "probs": {CLASS_NAMES[k]: float(preds[idx][k]) for k in range(4)}
            })
            
    return {
        "title": title,
        "total": len(labels),
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "misclassified": misclassified
    }

def print_eval_report(res):
    print(f"\n=======================================================")
    print(f" {res['title'].upper()}")
    print(f"=======================================================")
    print(f"Total Images: {res['total']}")
    print(f"Accuracy:     {res['accuracy']*100:.2f}%")
    print(f"Macro F1:     {res['macro_f1']:.4f}")
    print(f"Macro Prec:   {res['macro_precision']:.4f}")
    print(f"Macro Recall: {res['macro_recall']:.4f}")
    print("\nPer-Class Breakdown:")
    for c in CLASS_NAMES:
        m = res['per_class'][c]
        print(f"  {c.capitalize():12s} (n={m['support']:2d}) | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f}")
        
    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    print("True \\ Pred | " + " | ".join(f"{c[:4]:>4s}" for c in CLASS_NAMES))
    print("-" * 38)
    cm = res['confusion_matrix']
    for i in range(4):
        row = " | ".join(f"{cm[i][j]:4d}" for j in range(4))
        print(f"{CLASS_NAMES[i][:4]:>10s} | {row}")
        
    print(f"Total misclassifications: {len(res['misclassified'])}")
    for m in res['misclassified'][:5]:
        print(f"  - {os.path.basename(m['filepath'])}: True={m['true']}, Pred={m['pred']} ({m['confidence']*100:.1f}%)")
    if len(res['misclassified']) > 5:
        print(f"  ... and {len(res['misclassified']) - 5} more misclassified images.")

def train_candidate():
    print("=" * 70)
    print("TRAINING SEPARATE CANDIDATE MODEL: brain_tumor_mobilenet_v2_improved.keras")
    print("=" * 70)
    
    # 1. Load data
    print("\n[STEP 1] Loading preprocessed datasets from disk...")
    t0 = time.time()
    X_train, y_train, train_paths = load_split_data(TRAIN_DIR)
    X_val, y_val, val_paths = load_split_data(VAL_DIR)
    X_test_indep, y_test_indep, test_indep_paths = load_split_data(TEST_INDEP_DIR)
    X_bench, y_bench, bench_paths = load_split_data(BENCHMARK_DIR)
    print(f"Data loaded in {time.time()-t0:.2f}s:")
    print(f"  Train:            {len(X_train)} images (shape: {X_train.shape})")
    print(f"  Val:              {len(X_val)} images (shape: {X_val.shape})")
    print(f"  Test_Independent: {len(X_test_indep)} images (shape: {X_test_indep.shape})")
    print(f"  Benchmark Test:   {len(X_bench)} images (shape: {X_bench.shape})")

    # One-hot encode targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    # 2. Compute balanced class weights for training set (Standard formula: N / (n_classes * n_samples_c))
    total_samples = len(y_train)
    class_weight_dict = {}
    for i in range(NUM_CLASSES):
        count_i = int(np.sum(y_train == i))
        class_weight_dict[i] = float(total_samples / (NUM_CLASSES * max(count_i, 1)))
        
    print("\n[STEP 2] Computed Training Class Weights to address class imbalance:")
    for c in CLASS_NAMES:
        print(f"  {c:12s} (n={np.sum(y_train == CLASS_MAP[c]):3d}): weight = {class_weight_dict[CLASS_MAP[c]]:.3f}")

    # 3. Medical-grade data augmentation for training only
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04, fill_mode='constant', fill_value=-1.0), # subtle +/- 14 degrees
        tf.keras.layers.RandomFlip("horizontal"),                                     # bilateral axial brain symmetry
        tf.keras.layers.RandomZoom(0.03, fill_mode='constant', fill_value=-1.0),     # slight zoom
        tf.keras.layers.RandomContrast(0.08)                                         # slight contrast variation
    ], name='medical_augmentation')

    # Create tf.data datasets
    batch_size = 16
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=42)
    train_ds = train_ds.batch(batch_size).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 4. Build Model
    print("\n[STEP 3] Initializing MobileNetV2 architecture with ImageNet weights...")
    model, base_model = build_candidate_architecture()

    # Phase 1: Warmup classification head with base frozen
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Phase 1: Warming up top classification head (8 epochs)...")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Phase 2: Fine-tuning top 30 layers of MobileNetV2 with low learning rate
    base_model.trainable = True
    # Freeze all layers before the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    print("\nPhase 2: Fine-tuning MobileNetV2 upper layers with ReduceLROnPlateau & EarlyStopping...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=CANDIDATE_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    print(f"\nTraining complete! Best candidate model saved to: {CANDIDATE_MODEL_PATH}")

    # Combine history
    combined_history = {
        "loss": h1.history['loss'] + h2.history['loss'],
        "val_loss": h1.history['val_loss'] + h2.history['val_loss'],
        "accuracy": h1.history['accuracy'] + h2.history['accuracy'],
        "val_accuracy": h1.history['val_accuracy'] + h2.history['val_accuracy']
    }
    best_val_epoch = int(np.argmin(combined_history['val_loss'])) + 1
    best_val_loss = float(np.min(combined_history['val_loss']))
    best_val_acc = float(combined_history['val_accuracy'][best_val_epoch - 1])

    print(f"\nBest Validation Epoch: {best_val_epoch}")
    print(f"Best Validation Loss:  {best_val_loss:.4f}")
    print(f"Best Validation Acc:   {best_val_acc*100:.2f}%")

    # 5. Load best candidate model for evaluation
    best_candidate = tf.keras.models.load_model(CANDIDATE_MODEL_PATH)
    original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

    # 6. Evaluate both models on independent test set
    cand_indep_eval = evaluate_on_dataset(best_candidate, X_test_indep, y_test_indep, test_indep_paths, "Candidate Model on Independent Test Set (160 images)")
    orig_indep_eval = evaluate_on_dataset(original_model, X_test_indep, y_test_indep, test_indep_paths, "Original Model on Independent Test Set (160 images)")

    # 7. Evaluate both models on original 80-image benchmark test set
    cand_bench_eval = evaluate_on_dataset(best_candidate, X_bench, y_bench, bench_paths, "Candidate Model on Benchmark Test Set (80 images)")
    orig_bench_eval = evaluate_on_dataset(original_model, X_bench, y_bench, bench_paths, "Original Model on Benchmark Test Set (80 images)")

    # Print reports
    print_eval_report(orig_indep_eval)
    print_eval_report(cand_indep_eval)
    print_eval_report(orig_bench_eval)
    print_eval_report(cand_bench_eval)

    # 8. Test Grad-CAM compatibility
    print("\n" + "=" * 70)
    print("GRAD-CAM & API COMPATIBILITY VERIFICATION")
    print("=" * 70)
    import ai_engine
    # Test Grad-CAM using candidate model directly
    test_img = Image.open("sample_images/sample_mri_glioma.png").convert("RGB")
    
    # Temporarily check Grad-CAM on candidate
    orig_m = ai_engine.MODELS.get("mobilenet_v2")
    ai_engine.MODELS["mobilenet_v2_improved"] = best_candidate
    
    gradcam_res = ai_engine.run_classification_and_gradcam(test_img, model_type="mobilenet_v2_improved")
    print("Grad-CAM Execution Success!")
    print(f"  Prediction: {gradcam_res['prediction_display']}")
    print(f"  Confidence: {gradcam_res['confidence_percentage']}%")
    print(f"  Probabilities: {gradcam_res['probabilities']}")
    print(f"  Peak Attention Coordinate: {gradcam_res['peak_attention']}")
    print(f"  CAM map shape: {np.array(gradcam_res['cam_map']).shape}")
    print(f"  Inference Time: {gradcam_res['inference_time_ms']} ms")

    # Clean up model registry override
    if orig_m is not None:
        ai_engine.MODELS["mobilenet_v2"] = orig_m
    del ai_engine.MODELS["mobilenet_v2_improved"]

    # Save evaluation summary to file
    summary_data = {
        "candidate_path": CANDIDATE_MODEL_PATH,
        "best_epoch": best_val_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "history": combined_history,
        "evaluations": {
            "candidate_independent": cand_indep_eval,
            "original_independent": orig_indep_eval,
            "candidate_benchmark": cand_bench_eval,
            "original_benchmark": orig_bench_eval
        }
    }
    with open("training_and_evaluation_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved full evaluation payload to training_and_evaluation_results.json.")

if __name__ == "__main__":
    train_candidate()
