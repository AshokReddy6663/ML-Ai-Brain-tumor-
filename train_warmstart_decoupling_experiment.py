import os
import time
import json
import ssl
import numpy as np
import tensorflow as tf
from PIL import Image

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

ORIGINAL_MODEL_PATH = "brain_tumor_mobilenet_v2.keras"
TARGET_MODEL_PATH = "brain_tumor_mobilenet_v2_warmstart.keras"

def letterbox_preprocess(pil_img: Image.Image, target_size=(128, 128)) -> np.ndarray:
    """Exact replica of deployment preprocessing: aspect-preserving bicubic letterboxing to 128x128 with [-1, 1] normalization."""
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
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'support': int(np.sum(cm[i, :]))
        }
        
    acc = float(np.sum(np.diag(cm)) / len(labels))
    macro_prec = float(np.mean([per_class[n]['precision'] for n in CLASS_NAMES]))
    macro_rec = float(np.mean([per_class[n]['recall'] for n in CLASS_NAMES]))
    macro_f1 = float(np.mean([per_class[n]['f1'] for n in CLASS_NAMES]))
    
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
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
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

def run_experiment():
    print("=" * 75)
    print("CONTROLLED WARM-START TARGETED DECOUPLING EXPERIMENT")
    print("=" * 75)

    # 1. Load Data
    print("\n[STEP 1] Loading preprocessed datasets from disk...")
    t0 = time.time()
    X_train, y_train, train_paths = load_split_data(TRAIN_DIR)
    X_val, y_val, val_paths = load_split_data(VAL_DIR)
    X_test_indep, y_test_indep, test_indep_paths = load_split_data(TEST_INDEP_DIR)
    X_bench, y_bench, bench_paths = load_split_data(BENCHMARK_DIR)
    print(f"  Loaded in {time.time()-t0:.2f}s: Train={len(X_train)}, Val={len(X_val)}, Test_Indep={len(X_test_indep)}, Bench={len(X_bench)}")

    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    # 2. Warm-Start Model Initialization
    print(f"\n[STEP 2] Loading original production model for warm-start: {ORIGINAL_MODEL_PATH}...")
    model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
    base_model = model.layers[1]
    
    print("  Initial layer inspection:")
    print(f"    Total base layers: {len(base_model.layers)}")
    
    # Freeze all layers up to Block 15 (index 0 to 142)
    # Block 16 begins at index 143 ('block_16_expand')
    base_model.trainable = True
    for layer in base_model.layers[:143]:
        layer.trainable = False
    for layer in base_model.layers[143:]:
        layer.trainable = True
        
    # Classification head layers remain trainable
    for layer in model.layers[2:]:
        layer.trainable = True

    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable_params = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))
    base_trainable = [l.name for l in base_model.layers if l.trainable and len(l.weights) > 0]
    
    print("\n[STEP 3] Warm-Start Targeted Layer Configuration:")
    print(f"  Trainable Base Layers ({len(base_trainable)}): {base_trainable}")
    print(f"  Trainable Parameters:     {trainable_params:,} (Base: {trainable_params - 164484:,}, Head: 164,484)")
    print(f"  Non-Trainable Parameters: {non_trainable_params:,}")
    assert trainable_params == 1050564, f"Expected 1,050,564 trainable parameters, got {trainable_params}"

    # 3. Evaluate Baseline (Epoch 0) on Validation Set
    print("\n[STEP 4] Evaluating Baseline Original Model on Validation Set (Epoch 0)...")
    baseline_val_eval = evaluate_on_dataset(model, X_val, y_val, val_paths, "Baseline Original Model on Validation Set (Epoch 0)")
    print_eval_report(baseline_val_eval)

    # 4. Data Augmentation (Medical grade, training only)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04, fill_mode='constant', fill_value=-1.0),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.03, fill_mode='constant', fill_value=-1.0),
        tf.keras.layers.RandomContrast(0.08)
    ], name='medical_augmentation')

    batch_size = 16
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=42)
    train_ds = train_ds.batch(batch_size).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 5. Compile with low learning rate near 1e-5 and EQUAL class weights
    learning_rate = 1e-5
    print(f"\n[STEP 5] Compiling with Adam(learning_rate={learning_rate}) and EQUAL class weighting...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Callbacks with ModelCheckpoint from Epoch 1
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=TARGET_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # 7. Train
    print("\n[STEP 6] Fine-tuning warm-start model (up to 30 epochs, equal class weights)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    # 8. Reload Best Checkpoint and Verify on Validation Set
    print(f"\n[STEP 7] Training complete. Reloading saved checkpoint: {TARGET_MODEL_PATH}...")
    saved_warmstart = tf.keras.models.load_model(TARGET_MODEL_PATH)
    
    val_loss, val_acc = saved_warmstart.evaluate(X_val, y_val_cat, verbose=0)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_loss_reported = float(np.min(history.history['val_loss']))
    
    print(f"  Best Epoch in History: {best_epoch}")
    print(f"  Best Val Loss in Hist: {best_loss_reported:.4f}")
    print(f"  Reloaded Model Loss:   {val_loss:.4f}")
    print(f"  Reloaded Model Acc:    {val_acc*100:.2f}%")

    warmstart_val_eval = evaluate_on_dataset(saved_warmstart, X_val, y_val, val_paths, "Warm-Start Candidate on Validation Set (200 images)")
    print_eval_report(warmstart_val_eval)

    # 9. EVALUATION GATE on Independent Test Set (160 images)
    print("\n[STEP 8] EVALUATION GATE: Evaluating on Independent Test Set (160 images)...")
    orig_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
    
    warmstart_indep_eval = evaluate_on_dataset(saved_warmstart, X_test_indep, y_test_indep, test_indep_paths, "Warm-Start Candidate on Independent Test Set (160 images)")
    orig_indep_eval = evaluate_on_dataset(orig_model, X_test_indep, y_test_indep, test_indep_paths, "Original Deployed Model on Independent Test Set (160 images)")

    print_eval_report(orig_indep_eval)
    print_eval_report(warmstart_indep_eval)

    # 10. Benchmark Test Set Evaluation (80 images)
    print("\n[STEP 9] Evaluating on Benchmark Test Set (80 images)...")
    warmstart_bench_eval = evaluate_on_dataset(saved_warmstart, X_bench, y_bench, bench_paths, "Warm-Start Candidate on Benchmark Test Set (80 images)")
    orig_bench_eval = evaluate_on_dataset(orig_model, X_bench, y_bench, bench_paths, "Original Deployed Model on Benchmark Test Set (80 images)")

    print_eval_report(orig_bench_eval)
    print_eval_report(warmstart_bench_eval)

    # 11. Test Grad-CAM Compatibility
    print("\n[STEP 10] Testing Grad-CAM Compatibility...")
    import ai_engine
    test_img = Image.open("sample_images/sample_mri_glioma.png").convert("RGB")
    ai_engine.MODELS["warmstart_test"] = saved_warmstart
    gc_res = ai_engine.run_classification_and_gradcam(test_img, model_type="warmstart_test")
    del ai_engine.MODELS["warmstart_test"]

    print("Grad-CAM verification successful:")
    print(f"  Prediction: {gc_res['prediction_display']}")
    print(f"  Confidence: {gc_res['confidence_percentage']}%")
    print(f"  Peak attention: {gc_res['peak_attention']}")
    print(f"  Inference time: {gc_res['inference_time_ms']} ms")

    # 12. Save Results JSON
    summary = {
        "model_path": TARGET_MODEL_PATH,
        "best_epoch": best_epoch,
        "best_val_loss": float(val_loss),
        "best_val_accuracy": float(val_acc),
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "trainable_base_layers": base_trainable,
        "learning_rate": learning_rate,
        "history": {k: [float(v) for v in history.history[k]] for k in history.history},
        "evaluations": {
            "baseline_val": baseline_val_eval,
            "warmstart_val": warmstart_val_eval,
            "original_independent": orig_indep_eval,
            "warmstart_independent": warmstart_indep_eval,
            "original_benchmark": orig_bench_eval,
            "warmstart_benchmark": warmstart_bench_eval
        }
    }
    with open("warmstart_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved complete payload to warmstart_results.json.")

if __name__ == "__main__":
    run_experiment()
