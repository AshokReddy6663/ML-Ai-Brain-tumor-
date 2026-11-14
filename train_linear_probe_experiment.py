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

TARGET_MODEL_PATH = "brain_tumor_mobilenet_v2_linear_probe.keras"
ORIGINAL_MODEL_PATH = "brain_tumor_mobilenet_v2.keras"
PREVIOUS_CANDIDATE_PATH = "brain_tumor_mobilenet_v2_improved.keras"

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

def build_frozen_backbone_model():
    """Builds MobileNetV2 with 100% frozen backbone for the entire linear probe experiment."""
    inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer_1')
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model._name = 'mobilenetv2_1.00_128'
    base_model.trainable = False  # Strictly frozen
    
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

def run_experiment():
    print("=" * 75)
    print("TARGETED FROZEN-BACKBONE LINEAR PROBE EXPERIMENT")
    print("=" * 75)

    # 1. Load data
    print("\n[STEP 1] Loading preprocessed datasets...")
    X_train, y_train, train_paths = load_split_data(TRAIN_DIR)
    X_val, y_val, val_paths = load_split_data(VAL_DIR)
    X_test_indep, y_test_indep, test_indep_paths = load_split_data(TEST_INDEP_DIR)
    X_bench, y_bench, bench_paths = load_split_data(BENCHMARK_DIR)

    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    # 2. Balanced class weights
    total_samples = len(y_train)
    class_weight_dict = {}
    for i in range(NUM_CLASSES):
        count_i = int(np.sum(y_train == i))
        class_weight_dict[i] = float(total_samples / (NUM_CLASSES * max(count_i, 1)))

    print("\n[STEP 2] Class Distribution & Weights:")
    for c in CLASS_NAMES:
        print(f"  {c:12s} (n={np.sum(y_train == CLASS_MAP[c]):3d}): weight = {class_weight_dict[CLASS_MAP[c]]:.3f}")

    # 3. Data augmentation for training only
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

    # 4. Build Model with 100% Frozen Backbone
    print("\n[STEP 3] Initializing MobileNetV2 with 100% frozen backbone...")
    model, base_model = build_frozen_backbone_model()
    
    trainable_count = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_trainable_count = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    print(f"  Trainable Parameters (Head only):       {trainable_count:,}")
    print(f"  Non-Trainable Parameters (Backbone):   {non_trainable_count:,}")
    assert trainable_count == 164484, f"Expected 164,484 trainable parameters, got {trainable_count}"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks with ModelCheckpoint active from EPOCH 1
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
        min_lr=1e-6,
        verbose=1
    )

    print("\n[STEP 4] Training linear probe head (up to 30 epochs, checkpointing from Epoch 1)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    # 6. Post-training verification of saved checkpoint
    print(f"\n[STEP 5] Training finished. Reloading {TARGET_MODEL_PATH} for verification...")
    saved_probe_model = tf.keras.models.load_model(TARGET_MODEL_PATH)
    
    val_loss, val_acc = saved_probe_model.evaluate(X_val, y_val_cat, verbose=0)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_loss_reported = float(np.min(history.history['val_loss']))
    
    print(f"  Best Epoch in History: {best_epoch}")
    print(f"  Best Val Loss in Hist: {best_loss_reported:.4f}")
    print(f"  Reloaded Model Loss:   {val_loss:.4f}")
    print(f"  Reloaded Model Acc:    {val_acc*100:.2f}%")

    # Evaluate on Validation Set
    val_eval = evaluate_on_dataset(saved_probe_model, X_val, y_val, val_paths, "Linear Probe Model on Validation Set (200 images)")
    print_eval_report(val_eval)

    # 7. Evaluate on Independent Test Set (160 images)
    print("\n[STEP 6] Evaluating on Independent Test Set (160 images)...")
    orig_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
    prev_cand_model = tf.keras.models.load_model(PREVIOUS_CANDIDATE_PATH)

    probe_indep_eval = evaluate_on_dataset(saved_probe_model, X_test_indep, y_test_indep, test_indep_paths, "Linear Probe Candidate on Independent Test Set (160 images)")
    orig_indep_eval = evaluate_on_dataset(orig_model, X_test_indep, y_test_indep, test_indep_paths, "Original Model on Independent Test Set (160 images)")
    prev_cand_indep_eval = evaluate_on_dataset(prev_cand_model, X_test_indep, y_test_indep, test_indep_paths, "Previous Overfitted Candidate on Independent Test Set (160 images)")

    print_eval_report(orig_indep_eval)
    print_eval_report(prev_cand_indep_eval)
    print_eval_report(probe_indep_eval)

    # 8. Evaluate on Benchmark Test Set (80 images)
    print("\n[STEP 7] Evaluating on Benchmark Test Set (80 images)...")
    probe_bench_eval = evaluate_on_dataset(saved_probe_model, X_bench, y_bench, bench_paths, "Linear Probe Candidate on Benchmark Test Set (80 images)")
    orig_bench_eval = evaluate_on_dataset(orig_model, X_bench, y_bench, bench_paths, "Original Model on Benchmark Test Set (80 images)")
    
    print_eval_report(orig_bench_eval)
    print_eval_report(probe_bench_eval)

    # 9. Test Grad-CAM compatibility
    print("\n[STEP 8] Verifying Grad-CAM Compatibility...")
    import ai_engine
    test_img = Image.open("sample_images/sample_mri_glioma.png").convert("RGB")
    ai_engine.MODELS["linear_probe_test"] = saved_probe_model
    gc_res = ai_engine.run_classification_and_gradcam(test_img, model_type="linear_probe_test")
    del ai_engine.MODELS["linear_probe_test"]

    print("Grad-CAM verification successful:")
    print(f"  Prediction: {gc_res['prediction_display']}")
    print(f"  Confidence: {gc_res['confidence_percentage']}%")
    print(f"  Peak attention: {gc_res['peak_attention']}")
    print(f"  Inference time: {gc_res['inference_time_ms']} ms")

    # 10. Save JSON payload
    summary = {
        "model_path": TARGET_MODEL_PATH,
        "best_epoch": best_epoch,
        "best_val_loss": val_loss,
        "best_val_accuracy": val_acc,
        "history": {k: [float(v) for v in history.history[k]] for k in history.history},
        "evaluations": {
            "val_eval": val_eval,
            "probe_independent": probe_indep_eval,
            "original_independent": orig_indep_eval,
            "previous_candidate_independent": prev_cand_indep_eval,
            "probe_benchmark": probe_bench_eval,
            "original_benchmark": orig_bench_eval
        }
    }
    with open("linear_probe_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved complete payload to linear_probe_results.json.")

if __name__ == "__main__":
    run_experiment()
