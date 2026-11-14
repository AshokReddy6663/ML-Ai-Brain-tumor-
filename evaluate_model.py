import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import ai_engine

# Classes as mapped in the model
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

TEST_DIR = 'testing'

def evaluate():
    model = ai_engine.get_model('mobilenet_v2')
    
    y_true = []
    y_pred = []
    
    class_counts = {name: 0 for name in CLASS_NAMES}
    total_images = 0
    
    t0 = time.time()
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.startswith('.'):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                continue
                
            # Preprocessing EXACTLY as in working pipeline
            img_canvas, _ = ai_engine.letterbox_image(image, target_size=(128, 128))
            arr = np.array(img_canvas, dtype=np.float32)
            tensor_val = arr / 127.5 - 1.0
            input_tensor = np.expand_dims(tensor_val, axis=0)
            
            # Predict
            preds = model.predict(input_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            
            y_true.append(CLASS_MAP[class_name])
            y_pred.append(pred_idx)
            
            class_counts[class_name] += 1
            total_images += 1

    t1 = time.time()
    print(f"Evaluated {total_images} images in {t1-t0:.2f} seconds.")
    
    # Confusion Matrix
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
        
    # Per-class metrics
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(cm[i, :]))
        }
        
    # Overall metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    macro_precision = np.mean([per_class[n]['precision'] for n in CLASS_NAMES])
    macro_recall = np.mean([per_class[n]['recall'] for n in CLASS_NAMES])
    macro_f1 = np.mean([per_class[n]['f1'] for n in CLASS_NAMES])
    
    weighted_precision = sum(per_class[n]['precision'] * per_class[n]['support'] for n in CLASS_NAMES) / total_images
    weighted_recall = sum(per_class[n]['recall'] * per_class[n]['support'] for n in CLASS_NAMES) / total_images
    weighted_f1 = sum(per_class[n]['f1'] * per_class[n]['support'] for n in CLASS_NAMES) / total_images

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    plt.colorbar(cax)
    
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', 
                color='white' if cm[i, j] > (np.max(cm)/2) else 'black')
                
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels([n.capitalize() for n in CLASS_NAMES], rotation=45)
    ax.set_yticklabels([n.capitalize() for n in CLASS_NAMES])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # Save Report
    with open('model_evaluation.txt', 'w') as f:
        f.write("Model: brain_tumor_mobilenet_v2.keras\n")
        f.write("Dataset: testing/\n")
        f.write(f"Number of test images: {total_images}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Macro): {macro_precision:.4f}\n")
        f.write(f"Recall (Macro): {macro_recall:.4f}\n")
        f.write(f"F1-score (Macro): {macro_f1:.4f}\n")
        f.write(f"F1-score (Weighted): {weighted_f1:.4f}\n\n")
        
        f.write("Class Distribution:\n")
        for name in CLASS_NAMES:
            f.write(f"{name.capitalize()}: {class_counts[name]}\n")
            
        f.write("\nPer-class results:\n")
        for name in CLASS_NAMES:
            f.write(f"\n{name.capitalize()}:\n")
            f.write(f"Precision: {per_class[name]['precision']:.4f}\n")
            f.write(f"Recall: {per_class[name]['recall']:.4f}\n")
            f.write(f"F1: {per_class[name]['f1']:.4f}\n")
            
        f.write("\nConfusion matrix:\n")
        f.write("True \\ Pred | " + " | ".join(f"{n[:4]}" for n in CLASS_NAMES) + "\n")
        f.write("-" * 40 + "\n")
        for i in range(4):
            row_str = " | ".join(f"{cm[i][j]:4d}" for j in range(4))
            f.write(f"{CLASS_NAMES[i][:4]:11s} | {row_str}\n")
            
    print("Saved model_evaluation.txt")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    best_class = max(per_class.items(), key=lambda x: x[1]['f1'])[0]
    worst_class = min(per_class.items(), key=lambda x: x[1]['f1'])[0]
    print(f"Best performing: {best_class.capitalize()}")
    print(f"Worst performing: {worst_class.capitalize()}")

if __name__ == "__main__":
    evaluate()
