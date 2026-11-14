import os
import io
import tensorflow as tf
from PIL import Image
import numpy as np
import ai_engine

def debug_test():
    img_path = "sample_images/sample_mri_glioma.png"
    img = Image.open(img_path).convert("RGB")
    
    # 1. Run through ai_engine pipeline
    print("--- AI ENGINE PIPELINE ---")
    res = ai_engine.run_classification_and_gradcam(img, "mobilenet_v2")
    print(res["prediction"], res["confidence_percentage"])
    print(res["probabilities"])
    
    # 2. Run directly
    print("\n--- DIRECT MODEL PIPELINE ---")
    model = ai_engine.get_model("mobilenet_v2")
    
    # Same preprocessing
    img_canvas, _ = ai_engine.letterbox_image(img, target_size=(128, 128))
    arr = np.array(img_canvas, dtype=np.float32)
    input_tensor = np.expand_dims(arr / 127.5 - 1.0, axis=0)
    
    # Print tensor stats
    print("Shape:", input_tensor.shape)
    print("Min:", np.min(input_tensor))
    print("Max:", np.max(input_tensor))
    print("Mean:", np.mean(input_tensor))
    
    preds = model.predict(input_tensor, verbose=0)[0]
    classes = ai_engine.CLASS_NAMES
    probs = {classes[i]: float(preds[i]) for i in range(len(classes))}
    print("Direct Probabilities:", probs)

if __name__ == "__main__":
    debug_test()
