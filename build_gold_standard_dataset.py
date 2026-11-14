import os
import json
import shutil
import random
import ssl
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import certifi

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
HF_CLASS_MAP = {
    'glioma': 'glioma',
    'meningioma': 'meningioma',
    'no-tumor': 'notumor',
    'pituitary': 'pituitary'
}
DATASET_ROOT = "dataset_clean"
BENCHMARK_DIR = "testing"

def compute_dhash(image: Image.Image, hash_size=8) -> int:
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.int32)
    diff = arr[:, 1:] > arr[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

def compute_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:
    mu1, mu2 = np.mean(arr1), np.mean(arr2)
    sig1, sig2 = np.var(arr1), np.var(arr2)
    sig12 = np.mean((arr1 - mu1) * (arr2 - mu2))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu1 * mu2 + c1) * (2 * sig12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sig1 + sig2 + c2)))

def extract_base_id(fname: str) -> str:
    if "_jpg.rf." in fname:
        return fname.split("_jpg.rf.")[0]
    elif ".rf." in fname:
        return fname.split(".rf.")[0]
    return fname.rsplit('.', 1)[0]

def build_gold_standard_dataset():
    print("=" * 70)
    print("BUILDING GOLD STANDARD LEAK-FREE DATASET (PATIENT-LEVEL SSIM & DHASH)")
    print("=" * 70)

    # 1. Load benchmark test images
    bench_data = defaultdict(list)
    bench_bases = set()
    for cls in CLASSES:
        cdir = os.path.join(BENCHMARK_DIR, cls)
        for f in os.listdir(cdir):
            if f.startswith('.'): continue
            p = os.path.join(cdir, f)
            with Image.open(p) as img:
                arr = np.array(img.convert('L').resize((64, 64)), dtype=np.float32)
                dh = compute_dhash(img)
            base = extract_base_id(f)
            bench_bases.add(base)
            bench_data[cls].append({"fname": f, "base": base, "dh": dh, "arr": arr})
    print(f"Loaded {sum(len(v) for v in bench_data.values())} benchmark test images ({len(bench_bases)} unique base IDs).")

    # 2. Collect existing clean candidate images from dataset_clean if available
    existing_candidates = defaultdict(list)
    for root, dirs, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.startswith('.'): continue
            p = os.path.join(root, f)
            cls = os.path.basename(os.path.dirname(p))
            if cls in CLASSES:
                existing_candidates[cls].append((p, f))

    # Move existing candidates to temporary pool
    temp_pool = "temp_gold_pool"
    os.makedirs(temp_pool, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(temp_pool, cls), exist_ok=True)
        for p, f in existing_candidates[cls]:
            dst = os.path.join(temp_pool, cls, f)
            if p != dst and os.path.exists(p):
                shutil.move(p, dst)

    shutil.rmtree(DATASET_ROOT, ignore_errors=True)
    os.makedirs(DATASET_ROOT, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    partition_summary = defaultdict(lambda: defaultdict(int))

    for cls in CLASSES:
        print(f"\nProcessing class: {cls.upper()}...")
        cls_pool_dir = os.path.join(temp_pool, cls)
        all_pool_files = [os.path.join(cls_pool_dir, f) for f in os.listdir(cls_pool_dir) if not f.startswith('.')]
        
        # Load and verify candidates against benchmark
        verified_candidates = []
        for p in all_pool_files:
            fname = os.path.basename(p)
            base = extract_base_id(fname)
            if base in bench_bases:
                continue
                
            with Image.open(p) as img:
                dh = compute_dhash(img)
                arr = np.array(img.convert('L').resize((64, 64)), dtype=np.float32)
                
            # Check SSIM and dHash to benchmark test
            is_tainted = False
            for b in bench_data[cls]:
                if dh == b["dh"]:
                    is_tainted = True
                    break
                if compute_ssim(arr, b["arr"]) >= 0.82:
                    is_tainted = True
                    break
                    
            if not is_tainted:
                verified_candidates.append({"path": p, "fname": fname, "base": base, "dh": dh, "arr": arr})

        print(f"  Verified clean candidates (zero benchmark match): {len(verified_candidates)}")
        
        # Group by base scan to keep slices together
        base_to_items = defaultdict(list)
        for c in verified_candidates:
            base_to_items[c["base"]].append(c)

        unique_bases = list(base_to_items.keys())
        random.shuffle(unique_bases)

        # 1. Allocate 50 base scans to val
        val_items = []
        val_arrs = []
        val_dhs = []
        for b in unique_bases:
            if len(val_items) >= 50: break
            item = base_to_items[b][0]
            # Ensure no perceptual collision within val
            if not any(item["dh"] == vdh or compute_ssim(item["arr"], varr) >= 0.82 for varr, vdh in zip(val_arrs, val_dhs)):
                val_items.append(item)
                val_arrs.append(item["arr"])
                val_dhs.append(item["dh"])

        used_bases = set(x["base"] for x in val_items)
        remaining_bases = [b for b in unique_bases if b not in used_bases]

        # 2. Allocate 40 base scans to test_independent
        test_indep_items = []
        test_indep_arrs = []
        test_indep_dhs = []
        for b in remaining_bases:
            if len(test_indep_items) >= 40: break
            item = base_to_items[b][0]
            # Ensure no collision with val or within test_indep
            if not any(item["dh"] == vdh or compute_ssim(item["arr"], varr) >= 0.82 for varr, vdh in zip(val_arrs + test_indep_arrs, val_dhs + test_indep_dhs)):
                test_indep_items.append(item)
                test_indep_arrs.append(item["arr"])
                test_indep_dhs.append(item["dh"])

        used_bases.update(x["base"] for x in test_indep_items)
        remaining_bases = [b for b in unique_bases if b not in used_bases]

        # 3. Allocate remaining to train, strictly excluding any high SSIM/dHash to val or test_indep
        train_items = []
        train_arrs = []
        train_dhs = []
        eval_arrs = val_arrs + test_indep_arrs
        eval_dhs = val_dhs + test_indep_dhs

        for b in remaining_bases:
            for item in base_to_items[b][:2]:
                # Strict check against all val and test_indep images
                collides_eval = False
                for e_arr, e_dh in zip(eval_arrs, eval_dhs):
                    if item["dh"] == e_dh or compute_ssim(item["arr"], e_arr) >= 0.82:
                        collides_eval = True
                        break
                if not collides_eval:
                    train_items.append(item)
                    train_arrs.append(item["arr"])
                    train_dhs.append(item["dh"])

        print(f"  Allocated -> Train: {len(train_items)}, Val: {len(val_items)}, Test_Indep: {len(test_indep_items)}")

        # Copy to destination splits
        for split_name, items in [("train", train_items), ("val", val_items), ("test_independent", test_indep_items)]:
            out_dir = os.path.join(DATASET_ROOT, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for it in items:
                shutil.copy(it["path"], os.path.join(out_dir, it["fname"]))
                partition_summary[split_name][cls] += 1

    shutil.rmtree(temp_pool, ignore_errors=True)
    print("\nGold standard dataset creation complete!")
    for s in ["train", "val", "test_independent"]:
        print(f"  [{s}] Total: {sum(partition_summary[s].values())}")
        for c in CLASSES:
            print(f"    {c:12s}: {partition_summary[s][c]}")

if __name__ == "__main__":
    build_gold_standard_dataset()
