import os
import json
import random
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import certifi

CLASS_MAPPING = {
    'glioma': 'glioma',
    'meningioma': 'meningioma',
    'no-tumor': 'notumor',
    'pituitary': 'pituitary'
}

DATASET_ROOT = "dataset_clean"

def get_existing_test_bases():
    """Gathers base names of all 80 benchmark test images to prevent leakage."""
    test_bases = set()
    for local_cls in CLASS_MAPPING.values():
        cdir = os.path.join("testing", local_cls)
        if not os.path.isdir(cdir): continue
        for f in os.listdir(cdir):
            if f.startswith('.'): continue
            base = f.split('_jpg.rf.')[0] if '_jpg.rf.' in f else f.rsplit('.', 1)[0]
            test_bases.add(base)
    return test_bases

def download_file(item):
    url, out_path = item
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        return True
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = resp.read()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def build_leak_free_dataset():
    random.seed(42)
    ctx = ssl.create_default_context(cafile=certifi.where())
    
    test_bases_to_exclude = get_existing_test_bases()
    print(f"Benchmark test bases to exclude from train/val: {len(test_bases_to_exclude)}")
    
    # 1. Fetch all file info from HuggingFace
    print("Fetching repository tree from HuggingFace...")
    hf_data = defaultdict(lambda: defaultdict(list)) # local_cls -> base_id -> [(fname, raw_url)]
    
    for hf_cls, local_cls in CLASS_MAPPING.items():
        for split in ['train', 'valid', 'test']:
            url = f"https://huggingface.co/api/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/tree/main/timri/{split}/{hf_cls}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            res = urllib.request.urlopen(req, context=ctx, timeout=15)
            files = json.loads(res.read().decode())
            for f in files:
                fpath = f.get('path', '')
                if not fpath.endswith('.jpg'): continue
                fname = os.path.basename(fpath)
                base_id = fname.split('_jpg.rf.')[0] if '_jpg.rf.' in fname else fname.rsplit('.', 1)[0]
                raw_url = f"https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor/resolve/main/{fpath}"
                hf_data[local_cls][base_id].append((fname, raw_url))

    # 2. Partition base IDs cleanly
    # For each class:
    # - Exclude test_bases_to_exclude
    # - 50 unique base IDs -> val
    # - 40 unique base IDs -> test_independent
    # - Remaining (up to ~350 unique base IDs) -> train
    split_tasks = []
    split_summary = defaultdict(lambda: defaultdict(int))
    
    for local_cls, bases_dict in hf_data.items():
        all_bases = list(bases_dict.keys())
        # Filter out benchmark test bases
        available_bases = [b for b in all_bases if b not in test_bases_to_exclude]
        random.shuffle(available_bases)
        
        val_bases = available_bases[:50]
        test_indep_bases = available_bases[50:90]
        train_bases = available_bases[90:450] # up to 360 unique base images
        
        print(f"Class '{local_cls}': available unique bases = {len(available_bases)} | train bases = {len(train_bases)} | val bases = {len(val_bases)} | test_indep bases = {len(test_indep_bases)}")
        
        # Verify disjointness
        assert len(set(train_bases) & set(val_bases)) == 0
        assert len(set(train_bases) & set(test_indep_bases)) == 0
        assert len(set(val_bases) & set(test_indep_bases)) == 0
        assert len(set(train_bases) & test_bases_to_exclude) == 0
        assert len(set(val_bases) & test_bases_to_exclude) == 0
        
        # Assign files to splits
        for b in train_bases:
            # Take up to 2 augmented variants per base image for training diversity
            for fname, url in bases_dict[b][:2]:
                out = os.path.join(DATASET_ROOT, "train", local_cls, fname)
                split_tasks.append((url, out))
                split_summary["train"][local_cls] += 1
                
        for b in val_bases:
            # Exactly 1 representative variant per base for clean validation
            fname, url = bases_dict[b][0]
            out = os.path.join(DATASET_ROOT, "val", local_cls, fname)
            split_tasks.append((url, out))
            split_summary["val"][local_cls] += 1
            
        for b in test_indep_bases:
            # Exactly 1 representative variant per base for independent testing
            fname, url = bases_dict[b][0]
            out = os.path.join(DATASET_ROOT, "test_independent", local_cls, fname)
            split_tasks.append((url, out))
            split_summary["test_independent"][local_cls] += 1

    print("\nDataset Partition Plan:")
    for split_name in ["train", "val", "test_independent"]:
        print(f"  Split '{split_name}':")
        for local_cls in CLASS_MAPPING.values():
            print(f"    {local_cls:12s}: {split_summary[split_name][local_cls]} images")
            
    print(f"\nTotal images to download: {len(split_tasks)}. Starting parallel downloads (25 threads)...")
    with ThreadPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(download_file, split_tasks))
        
    success_count = sum(1 for r in results if r)
    print(f"Download complete: {success_count}/{len(split_tasks)} downloaded successfully.")

if __name__ == "__main__":
    build_leak_free_dataset()
