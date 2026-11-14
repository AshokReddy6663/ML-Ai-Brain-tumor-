import os
import shutil
from collections import defaultdict
from PIL import Image
import numpy as np

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
DATASET_ROOT = "dataset_clean"
BENCHMARK_DIR = "testing"

def compute_dhash(image: Image.Image, hash_size=8) -> int:
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.int32)
    diff = arr[:, 1:] > arr[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def repartition_with_perceptual_guarantee():
    print("=" * 70)
    print("PERCEPTUAL DEDUPLICATION & LEAK-FREE PARTITIONING")
    print("=" * 70)
    
    # 1. Load all downloaded candidate images across all folders
    print("Scanning all existing candidate images...")
    pool = defaultdict(list)
    for root, dirs, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.startswith('.'): continue
            p = os.path.join(root, f)
            cls = os.path.basename(os.path.dirname(p))
            if cls not in CLASSES: continue
            pool[cls].append((p, f))

    # Benchmark test images
    benchmark_pool = defaultdict(list)
    for cls in CLASSES:
        cdir = os.path.join(BENCHMARK_DIR, cls)
        for f in os.listdir(cdir):
            if f.startswith('.'): continue
            p = os.path.join(cdir, f)
            benchmark_pool[cls].append((p, f))

    # Create temporary backup directory to rearrange files cleanly
    temp_pool_dir = "temp_candidate_pool"
    os.makedirs(temp_pool_dir, exist_ok=True)
    
    # Move all candidates into temp pool
    all_candidate_files = []
    for cls in CLASSES:
        os.makedirs(os.path.join(temp_pool_dir, cls), exist_ok=True)
        for src_path, fname in pool[cls]:
            dst = os.path.join(temp_pool_dir, cls, fname)
            if os.path.exists(src_path) and src_path != dst:
                shutil.move(src_path, dst)
            all_candidate_files.append((dst, fname, cls))

    # Clean existing DATASET_ROOT
    shutil.rmtree(DATASET_ROOT, ignore_errors=True)
    os.makedirs(DATASET_ROOT, exist_ok=True)

    # 2. For each class, cluster candidates and identify benchmark matches
    for cls in CLASSES:
        print(f"\nProcessing class: {cls.upper()}...")
        cls_candidates = [x for x in all_candidate_files if x[2] == cls]
        bench_files = benchmark_pool[cls]
        
        # Load dhash and arrays
        cand_data = []
        for p, fn, _ in cls_candidates:
            with Image.open(p) as img:
                dh = compute_dhash(img)
                arr = np.array(img, dtype=np.float32)
                base = fn.split('_jpg.rf.')[0] if '_jpg.rf.' in fn else fn.rsplit('.', 1)[0]
                cand_data.append({"path": p, "fname": fn, "base": base, "dhash": dh, "arr": arr})
                
        bench_data = []
        for p, fn in bench_files:
            with Image.open(p) as img:
                dh = compute_dhash(img)
                arr = np.array(img, dtype=np.float32)
                base = fn.split('_jpg.rf.')[0] if '_jpg.rf.' in fn else fn.rsplit('.', 1)[0]
                bench_data.append({"path": p, "fname": fn, "base": base, "dhash": dh, "arr": arr})

        n_cand = len(cand_data)
        dsu = DSU(n_cand)
        
        # Cluster within candidates (by base ID or perceptual match)
        # First group by base_id
        base_map = defaultdict(list)
        for i, c in enumerate(cand_data):
            base_map[c["base"]].append(i)
        for indices in base_map.values():
            for k in range(1, len(indices)):
                dsu.union(indices[0], indices[k])
                
        # Second, group by visual match (hamming distance <= 1 or dist==2 and MAD < 10)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                dist = hamming_distance(cand_data[i]["dhash"], cand_data[j]["dhash"])
                if dist <= 2:
                    mad = np.mean(np.abs(cand_data[i]["arr"] - cand_data[j]["arr"]))
                    if mad < 12.0:
                        dsu.union(i, j)

        # Flag any candidate that matches benchmark test scans
        bench_tainted_roots = set()
        for i in range(n_cand):
            for b in bench_data:
                # Same base ID
                if cand_data[i]["base"] == b["base"]:
                    bench_tainted_roots.add(dsu.find(i))
                    continue
                # Visual match
                dist = hamming_distance(cand_data[i]["dhash"], b["dhash"])
                if dist <= 2:
                    mad = np.mean(np.abs(cand_data[i]["arr"] - b["arr"]))
                    if mad < 12.0:
                        bench_tainted_roots.add(dsu.find(i))

        # Group candidates by root
        clusters = defaultdict(list)
        for i in range(n_cand):
            root = dsu.find(i)
            if root not in bench_tainted_roots:
                clusters[root].append(cand_data[i])

        print(f"  Total candidate files: {n_cand}")
        print(f"  Clusters tainted by benchmark test: {len(bench_tainted_roots)} (excluded completely)")
        print(f"  Clean independent clusters available: {len(clusters)}")

        # Partition clean clusters
        cluster_list = list(clusters.values())
        np.random.seed(42)
        np.random.shuffle(cluster_list)

        val_clusters = cluster_list[:50]
        test_indep_clusters = cluster_list[50:90]
        train_clusters = cluster_list[90:]

        print(f"  Allocated: train={len(train_clusters)} clusters, val={len(val_clusters)} clusters, test_indep={len(test_indep_clusters)} clusters")

        # Copy to clean splits
        for split_name, assigned_clusters, max_per_cluster in [
            ("val", val_clusters, 1),
            ("test_independent", test_indep_clusters, 1),
            ("train", train_clusters, 2)
        ]:
            out_dir = os.path.join(DATASET_ROOT, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for cluster in assigned_clusters:
                for item in cluster[:max_per_cluster]:
                    shutil.copy(item["path"], os.path.join(out_dir, item["fname"]))

    # Cleanup temp pool
    shutil.rmtree(temp_pool_dir, ignore_errors=True)
    print("\nRepartitioning complete! Clean dataset ready for final audit.")

if __name__ == "__main__":
    repartition_with_perceptual_guarantee()
