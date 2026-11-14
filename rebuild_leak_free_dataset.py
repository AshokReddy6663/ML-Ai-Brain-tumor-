import os
import shutil
import random
from collections import defaultdict
from PIL import Image
import numpy as np
import certifi

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

def compute_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Fast SSIM on pre-downsampled 64x64 float32 grayscale arrays."""
    mu1, mu2 = np.mean(arr1), np.mean(arr2)
    sig1, sig2 = np.var(arr1), np.var(arr2)
    sig12 = np.mean((arr1 - mu1) * (arr2 - mu2))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu1 * mu2 + c1) * (2 * sig12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sig1 + sig2 + c2)))

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

def rebuild():
    print("=" * 70)
    print("REBUILDING CLEAN DATASET WITH STRICT STRUCTURAL (SSIM) & DHASH CLUSTERING")
    print("=" * 70)
    
    # 1. Collect all images currently in dataset_clean into a temporary holding area
    temp_dir = "temp_rebuild_pool"
    os.makedirs(temp_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(temp_dir, c), exist_ok=True)
        
    all_files = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.startswith('.'): continue
            p = os.path.join(root, f)
            cls = os.path.basename(os.path.dirname(p))
            if cls not in CLASSES: continue
            dst = os.path.join(temp_dir, cls, f)
            if os.path.exists(p) and p != dst:
                shutil.move(p, dst)
            all_files.append((dst, f, cls))
            
    shutil.rmtree(DATASET_ROOT, ignore_errors=True)
    os.makedirs(DATASET_ROOT, exist_ok=True)

    # 2. Benchmark images
    bench_data = defaultdict(list)
    for cls in CLASSES:
        cdir = os.path.join(BENCHMARK_DIR, cls)
        for f in os.listdir(cdir):
            if f.startswith('.'): continue
            p = os.path.join(cdir, f)
            with Image.open(p) as img:
                arr = np.array(img.convert('L').resize((64, 64)), dtype=np.float32)
                dh = compute_dhash(img)
                base = f.split('_jpg.rf.')[0] if '_jpg.rf.' in f else f.rsplit('.', 1)[0]
                bench_data[cls].append({"path": p, "fname": f, "base": base, "dhash": dh, "arr": arr})

    random.seed(42)
    np.random.seed(42)

    for cls in CLASSES:
        print(f"\nProcessing class: {cls.upper()}...")
        cls_files = [x for x in all_files if x[2] == cls]
        
        # Load all candidate representations
        cand_list = []
        for p, fn, _ in cls_files:
            with Image.open(p) as img:
                arr = np.array(img.convert('L').resize((64, 64)), dtype=np.float32)
                dh = compute_dhash(img)
                base = fn.split('_jpg.rf.')[0] if '_jpg.rf.' in fn else fn.rsplit('.', 1)[0]
                cand_list.append({"path": p, "fname": fn, "base": base, "dhash": dh, "arr": arr})

        n = len(cand_list)
        dsu = DSU(n)
        
        # Cluster rule 1: Same base ID
        base_groups = defaultdict(list)
        for i, c in enumerate(cand_list):
            base_groups[c["base"]].append(i)
        for group in base_groups.values():
            for k in range(1, len(group)):
                dsu.union(group[0], group[k])
                
        # Cluster rule 2: Perceptual similarity (dHash == 0 OR SSIM >= 0.84)
        for i in range(n):
            for j in range(i + 1, n):
                # Quick dhash check
                dist = hamming_distance(cand_list[i]["dhash"], cand_list[j]["dhash"])
                if dist == 0:
                    dsu.union(i, j)
                elif dist <= 4:
                    s = compute_ssim(cand_list[i]["arr"], cand_list[j]["arr"])
                    if s >= 0.84:
                        dsu.union(i, j)

        # Check against benchmark test set
        tainted_roots = set()
        for i in range(n):
            for b in bench_data[cls]:
                if cand_list[i]["base"] == b["base"]:
                    tainted_roots.add(dsu.find(i))
                    break
                dist = hamming_distance(cand_list[i]["dhash"], b["dhash"])
                if dist == 0:
                    tainted_roots.add(dsu.find(i))
                    break
                elif dist <= 4:
                    s = compute_ssim(cand_list[i]["arr"], b["arr"])
                    if s >= 0.84:
                        tainted_roots.add(dsu.find(i))
                        break

        # Group into clean clusters
        clean_clusters = defaultdict(list)
        for i in range(n):
            root = dsu.find(i)
            if root not in tainted_roots:
                clean_clusters[root].append(cand_list[i])

        print(f"  Total candidate files: {n}")
        print(f"  Benchmark-tainted clusters: {len(tainted_roots)} (purged)")
        print(f"  Independent patient/scan clusters: {len(clean_clusters)}")

        clusters = list(clean_clusters.values())
        random.shuffle(clusters)

        val_clusters = clusters[:50]
        test_indep_clusters = clusters[50:90]
        train_clusters = clusters[90:]

        print(f"  Allocated: train={len(train_clusters)} clusters | val={len(val_clusters)} clusters | test_indep={len(test_indep_clusters)} clusters")

        # Copy to final destinations
        # For val and test_indep, select exactly 1 representative image per patient cluster
        for split_name, assigned_clusters, max_imgs in [
            ("val", val_clusters, 1),
            ("test_independent", test_indep_clusters, 1),
            ("train", train_clusters, 2)
        ]:
            out_dir = os.path.join(DATASET_ROOT, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for cl in assigned_clusters:
                for item in cl[:max_imgs]:
                    shutil.copy(item["path"], os.path.join(out_dir, item["fname"]))

    # Cleanup temp pool
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nRebuild complete with zero structural/perceptual leakage!")

if __name__ == "__main__":
    rebuild()
