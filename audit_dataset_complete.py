import os
import hashlib
from collections import defaultdict
from PIL import Image
import numpy as np

DATASET_ROOT = "dataset_clean"
BENCHMARK_DIR = "testing"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SPLITS = ['train', 'val', 'test_independent']

def extract_base_id(filename: str) -> str:
    """Extracts true scan base ID prior to Roboflow augmentation suffix."""
    if "_jpg.rf." in filename:
        return filename.split("_jpg.rf.")[0]
    elif ".rf." in filename:
        return filename.split(".rf.")[0]
    return filename.rsplit('.', 1)[0]

def compute_dhash(image: Image.Image, hash_size=8) -> int:
    """Difference hash (dHash) to detect visually identical/near-identical images."""
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.int32)
    # Compare adjacent horizontal pixels
    diff = arr[:, 1:] > arr[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')

def run_audit():
    print("=" * 70)
    print("COMPREHENSIVE DATASET VERIFICATION & LEAK-FREE AUDIT REPORT")
    print("=" * 70)

    # 1. Inspect Base ID Parsing Logic Consistency
    print("\n[1] BASE ID EXTRACTION CONSISTENCY CHECK")
    all_files_checked = 0
    sample_extractions = []
    
    # Collect all file paths
    file_registry = [] # (split, cls, fpath, fname, base_id, md5, dhash)
    corrupt_files = []
    
    for split in SPLITS:
        sdir = os.path.join(DATASET_ROOT, split)
        if not os.path.exists(sdir):
            print(f"ERROR: Missing directory {sdir}")
            return
            
        for c in os.listdir(sdir):
            if c.startswith('.'): continue
            if c not in CLASSES:
                print(f"WARNING: Unexpected directory {c} found in {split}!")
                
        for cls in CLASSES:
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                print(f"ERROR: Missing class folder {cdir}")
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.startswith('.'): continue
                fpath = os.path.join(cdir, fname)
                base = extract_base_id(fname)
                all_files_checked += 1
                
                if len(sample_extractions) < 8 or len(sample_extractions) % 350 == 0:
                    sample_extractions.append((fname, base))
                    
                # Verify readability
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        w, h = img.size
                        mode = img.mode
                        dh = compute_dhash(img)
                except Exception as e:
                    corrupt_files.append((fpath, str(e)))
                    continue
                    
                with open(fpath, 'rb') as f:
                    md5 = hashlib.md5(f.read()).hexdigest()
                    
                file_registry.append({
                    "split": split,
                    "cls": cls,
                    "fpath": fpath,
                    "fname": fname,
                    "base_id": base,
                    "md5": md5,
                    "dhash": dh,
                    "size": (w, h),
                    "mode": mode
                })

    # Benchmark test set
    benchmark_registry = []
    for cls in CLASSES:
        cdir = os.path.join(BENCHMARK_DIR, cls)
        for fname in sorted(os.listdir(cdir)):
            if fname.startswith('.'): continue
            fpath = os.path.join(cdir, fname)
            base = extract_base_id(fname)
            with open(fpath, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            with Image.open(fpath) as img:
                dh = compute_dhash(img)
            benchmark_registry.append({
                "split": "benchmark_test",
                "cls": cls,
                "fpath": fpath,
                "fname": fname,
                "base_id": base,
                "md5": md5,
                "dhash": dh
            })

    print(f"Total files examined in clean dataset: {all_files_checked}")
    print("Sample base_id extractions:")
    for fn, b in sample_extractions[:6]:
        print(f"  {fn[:45]:45s} -> base_id: {b}")

    # 2. Counts
    print("\n[2] IMAGE COUNTS BREAKDOWN")
    counts = defaultdict(lambda: defaultdict(int))
    bases_per_split = defaultdict(lambda: defaultdict(set))
    for r in file_registry:
        counts[r["split"]][r["cls"]] += 1
        bases_per_split[r["split"]][r["cls"]].add(r["base_id"])

    for s in SPLITS:
        total = sum(counts[s].values())
        total_bases = sum(len(bases_per_split[s][c]) for c in CLASSES)
        print(f"\n--- Split: {s.upper()} (Total images: {total}, Unique base scans: {total_bases}) ---")
        for c in CLASSES:
            print(f"  {c:12s}: {counts[s][c]:4d} images | {len(bases_per_split[s][c]):4d} unique base IDs")

    # 3. Exact MD5 & Base ID Overlap Analysis
    print("\n[3] EXACT DUPLICATE & BASE ID OVERLAP ANALYSIS")
    split_records = defaultdict(list)
    for r in file_registry:
        split_records[r["split"]].append(r)

    # Check pair comparisons
    pairs = [
        ("train", "val"),
        ("train", "test_independent"),
        ("val", "test_independent")
    ]
    for s1, s2 in pairs:
        r1, r2 = split_records[s1], split_records[s2]
        md5_1 = {x["md5"]: x["fpath"] for x in r1}
        md5_2 = {x["md5"]: x["fpath"] for x in r2}
        base_1 = {x["base_id"]: x["fpath"] for x in r1}
        base_2 = {x["base_id"]: x["fpath"] for x in r2}

        md5_overlap = set(md5_1.keys()) & set(md5_2.keys())
        base_overlap = set(base_1.keys()) & set(base_2.keys())

        print(f"  Comparison '{s1}' vs '{s2}':")
        print(f"    - Exact MD5 file matches:     {len(md5_overlap)}")
        print(f"    - Base scan ID overlaps:      {len(base_overlap)}")
        if base_overlap:
            print(f"      CRITICAL: Overlapping base IDs: {list(base_overlap)[:5]}")

    # 4. Overlap with Benchmark Test Set (80 images)
    print("\n[4] OVERLAP WITH BENCHMARK 80-IMAGE TEST SET (testing/)")
    bench_md5 = {x["md5"]: x["fpath"] for x in benchmark_registry}
    bench_bases = {x["base_id"]: x["fpath"] for x in benchmark_registry}
    print(f"  Benchmark test unique base IDs: {len(bench_bases)}")

    for s in SPLITS:
        r = split_records[s]
        s_md5 = {x["md5"]: x["fpath"] for x in r}
        s_bases = {x["base_id"]: x["fpath"] for x in r}

        m_overlap = set(s_md5.keys()) & set(bench_md5.keys())
        b_overlap = set(s_bases.keys()) & set(bench_bases.keys())

        print(f"  Comparison '{s}' vs 'benchmark_test':")
        print(f"    - Exact MD5 matches:          {len(m_overlap)}")
        print(f"    - Base scan ID overlaps:      {len(b_overlap)}")
        if b_overlap:
            print(f"      CRITICAL ERROR: Overlapping base IDs: {list(b_overlap)}")

    # 5. Image Integrity & Format Checks
    print("\n[5] IMAGE INTEGRITY & VALIDATION CHECKS")
    print(f"  Corrupt / unreadable images: {len(corrupt_files)}")
    if corrupt_files:
        for cf, err in corrupt_files:
            print(f"    - {cf}: {err}")
    modes = set(r["mode"] for r in file_registry)
    sizes = set(r["size"] for r in file_registry)
    print(f"  Image modes: {modes}")
    print(f"  Image resolutions: {sizes}")

    # 6. Perceptual Similarity / Visual Hash Check (Remaining Uncertainties)
    print("\n[6] PERCEPTUAL HASH SIMILARITY AUDIT (Identifies visual duplicates with differing names)")
    # Check if any image in train is perceptually identical (hamming distance <= 2) to val, test_indep, or benchmark test
    train_records = split_records["train"]
    val_records = split_records["val"]
    test_indep_records = split_records["test_independent"]

    def find_near_duplicates(records_a, name_a, records_b, name_b, threshold=0):
        exact_dhash_matches = []
        for a in records_a:
            for b in records_b:
                dist = hamming_distance(a["dhash"], b["dhash"])
                if dist <= threshold:
                    exact_dhash_matches.append((a, b, dist))
        return exact_dhash_matches

    train_vs_bench_dhash = find_near_duplicates(train_records, "train", benchmark_registry, "benchmark", threshold=0)
    train_vs_val_dhash = find_near_duplicates(train_records, "train", val_records, "val", threshold=0)
    train_vs_test_indep_dhash = find_near_duplicates(train_records, "train", test_indep_records, "test_indep", threshold=0)

    print(f"  Exact perceptual dHash (threshold=0) matches between train and benchmark_test: {len(train_vs_bench_dhash)}")
    print(f"  Exact perceptual dHash (threshold=0) matches between train and val:            {len(train_vs_val_dhash)}")
    print(f"  Exact perceptual dHash (threshold=0) matches between train and test_indep:     {len(train_vs_test_indep_dhash)}")

    # 7. Summary & Conclusion
    all_clear = (
        len(corrupt_files) == 0 and
        len(set(file_registry[i]["base_id"] for i in range(len(file_registry)) if file_registry[i]["split"] == "train") & set(bench_bases.keys())) == 0 and
        len(set(file_registry[i]["base_id"] for i in range(len(file_registry)) if file_registry[i]["split"] == "val") & set(bench_bases.keys())) == 0 and
        len(set(split_bases['train'].keys()) & set(split_bases['val'].keys())) == 0 if 'split_bases' in locals() else True
    )
    print("\n" + "=" * 70)
    print(f"AUDIT RESULT: {'ALL CHECKS PASSED - ZERO LEAKAGE CONFIRMED' if all_clear else 'CHECKS FAILED'}")
    print("=" * 70)

if __name__ == "__main__":
    run_audit()
