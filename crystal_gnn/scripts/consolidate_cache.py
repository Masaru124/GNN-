"""One-time script to consolidate individual .pt cache files into a single fast-loading archive.

Creates data/cache_consolidated.pt containing all graphs indexed by (material_id, radius).
This eliminates the overhead of 168K individual torch.load() calls during training.
"""
import os
import sys
import time
import torch
from pathlib import Path

def main():
    cache_dir = Path("data/cache")
    out_path = Path("data/cache_consolidated.pt")
    
    if out_path.exists():
        print(f"[consolidate] {out_path} already exists. Delete it first to rebuild.")
        return
    
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    print(f"[consolidate] Found {len(files)} cache files to consolidate...")
    
    archive = {}
    t0 = time.time()
    for i, fname in enumerate(files):
        fpath = cache_dir / fname
        data = torch.load(fpath, weights_only=False)
        # Key is the filename without .pt extension
        key = fname[:-3]  # strip ".pt"
        archive[key] = data
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(files) - i - 1) / rate
            print(f"[consolidate] {i+1}/{len(files)} loaded ({rate:.0f} files/s, ETA {eta:.0f}s)", flush=True)
    
    elapsed = time.time() - t0
    print(f"[consolidate] All {len(files)} files loaded in {elapsed:.1f}s. Saving archive...", flush=True)
    
    torch.save(archive, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[consolidate] Saved {out_path} ({size_mb:.0f} MB)")
    print(f"[consolidate] Done! Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
