import os, sys
cache_dir = "data/cache"
files = [f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
total = len(files)
sizes = [os.path.getsize(os.path.join(cache_dir, f)) for f in files[:100]]
avg_kb = sum(sizes) / len(sizes) / 1024 if sizes else 0
print(f"Total cache files: {total}")
print(f"Avg file size (sample of 100): {avg_kb:.1f} KB")
print(f"Estimated total cache size: {total * avg_kb / 1024:.0f} MB")
