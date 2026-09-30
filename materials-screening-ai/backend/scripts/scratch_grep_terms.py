import sys, os, glob

search_terms = ["0.4595", "0.46", "0.361"]
root_dir = "materials-screening-ai"

print("=== GREP SEARCH FOR 0.4595, 0.46, 0.361 IN materials-screening-ai ===")
for root, dirs, files in os.walk(root_dir):
    # skip .git, venv, __pycache__
    if any(x in root for x in [".git", "venv", "__pycache__", "node_modules", ".pytest_cache"]):
        continue
    for file in files:
        if file.endswith((".md", ".py", ".csv", ".txt", ".json", ".ts", ".tsx")):
            fpath = os.path.join(root, file)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        for term in search_terms:
                            if term in line:
                                print(f"{fpath}:{line_no} [{term}] -> {line.strip()[:120]}")
            except Exception:
                pass
