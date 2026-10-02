#!/usr/bin/env python3
"""Restore crystal_gnn/model_inference.pt (untracked in git; 77,454,219 bytes).

SHA256: ebd1afcb77f70c58326a71c4d2dba2cf93044bb1e540d4618bf1461ecd1e6489

Usage:
    python scripts/fetch_model.py [SOURCE]

SOURCE may be:
  * a checkpoint file,
  * a directory containing crystal_gnn/model_inference.pt (or model_inference.pt),
  * a git mirror/bare repo whose history still contains the file
    (e.g. the pre-rewrite backup clone ../GNN-backup.git).

If omitted, $MODEL_SOURCE is used, then default mirror locations.
Exits 0 when the file is already present and hashes correctly.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHA256 = "ebd1afcb77f70c58326a71c4d2dba2cf93044bb1e540d4618bf1461ecd1e6489"
SIZE = 77454219
REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "crystal_gnn" / "model_inference.pt"
BLOB_PATH = "crystal_gnn/model_inference.pt"
DEFAULT_MIRRORS = [REPO.parent / "GNN-backup.git"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    return sha256(path) == SHA256


def extract_from_git(git_dir: Path) -> Path:
    """Extract the checkpoint from git history into a temp file."""
    revs = subprocess.run(
        ["git", f"--git-dir={git_dir}", "log", "--all", "--diff-filter=AM", "--format=%H", "--", BLOB_PATH],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not revs:
        raise FileNotFoundError(f"{git_dir} history has no {BLOB_PATH}")
    rev = next(
        (
            r for r in revs
            if subprocess.run(
                ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{r}:{BLOB_PATH}"],
                capture_output=True,
            ).returncode == 0
        ),
        None,
    )
    if rev is None:
        raise FileNotFoundError(f"{git_dir}: no revision actually contains {BLOB_PATH}")
    tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name)
    with open(tmp, "wb") as out:
        subprocess.run(
            ["git", f"--git-dir={git_dir}", "show", f"{rev}:{BLOB_PATH}"],
            stdout=out, check=True,
        )
    return tmp


def candidates_from(source: str):
    p = Path(source)
    if p.is_file():
        yield p
    elif p.is_dir():
        for guess in (p / BLOB_PATH, p / "model_inference.pt"):
            if guess.is_file():
                yield guess
        if (p / "HEAD").exists() and (p / "objects").exists():
            yield extract_from_git(p)
    else:
        print(f"[fetch_model] source not found: {p}", file=sys.stderr)


def main(argv) -> int:
    if verify(TARGET):
        print(f"[fetch_model] OK: {TARGET} already present, sha256 verified")
        return 0

    sources = [s for s in [argv[1] if len(argv) > 1 else None, os.environ.get("MODEL_SOURCE")] if s]
    sources += [str(m) for m in DEFAULT_MIRRORS if m.exists()]

    for source in sources:
        for cand in candidates_from(source):
            if verify(cand):
                TARGET.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cand, TARGET)
                print(f"[fetch_model] installed {TARGET} ({SIZE} bytes, sha256 verified)")
                return 0
            print(f"[fetch_model] hash/size mismatch, skipping: {cand}", file=sys.stderr)

    print(
        "[fetch_model] checkpoint not found. Provide a source:\n"
        f"  python scripts/fetch_model.py <path to {BLOB_PATH} or mirror repo>\n"
        f"  expected sha256: {SHA256}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
