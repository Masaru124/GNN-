# -*- coding: utf-8 -*-
"""
Native Windows Quantum ESPRESSO Installer (No WSL Required).

Downloads and unpacks the precompiled native Windows Intel oneAPI / MS-MPI
binaries of Quantum ESPRESSO 7.5 directly to <repo>/qe (override with QE_DIR env var).

Release source:
  QMatSuite/quantum-espresso-windows-exe
  https://github.com/QMatSuite/quantum-espresso-windows-exe/releases/tag/qe-7.5-win-oneapi-msmpi
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

QE_ZIP_URL = "https://github.com/QMatSuite/quantum-espresso-windows-exe/releases/download/qe-7.5-win-oneapi-msmpi/qe-7.5-win-oneapi-msmpi.zip"
_REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_DIR = os.environ.get("QE_DIR") or str(_REPO_ROOT / "qe")
ZIP_PATH = str(_REPO_ROOT / "qe_download.zip")


def download_with_progress(url: str, output_path: str):
    print(f"Downloading native Windows QE 7.5 from:\n{url}")
    print(f"Target: {output_path}")

    class ProgressHook:
        def __init__(self):
            self.last_reported = 0

        def __call__(self, block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = int(downloaded * 100 / total_size)
                if percent >= self.last_reported + 5:
                    self.last_reported = percent
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"  [Progress] {percent}% ({mb_down:.1f} MB / {mb_tot:.1f} MB)", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=ProgressHook())
    print("\nDownload complete!")


def install_qe():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. Download
    if not os.path.exists(ZIP_PATH):
        download_with_progress(QE_ZIP_URL, ZIP_PATH)
    else:
        print(f"Found existing zip at {ZIP_PATH} (size: {os.path.getsize(ZIP_PATH)/(1024*1024):.1f} MB)")

    # 2. Extract
    print(f"Extracting to {TARGET_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print("Extraction complete!")

    # 3. Locate pw.exe
    pw_exe = None
    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            if f.lower() in ("pw.exe", "pw.x.exe"):
                pw_exe = os.path.join(root, f)
                break
        if pw_exe:
            break

    if pw_exe:
        bin_dir = os.path.dirname(pw_exe)
        print(f"\nSUCCESS! Found native Quantum ESPRESSO executable:")
        print(f"  pw.exe: {pw_exe}")
        print(f"  bin directory: {bin_dir}")
        print(f"\nTo use globally, add to PowerShell profile or environment:")
        print(f"  $env:QE_BIN_DIR = '{bin_dir}'")
        print(f"  $env:PATH += ';{bin_dir}'")
    else:
        print(f"Extraction finished, checking {TARGET_DIR}:")
        for item in os.listdir(TARGET_DIR):
            print(f"  {item}")


if __name__ == "__main__":
    install_qe()
