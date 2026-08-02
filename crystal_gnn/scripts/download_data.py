"""Download Materials Project dataset to local cache."""

from __future__ import annotations

import argparse
import os

from crystal_gnn.data.download import download_mp_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MP data for Crystal GNN")
    parser.add_argument("--api_key", type=str, default=os.getenv("MP_API_KEY", ""))
    parser.add_argument("--save_path", type=str, default="data/raw")
    parser.add_argument("--max_structures", type=int, default=None)
    args = parser.parse_args()

    stats = download_mp_dataset(api_key=args.api_key, save_path=args.save_path, max_structures=args.max_structures)
    print(stats)


if __name__ == "__main__":
    main()
