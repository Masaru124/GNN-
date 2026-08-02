"""Build and persist all split strategies."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import time
from pathlib import Path

from pymatgen.core import Structure

from crystal_gnn.data.splits import composition_split, crystal_system_split, random_split, save_splits, soap_loco_split


LOGGER = logging.getLogger(__name__)


def _load_dataset(data_dir: str):
    structures_path = Path(data_dir) / "mp_structures.json.gz"
    labels_path = Path(data_dir) / "mp_labels.csv"

    with gzip.open(structures_path, "rt", encoding="utf-8") as f_in:
        structures_raw = json.load(f_in)
    structures = [Structure.from_dict(d["structure"]) for d in structures_raw]

    labels = {}
    with labels_path.open("r", encoding="utf-8") as f_in:
        for row in csv.DictReader(f_in):
            labels[row["material_id"]] = {
                "formation_energy_per_atom": float(row["formation_energy_per_atom"]),
                "band_gap": float(row["band_gap"]),
                "crystal_system": row.get("crystal_system", "triclinic"),
                "nelements": int(row.get("nelements", 0)),
            }
    return structures, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random and OOD splits.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soap_n_jobs", type=int, default=1, help="Parallel jobs per SOAP call (higher may be faster on strong CPUs).")
    parser.add_argument("--log_every", type=int, default=1000, help="Progress log interval during SOAP descriptor generation.")
    parser.add_argument("--soap_r_cut", type=float, default=6.0)
    parser.add_argument("--soap_n_max", type=int, default=9)
    parser.add_argument("--soap_l_max", type=int, default=9)
    parser.add_argument("--soap_sigma", type=float, default=0.5)
    parser.add_argument("--soap_average", type=str, default="inner", choices=["off", "inner", "outer"])
    parser.add_argument("--soap_max_species", type=int, default=None, help="Cap SOAP species to top-K frequent elements for speed/memory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    t0 = time.time()

    LOGGER.info("Loading dataset from %s", args.data_dir)
    structures, labels = _load_dataset(args.data_dir)
    LOGGER.info("Loaded %d structures and %d labels", len(structures), len(labels))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building random split")
    save_splits(random_split(structures, seed=args.seed), str(out_dir / "random.json"))

    LOGGER.info("Building SOAP-LOCO split")
    save_splits(
        {
            "splits": soap_loco_split(
                structures,
                labels,
                n_clusters=10,
                seed=args.seed,
                soap_n_jobs=args.soap_n_jobs,
                log_every=args.log_every,
                soap_r_cut=args.soap_r_cut,
                soap_n_max=args.soap_n_max,
                soap_l_max=args.soap_l_max,
                soap_sigma=args.soap_sigma,
                soap_average=args.soap_average,
                soap_max_species=args.soap_max_species,
            )
        },
        str(out_dir / "soap_loco.json"),
    )

    LOGGER.info("Building crystal-system split")
    save_splits(crystal_system_split(structures, labels), str(out_dir / "crystal_system.json"))

    LOGGER.info("Building composition split")
    save_splits(composition_split(structures, labels, seed=args.seed), str(out_dir / "composition.json"))
    LOGGER.info("Saved splits to %s in %.1fs", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
