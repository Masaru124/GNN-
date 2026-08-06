"""Materials Project dataset download and filtering."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mp_api.client import MPRester
except ImportError:
    MPRester = None
from pymatgen.core import Structure

LOGGER = logging.getLogger(__name__)


FIELDS = [
    "material_id",
    "structure",
    "formation_energy_per_atom",
    "band_gap",
    "energy_above_hull",
    "symmetry",
    "formula_pretty",
    "nelements",
    "nsites",
]


def _parse_crystal_system(doc: Any) -> str:
    cs = getattr(doc, "crystal_system", None)
    if cs is None and getattr(doc, "symmetry", None) is not None:
        cs = getattr(doc.symmetry, "crystal_system", None)
    if cs is None and getattr(doc, "spacegroup", None) is not None:
        cs = getattr(doc.spacegroup, "crystal_system", None)
    return str(cs) if cs is not None else "triclinic"


def _parse_spacegroup_number(doc: Any) -> int:
    sg = getattr(doc, "spacegroup", None)
    num = None
    if sg is not None:
        num = getattr(sg, "number", None)
    if num is None and getattr(doc, "symmetry", None) is not None:
        num = getattr(doc.symmetry, "number", None)
    return int(num) if num is not None else -1


def _load_existing(struct_file: Path, labels_file: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not struct_file.exists() or not labels_file.exists():
        return [], []
    with gzip.open(struct_file, "rt", encoding="utf-8") as f_in:
        structures = json.load(f_in)
    labels: list[dict[str, Any]] = []
    with labels_file.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            labels.append(row)
    return structures, labels


def download_mp_dataset(api_key: str, save_path: str, max_structures: Optional[int] = None) -> dict[str, int]:
    """Download and filter MP structures, writing gzipped structures and CSV labels."""
    if not api_key:
        raise ValueError("Missing API key. Set MP_API_KEY or pass --api_key explicitly.")

    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    struct_file = out_dir / "mp_structures.json.gz"
    labels_file = out_dir / "mp_labels.csv"

    existing_structs, existing_labels = _load_existing(struct_file, labels_file)
    if existing_structs and existing_labels:
        LOGGER.info("Existing dataset found. Resuming skipped: %s", out_dir)
        return {
            "downloaded": len(existing_structs),
            "kept": len(existing_labels),
            "filtered_e_hull": 0,
            "filtered_missing_fe": 0,
            "filtered_missing_bg": 0,
            "filtered_nsites": 0,
        }

    attempts = 0
    docs: List[Any] = []
    while attempts < 3:
        try:
            with MPRester(api_key=api_key) as mpr:
                docs = list(
                    mpr.summary.search(
                        fields=FIELDS,
                        num_chunks=None,
                        chunk_size=2000,
                    )
                )
            break
        except Exception as exc:  # noqa: BLE001
            attempts += 1
            if attempts >= 3:
                raise RuntimeError(f"MP API download failed after {attempts} attempts: {exc}") from exc
            sleep_s = 2**attempts
            LOGGER.warning("MP API timeout/failure (%s). Retrying in %ss...", exc, sleep_s)
            time.sleep(sleep_s)

    if max_structures is not None:
        docs = docs[: int(max_structures)]

    downloaded = len(docs)
    filtered_e_hull = 0
    filtered_missing_fe = 0
    filtered_missing_bg = 0
    filtered_nsites = 0

    dedup: Dict[str, dict[str, Any]] = {}
    labels_rows: List[dict[str, Any]] = []

    for doc in docs:
        e_hull = getattr(doc, "energy_above_hull", None)
        if e_hull is None or float(e_hull) >= 0.1:
            filtered_e_hull += 1
            continue

        fe = getattr(doc, "formation_energy_per_atom", None)
        if fe is None:
            filtered_missing_fe += 1
            continue

        bg = getattr(doc, "band_gap", None)
        if bg is None:
            filtered_missing_bg += 1
            continue

        nsites = getattr(doc, "nsites", None)
        if nsites is None or int(nsites) > 50:
            filtered_nsites += 1
            continue

        material_id = str(getattr(doc, "material_id"))
        if material_id in dedup:
            LOGGER.warning("Duplicate material_id encountered and deduplicated: %s", material_id)
            continue

        structure = getattr(doc, "structure", None)
        if structure is None:
            continue

        if not isinstance(structure, Structure):
            structure = Structure.from_dict(structure)

        cs = _parse_crystal_system(doc)
        sg_num = _parse_spacegroup_number(doc)
        formula_pretty = str(getattr(doc, "formula_pretty", ""))
        nelements = int(getattr(doc, "nelements", 0) or 0)

        dedup[material_id] = {
            "material_id": material_id,
            "structure": structure.as_dict(),
        }
        labels_rows.append(
            {
                "material_id": material_id,
                "formation_energy_per_atom": float(fe),
                "band_gap": float(bg),
                "energy_above_hull": float(e_hull),
                "crystal_system": cs,
                "spacegroup_number": sg_num,
                "formula_pretty": formula_pretty,
                "nelements": nelements,
                "nsites": int(nsites),
            }
        )

    kept = len(labels_rows)
    if kept == 0:
        raise ValueError(
            "Zero structures after filtering. "
            f"Stats: downloaded={downloaded}, filtered_e_hull={filtered_e_hull}, "
            f"filtered_missing_fe={filtered_missing_fe}, filtered_missing_bg={filtered_missing_bg}, "
            f"filtered_nsites={filtered_nsites}."
        )

    with gzip.open(struct_file, "wt", encoding="utf-8") as f_out:
        json.dump(list(dedup.values()), f_out)

    with labels_file.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "material_id",
                "formation_energy_per_atom",
                "band_gap",
                "energy_above_hull",
                "crystal_system",
                "spacegroup_number",
                "formula_pretty",
                "nelements",
                "nsites",
            ],
        )
        writer.writeheader()
        writer.writerows(labels_rows)

    LOGGER.info(
        "Downloaded %d structures, kept %d after filtering. Filtered: e_hull=%d, missing_fe=%d, missing_bg=%d, nsites=%d",
        downloaded,
        kept,
        filtered_e_hull,
        filtered_missing_fe,
        filtered_missing_bg,
        filtered_nsites,
    )

    return {
        "downloaded": downloaded,
        "kept": kept,
        "filtered_e_hull": filtered_e_hull,
        "filtered_missing_fe": filtered_missing_fe,
        "filtered_missing_bg": filtered_missing_bg,
        "filtered_nsites": filtered_nsites,
    }
