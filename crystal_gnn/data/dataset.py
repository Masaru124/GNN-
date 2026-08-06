"""PyTorch dataset for multi-scale crystal graphs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .preprocessing import build_node_features, rbf_encode_distance

LOGGER = logging.getLogger(__name__)


class MultiScaleDataset(Dataset):
    """Dataset that returns one graph per radius for each crystal."""

    def __init__(
        self,
        structures: Sequence[Any],
        labels: Dict[str, Dict[str, float]],
        radii: List[float] | None = None,
        target: str = "formation_energy_per_atom",
        cache_dir: str = "./data/cache",
        max_neighbors: int = 64,
    ) -> None:
        self.radii = radii or [4.0, 6.0, 8.0]
        self.target = target
        self.max_neighbors = int(max_neighbors)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Store entries as (material_id, raw_structure_dict_or_Structure)
        self._entries: List[tuple[str, Any]] = []
        for idx, structure in enumerate(structures):
            material_id = self._resolve_material_id(idx, labels, structure)
            if material_id is None:
                continue
            if material_id not in labels or target not in labels[material_id]:
                continue
            self._entries.append((material_id, structure))

        self.labels = labels

    @staticmethod
    def _resolve_material_id(idx: int, labels: Dict[str, Dict[str, float]], structure: Any) -> str | None:
        sid = None
        # structure may be a dict (raw) or a pymatgen.Structure
        if isinstance(structure, dict):
            sid = structure.get("material_id") or (structure.get("properties") or {}).get("material_id")
        else:
            sid = getattr(structure, "material_id", None)
            if sid is None and hasattr(structure, "properties"):
                sid = structure.properties.get("material_id")
        if sid is not None and str(sid) in labels:
            return str(sid)
        fallback = f"mp-fake-{idx}"
        if fallback in labels:
            return fallback
        keys = list(labels.keys())
        if idx < len(keys):
            return keys[idx]
        return None

    # Neighbor checks are performed during graph building inside worker processes to avoid
    # creating pymatgen.Structure objects in the main process (which would bloat pickling)

    def __len__(self) -> int:
        return len(self._entries)

    def _cache_path(self, material_id: str, radius: float) -> Path:
        r = f"{radius:.2f}".replace(".", "p")
        return self.cache_dir / f"{material_id}_r{r}.pt"

    def _build_graph(self, material_id: str, structure: Structure, radius: float) -> Data:
        # If a raw dict was passed, convert to Structure here (inside worker)
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)

        x = build_node_features(structure)
        pos = torch.tensor(structure.frac_coords, dtype=torch.float32)
        y = torch.tensor([float(self.labels[material_id][self.target])], dtype=torch.float32)

        src: List[int] = []
        dst: List[int] = []
        dists: List[float] = []

        for i, neighs in enumerate(structure.get_all_neighbors(r=radius, include_index=True, include_image=True)):
            if len(neighs) == 0:
                continue
            if len(neighs) > self.max_neighbors:
                idxs = np.random.choice(len(neighs), size=self.max_neighbors, replace=False)
                neighs = [neighs[int(j)] for j in idxs]
                LOGGER.info("Subsampled neighbors for %s at r=%.2f (site=%d)", material_id, radius, i)
            for n in neighs:
                src.append(i)
                dst.append(int(n.index))
                dists.append(float(n.nn_distance))

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = rbf_encode_distance(dists, radius=radius, n_rbf=50, width=0.5)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            material_id=material_id,
        )

    def _load_or_build_graph(self, material_id: str, structure: Structure, radius: float) -> Data:
        cache_path = self._cache_path(material_id, radius)
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)
        data = self._build_graph(material_id, structure, radius)
        torch.save(data, cache_path)
        return data

    def __getitem__(self, idx: int) -> tuple[Data, Data, Data, torch.Tensor, str]:
        material_id, structure = self._entries[idx]
        # Convert to Structure inside the worker when building graphs
        graphs = [self._load_or_build_graph(material_id, structure, r) for r in self.radii]

        if len(graphs) == 1:
            graphs = [graphs[0], graphs[0], graphs[0]]
        elif len(graphs) == 2:
            graphs = [graphs[0], graphs[1], graphs[1]]
        elif len(graphs) > 3:
            graphs = graphs[:3]

        y = torch.tensor([float(self.labels[material_id][self.target])], dtype=torch.float32)
        return graphs[0], graphs[1], graphs[2], y, material_id


class MultiScaleCollate:
    """Collate function for batching multi-scale graph tuples."""

    def __call__(self, batch: Iterable[tuple[Data, Data, Data, torch.Tensor, str]]) -> tuple[Batch, Batch, Batch, torch.Tensor, list[str]]:
        r1, r2, r3, y, ids = zip(*batch)
        b1 = Batch.from_data_list(list(r1))
        b2 = Batch.from_data_list(list(r2))
        b3 = Batch.from_data_list(list(r3))
        y_batch = torch.cat(list(y), dim=0).reshape(-1, 1)
        return b1, b2, b3, y_batch, list(ids)
