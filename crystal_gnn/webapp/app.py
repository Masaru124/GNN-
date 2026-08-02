from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data

from crystal_gnn.data.preprocessing import build_node_features, rbf_encode_distance
from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DEFAULT_TARGET = "formation_energy_per_atom"

app = FastAPI(title="Crystal GNN 3D Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


class PredictRequest(BaseModel):
    structure_text: str = Field(..., min_length=10)
    structure_format: str = Field(default="auto", pattern="^(auto|cif|poscar)$")
    checkpoint: str | None = None
    mc_samples: int = Field(default=10, ge=1, le=100)


class NodePayload(BaseModel):
    id: int
    element: str
    frac: list[float]
    cart: list[float]


class EdgePayload(BaseModel):
    source: int
    target: int
    distance: float


class PredictResponse(BaseModel):
    checkpoint: str
    target: str
    prediction: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    total_uncertainty: float
    interval_90: list[float]
    crystal: dict[str, Any]
    graph: dict[str, Any]


def _discover_latest_checkpoint() -> Path:
    candidates = sorted(CHECKPOINTS_DIR.glob("*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No checkpoint found under checkpoints/*/best.pt")
    return candidates[0]


def _resolve_checkpoint_path(checkpoint: str | None) -> Path:
    if checkpoint:
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / checkpoint
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate
    return _discover_latest_checkpoint()


def _parse_structure(text: str, fmt: str) -> Structure:
    formats = [fmt] if fmt != "auto" else ["cif", "poscar"]
    errors: list[str] = []
    for one_fmt in formats:
        try:
            return Structure.from_str(text, fmt=one_fmt)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{one_fmt}: {exc}")
    raise ValueError("Could not parse structure. Tried formats -> " + " | ".join(errors))


def _build_model(config: dict[str, Any]) -> torch.nn.Module:
    mcfg = config["model"]
    radii = mcfg.get("radii", [4.0, 6.0, 8.0])
    if len(radii) == 1:
        return SingleScaleGNN(
            hidden_dim=mcfg["hidden_dim"],
            num_encoder_layers=mcfg["num_encoder_layers"],
            dropout_rate=mcfg["dropout_rate"],
            use_der=mcfg["use_der"],
        )
    return MultiScaleGNN(
        hidden_dim=mcfg["hidden_dim"],
        num_encoder_layers=mcfg["num_encoder_layers"],
        dropout_rate=mcfg["dropout_rate"],
        use_attention_fusion=mcfg.get("use_attention_fusion", True),
        use_der=mcfg.get("use_der", True),
        radii=radii,
    )


def _build_graph_for_radius(structure: Structure, radius: float, max_neighbors: int) -> tuple[Data, list[EdgePayload]]:
    x = build_node_features(structure)
    pos = torch.tensor(structure.frac_coords, dtype=torch.float32)

    src: list[int] = []
    dst: list[int] = []
    dists: list[float] = []
    edges: list[EdgePayload] = []

    for i, neighs in enumerate(structure.get_all_neighbors(r=radius, include_index=True, include_image=True)):
        if len(neighs) == 0:
            continue
        if len(neighs) > max_neighbors:
            chosen = np.random.choice(len(neighs), size=max_neighbors, replace=False)
            neighs = [neighs[int(j)] for j in chosen]
        for n in neighs:
            j = int(n.index)
            d = float(n.nn_distance)
            src.append(i)
            dst.append(j)
            dists.append(d)
            edges.append(EdgePayload(source=i, target=j, distance=d))

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = rbf_encode_distance(dists, radius=radius, n_rbf=50, width=0.5)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor([[0.0]], dtype=torch.float32),
        material_id="user-input",
    )
    return data, edges


def _structure_payload(structure: Structure) -> dict[str, Any]:
    lattice = structure.lattice.matrix.tolist()
    nodes: list[NodePayload] = []
    for i, site in enumerate(structure.sites):
        nodes.append(
            NodePayload(
                id=i,
                element=str(site.specie),
                frac=[float(v) for v in structure.frac_coords[i]],
                cart=[float(v) for v in site.coords],
            )
        )
    return {
        "lattice": [[float(v) for v in row] for row in lattice],
        "nodes": [n.model_dump() for n in nodes],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/api/checkpoint/default")
def default_checkpoint() -> dict[str, str]:
    ckpt = _discover_latest_checkpoint()
    return {"checkpoint": str(ckpt.relative_to(ROOT_DIR)).replace("\\", "/")}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        structure = _parse_structure(req.structure_text, req.structure_format)
        ckpt_path = _resolve_checkpoint_path(req.checkpoint)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        cfg = checkpoint["config"]
        mcfg = cfg["model"]
        dcfg = cfg.get("data", {})
        target = dcfg.get("target", DEFAULT_TARGET)
        radii = mcfg.get("radii", [4.0, 6.0, 8.0])
        max_neighbors = int(dcfg.get("max_neighbors", 64))

        model = _build_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        per_radius_graphs: list[Data] = []
        first_radius_edges: list[EdgePayload] = []
        for idx, r in enumerate(radii[:3]):
            g, edges = _build_graph_for_radius(structure, float(r), max_neighbors=max_neighbors)
            per_radius_graphs.append(g)
            if idx == 0:
                first_radius_edges = edges

        while len(per_radius_graphs) < 3:
            per_radius_graphs.append(per_radius_graphs[-1])

        b1 = Batch.from_data_list([per_radius_graphs[0]])
        b2 = Batch.from_data_list([per_radius_graphs[1]])
        b3 = Batch.from_data_list([per_radius_graphs[2]])

        with torch.no_grad():
            out = model.predict_with_uncertainty(b1, b2, b3, T=req.mc_samples)

        pred = float(out["prediction"].reshape(-1)[0].cpu().item())
        ale = float(out["aleatoric"].reshape(-1)[0].cpu().item())
        epi = float(out["epistemic"].reshape(-1)[0].cpu().item())
        total = max(ale + epi, 0.0)
        total_std = float(np.sqrt(total))
        z90 = 1.6448536269514722

        crystal_payload = _structure_payload(structure)
        graph_payload = {
            "radius": float(radii[0]),
            "nodes": [
                {
                    "id": n["id"],
                    "name": n["element"],
                    "x": n["cart"][0],
                    "y": n["cart"][1],
                    "z": n["cart"][2],
                }
                for n in crystal_payload["nodes"]
            ],
            "links": [e.model_dump() for e in first_radius_edges],
        }

        return PredictResponse(
            checkpoint=str(ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            target=target,
            prediction=pred,
            aleatoric_uncertainty=ale,
            epistemic_uncertainty=epi,
            total_uncertainty=total_std,
            interval_90=[pred - z90 * total_std, pred + z90 * total_std],
            crystal=crystal_payload,
            graph=graph_payload,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000, reload=False)
