import sys
import os
import glob
import torch
import torch.nn as nn
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch

sys.path.append(os.path.join(os.getcwd(), "scripts"))
sys.path.append(os.getcwd())

from train import _build_model
from crystal_gnn.data.preprocessing import build_node_features, rbf_encode_distance

KNOWN_TARGETS = {
    "mp-754388": -1.91428489,
    "mp-1178504": -1.98049591,
    "mp-1949090": -0.04146314,
    "mp-1039332": 0.02474444,
}

CIF_FILES = {
    "mp-754388": "comformer_check_structures/mp-754388.cif",
    "mp-1178504": "comformer_check_structures/mp-1178504.cif",
    "mp-1949090": "comformer_check_structures/mp-1949090.cif",
    "mp-1039332": "comformer_check_structures/mp-1039332.cif",
}

def load_cif_as_graph(cif_path, radius, max_neighbors=24):
    struct = Structure.from_file(cif_path)
    all_neighs = struct.get_all_neighbors(r=radius, include_index=True)
    
    edge_index = []
    distances = []
    
    for i, neighs in enumerate(all_neighs):
        sorted_neighs = sorted(neighs, key=lambda n: n.nn_distance)[:max_neighbors]
        for n in sorted_neighs:
            edge_index.append([i, n.index])
            distances.append(n.nn_distance)
            
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        distances = [0.0]
        
    x = build_node_features(struct)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = rbf_encode_distance(distances, radius=radius)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(struct))
    return data

def evaluate_checkpoint(ckpt_dir, cif_files, device='cuda'):
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt_path):
        return None
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    use_der = config["model"].get("use_der", False)
    radii = config["model"].get("radii", [4.0, 6.0, 8.0])
    
    preds = {}
    with torch.no_grad():
        for mp_id, path in cif_files.items():
            if len(radii) == 1:
                g1 = load_cif_as_graph(path, radius=radii[0]).to(device)
                b1 = Batch.from_data_list([g1])
                out = model(b1)
            else:
                g1 = load_cif_as_graph(path, radius=radii[0]).to(device)
                g2 = load_cif_as_graph(path, radius=radii[1]).to(device)
                g3 = load_cif_as_graph(path, radius=radii[2]).to(device)
                
                b1 = Batch.from_data_list([g1])
                b2 = Batch.from_data_list([g2])
                b3 = Batch.from_data_list([g3])
                out = model(b1, b2, b3)
                
            if use_der:
                mu = out[0].squeeze().item()
            else:
                mu = out.squeeze().item()
                
            preds[mp_id] = mu
            
    return preds

def main():
    print("--- INFERENCE CHECK ON COLLIDING CIF PAIRS ACROSS MODEL CHECKPOINTS ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    
    ckpt_dirs = {
        "A1 (Single-Scale r=4.0, MSE)": "checkpoints/paper_A1_soap_loco_formation_energy_per_atom",
        "A2 (Single-Scale r=4.0, MSE)": "checkpoints/paper_A2_soap_loco_formation_energy_per_atom",
        "A3 (Multi-Scale, Concatenation)": "checkpoints/paper_A3_soap_loco_formation_energy_per_atom",
        "A4 (Multi-Scale, Attention)": "checkpoints/paper_A4_soap_loco_formation_energy_per_atom",
        "A6 (Multi-Scale, DER T=1)": "checkpoints/paper_A6_soap_loco_formation_energy_per_atom",
        "A7 (Main Model Multi-Scale, DER)": "checkpoints/paper_A7_soap_loco_formation_energy_per_atom",
    }
    
    print("\nGround Truth Energies:")
    print("  BaSrI4 pair: mp-754388 (-1.914285) vs mp-1178504 (-1.980496) -> Real Gap = 0.066211 eV/atom")
    print("  Ca2Mg  pair: mp-1949090 (-0.041463) vs mp-1039332 (0.024744) -> Real Gap = 0.066208 eV/atom")
    
    print("\n" + "=" * 95)
    print(f"{'Model Checkpoint':<35} | {'BaSrI4 Pred Gap':<22} | {'Ca2Mg Pred Gap':<22}")
    print("-" * 95)
    
    for name, c_dir in ckpt_dirs.items():
        if not os.path.exists(c_dir):
            continue
        preds = evaluate_checkpoint(c_dir, CIF_FILES, device=device)
        if preds is None:
            continue
            
        gap_basri = abs(preds["mp-754388"] - preds["mp-1178504"])
        gap_ca2mg = abs(preds["mp-1949090"] - preds["mp-1039332"])
        
        print(f"{name:<35} | {gap_basri:<22.6f} | {gap_ca2mg:<22.6f}")
        print(f"  -> Predictions BaSrI4: mp-754388={preds['mp-754388']:.4f}, mp-1178504={preds['mp-1178504']:.4f}")
        print(f"  -> Predictions Ca2Mg:  mp-1949090={preds['mp-1949090']:.4f}, mp-1039332={preds['mp-1039332']:.4f}")
        print("-" * 95)

if __name__ == "__main__":
    main()
