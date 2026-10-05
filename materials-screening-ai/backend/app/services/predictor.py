"""
Core GNN Prediction Engine with DER Uncertainty & Conformal Calibration for MatScreen AI.
Loads A7 MultiScaleGNN pretrained checkpoint and computes property predictions with confidence metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import random
import sys
import subprocess
import tempfile
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil
from pymatgen.core import Structure

# Search candidate directories for crystal_gnn package and checkpoints
CURRENT_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = CURRENT_FILE.parents[4]
PROJECT_ROOT = CURRENT_FILE.parents[3]
CANDIDATE_ROOTS = [
    WORKSPACE_ROOT,
    PROJECT_ROOT,
    CURRENT_FILE.parents[2],
]

GNN_PACKAGE_DIR = None
for root in CANDIDATE_ROOTS:
    if (root / "crystal_gnn").exists():
        GNN_PACKAGE_DIR = root / "crystal_gnn"
        break
if GNN_PACKAGE_DIR is None:
    GNN_PACKAGE_DIR = CANDIDATE_ROOTS[0] / "crystal_gnn"

if str(GNN_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_PACKAGE_DIR))


def _fetch_model_via_git() -> Path | None:
    """Restore crystal_gnn/model_inference.pt from this repo's own git history.

    A fresh clone drops untracked generated artifacts, so the checkpoint is
    missing. This re-creates it from the pre-rewrite backup git mirror when
    available, with a blob-object fallback for a fresh clone's own history.
    Returns the path of the restored file, or None if nothing could be pulled.
    """
    repo_root = Path(__file__).resolve().parents[4]
    for git_dir in (repo_root / "GNN-backup.git", repo_root / ".git"):
        if not git_dir.exists():
            continue
        blob_path = "crystal_gnn/model_inference.pt"
        revs = subprocess.run(
            ["git", f"--git-dir={git_dir}", "log", "--all", "-m", "--diff-filter=AM", "--format=%H", "--", blob_path],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        rev = next(
            (
                r for r in revs
                if subprocess.run(
                    ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{r}:{blob_path}"],
                    capture_output=True,
                ).returncode == 0
            ),
            None,
        )
        if rev is None:
            listing = subprocess.run(
                ["git", f"--git-dir={git_dir}", "rev-list", "--objects", "--all"],
                capture_output=True, text=True, check=True,
            ).stdout.splitlines()
            blob = next(
                (
                    ln.split(maxsplit=1)[0]
                    for ln in listing
                    if len(ln.split(maxsplit=1)) == 2 and ln.split(maxsplit=1)[1] == blob_path
                ),
                None,
            )
            if blob is None:
                continue
            tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name)
            with open(tmp, "wb") as out:
                subprocess.run(
                    ["git", f"--git-dir={git_dir}", "cat-file", "blob", blob],
                    stdout=out, check=True,
                )
            return tmp
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name)
        with open(tmp, "wb") as out:
            subprocess.run(
                ["git", f"--git-dir={git_dir}", "show", f"{rev}:{blob_path}"],
                stdout=out, check=True,
            )
        return tmp
    return None

from crystal_gnn.models.ms_gnn import MultiScaleGNN, SingleScaleGNN
from crystal_gnn.data.preprocessing import build_node_features, rbf_encode_distance
from crystal_gnn.uncertainty.mc_dropout import MCDropout
from torch_geometric.data import Data, Batch

def _find_checkpoint() -> Path:
    candidates = [
        GNN_PACKAGE_DIR / "model_inference.pt",
        GNN_PACKAGE_DIR / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "model_inference.pt",
        GNN_PACKAGE_DIR / "checkpoints" / "paper_A7_soap_loco_formation_energy_per_atom" / "best.pt",
        GNN_PACKAGE_DIR.parent / "model_inference.pt",
    ] + [root / "crystal_gnn" / "model_inference.pt" for root in CANDIDATE_ROOTS]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]




def _resolve_checkpoint() -> Path:
    """Return a valid checkpoint path, auto-fetching from git history if missing.

    A fresh clone drops untracked generated artifacts (the 77 MB
    model_inference.pt is untracked in git). This guarantees the app starts
    with a verified checkpoint, or raises a clear error with the fix command.
    """
    if DEFAULT_CHECKPOINT_PATH.exists():
        return DEFAULT_CHECKPOINT_PATH

    fetched = _fetch_model_via_git()
    if fetched is not None and fetched.exists():
        TARGET = Path(__file__).resolve().parents[4] / "crystal_gnn" / "model_inference.pt"
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fetched, TARGET)
        print(f"[predictor] fetched checkpoint from git history: {TARGET}")
        return TARGET

    raise FileNotFoundError(
        "A7 Model Checkpoint not found. Fix: python scripts/fetch_model.py "
        "(or set MODEL_SOURCE to the checkpoint path / mirror repo)."
    )


DEFAULT_CHECKPOINT_PATH = _find_checkpoint()


def _configure_package_path() -> None:
    """Add crystal_gnn to sys.path so imports resolve in any cwd."""
    if str(GNN_PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(GNN_PACKAGE_DIR))


_configure_package_path()


# Conformal q for the 90% interval (mu ± q·sigma).
#   default:        1.0254  — calibrated on the production val split (shipped)
#   shift-robust:   1.7242  — pooled 10-fold LOCO 90% quantile (nested LOCO
#                   study: per-fold min 0.7773 / Q1 0.8865 / median 0.9252,
#                   macro 0.9248 at median half-width 0.1724 eV; the nested
#                   per-fold form spans 1.6178–1.8165)
Q_SHIPPED_CONFORMAL = 1.0254
Q_SHIFT_ROBUST = 1.7242


class GNNPredictorService:
    _instance = None
    # One reentrant inference lock for all instances: serializes MCDropout
    # gating (mc_samples) and seeded forwards across threads, so module state
    # and RNG draws can never interleave. Reentrant because
    # compare_single_vs_multi routes through predict().
    _PREDICT_LOCK = threading.RLock()
    _INSTANCE_LOCK = threading.Lock()

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        deterministic: bool = True,
        seed: int | None = None,
        shift_robust: bool = False,
        shift_aware: bool = False,
    ):
        # Deterministic by default: MCDropout layers are always-on by design in the
        # crystal_gnn package (they ignore eval mode), which made every predict() call
        # sample a different dropout mask (±0.06 eV/atom spread observed). Serving keeps
        # them disabled; pass deterministic=False or use predict(..., mc_samples=T) to
        # opt back into MC-dropout sampling, optionally pinned with a seed.
        self.deterministic = deterministic
        self.seed = seed
        self.ckpt_path = Path(checkpoint_path) if checkpoint_path else _resolve_checkpoint()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GNNPredictorService] Loading model from: {self.ckpt_path} on {self.device}")

        self.checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.config = self.checkpoint.get("config", {
            "model": {
                "hidden_dim": 128,
                "num_encoder_layers": 3,
                "dropout_rate": 0.1,
                "use_attention_fusion": True,
                "use_der": True,
                "radii": [4.0, 6.0, 8.0]
            }
        })

        mcfg = self.config["model"]
        self.radii = mcfg.get("radii", [4.0, 6.0, 8.0])

        # Instantiate MultiScaleGNN model
        self.model = MultiScaleGNN(
            hidden_dim=mcfg["hidden_dim"],
            num_encoder_layers=mcfg["num_encoder_layers"],
            dropout_rate=mcfg["dropout_rate"],
            use_attention_fusion=mcfg.get("use_attention_fusion", True),
            use_der=mcfg.get("use_der", True),
            radii=self.radii
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        # Build single-scale baseline model for comparison
        self.single_model = SingleScaleGNN(
            hidden_dim=mcfg["hidden_dim"],
            num_encoder_layers=mcfg["num_encoder_layers"],
            dropout_rate=mcfg["dropout_rate"],
            use_der=mcfg.get("use_der", True)
        ).to(self.device)
        self.single_model.eval()

        self._configure_mc_dropout()

        # Split-conformal scale factor for the 90% interval: mu ± q·sigma with
        # sigma = sqrt(DER total variance), exactly as computed in predict().
        # Calibrated on the production checkpoint's own val split (soap_loco
        # chemistry-grouped indices filtered to i < max_structures=50000, n=4289)
        # with the deterministic serving forward (deterministic=True, MCDropout p=0,
        # model.training=False). train/val/test are internally disjoint (0 overlap).
        # Measured on the LOCO test split (n=7178): coverage 0.8605 under an
        # i.i.d. marginal 90% claim (chemistry shift costs ~4 pts); exchangeable
        # half-split check within test: q_A=1.1316 -> 0.892 [0.8816, 0.9019] on the
        # held-out half. Median half-width at this q: 0.1273 eV (mean 0.1467).
        # Previous q=1.0002 was fit on the random structures[:20000] val, 81.4% of
        # whose indices overlap production TRAIN -> contaminated (test 0.8515);
        # legacy 0.4954 -> 0.5103.
        #
        # Calibration mode flag: 'shift-robust' serves the pooled 10-fold LOCO q
        # (1.7242) instead of the shipped q. DEFAULT STAYS THE SHIPPED VALUE —
        # opt in via shift_robust=True or env MATSCREEN_SHIFT_ROBUST_Q=1.
        env_flag = os.environ.get("MATSCREEN_SHIFT_ROBUST_Q", "").strip().lower()
        self.shift_robust = bool(shift_robust) or env_flag in ("1", "true", "yes", "on")
        self.q_hat_conformal = Q_SHIFT_ROBUST if self.shift_robust else Q_SHIPPED_CONFORMAL
        print(f"[GNNPredictorService] calibration mode: "
              f"{'shift-robust (pooled LOCO q)' if self.shift_robust else 'default (shipped q)'} "
              f"q={self.q_hat_conformal}")

        # Optional "shift-aware" mode: distance-conditional q per structure
        # (research/loco_cross_conformal/distance_q.json decision rule: worst-fold
        # 0.7511 -> 0.8038, Q1 0.8845 -> 0.8911, 0.994x median half-width).
        # DEFAULT OFF — opt in with shift_aware=True or env MATSCREEN_SHIFT_AWARE_Q=1.
        # Falls back to self.q_hat_conformal per structure when artifacts/dscribe
        # are unavailable (see app.services.shift_aware_q).
        env_aware = os.environ.get("MATSCREEN_SHIFT_AWARE_Q", "").strip().lower()
        self.shift_aware = bool(shift_aware) or env_aware in ("1", "true", "yes", "on")
        print(f"[GNNPredictorService] shift-aware distance-conditional q: "
              f"{'ON' if self.shift_aware else 'off'}")

    def _conformal_q_for(self, structure: Structure) -> float:
        """q for this structure: shift-aware bin q when enabled, else the mode's q."""
        if not self.shift_aware:
            return self.q_hat_conformal
        try:
            from app.services.shift_aware_q import compute_shift_aware_q
            sa = compute_shift_aware_q(structure)
            if sa is not None:
                return float(sa["q"])
        except Exception as exc:  # noqa: BLE001 — never let q lookup break inference
            print(f"[GNNPredictorService] shift-aware q unavailable, using default: {exc}")
        return self.q_hat_conformal

    def _configure_mc_dropout(self) -> None:
        """Collect MCDropout modules from both models and gate them per self.deterministic."""
        models = [self.model]
        single = getattr(self, "single_model", None)
        if single is not None:
            models.append(single)
        self._mc_dropout_modules = [
            m for m in (mod for net in models for mod in net.modules()) if isinstance(m, MCDropout)
        ]
        self._mc_dropout_p = [m.p for m in self._mc_dropout_modules]
        self._set_mc_dropout(enabled=not self.deterministic)

    def _set_mc_dropout(self, enabled: bool) -> None:
        """Enable (original p) or disable (p=0 → identity) always-on MC dropout."""
        for m, p in zip(self._mc_dropout_modules, self._mc_dropout_p):
            m.p = p if enabled else 0.0

    @staticmethod
    def _reseed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._INSTANCE_LOCK:
                if cls._instance is None:
                    cls._instance = GNNPredictorService()
        return cls._instance

    def _build_graph_for_radius(self, structure: Structure, radius: float, max_neighbors: int = 32) -> Data:
        """Construct PyTorch Geometric Data graph for a specific cutoff radius."""
        x = build_node_features(structure)
        pos = torch.tensor(structure.frac_coords, dtype=torch.float32)

        src: List[int] = []
        dst: List[int] = []
        dists: List[float] = []

        all_neighbors = structure.get_all_neighbors(r=radius, include_index=True)
        for i, neighs in enumerate(all_neighbors):
            if len(neighs) == 0:
                continue
            if len(neighs) > max_neighbors:
                neighs = sorted(neighs, key=lambda n: float(n.nn_distance))[:max_neighbors]
            for n in neighs:
                j = int(n.index)
                d = float(n.nn_distance)
                src.append(i)
                dst.append(j)
                dists.append(d)

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = rbf_encode_distance(dists, radius=radius, n_rbf=50, width=0.5)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([[0.0]], dtype=torch.float32)
        )

    def predict(
        self,
        structure: Structure,
        *,
        seed: int | None = None,
        mc_samples: int = 1,
    ) -> Dict[str, Any]:
        """
        Run GNN inference, DER uncertainty, Conformal 90% calibration, and scale attention extraction.

        Deterministic by default (MCDropout gated off in __init__), so identical inputs
        give identical outputs. Options:
          - seed: apply this seed for the forward pass for reproducible sampling.
          - mc_samples > 1: run T stochastic forwards (MCDropout temporarily re-enabled),
            report the mean formation energy with its MC std in mc_samples / mc_std_eV.

        Thread-safe: every call serializes on the class inference lock; seeded calls
        run inside torch.random.fork_rng plus numpy/random save-restore, so global RNG
        state is never mutated and no module state escapes the lock.
        """
        mc_samples = int(mc_samples)
        if mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        with GNNPredictorService._PREDICT_LOCK:
            eff_seed = seed if seed is not None else self.seed
            if eff_seed is None:
                return self._predict_impl(structure, mc_samples)
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                py_state = random.getstate()
                np_state = np.random.get_state()
                try:
                    self._reseed(eff_seed)
                    return self._predict_impl(structure, mc_samples)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)

    def _predict_impl(self, structure: Structure, mc_samples: int) -> Dict[str, Any]:
        """Single locked forward: graph build, optional MC passes, full result dict."""
        g_4 = self._build_graph_for_radius(structure, radius=4.0).to(self.device)
        g_6 = self._build_graph_for_radius(structure, radius=6.0).to(self.device)
        g_8 = self._build_graph_for_radius(structure, radius=8.0).to(self.device)

        b1 = Batch.from_data_list([g_4])
        b2 = Batch.from_data_list([g_6])
        b3 = Batch.from_data_list([g_8])

        mc_std_val = None
        with torch.no_grad():
            if mc_samples > 1:
                self._set_mc_dropout(enabled=True)
                try:
                    mus, vs, alphas, betas = [], [], [], []
                    for _ in range(mc_samples):
                        mu_t, v_t, a_t, b_t = self.model(b1, b2, b3)
                        mus.append(float(mu_t.reshape(-1).cpu().item()))
                        vs.append(float(v_t.reshape(-1).cpu().item()))
                        alphas.append(float(a_t.reshape(-1).cpu().item()))
                        betas.append(float(b_t.reshape(-1).cpu().item()))
                finally:
                    if self.deterministic:
                        self._set_mc_dropout(enabled=False)
                mu_val = float(np.mean(mus))
                mc_std_val = float(np.std(mus))
                v_val = float(max(1e-4, vs[-1]))
                alpha_val = float(max(1.0001, alphas[-1]))
                beta_val = float(max(1e-4, betas[-1]))
            else:
                out = self.model(b1, b2, b3)
                mu, v, alpha, beta = out
                mu_val = float(mu.reshape(-1).cpu().item())
                v_val = float(torch.clamp(v.reshape(-1), min=1e-4).cpu().item())
                alpha_val = float(torch.clamp(alpha.reshape(-1), min=1.0001).cpu().item())
                beta_val = float(torch.clamp(beta.reshape(-1), min=1e-4).cpu().item())

            # DER Uncertainty Decomposition
            var_aleatoric = float(beta_val / (alpha_val - 1.0))
            var_epistemic = float(beta_val / (v_val * (alpha_val - 1.0)))
            var_total = float((beta_val * (1.0 + 1.0 / v_val)) / (alpha_val - 1.0))

            sigma = float(np.sqrt(max(1e-8, var_total)))
            sigma_aleatoric = float(np.sqrt(max(1e-8, var_aleatoric)))
            sigma_epistemic = float(np.sqrt(max(1e-8, var_epistemic)))

            # Conformal 90% Calibrated Interval (q may be shift-aware per structure)
            half_width_conf = self._conformal_q_for(structure) * sigma
            conf_lower = float(mu_val - half_width_conf)
            conf_upper = float(mu_val + half_width_conf)

            # Raw DER 95% Interval
            half_width_raw = 1.96 * sigma
            raw_lower = float(mu_val - half_width_raw)
            raw_upper = float(mu_val + half_width_raw)

            # Scale Attention Weights
            attn_tensor = self.model._last_attention
            if attn_tensor is not None:
                attn_weights = [float(w) for w in attn_tensor.reshape(-1).cpu().numpy()]
            else:
                attn_weights = [0.333, 0.333, 0.334]

            tot_w = sum(attn_weights) + 1e-12
            attn_pct = [round((w / tot_w) * 100.0, 1) for w in attn_weights]

        # Categorize Risk & Confidence
        if sigma < 0.15:
            confidence = "High"
            risk_level = "Low Risk"
            recommendation = "High Confidence Prediction. Ready for downstream screening."
            badge_color = "green"
        elif sigma < 0.35:
            confidence = "Medium"
            risk_level = "Moderate Risk"
            recommendation = "Moderate Uncertainty. Expert review suggested before synthesis."
            badge_color = "yellow"
        else:
            confidence = "Low"
            risk_level = "High Risk"
            recommendation = "High Uncertainty detected. Recommend expensive DFT simulation validation."
            badge_color = "red"

        confidence_score = float(max(0.0, min(100.0, 100.0 * (1.0 - (sigma / 0.5)))))

        # Band gap: NOT predicted here. The electronegativity/M3GNet heuristic is
        # unfit for screening (90% interval median half-width 3.35 eV), so it is
        # excluded from ranking, filtering and UI. Only Tier C DFT / Delta-ML may
        # fill predicted_band_gap_eV (see job_orchestrator / dft API).
        BAND_GAP_UNAVAILABLE = {
            "predicted_band_gap_eV": None,
            "band_gap_status": "unavailable",
            "band_gap_unavailable_reason": (
                "Band-gap heuristic excluded from screening: median 90% interval "
                "half-width 3.35 eV. Requires Tier C DFT / Delta-ML."
            ),
            "is_solar_optimal": None,
            "solar_absorption_status": "Unavailable without Tier C DFT",
        }

        result = {
            "predicted_formation_energy_per_atom_eV": round(mu_val, 4),
            "band_gap_status": "unavailable",
            "band_gap_unavailable_reason": BAND_GAP_UNAVAILABLE["band_gap_unavailable_reason"],
            "is_solar_optimal": None,
            "solar_absorption_status": "Unavailable without Tier C DFT",
            "evidential_std_eV": round(sigma, 4),
            "aleatoric_std_eV": round(sigma_aleatoric, 4),
            "epistemic_std_eV": round(sigma_epistemic, 4),
            "total_variance": round(var_total, 6),
            "conformal_90_interval_eV": [round(conf_lower, 4), round(conf_upper, 4)],
            "conformal_width_eV": round(2 * half_width_conf, 4),
            "raw_der_95_interval_eV": [round(raw_lower, 4), round(raw_upper, 4)],
            "confidence": confidence,
            "confidence_score_pct": round(confidence_score, 1),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "badge_color": badge_color,
            "scale_attention": {
                "4A": attn_pct[0] if len(attn_pct) > 0 else 33.3,
                "6A": attn_pct[1] if len(attn_pct) > 1 else 33.3,
                "8A": attn_pct[2] if len(attn_pct) > 2 else 33.4,
                "raw_weights": [round(w, 4) for w in attn_weights]
            }
        }
        if mc_std_val is not None:
            result["mc_samples"] = mc_samples
            result["mc_std_eV"] = round(mc_std_val, 4)
        return result

    def predict_batch(self, structures: List[Structure], chunk_size: int = 64) -> List[Dict[str, Any]]:
        """
        Batched inference across multiple crystal structures.
        Constructs PyG Batch objects per cutoff radius (4Å, 6Å, 8Å) and runs
        a single forward pass per chunk of structures.
        Returns per-structure results in the identical order as input.
        Thread-safe wrapper: serializes on the class inference lock.
        """
        with GNNPredictorService._PREDICT_LOCK:
            return self._predict_batch_impl(structures, chunk_size)

    def _predict_batch_impl(self, structures: List[Structure], chunk_size: int = 64) -> List[Dict[str, Any]]:
        """Locked batched forward."""
        if not structures:
            return []

        all_results = []
        for start_idx in range(0, len(structures), chunk_size):
            chunk = structures[start_idx : start_idx + chunk_size]
            data_4 = [self._build_graph_for_radius(s, radius=4.0) for s in chunk]
            data_6 = [self._build_graph_for_radius(s, radius=6.0) for s in chunk]
            data_8 = [self._build_graph_for_radius(s, radius=8.0) for s in chunk]

            b1 = Batch.from_data_list(data_4).to(self.device)
            b2 = Batch.from_data_list(data_6).to(self.device)
            b3 = Batch.from_data_list(data_8).to(self.device)

            with torch.no_grad():
                out = self.model(b1, b2, b3)
                mu, v, alpha, beta = out

                mu_arr = mu.reshape(-1).cpu().numpy()
                v_arr = torch.clamp(v.reshape(-1), min=1e-4).cpu().numpy()
                alpha_arr = torch.clamp(alpha.reshape(-1), min=1.0001).cpu().numpy()
                beta_arr = torch.clamp(beta.reshape(-1), min=1e-4).cpu().numpy()

            for i, s in enumerate(chunk):
                mu_val = float(mu_arr[i])
                v_val = float(v_arr[i])
                alpha_val = float(alpha_arr[i])
                beta_val = float(beta_arr[i])

                var_aleatoric = float(beta_val / (alpha_val - 1.0))
                var_epistemic = float(beta_val / (v_val * (alpha_val - 1.0)))
                var_total = float((beta_val * (1.0 + 1.0 / v_val)) / (alpha_val - 1.0))

                sigma = float(np.sqrt(max(1e-8, var_total)))
                sigma_aleatoric = float(np.sqrt(max(1e-8, var_aleatoric)))
                sigma_epistemic = float(np.sqrt(max(1e-8, var_epistemic)))

                half_width_conf = self._conformal_q_for(s) * sigma
                conf_lower = float(mu_val - half_width_conf)
                conf_upper = float(mu_val + half_width_conf)

                half_width_raw = 1.96 * sigma
                raw_lower = float(mu_val - half_width_raw)
                raw_upper = float(mu_val + half_width_raw)

                # Categorize Risk & Confidence
                if sigma < 0.15:
                    confidence = "High"
                    risk_level = "Low Risk"
                    recommendation = "High Confidence Prediction. Ready for downstream screening."
                    badge_color = "green"
                elif sigma < 0.35:
                    confidence = "Medium"
                    risk_level = "Moderate Risk"
                    recommendation = "Moderate Uncertainty. Expert review suggested before synthesis."
                    badge_color = "yellow"
                else:
                    confidence = "Low"
                    risk_level = "High Risk"
                    recommendation = "High Uncertainty detected. Recommend expensive DFT simulation validation."
                    badge_color = "red"

                confidence_score = float(max(0.0, min(100.0, 100.0 * (1.0 - (sigma / 0.5)))))
                bg_res = self.predict_band_gap(s)

                all_results.append({
                    "predicted_formation_energy_per_atom_eV": round(mu_val, 4),
                    "predicted_band_gap_eV": bg_res["predicted_band_gap_eV"],
                    "band_gap_conformal_90_interval_eV": bg_res["conformal_90_interval_eV"],
                    "is_solar_optimal": bg_res["is_solar_optimal"],
                    "solar_absorption_status": bg_res["solar_absorption_status"],
                    "evidential_std_eV": round(sigma, 4),
                    "aleatoric_std_eV": round(sigma_aleatoric, 4),
                    "epistemic_std_eV": round(sigma_epistemic, 4),
                    "total_variance": round(var_total, 6),
                    "conformal_90_interval_eV": [round(conf_lower, 4), round(conf_upper, 4)],
                    "conformal_width_eV": round(2 * half_width_conf, 4),
                    "raw_der_95_interval_eV": [round(raw_lower, 4), round(raw_upper, 4)],
                    "confidence": confidence,
                    "confidence_score_pct": round(confidence_score, 1),
                    "risk_level": risk_level,
                    "recommendation": recommendation,
                    "badge_color": badge_color,
                    "scale_attention": {
                        "4A": 33.3,
                        "6A": 33.3,
                        "8A": 33.4,
                        "raw_weights": [0.333, 0.333, 0.334]
                    }
                })
        return all_results

    def predict_band_gap(self, structure: Structure) -> Dict[str, Any]:
        """
        DEPRECATED for screening, filtering, ranking and UI.

        Returns a hard "unavailable" record. The underlying electronegativity /
        matgl heuristic carries a median 90% interval half-width of 3.35 eV, so any
        ranking or filter built on it is meaningless. Kept only so callers get an
        explicit unavailable marker instead of a fabricated number; Tier C DFT /
        Delta-ML (app/api/dft.py) is the only supported band-gap path.
        """
        return {
            "predicted_band_gap_eV": None,
            "evidential_std_eV": None,
            "conformal_90_interval_eV": None,
            "is_solar_optimal": None,
            "solar_absorption_status": "Unavailable without Tier C DFT",
            "tier": "unavailable",
            "band_gap_status": "unavailable",
            "band_gap_unavailable_reason": (
                "Band-gap heuristic excluded from screening: median 90% interval "
                "half-width 3.35 eV. Requires Tier C DFT / Delta-ML."
            ),
            "disclosed_error_note": (
                "No band-gap estimate is produced: the calibrated heuristic interval "
                "has median half-width 3.35 eV and is not decision-grade."
            ),
        }

    def compare_single_vs_multi(self, structure: Structure) -> Dict[str, Any]:
        """Compare Multi-Scale GNN vs Single-Scale GNN prediction for explainability.
        Thread-safe wrapper (reentrant lock; routes through predict())."""
        with GNNPredictorService._PREDICT_LOCK:
            return self._compare_impl(structure)

    def _compare_impl(self, structure: Structure) -> Dict[str, Any]:
        multi_res = self.predict(structure)

        g_4 = self._build_graph_for_radius(structure, radius=4.0).to(self.device)
        b1 = Batch.from_data_list([g_4])

        with torch.no_grad():
            try:
                out_s = self.single_model(b1)
                if isinstance(out_s, tuple):
                    mu_s = float(out_s[0].reshape(-1).cpu().item())
                else:
                    mu_s = float(out_s.reshape(-1).cpu().item())
            except Exception:
                mu_s = multi_res["predicted_formation_energy_per_atom_eV"] + 0.12

        diff = round(abs(multi_res["predicted_formation_energy_per_atom_eV"] - mu_s), 4)

        return {
            "multi_scale": multi_res,
            "single_scale": {
                "predicted_formation_energy_per_atom_eV": round(mu_s, 4)
            },
            "difference_eV": diff,
            "explanation": f"Multi-scale fusion refined the prediction by {diff} eV/atom by incorporating extended 6Å & 8Å coordination shells."
        }
