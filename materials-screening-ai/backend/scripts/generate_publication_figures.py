# -*- coding: utf-8 -*-
"""
Publication Figure Generator for Delta-ML Validation Matrix (2x2 Grid).
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style configuration adhering to scientific-visualization / academic-plotting guidelines
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette (colorblind-safe)
palette = {
    "halide_perovskite": "#0072B2",        # Blue
    "transition_metal_perovskite": "#E69F00", # Orange
    "alkaline_earth_oxide": "#009E73",      # Green
    "alkali_halide": "#D55E00",             # Vermillion
    "fit_line": "#333333",
    "leverage_thresh": "#CC79A7",
    "accent": "#56B4E9",
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.delta_ml_corrector import CALIBRATION_RECORDS, get_delta_ml_corrector


def create_publication_figure() -> plt.Figure:
    """Create and return the authoritative 4-panel publication figure."""
    corrector = get_delta_ml_corrector()

    pbe_gaps = []
    target_gaps = []
    pred_gaps = []
    families = []
    formulas = []
    features_list = []

    for r in CALIBRATION_RECORDS:
        pbe, target, chi, r_rat, z_avg, eps, formula, src, mp_id, fam = r
        pbe_gaps.append(pbe)
        target_gaps.append(target)
        families.append(fam)
        formulas.append(formula)
        feat = corrector._make_features(pbe, chi, r_rat, z_avg, eps)
        features_list.append(feat)
        pred_delta = corrector.model.predict(feat.reshape(1, -1))[0]
        pred_gaps.append(pbe + pred_delta)

    pbe_gaps = np.array(pbe_gaps)
    target_gaps = np.array(target_gaps)
    pred_gaps = np.array(pred_gaps)
    X = np.array(features_list)

    # Compute Hat Matrix & Statistical Leverage
    H = X @ np.linalg.pinv(X)
    leverages = np.diag(H)
    n = len(formulas)
    p = X.shape[1]
    h_thresh = 2.0 * p / n

    # MgO 4-point sensitivity data
    mgo_a = np.array([4.200, 4.253, 4.300, 4.410])
    mgo_eg = np.array([4.835, 4.475, 4.169, 3.515])
    slope, intercept = np.polyfit(mgo_a, mgo_eg, 1)

    # Family LOCO MAEs under production Delta formulation (alpha=1.0)
    loco_data = {
        "Halide\nPerovskite": (0.954, 10, palette["halide_perovskite"]),
        "TM\nPerovskite": (0.573, 2, palette["transition_metal_perovskite"]),
        "Alkaline Earth\nOxide": (0.596, 3, palette["alkaline_earth_oxide"]),
        "Alkali\nHalide": (1.054, 6, palette["alkali_halide"]),
        "Pooled\nLOOCV": (0.360, 21, "#666666"),
    }

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    plt.subplots_adjust(wspace=0.28, hspace=0.34)

    # Panel (a): In-Sample Calibration Parity Plot
    ax_a = axs[0, 0]
    ax_a.text(-0.16, 1.08, "(a)", transform=ax_a.transAxes, size=11, weight="bold")
    ax_a.set_title("In-Sample Calibration Parity", fontsize=9.5, pad=6)

    for fam, col, label in [
        ("halide_perovskite", palette["halide_perovskite"], "Halide Perovskites"),
        ("transition_metal_perovskite", palette["transition_metal_perovskite"], "TM Perovskites"),
        ("alkaline_earth_oxide", palette["alkaline_earth_oxide"], "Alkaline Earth Oxides"),
        ("alkali_halide", palette["alkali_halide"], "Alkali Halides"),
    ]:
        mask = [f == fam for f in families]
        ax_a.scatter(
            target_gaps[mask],
            pred_gaps[mask],
            color=col,
            s=40,
            alpha=0.88,
            edgecolors="k",
            linewidths=0.5,
            label=label,
            zorder=4,
        )

    min_val, max_val = 0.5, 12.5
    ax_a.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.2, label="Identity ($y=x$)", zorder=2)
    q_hat = corrector.q_hat
    ax_a.fill_between(
        [min_val, max_val],
        [min_val - q_hat, max_val - q_hat],
        [min_val + q_hat, max_val + q_hat],
        color="gray",
        alpha=0.15,
        label=f"90% Conformal Band (±{q_hat:.2f} eV)",
        zorder=1,
    )

    ax_a.annotate("LiF", xy=(11.45, 11.23), xytext=(10.0, 11.5),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333"), fontsize=7.5)
    ax_a.annotate("CsPbI$_3$", xy=(1.73, 1.99), xytext=(2.5, 1.2),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333"), fontsize=7.5)
    ax_a.annotate("BaTiO$_3$", xy=(3.20, 3.42), xytext=(4.0, 2.7),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333"), fontsize=7.5)

    ax_a.set_xlabel("High-Fidelity Reference Gap $E_g^{\\mathrm{ref}}$ (eV)")
    ax_a.set_ylabel("$\\Delta$-ML Corrected Gap $E_g^{\\mathrm{pred}}$ (eV)")
    ax_a.set_xlim(0.0, 12.5)
    ax_a.set_ylim(0.0, 12.5)
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.8)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel (b): Statistical Leverage (Hat Matrix Diagonal)
    ax_b = axs[0, 1]
    ax_b.text(-0.16, 1.08, "(b)", transform=ax_b.transAxes, size=11, weight="bold")
    ax_b.set_title("Statistical Leverage ($h_{ii}$)", fontsize=9.5, pad=6)

    idx_sort = np.argsort(leverages)[::-1]
    x_pos = np.arange(len(formulas))
    colors_b = [palette[families[i]] for i in idx_sort]

    bars = ax_b.bar(x_pos, leverages[idx_sort], color=colors_b, edgecolor="k", linewidth=0.4, width=0.7, zorder=3)
    ax_b.axhline(h_thresh, color="red", ls="--", lw=1.2, label=f"Leverage Cutoff ($2p/n = {h_thresh:.3f}$)", zorder=4)

    top1_idx = idx_sort[0]
    top2_idx = idx_sort[1]
    ax_b.annotate(f"{formulas[top1_idx]}\n($h_{{ii}}={leverages[top1_idx]:.3f}$)", xy=(0, leverages[top1_idx]), xytext=(1.8, 0.88),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="red"), fontsize=7.2, weight="bold")
    ax_b.annotate(f"{formulas[top2_idx]}\n($h_{{ii}}={leverages[top2_idx]:.3f}$)", xy=(1, leverages[top2_idx]), xytext=(3.5, 0.70),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="red"), fontsize=7.2, weight="bold")

    ax_b.set_xlabel("Calibration Dataset Entries (Ranked by Influence)")
    ax_b.set_ylabel("Hat Matrix Leverage $h_{ii}$")
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_xticks([0, 5, 10, 15, 20])
    ax_b.legend(loc="upper right", frameon=False, fontsize=7.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel (c): MgO Lattice Sensitivity & Deformation Potential
    ax_c = axs[1, 0]
    ax_c.text(-0.16, 1.08, "(c)", transform=ax_c.transAxes, size=11, weight="bold")
    ax_c.set_title("MgO Lattice Deformation Scan", fontsize=9.5, pad=6)

    a_fine = np.linspace(4.18, 4.43, 100)
    ax_c.plot(a_fine, slope * a_fine + intercept, color="#222222", lw=1.4, ls="-",
             label=f"Fit: $dE_g/da = {slope:.3f}$ eV/Å ($R^2=0.998$)", zorder=2)
    ax_c.scatter(mgo_a, mgo_eg, color=palette["alkaline_earth_oxide"], s=55, edgecolor="k", linewidth=0.6,
                label="Direct QE-PBE SCF Scan", zorder=4)

    ax_c.scatter([4.253], [4.475], color="#0072B2", s=80, marker="*", edgecolor="k", zorder=5, label="PBE Relaxed ($a=4.253$ Å)")
    ax_c.scatter([4.212], [slope * 4.212 + intercept], color="#D55E00", s=60, marker="X", edgecolor="k", zorder=5, label="Exp Compressed ($a=4.212$ Å)")

    ax_c.annotate("Relaxed PBE\n$E_g=4.48$ eV", xy=(4.253, 4.475), xytext=(4.26, 4.65),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#0072B2"), fontsize=7.0)
    ax_c.annotate("Heuristic Volume Expansion\n$a=4.41$ Å $\\to E_g=3.52$ eV", xy=(4.410, 3.515), xytext=(4.28, 3.70),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#333"), fontsize=7.0)

    ax_c.set_xlabel("Lattice Parameter $a$ (Å)")
    ax_c.set_ylabel("Direct Band Gap $E_g$ (eV)")
    ax_c.set_xlim(4.18, 4.43)
    ax_c.set_ylim(3.3, 5.1)
    ax_c.legend(loc="upper right", frameon=False, fontsize=6.8)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel (d): Leave-One-Chemistry-Out (LOCO) Generalization
    ax_d = axs[1, 1]
    ax_d.text(-0.16, 1.08, "(d)", transform=ax_d.transAxes, size=11, weight="bold")
    ax_d.set_title("Leave-One-Family-Out Validation", fontsize=9.5, pad=6)

    fam_names = list(loco_data.keys())
    maes = [loco_data[k][0] for k in fam_names]
    counts = [loco_data[k][1] for k in fam_names]
    bar_cols = [loco_data[k][2] for k in fam_names]

    bar_d = ax_d.bar(range(len(fam_names)), maes, color=bar_cols, edgecolor="k", linewidth=0.5, width=0.6, zorder=3)

    for i, (b, val, cnt) in enumerate(zip(bar_d, maes, counts)):
        ax_d.text(b.get_x() + b.get_width() / 2.0, val + 0.025, f"{val:.3f} eV\n($n={cnt}$)",
                 ha="center", va="bottom", fontsize=6.8, weight="bold" if i == 4 else "normal")

    ax_d.axhline(0.360, color="gray", ls=":", lw=1.0, zorder=2)
    ax_d.set_xticks(range(len(fam_names)))
    ax_d.set_xticklabels(fam_names, fontsize=7.5)
    ax_d.set_ylabel("Leave-One-Family-Out MAE (eV)")
    ax_d.set_ylim(0.0, 1.25)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    return fig


if __name__ == "__main__":
    fig = create_publication_figure()
    
    out_dir = Path("materials-screening-ai/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_delta_ml_validation_matrix.png"
    pdf_path = out_dir / "fig_delta_ml_validation_matrix.pdf"

    fig.savefig(png_path)
    fig.savefig(pdf_path)
    
    # Also save to artifact directory for display
    artifact_dir = Path(r"C:\Users\User\.gemini\antigravity-ide\brain\6afe051c-794c-4ead-8e8b-6bff179594b5")
    if artifact_dir.exists():
        fig.savefig(artifact_dir / "fig_delta_ml_validation_matrix.png")

    plt.close(fig)

    print(f"[SUCCESS] Exported publication figures:")
    print(f"  - PNG: {png_path} ({png_path.stat().st_size} bytes)")
    print(f"  - PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")
