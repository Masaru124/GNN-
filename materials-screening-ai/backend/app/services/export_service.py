"""
Export Service for MatScreen AI.
Generates downloadable CSV spreadsheets and structured PDF screening reports for material discovery candidates.
"""

import csv
import io
from typing import Any, Dict, List


class ExportService:
    @staticmethod
    def generate_csv_report(predictions: List[Dict[str, Any]]) -> str:
        """Generate CSV spreadsheet from batch prediction results."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Material Name",
            "Formula",
            "Predicted E_f (eV/atom)",
            "Evidential Std Dev (eV)",
            "Aleatoric Uncertainty (eV)",
            "Epistemic Uncertainty (eV)",
            "Conformal 90% Lower, i.i.d. marginal (eV)",
            "Conformal 90% Upper, i.i.d. marginal (eV)",
            "Conformal Interval Width (eV)",
            "Confidence Level",
            "Risk Assessment",
            "Recommendation",
            "Attention 4A (%)",
            "Attention 6A (%)",
            "Attention 8A (%)"
        ])

        for p in predictions:
            attn = p.get("scale_attention", {})
            conf_int = p.get("conformal_90_interval_eV", [0.0, 0.0])

            writer.writerow([
                p.get("filename", "Material"),
                p.get("formula", "N/A"),
                p.get("predicted_formation_energy_per_atom_eV", 0.0),
                p.get("evidential_std_eV", 0.0),
                p.get("aleatoric_std_eV", 0.0),
                p.get("epistemic_std_eV", 0.0),
                conf_int[0] if len(conf_int) > 0 else 0.0,
                conf_int[1] if len(conf_int) > 1 else 0.0,
                p.get("conformal_width_eV", 0.0),
                p.get("confidence", "High"),
                p.get("risk_level", "Low Risk"),
                p.get("recommendation", ""),
                attn.get("4A", 33.3),
                attn.get("6A", 33.3),
                attn.get("8A", 33.4)
            ])

        return output.getvalue()
