import sys, os
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from pymatgen.core import Composition, Element
from app.services.delta_ml_corrector import _composition_features, CALIBRATION_RECORDS

print("=== VERIFYING _composition_features FOR ALL 21 BENCHMARK COMPOUNDS ===")
print(f"{'Formula':10s} | {'chi_diff':8s} | {'r_ratio':8s} | {'Z_avg':8s} | {'Calib chi':9s} | {'Calib r':8s} | {'Calib Z':8s} | {'Calib eps':9s} | {'Status'}")
print("-" * 105)

for r in CALIBRATION_RECORDS:
    pbe, hse, c_chi, c_r, c_Z, c_eps, form, src, mp, fam = r
    chi, r_rat, z_avg = _composition_features(form)
    
    match = (abs(chi - c_chi) < 1e-4 and abs(r_rat - c_r) < 1e-4 and abs(z_avg - c_Z) < 1e-4)
    status = "EXACT MATCH" if match else "MISMATCH"
    
    print(f"{form:10s} | {chi:8.4f} | {r_rat:8.4f} | {z_avg:8.4f} | {c_chi:9.4f} | {c_r:8.4f} | {c_Z:8.4f} | {c_eps:9.4f} | {status}")
