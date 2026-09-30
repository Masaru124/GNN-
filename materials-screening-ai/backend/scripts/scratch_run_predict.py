import sys, os, json
sys.path.insert(0, os.path.abspath("materials-screening-ai/backend"))
from app.services.delta_ml_corrector import DeltaMLGapCorrector

corrector = DeltaMLGapCorrector()

print("=== REAL CALLS TO predict_corrected_gap ===")

# 1. SrTiO3 (Calibration PBE = 2.179 eV, eps_inf = 6.10)
res_srtio3 = corrector.predict_corrected_gap(pbe_gap_ev=2.179, formula="SrTiO3", features={"eps_inf": 6.10})
print("\n[SrTiO3 (PBE=2.179, eps_inf=6.10)]:")
print(json.dumps(res_srtio3, indent=2))

# 2. BaTiO3 (Calibration PBE = 2.068 eV, eps_inf = 6.30)
res_batio3 = corrector.predict_corrected_gap(pbe_gap_ev=2.068, formula="BaTiO3", features={"eps_inf": 6.30})
print("\n[BaTiO3 (PBE=2.068, eps_inf=6.30)]:")
print(json.dumps(res_batio3, indent=2))

# 3. KZrCl3 (Test PBE = 1.50 eV, eps_inf = 4.5)
res_kzrcl3 = corrector.predict_corrected_gap(pbe_gap_ev=1.50, formula="KZrCl3", features={"eps_inf": 4.5})
print("\n[KZrCl3 (PBE=1.50, eps_inf=4.5)]:")
print(json.dumps(res_kzrcl3, indent=2))

# 4. ZrCl4 (Test PBE = 3.80 eV, eps_inf = 3.2)
res_zrcl4 = corrector.predict_corrected_gap(pbe_gap_ev=3.80, formula="ZrCl4", features={"eps_inf": 3.2})
print("\n[ZrCl4 (PBE=3.80, eps_inf=3.2)]:")
print(json.dumps(res_zrcl4, indent=2))
