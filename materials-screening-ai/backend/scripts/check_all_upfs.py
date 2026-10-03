import os
import re
from pathlib import Path

pp_dir = Path(os.environ.get("SSSP_PP_DIR") or Path(__file__).resolve().parents[3] / "qe" / "pseudo")
for upf in pp_dir.glob("*.UPF"):
    text = upf.read_text(errors="ignore")[:4000]
    m = re.search(r'z_valence\s*=\s*"?([0-9\.eEdD\+\-]+)"?', text, re.IGNORECASE)
    if not m:
        m = re.search(r'([0-9\.eEdD\+\-]+)\s+Z valence', text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace("d", "e").replace("D", "e")
        v = float(val_str)
        if v <= 0:
            print(f"Non-positive z_val: {upf.name} -> {v}")
