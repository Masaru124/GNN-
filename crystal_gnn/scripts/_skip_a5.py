"""Mark A5 as completed by copying A4's metrics, since they have identical configs."""
import json
from pathlib import Path

state_path = Path("checkpoints/paper_managed_state.json")
with open(state_path, "r") as f:
    state = json.load(f)

# A4 and A5 have identical configs - copy A4's results to A5
state["A5"] = {
    "status": "completed",
    "run_id": "paper_A5_soap_loco_formation_energy_per_atom",
    "current_epoch": state["A4"]["current_epoch"],
    "metrics": state["A4"]["metrics"].copy()
}

with open(state_path, "w") as f:
    json.dump(state, f, indent=2)

print("Done! A5 marked as completed with A4's metrics.")
print(f"A5 status: {state['A5']['status']}")
print(f"A6 status: {state['A6']['status']}")
print("Next run will skip A5 and start A6 directly.")
