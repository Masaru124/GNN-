#!/bin/bash
# LOCO run monitor - progress, health, and the exact resume command.
# State is derived from on-disk artifacts, so this is safe to run any time,
# even while nothing is running (after a power cut, etc).
cd "$(dirname "$0")" || exit 1
S=crystal_gnn/.loco_scheduler_state.json
NPZ=materials-screening-ai/research/loco_cross_conformal

echo "=== time: $(date) ==="
echo "--- scheduler state ---"
cat "$S" 2>/dev/null || echo "(no state file - scheduler has not run yet)"
echo "--- trained / scored ---"
for c in 0 1 2 3 4 5 6 7 8 9; do
  if [ "$c" = 0 ]; then CK=crystal_gnn/checkpoints/paper_A7_soap_loco_formation_energy_per_atom/best.pt
  else CK=crystal_gnn/checkpoints/loco_A7_soap_cluster${c}_seed42/best.pt; fi
  T="."; [ -f "$CK" ] && T="T"
  G="."; [ -f "$NPZ/fold${c}_scores.npz" ] && G="S"
  printf "fold %s [%s%s] " "$c" "$T" "$G"
done
echo "   (T=trained S=scored)"
echo "--- report n_folds ---"
grep -o '"n_folds": [0-9]*' "$NPZ/cross_conformal_summary.json" 2>/dev/null || echo "(no report)"
echo "--- scheduler log tail ---"
tail -4 crystal_gnn/loco_scheduler.log 2>/dev/null
echo "--- running procs ---"
ps aux 2>/dev/null | grep -E "loco_scheduler|train.py" | grep -v grep | head -3
echo "--- resume command ---"
echo "  ./venv311/Scripts/python.exe crystal_gnn/scripts/loco_scheduler.py  (bg)"
echo "=== monitor done ==="
