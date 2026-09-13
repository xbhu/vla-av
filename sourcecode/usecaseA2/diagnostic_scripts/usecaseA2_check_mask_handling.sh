#!/bin/bash
set -uo pipefail
AUTOVLA_DIR="/home/xzh5180/Research/vla-av/sourcecode/AutoVLA"
cd "$AUTOVLA_DIR"

echo "===================================================================="
echo "[M1] Full body of compute_L2, update, evaluate_coll in planning_metrics.py"
echo "(need to see whether gt_trajs_mask / sdc_planning_mask actually gates"
echo "which timesteps contribute to the accumulated obj_col/obj_box_col/L2 sums)"
echo "===================================================================="
sed -n '1,180p' tools/eval/planning_metrics.py

echo
echo "===================================================================="
echo "[M2] Confirm what gets passed as gt_trajs_mask in nusc_eval_a2.py's call"
echo "===================================================================="
grep -n -B2 -A2 "planning_metrics(" tools/eval/nusc_eval_a2.py

echo
echo "CHECK MASK HANDLING DONE"
