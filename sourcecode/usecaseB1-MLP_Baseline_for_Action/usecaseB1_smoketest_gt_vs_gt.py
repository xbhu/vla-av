"""
Smoke test: feed GT trajectory as its own 'prediction' through the B1
eval pipeline. L2 must be ~0 across all horizons; if not, the
coordinate transform or shape handling in usecaseB1_common.py has a
bug relative to nusc_eval_a2.py and must be fixed before trusting any
other B1 number.
"""
import sys
sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import VAL_DIR, load_all_samples, evaluate_predictions, print_planning_tables

val_samples = load_all_samples(VAL_DIR)
print(f"Loaded {len(val_samples)} val samples")

pred_trajs = {token: s["gt_trajectory"] for token, s in val_samples.items()}
result, n_used, n_skipped = evaluate_predictions(pred_trajs, val_samples, label="SMOKETEST_GT_VS_GT")
print_planning_tables(result, "SMOKETEST (GT vs GT, expect ~0)")

max_l2 = float(result["L2"].max())
if max_l2 > 1e-3:
    print(f"[FAIL] max L2 = {max_l2:.6f}, expected ~0. DO NOT TRUST downstream B1 numbers until fixed.")
    sys.exit(1)
else:
    print(f"[PASS] max L2 = {max_l2:.6f}, pipeline confirmed correct.")
