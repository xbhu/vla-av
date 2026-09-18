"""
Tier 0 baseline: zero-learning kinematic extrapolation, two variants.
No training involved -- this is a closed-form computation from each
val sample's current velocity/acceleration, evaluated on the same 126
val_filtered samples and the same PlanningMetric pipeline as A1/A2.
"""
import sys
import json

sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import (
    VAL_DIR, load_all_samples, kinematic_extrapolation,
    evaluate_predictions, print_planning_tables,
)

OUT_DIR = "/home/xzh5180/Research/vla-av/outputs/usecaseB1"

val_samples = load_all_samples(VAL_DIR)
print(f"Loaded {len(val_samples)} val samples")

for variant_name, use_acc in [("const_velocity", False), ("const_accel", True)]:
    pred_trajs = {
        token: kinematic_extrapolation(s, use_acceleration=use_acc).tolist()
        for token, s in val_samples.items()
    }

    with open(f"{OUT_DIR}/usecaseB1_tier0_{variant_name}_preds.json", "w") as f:
        json.dump(pred_trajs, f, indent=2)

    result, n_used, n_skipped = evaluate_predictions(
        pred_trajs, val_samples, label=f"TIER0_{variant_name}"
    )
    print_planning_tables(result, f"Tier0 ({variant_name})")

    serializable = {k: v.tolist() for k, v in result.items()}
    with open(f"{OUT_DIR}/usecaseB1_tier0_{variant_name}_eval.json", "w") as f:
        json.dump({"n_used": n_used, "n_skipped": n_skipped, "metrics": serializable}, f, indent=2)
