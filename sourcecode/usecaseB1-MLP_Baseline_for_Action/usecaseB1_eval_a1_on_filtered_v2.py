"""
Corrected re-evaluation of A1 (OpenDriveVLA) cached predictions on the
126-token val_filtered set.

Fix vs v1: A1's pred_trajs_dict.json values are already in UniAD-coord
frame (-y, x), confirmed by direct numeric comparison against GT's
known raw-frame values run through the same transform (see
usecaseB1_diagnose_a1_coord_frame.py output). v1 fed these
already-transformed values into evaluate_predictions(), which applies
the transform AGAIN -- a double transform that produced implausible
~4-24m L2 values. This version first inverse-transforms A1's cached
pred back to raw ego-relative frame, then lets evaluate_predictions()
apply the standard forward transform once, exactly as it does for GT
and for every other model in this comparison (Tier0, Tier1, A2).

Inverse of (raw_x, raw_y) -> (-raw_y, raw_x) is:
    raw_x = transformed[1]
    raw_y = -transformed[0]
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import VAL_DIR, load_all_samples, evaluate_predictions, print_planning_tables

OUT_DIR = "/home/xzh5180/Research/vla-av/outputs/usecaseB1"
A1_PRED_PATH = "/home/xzh5180/Research/vla-av/outputs/usecaseA1/pred_trajs_dict.json"

val_samples = load_all_samples(VAL_DIR)
val_tokens = set(val_samples.keys())

with open(A1_PRED_PATH) as f:
    a1_raw = json.load(f)

a1_preds_raw_frame = {}
for k, v in a1_raw.items():
    token = k.replace("_trajectory", "")
    if token not in val_tokens:
        continue
    transformed = np.array(v[0], dtype=np.float32)  # (6, 2), already in (-y, x) form
    raw_x = transformed[:, 1]
    raw_y = -transformed[:, 0]
    raw_xy = np.stack([raw_x, raw_y], axis=-1)
    a1_preds_raw_frame[token] = raw_xy.tolist()

print(f"val_filtered tokens: {len(val_tokens)}")
print(f"A1 cached preds overlapping val_filtered: {len(a1_preds_raw_frame)}")
missing = val_tokens - set(a1_preds_raw_frame.keys())
print(f"missing from A1 cache: {sorted(missing)}")

# sanity print: compare inverse-transformed pred[0] against GT[0] for one token, should be close now
sample_token = list(a1_preds_raw_frame.keys())[0]
print(f"\n[SANITY] token={sample_token}")
print(f"  A1 pred, raw frame (first 3 steps): {a1_preds_raw_frame[sample_token][:3]}")
print(f"  GT, raw frame (first 3 steps): {[p[:2] for p in val_samples[sample_token]['gt_trajectory'][:3]]}")

val_samples_subset = {t: s for t, s in val_samples.items() if t in a1_preds_raw_frame}
result, n_used, n_skipped = evaluate_predictions(a1_preds_raw_frame, val_samples_subset, label="A1_v2_corrected")
print_planning_tables(result, f"A1 OpenDriveVLA CORRECTED (n={n_used}, subset of val_filtered)")

serializable = {k: v.tolist() for k, v in result.items()}
with open(f"{OUT_DIR}/usecaseB1_a1_rescored_on_filtered_eval_v2.json", "w") as f:
    json.dump(
        {"n_used": n_used, "n_skipped": n_skipped, "missing_tokens": sorted(missing), "metrics": serializable},
        f, indent=2,
    )
