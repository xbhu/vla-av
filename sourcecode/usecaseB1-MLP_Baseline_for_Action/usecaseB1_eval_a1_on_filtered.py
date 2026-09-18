"""
Re-evaluate A1 (OpenDriveVLA) cached predictions on our 126-token
val_filtered set, reusing the same evaluate_predictions() pipeline as
B1 (already smoke-tested against GT-vs-GT = 0). This gives an A1 number
on an exactly known, stated sample count -- not a re-run of inference,
just re-scoring cached predictions against the filtered token list.

Known limitation (stated, not hidden): only 122/126 val_filtered tokens
are present in A1's cached pred_trajs_dict.json (146 total cached preds,
4 of our 126 are missing). Reported n=122, not 126.
"""
import sys
import json

sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import VAL_DIR, load_all_samples, evaluate_predictions, print_planning_tables

OUT_DIR = "/home/xzh5180/Research/vla-av/outputs/usecaseB1"
A1_PRED_PATH = "/home/xzh5180/Research/vla-av/outputs/usecaseA1/pred_trajs_dict.json"

val_samples = load_all_samples(VAL_DIR)
val_tokens = set(val_samples.keys())

with open(A1_PRED_PATH) as f:
    a1_raw = json.load(f)

# keys look like "{token}_trajectory" -> value is [[ [x,y], [x,y], ... ]] (nested one level)
a1_preds = {}
for k, v in a1_raw.items():
    token = k.replace("_trajectory", "")
    if token in val_tokens:
        a1_preds[token] = v[0]  # unwrap outer list -> list of [x, y] pairs

print(f"val_filtered tokens: {len(val_tokens)}")
print(f"A1 cached preds overlapping val_filtered: {len(a1_preds)}")
missing = val_tokens - set(a1_preds.keys())
print(f"missing from A1 cache (excluded from this eval): {sorted(missing)}")

# Restrict val_samples to only the tokens we have A1 predictions for,
# so evaluate_predictions() reports n_used = len(a1_preds), not 126
# with silent skips.
val_samples_subset = {t: s for t, s in val_samples.items() if t in a1_preds}

result, n_used, n_skipped = evaluate_predictions(a1_preds, val_samples_subset, label="A1_on_val_filtered_subset")
print_planning_tables(result, f"A1 OpenDriveVLA (n={n_used}, subset of val_filtered)")

serializable = {k: v.tolist() for k, v in result.items()}
with open(f"{OUT_DIR}/usecaseB1_a1_rescored_on_filtered_eval.json", "w") as f:
    json.dump(
        {"n_used": n_used, "n_skipped": n_skipped, "missing_tokens": sorted(missing), "metrics": serializable},
        f, indent=2,
    )
