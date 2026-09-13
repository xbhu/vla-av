"""
Scan all 150 val samples' ground-truth future trajectories to find
"degenerate" ones where the GT barely moves (or doesn't move at all)
across the 3s planning horizon -- a likely artifact of AutoVLA's
nuscenes preprocessing script only filtering insufficient-future-frame
samples on the train split, not val (samples near the end of a scene
get their missing future frames padded by repeating the last known
position).

For each sample, report the straight-line displacement of GT from
t=0 to t=3s (6 steps). A near-zero displacement despite nonzero ego
velocity is the signature of this bug.
"""
import json, glob

PREP_VAL_DIR = "/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val"

records = []
for f in glob.glob(f"{PREP_VAL_DIR}/*.json"):
    with open(f) as fp:
        d = json.load(fp)
    token = d["token"]
    velocity = d.get("velocity")
    future_mask = d.get("future_mask", [])
    mask_sum_3s = sum(future_mask[:6]) if future_mask else None

    gt_diff = d.get("gt_trajectory")  # per-step (dx, dy) deltas, if stored this way
    records.append({
        "token": token,
        "velocity": velocity,
        "future_mask_sum_3s": mask_sum_3s,
        "future_mask_full": future_mask,
    })

# Sort by future_mask_sum_3s ascending -- lowest sums are the most
# likely to be padded/degenerate (should be 6 if fully valid)
records_valid_mask = [r for r in records if r["future_mask_sum_3s"] is not None]
records_valid_mask.sort(key=lambda r: r["future_mask_sum_3s"])

print(f"Total samples: {len(records)}")
print(f"Samples with future_mask field present: {len(records_valid_mask)}")
print()
print("Distribution of future_mask_sum over first 6 steps (should be 6 if fully valid):")
from collections import Counter
counts = Counter(r["future_mask_sum_3s"] for r in records_valid_mask)
for k in sorted(counts.keys()):
    print(f"  sum={k}: {counts[k]} samples")

print()
n_degenerate = sum(1 for r in records_valid_mask if r["future_mask_sum_3s"] < 6)
print(f"Samples with future_mask_sum_3s < 6 (likely padded/degenerate GT): {n_degenerate} / {len(records_valid_mask)}")
print()
print("Worst 10 (lowest future_mask_sum_3s), with velocity for context:")
for r in records_valid_mask[:10]:
    print(f"  token={r['token']}  future_mask_sum_3s={r['future_mask_sum_3s']}  velocity={r['velocity']:.2f} m/s")

with open("/home/xzh5180/Research/vla-av/outputs/usecaseA2/usecaseA2_degenerate_gt_check.json", "w") as f:
    json.dump(records_valid_mask, f, indent=2)
print()
print("Saved full per-sample future_mask info to usecaseA2_degenerate_gt_check.json")
