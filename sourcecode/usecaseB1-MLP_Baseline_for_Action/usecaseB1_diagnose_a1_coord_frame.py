"""
Diagnose whether A1's cached pred_trajs_dict.json is already in
UniAD-coordinate frame (post-transform) rather than the raw
ego-relative frame that B1's evaluate_predictions() expects and
transforms internally. If so, applying our transform again
double-transforms it, explaining the implausible ~4-24m L2 numbers.
"""
import json

VAL_DIR_TOKEN = "a51146d2b181450292623ec03ade9f03"  # not in A1 preds (excluded), use a token that IS
with open("/home/xzh5180/Research/vla-av/outputs/usecaseA1/pred_trajs_dict.json") as f:
    a1_raw = json.load(f)

# grab any one token's pred and compare directly against its GT in raw frame
sample_key = list(a1_raw.keys())[0]
token = sample_key.replace("_trajectory", "")
pred = a1_raw[sample_key][0]
print(f"token: {token}")
print(f"A1 cached pred (6 steps): {pred}")

with open(f"/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val_filtered/{token}.json") as f:
    gt_sample = json.load(f)
print(f"\nGT gt_trajectory (raw ego frame, first 6 of 10 steps): {gt_sample['gt_trajectory'][:6]}")

# Apply our UniAD transform to GT only, see which one A1's pred actually resembles
import torch
gt_xy = torch.tensor([p[:2] for p in gt_sample["gt_trajectory"][:6]], dtype=torch.float32)
out = gt_xy.unsqueeze(0).clone()
out[:, [0, 1]] = out[:, [1, 0]]
out[:, 0] = -out[:, 0]
print(f"\nGT after UniAD-coord transform (swap+negate): {out.tolist()}")

print("\n[COMPARE] A1 cached pred vs GT-raw vs GT-UniAD-transformed")
print("If A1 pred numerically resembles 'GT-UniAD-transformed' shape/scale more than 'GT-raw',")
print("then A1's cache is ALREADY post-transform, and evaluate_predictions() must NOT")
print("re-apply the transform to it (only to GT).")
