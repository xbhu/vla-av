"""
Build a degenerate-GT-filtered copy of autovla_nusc_preprocessed_val, reusing the exact
criterion from A2's usecaseA2_check_degenerate_gt.py: a sample is valid only if the sum
of the first 6 entries of its future_mask equals 6 (full 3-second future horizon present,
no padding). Samples failing this are excluded, matching the filtering nusc_eval_a2.py
applies at evaluation time -- this filtered directory is needed because SFTDataset in
run_sft_a3.py reads the raw preprocessed directory directly and has no equivalent filter
built in, which would otherwise contaminate val_loss during A3 training with degenerate
(padded/static) GT labels.
"""
import json
import shutil
from pathlib import Path

SRC_DIR = Path("/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val")
DST_DIR = Path("/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val_filtered")

DST_DIR.mkdir(parents=True, exist_ok=True)

total = 0
kept = 0
skipped = 0
skipped_no_mask = 0

for f in sorted(SRC_DIR.glob("*.json")):
    total += 1
    with open(f, "r") as fh:
        d = json.load(fh)
    future_mask = d.get("future_mask", [])
    if not future_mask:
        skipped_no_mask += 1
        continue
    mask_sum_3s = sum(future_mask[:6])
    if mask_sum_3s == 6:
        shutil.copy2(f, DST_DIR / f.name)
        kept += 1
    else:
        skipped += 1

print(f"Total source files: {total}")
print(f"Kept (future_mask_sum_3s == 6): {kept}")
print(f"Skipped (degenerate, future_mask_sum_3s < 6): {skipped}")
print(f"Skipped (no future_mask field present): {skipped_no_mask}")
print(f"Filtered val set written to: {DST_DIR}")
