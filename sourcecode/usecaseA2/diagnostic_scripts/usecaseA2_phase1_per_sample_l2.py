"""
Re-run inference but record per-sample 3s L2 error, to check whether
the Phase 1 average is being pulled up by a small number of outlier
failures rather than reflecting a uniform degradation.
"""
import sys, json, torch
from pathlib import Path

PROJECT_ROOT = Path("/home/xzh5180/Research/vla-av/sourcecode/AutoVLA")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
from transformers import AutoProcessor
from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA

CONFIG_PATH = "/home/xzh5180/Research/vla-av/sourcecode/clusterA_action_representation/configs/usecaseA2_nusc_fast_eval.yaml"
CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints/AutoVLA_PDMS_89.ckpt")
DEVICE = "cuda:0"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
val_dataset = SFTDataset(config['data']['val'], config['model'], processor)

model = SFTAutoVLA(config)
model.autovla.vlm.resize_token_embeddings(len(processor.tokenizer))
state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')['state_dict']
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
model.to(DEVICE)
model.autovla.device = DEVICE
model.eval()

results = []
for idx in range(len(val_dataset.scenes)):
    scene_path, _ = val_dataset.scenes[idx]
    with open(scene_path) as f:
        scene_data = json.load(f)

    input_features = {}
    target_trajectory = {}
    for builder in val_dataset._agent.get_feature_builders():
        input_features.update(builder.compute_features(scene_data))
    for builder in val_dataset._agent.get_target_builders():
        target_trajectory.update(builder.compute_targets(scene_data))

    pred_trajectory, output_text = model.autovla.predict(input_features)
    if pred_trajectory == [] or len(pred_trajectory) == 0:
        results.append({"idx": idx, "token": scene_data["token"], "l2_3s": None, "status": "empty_prediction"})
        continue

    gt_raw = target_trajectory["gt_pos_raw"]
    pred_xy = pred_trajectory[:, :2].to(gt_raw.device)
    n = min(pred_xy.shape[0], gt_raw.shape[0], 6)
    l2_final = torch.norm(pred_xy[n-1] - gt_raw[n-1, :2]).item()
    results.append({"idx": idx, "token": scene_data["token"], "l2_3s": l2_final, "status": "ok"})

valid = [r for r in results if r["l2_3s"] is not None]
valid_sorted = sorted(valid, key=lambda r: r["l2_3s"], reverse=True)

print(f"\nTotal samples: {len(results)}, valid: {len(valid)}, empty/failed: {len(results)-len(valid)}")
print(f"\nTop 10 worst 3s-L2 samples:")
for r in valid_sorted[:10]:
    print(f"  token={r['token']} l2_3s={r['l2_3s']:.3f}m")

import statistics
l2_values = [r["l2_3s"] for r in valid]
print(f"\nmean={statistics.mean(l2_values):.3f}  median={statistics.median(l2_values):.3f}  "
      f"stdev={statistics.stdev(l2_values):.3f}  max={max(l2_values):.3f}  min={min(l2_values):.3f}")

out_path = "/home/xzh5180/Research/vla-av/outputs/usecaseA2/usecaseA2_phase1_per_sample_l2.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved per-sample results to {out_path}")
