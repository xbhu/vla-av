"""
Check whether the 34m-L2 "outlier" samples are a real model failure, or
an artifact of usecaseA2_phase1_per_sample_l2.py comparing pred and GT
trajectories in different coordinate frames (that diagnostic script did
NOT apply the pred/GT -> UniAD-coordinate transform that the official
nusc_eval_a2.py pipeline does before computing L2/collision).
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

worst_tokens = [
    "63c24b51feb94f14bec29022dae4975d",
    "a51146d2b181450292623ec03ade9f03",
    "de9432d4fc7a4e5a985e2bc628eb614c",
]

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
val_dataset = SFTDataset(config['data']['val'], config['model'], processor)

model = SFTAutoVLA(config)
model.autovla.vlm.resize_token_embeddings(len(processor.tokenizer))
state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')['state_dict']
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.autovla.device = DEVICE
model.eval()

for sp, _ in val_dataset.scenes:
    with open(sp) as f:
        scene_data = json.load(f)
    if scene_data["token"] not in worst_tokens:
        continue

    input_features = {}
    target_trajectory = {}
    for builder in val_dataset._agent.get_feature_builders():
        input_features.update(builder.compute_features(scene_data))
    for builder in val_dataset._agent.get_target_builders():
        target_trajectory.update(builder.compute_targets(scene_data))

    pred_trajectory, _ = model.autovla.predict(input_features)
    gt_raw_trajectory = target_trajectory["gt_pos_raw"]
    pred_xy = pred_trajectory[:, :2].to(gt_raw_trajectory.device)

    print(f"--- token={scene_data['token']} ---")
    print(f"Pred xy (model's own local frame): \n{pred_xy}")
    print(f"GT xy (gt_pos_raw, no transform):  \n{gt_raw_trajectory[:, :2]}")

    # Reproduce the EXACT transform nusc_eval_a2.py applies before computing L2
    gt_traj_uniadcoord = gt_raw_trajectory.unsqueeze(0).clone()
    gt_traj_uniadcoord[:, :, [0, 1]] = gt_traj_uniadcoord[:, :, [1, 0]]
    gt_traj_uniadcoord[:, :, 0] = -gt_traj_uniadcoord[:, :, 0]

    pred_traj_uniadcoord = pred_xy.unsqueeze(0).clone()
    pred_traj_uniadcoord[:, :, [0, 1]] = pred_traj_uniadcoord[:, :, [1, 0]]
    pred_traj_uniadcoord[:, :, 0] = -pred_traj_uniadcoord[:, :, 0]

    print(f"Pred xy (after UniAD-coord transform): \n{pred_traj_uniadcoord[0]}")
    print(f"GT xy (after UniAD-coord transform):   \n{gt_traj_uniadcoord[0, :, :2]}")

    n = min(pred_xy.shape[0], gt_raw_trajectory.shape[0], 6)
    l2_no_transform = torch.norm(pred_xy[n-1] - gt_raw_trajectory[n-1, :2]).item()
    l2_with_transform = torch.norm(
        pred_traj_uniadcoord[0, n-1] - gt_traj_uniadcoord[0, n-1, :2]
    ).item()
    print(f"3s L2 WITHOUT UniAD-coord transform (what the diagnostic script computed): {l2_no_transform:.3f}m")
    print(f"3s L2 WITH UniAD-coord transform (matches official nusc_eval_a2.py):        {l2_with_transform:.3f}m")
    print()

print("CHECK COORDINATE FRAME DONE")
