"""
Inspect the worst-performing samples from Phase 1 per-sample L2 analysis:
what instruction/command were they given, what raw action tokens did the
model output, and do the failures cluster in one scene or one maneuver
type.
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

RESULTS_PATH = "/home/xzh5180/Research/vla-av/outputs/usecaseA2/usecaseA2_phase1_per_sample_l2.json"
PREP_VAL_DIR = "/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val"

with open(RESULTS_PATH) as f:
    results = json.load(f)

worst = sorted([r for r in results if r["l2_3s"] is not None], key=lambda r: r["l2_3s"], reverse=True)[:10]
worst_tokens = set(r["token"] for r in worst)

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

print(f"Investigating {len(worst)} worst samples (by 3s L2):\n")
for r in worst:
    scene_path = None
    for sp, _ in val_dataset.scenes:
        with open(sp) as f:
            sd = json.load(f)
        if sd["token"] == r["token"]:
            scene_path = sp
            scene_data = sd
            break
    if scene_path is None:
        continue

    input_features = {}
    for builder in val_dataset._agent.get_feature_builders():
        input_features.update(builder.compute_features(scene_data))

    pred_trajectory, output_text = model.autovla.predict(input_features)

    print(f"--- token={r['token']}  l2_3s={r['l2_3s']:.2f}m ---")
    print(f"  instruction: {scene_data.get('instruction')}")
    print(f"  velocity: {input_features.get('vehicle_velocity')}")
    print(f"  raw output: {output_text[:300]}")
    if isinstance(pred_trajectory, torch.Tensor):
        print(f"  predicted trajectory:\n{pred_trajectory}")
    print()

print("INVESTIGATE OUTLIERS DONE")
