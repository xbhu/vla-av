"""
Cluster C1 main driver script: runs all 4 conditions for each scene
  A. Baseline (normal generation)
  B. Experiment 1 - CoT off (prefilled straightforward template)
  C. Experiment 2 - Instruction counterfactual
  D. Experiment 3 - Perception masking (blackout all three camera frames)
The first 20 scenes additionally re-run condition A once, as the noise floor for Experiment 0.

Usage:
  python usecaseC1_run_battery.py --debug
  python usecaseC1_run_battery.py
"""

import argparse
import csv
import copy
import time
from pathlib import Path

from usecaseC1_common import (
    AutoVLA_C1, build_c1_agent, load_c1_scenes,
    AutoVLAAgentFeatureBuilder, INSTRUCTION_CLASSES,
    FORCED_NO_COT_PREFIX, endpoint_l2, mean_pointwise_l2,
)

CONFIG_PATH = "/home/xzh5180/Research/vla-av/sourcecode/AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
CHECKPOINT_PATH = "/home/xzh5180/Research/vla-av/sourcecode/AutoVLA/checkpoints/AutoVLA_PDMS_89.ckpt"
SENSOR_DATA_PATH = "/home/xzh5180/Research/vla-av/datasets/navsim_mini_subset/mini_sensor_blobs/mini"
JSON_DATA_PATH = "/home/xzh5180/Research/vla-av/datasets/navsim_mini_subset/autovla_json_input"
CODEBOOK_CACHE_PATH = "codebook_cache/agent_vocab.pkl"

NOISE_FLOOR_N_SCENES = 20


def build_features(feature_builder, scene_data, sensor_data_path, instruction_override=None):
    sd = copy.deepcopy(scene_data)
    if instruction_override is not None:
        sd['instruction'] = instruction_override
    features = feature_builder.compute_features(sd)
    features['sensor_data_path'] = sensor_data_path
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_csv", default="./outputs/usecaseC1/c1_battery_results.csv")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print("Loading agent...")
    agent = build_c1_agent(CONFIG_PATH, CHECKPOINT_PATH, SENSOR_DATA_PATH, CODEBOOK_CACHE_PATH)
    model = agent.autovla
    feature_builder = AutoVLAAgentFeatureBuilder(sensor_data_path=SENSOR_DATA_PATH)

    scenes = load_c1_scenes(JSON_DATA_PATH)
    print(f"Loaded {len(scenes)} scenes from {JSON_DATA_PATH}")
    if args.debug:
        scenes = scenes[:5]
        print(f"[DEBUG MODE] only running first {len(scenes)} scenes")

    fieldnames = [
        "token", "orig_instruction",
        "endpoint_l2_cot_off", "mean_l2_cot_off",
        "endpoint_l2_instr_cf1", "mean_l2_instr_cf1", "instr_cf1_target",
        "endpoint_l2_instr_cf2", "mean_l2_instr_cf2", "instr_cf2_target",
        "endpoint_l2_perception_mask", "mean_l2_perception_mask",
        "endpoint_l2_noise_floor", "mean_l2_noise_floor",
        "cot_text_baseline", "cot_text_cot_off",
        "elapsed_sec",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (token, scene_data) in enumerate(scenes):
            t0 = time.time()
            row = {"token": token, "orig_instruction": scene_data.get("instruction", "")}

            try:
                features_a = build_features(feature_builder, scene_data, SENSOR_DATA_PATH)
                traj_a, cot_a = model.predict_c1(features_a)
                row["cot_text_baseline"] = cot_a.replace("\n", " \\n ")

                traj_b, cot_b = model.predict_c1(features_a, forced_cot_prefix=FORCED_NO_COT_PREFIX)
                row["endpoint_l2_cot_off"] = endpoint_l2(traj_a, traj_b)
                row["mean_l2_cot_off"] = mean_pointwise_l2(traj_a, traj_b)
                row["cot_text_cot_off"] = cot_b.replace("\n", " \\n ")

                orig_instr = scene_data.get("instruction", "").lower()
                if orig_instr in INSTRUCTION_CLASSES:
                    cf_targets = [c for c in INSTRUCTION_CLASSES if c != orig_instr]
                else:
                    cf_targets = []  # instruction not in the known vocabulary (e.g. 'unknown'); skip the counterfactual experiment to avoid using an incorrect original instruction as the control
                for i, cf_target in enumerate(cf_targets[:2], start=1):
                    features_cf = build_features(feature_builder, scene_data, SENSOR_DATA_PATH,
                                                  instruction_override=cf_target)
                    traj_cf, _ = model.predict_c1(features_cf)
                    row[f"endpoint_l2_instr_cf{i}"] = endpoint_l2(traj_a, traj_cf)
                    row[f"mean_l2_instr_cf{i}"] = mean_pointwise_l2(traj_a, traj_cf)
                    row[f"instr_cf{i}_target"] = cf_target

                traj_d, _ = model.predict_c1(
                    features_a, camera_mask=["front_camera", "front_left_camera", "front_right_camera"]
                )
                row["endpoint_l2_perception_mask"] = endpoint_l2(traj_a, traj_d)
                row["mean_l2_perception_mask"] = mean_pointwise_l2(traj_a, traj_d)

                if idx < NOISE_FLOOR_N_SCENES:
                    traj_a2, _ = model.predict_c1(features_a)
                    row["endpoint_l2_noise_floor"] = endpoint_l2(traj_a, traj_a2)
                    row["mean_l2_noise_floor"] = mean_pointwise_l2(traj_a, traj_a2)

            except Exception as e:
                print(f"[ERROR] scene {token}: {e}")
                row["cot_text_baseline"] = f"ERROR: {e}"

            row["elapsed_sec"] = round(time.time() - t0, 1)
            writer.writerow(row)
            f.flush()
            print(f"[{idx+1}/{len(scenes)}] {token} done in {row['elapsed_sec']}s")

    print(f"\nDone. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
