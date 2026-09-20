"""
Reasoning-action coherence noise floor: for a subset of scenes, run the
SAME forced-full-CoT generation multiple times and check how often the
stated-vs-trajectory agreement label flips purely due to sampling
randomness (do_sample=True). This mirrors the noise-floor design used
in C1 (usecaseC1_run_battery.py) for trajectory displacement.

NOTE: this script was designed before C2 switched to greedy decoding
(see predict_c1's `greedy` argument in usecaseC1_common.py). Once C2
calls predict_c1(..., greedy=True), repeated calls on the same input are
deterministic by construction, so this noise-floor check is no longer
necessary for the current pipeline. Kept for reference / as a fallback
if C2 is ever re-run under sampling.

Usage:
  python usecaseC2_run_noise_floor.py --n_scenes 20 --repeats 3
"""

import argparse
import csv
import time
from pathlib import Path

from usecaseC1_common import build_c1_agent, load_c1_scenes, AutoVLAAgentFeatureBuilder
from usecaseC2_common import FORCED_FULL_COT_PREFIX, parse_stated_category, classify_trajectory

CONFIG_PATH = "/home/xzh5180/Research/vla-av/sourcecode/AutoVLA/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"
CHECKPOINT_PATH = "/home/xzh5180/Research/vla-av/sourcecode/AutoVLA/checkpoints/AutoVLA_PDMS_89.ckpt"
SENSOR_DATA_PATH = "/home/xzh5180/Research/vla-av/datasets/navsim_mini_subset/mini_sensor_blobs/mini"
JSON_DATA_PATH = "/home/xzh5180/Research/vla-av/datasets/navsim_mini_subset/autovla_json_input"
CODEBOOK_CACHE_PATH = "codebook_cache/agent_vocab.pkl"


def run_once(model, feature_builder, token, scene_data):
    features = feature_builder.compute_features(scene_data)
    features['sensor_data_path'] = SENSOR_DATA_PATH
    trajectory, cot_text = model.predict_c1(features, forced_cot_prefix=FORCED_FULL_COT_PREFIX)
    stated = parse_stated_category(cot_text)
    traj_cat = classify_trajectory(trajectory)

    if stated["no_decision_stated"] or stated["lateral_is_na_due_to_stop"]:
        lateral_match = None
    elif stated["lateral"] is not None:
        lateral_match = (stated["lateral"] == traj_cat["lateral"])
    else:
        lateral_match = None

    if stated["no_decision_stated"] or stated["longitudinal"] is None:
        longitudinal_match = None
    else:
        longitudinal_match = (stated["longitudinal"] == traj_cat["longitudinal"])

    return {
        "token": token,
        "stated_lateral": stated["lateral"],
        "stated_longitudinal": stated["longitudinal"],
        "no_decision_stated": stated["no_decision_stated"],
        "traj_lateral": traj_cat["lateral"],
        "traj_longitudinal": traj_cat["longitudinal"],
        "lateral_match": lateral_match,
        "longitudinal_match": longitudinal_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scenes", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_csv", default="./outputs/usecaseC2/c2_noise_floor_results.csv")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    print("Loading agent...")
    agent = build_c1_agent(CONFIG_PATH, CHECKPOINT_PATH, SENSOR_DATA_PATH, CODEBOOK_CACHE_PATH)
    model = agent.autovla
    feature_builder = AutoVLAAgentFeatureBuilder(sensor_data_path=SENSOR_DATA_PATH)

    scenes = load_c1_scenes(JSON_DATA_PATH)[:args.n_scenes]
    print(f"Running {args.repeats} repeats on {len(scenes)} scenes "
          f"({len(scenes) * args.repeats} total generations)")

    fieldnames = ["token", "repeat_idx", "stated_lateral", "stated_longitudinal",
                  "no_decision_stated", "traj_lateral", "traj_longitudinal",
                  "lateral_match", "longitudinal_match", "elapsed_sec"]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (token, scene_data) in enumerate(scenes):
            for r in range(args.repeats):
                t0 = time.time()
                try:
                    row = run_once(model, feature_builder, token, scene_data)
                except Exception as e:
                    print(f"[ERROR] scene {token} repeat {r}: {e}")
                    row = {"token": token, "stated_lateral": None, "stated_longitudinal": None,
                           "no_decision_stated": None, "traj_lateral": None, "traj_longitudinal": None,
                           "lateral_match": None, "longitudinal_match": None}
                row["repeat_idx"] = r
                row["elapsed_sec"] = round(time.time() - t0, 1)
                writer.writerow(row)
                f.flush()
                print(f"[scene {idx+1}/{len(scenes)}, repeat {r+1}/{args.repeats}] {token} "
                      f"lateral_match={row['lateral_match']} longitudinal_match={row['longitudinal_match']}")

    print(f"\nDone. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
