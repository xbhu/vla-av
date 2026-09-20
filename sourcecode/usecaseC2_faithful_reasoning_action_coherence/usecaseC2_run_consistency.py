"""
Cluster C2 main driver script: force full CoT reasoning on every scene,
parse the lateral/longitudinal action category stated in the CoT, and
compare it against the category the trajectory classifier assigns to
the decoded trajectory.

Each scene needs only 1 inference call (unlike C1, no cross-condition
pairing needed).

Usage:
  python usecaseC2_run_consistency.py --debug        # run first 5 scenes to sanity-check
  python usecaseC2_run_consistency.py                # run all 138 scenes
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
CODEBOOK_CACHE_PATH = "codebook_cache/agent_vocab.pkl"  # adjust to your actual path, same as C1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="only run the first 5 scenes to sanity-check the code")
    parser.add_argument("--output_csv", default="./outputs/usecaseC2/c2_consistency_results.csv")
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
        "stated_lateral", "stated_longitudinal", "used_final_action_section",
        "no_decision_stated", "lateral_is_na_due_to_stop",
        "traj_lateral", "traj_longitudinal",
        "lateral_match", "longitudinal_match", "both_match",
        "heading_change_deg", "lateral_offset_m", "delta_v_mps",
        "cot_text_full", "elapsed_sec",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (token, scene_data) in enumerate(scenes):
            t0 = time.time()
            row = {"token": token, "orig_instruction": scene_data.get("instruction", "")}

            try:
                features = feature_builder.compute_features(scene_data)
                features['sensor_data_path'] = SENSOR_DATA_PATH

                trajectory, cot_text = model.predict_c1(
                    features, forced_cot_prefix=FORCED_FULL_COT_PREFIX, greedy=True
                )

                stated = parse_stated_category(cot_text)
                traj_cat = classify_trajectory(trajectory)

                row["stated_lateral"] = stated["lateral"]
                row["stated_longitudinal"] = stated["longitudinal"]
                row["used_final_action_section"] = stated["used_final_action_section"]
                row["no_decision_stated"] = stated["no_decision_stated"]
                row["lateral_is_na_due_to_stop"] = stated["lateral_is_na_due_to_stop"]
                row["traj_lateral"] = traj_cat["lateral"]
                row["traj_longitudinal"] = traj_cat["longitudinal"]
                row["heading_change_deg"] = traj_cat["heading_change_deg"]
                row["lateral_offset_m"] = traj_cat["lateral_offset_m"]
                row["delta_v_mps"] = traj_cat["delta_v_mps"]

                # lateral_match is left blank (None) -- not scored as
                # a failure -- when there's no stated decision at all,
                # or when lateral is legitimately N/A because the
                # stated action is simply "stop".
                if stated["no_decision_stated"] or stated["lateral_is_na_due_to_stop"]:
                    lateral_match = None
                elif stated["lateral"] is not None:
                    lateral_match = (stated["lateral"] == traj_cat["lateral"])
                else:
                    lateral_match = None

                if stated["no_decision_stated"]:
                    longitudinal_match = None
                elif stated["longitudinal"] is not None:
                    longitudinal_match = (stated["longitudinal"] == traj_cat["longitudinal"])
                else:
                    longitudinal_match = None

                row["lateral_match"] = lateral_match
                row["longitudinal_match"] = longitudinal_match
                row["both_match"] = (
                    None if (lateral_match is None or longitudinal_match is None)
                    else (lateral_match and longitudinal_match)
                )

                row["cot_text_full"] = cot_text.replace("\n", " \\n ")

            except Exception as e:
                print(f"[ERROR] scene {token}: {e}")
                row["cot_text_full"] = f"ERROR: {e}"

            row["elapsed_sec"] = round(time.time() - t0, 1)
            writer.writerow(row)
            f.flush()
            print(f"[{idx+1}/{len(scenes)}] {token} done in {row['elapsed_sec']}s "
                  f"| stated=({row.get('stated_lateral')}, {row.get('stated_longitudinal')}) "
                  f"| traj=({row.get('traj_lateral')}, {row.get('traj_longitudinal')}) "
                  f"| no_decision={row.get('no_decision_stated')}")

    print(f"\nDone. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
