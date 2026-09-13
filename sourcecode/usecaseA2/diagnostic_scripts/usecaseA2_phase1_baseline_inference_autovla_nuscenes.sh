#!/bin/bash
set -uo pipefail
AUTOVLA_DIR="/home/xzh5180/Research/vla-av/sourcecode/AutoVLA"
CONFIG_DIR="/home/xzh5180/Research/vla-av/sourcecode/clusterA_action_representation/configs"
SEG_DIR="/home/xzh5180/Research/vla-av/datasets/autovla_nusc_seg/nusc_eval_seg_6s"
OUT_DIR="/home/xzh5180/Research/vla-av/outputs/usecaseA2"

cd "$AUTOVLA_DIR"
conda run -n autovla_codeclean python tools/eval/nusc_eval_a2.py \
    --config "$CONFIG_DIR/usecaseA2_nusc_fast_eval.yaml" \
    --checkpoint "$AUTOVLA_DIR/checkpoints/AutoVLA_PDMS_89.ckpt" \
    --seg_data_path "$SEG_DIR" \
    --output "$OUT_DIR/usecaseA2_phase1_fast_planning_table.txt" \
    --device cuda:0 2>&1 | tee "$OUT_DIR/usecaseA2_phase1_fast_inference.log"
