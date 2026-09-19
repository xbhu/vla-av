# VLA-B2: NAVSIM Pseudo-Closed-Loop Evaluation (PDM-Score)

## Overview

This use case evaluates **AutoVLA** (a Qwen2.5-VL-3B-based Vision-Language-Action
model, trained with GRPO-CoT) under NAVSIM's pseudo-closed-loop protocol, using
the **PDM-Score (PDMS)** metric. The goal is to check whether AutoVLA's
open-loop performance (measured with L2 trajectory error in a prior use case,
B1) is consistent with its performance under a simulator that allows nearby
agents to react to the ego vehicle's short-horizon plan.

The checkpoint used, `AutoVLA_PDMS_89.ckpt`, is named after a PDMS of ~0.89,
suggesting PDM-Score was used for model selection during training — on the
full official NAVSIM benchmark, not on the reduced dataset used here.

## Motivation

Open-loop metrics (e.g., L2 distance to the human trajectory) only penalize
deviation from what a human driver did — not whether that deviation is
actually unsafe. A model can have low L2 error and still be unsafe under
closed-loop rollout, or have higher L2 error while following a safer
alternative path. NAVSIM's PDM-Score directly measures closed-loop
consequences (collision, off-road, time-to-collision, progress, comfort)
rather than trajectory similarity.

In B1, AutoVLA had the worst open-loop L2 error (1.420 m) among all models
tested. This use case checks whether that open-loop weakness translates to,
diverges from, or is independent of closed-loop safety performance.

## Method

### Data
- **Dataset**: NAVSIM `navmini` split (OpenScene-derived), **shard 0 only**
  (~8.4 GB), covering 2 logs / 138 scenario tokens — a deliberately minimal
  subset (the full navmini split is 396 scenarios, ~200+ GB), chosen to test
  the open-loop-vs-closed-loop hypothesis at minimal cost rather than to
  reproduce a fully powered benchmark number.
- Only AutoVLA (A2) was evaluated; OpenDriveVLA (A1) was excluded because it
  would require a separate nuScenes → OpenScene data-format adaptation with
  no bearing on this use case's core question (single-model open-loop vs.
  closed-loop consistency).

### Model
- **AutoVLA**: Qwen2.5-VL-3B-Instruct backbone, vocabulary expanded by 2,048
  discrete action tokens (IDs 151665–153712), trained with a GRPO-CoT
  objective. Inference produces a structured
  `<think>...</think><answer>The final output action is: <action_X>×10</answer>`
  output; the 10 action tokens are decoded to a trajectory via a codebook
  (`codebook_cache/agent_vocab.pkl`).
- Checkpoint: `checkpoints/AutoVLA_PDMS_89.ckpt`.

### Pipeline
- Reused AutoVLA's own `navsim/` subdirectory (agent wrapper + Hydra configs)
  rather than reimplementing the NAVSIM evaluation harness.
- Environment: dedicated conda env `navsim_eval` (Python 3.9), with
  `nuplan-devkit`, `navsim` (editable install), and AutoVLA's inference stack
  (`transformers==4.49.0` — see Key Issues below).
- Evaluation invoked via `navsim/navsim/planning/script/run_pdm_score_cot.py`
  with a custom `train_test_split` (`b2_mini_subset`) and `scene_filter`
  pointing at the two downloaded logs.

## Key Technical Issues Resolved

1. **Relative-path resolution** — config/codebook paths resolved incorrectly
   depending on which directory the script was launched from; fixed by
   always launching from the AutoVLA repo root.
2. **Missing dependencies** (`peft`, `lmdb`) — installed; an unconditional
   top-level import of a `tensorflow`-dependent Waymo dataset module (never
   exercised by this pipeline) was patched out rather than installing the
   heavy dependency.
3. **CUDA OOM on checkpoint load** — `torch.load(..., map_location=self.autovla.device)`
   loaded the full checkpoint onto an already near-full GPU, stacking a second
   memory peak on top of the resident model. Fixed by loading to CPU first
   (`map_location="cpu"`); `load_state_dict` then copies tensors to the
   model's existing device.
4. **Silent checkpoint/architecture mismatch (root cause of inference failure)** —
   the model generated fluent CoT text but never emitted any action tokens,
   causing a downstream `IndexError`/`TypeError` when decoding an empty action
   tensor. The visible errors were misleading; the actual cause was verified
   by inspecting `load_state_dict(..., strict=False)`'s `missing_keys` /
   `unexpected_keys`, which showed 824 unmatched keys of the form
   `vlm.visual.*` vs. `vlm.model.visual.*`. This is a Qwen2.5-VL internal
   module path introduced between `transformers` 4.49 and 4.57; the
   checkpoint was saved under the older path structure. Pinning
   `transformers==4.49.0` (as specified in AutoVLA's own `requirements.txt`)
   resolved it — after the fix, `missing_keys` and `unexpected_keys` were
   both empty, and all 138 scenarios produced well-formed structured output.

## Results

138 / 138 scenarios evaluated successfully, 0 failures.

| Agent | PDM-Score |
|---|---|
| Human upper bound | 0.862 |
| CV baseline (rule-based, no learning) | 0.699 |
| **AutoVLA (A2)** | **0.639** |

## Interpretation

The original hypothesis was that AutoVLA would show a "good open-loop /
poor closed-loop" divergence. The actual result does not show a divergence:
AutoVLA's closed-loop PDMS (0.639) is **below the non-learned CV baseline**
(0.699), which is directionally consistent with its worst-in-B1 open-loop L2
(1.420 m). Both metrics point the same way — this checkpoint underperforms
on this particular mini-subset, rather than "looking good open-loop but
failing closed-loop."

**Caveat**: the checkpoint's name references a PDMS of ~0.89 presumably
measured on the full official NAVSIM `navtest` split. The 0.639 here is
measured on a 138-scenario subset drawn from only 2 logs — a much smaller
and narrower sample. The gap could reflect genuine model weakness, or could
partly reflect the mini-subset's scenario distribution; this evaluation does
not distinguish between the two, and any generalization beyond this subset
should account for that.

## Repository Contents

```
source_code/
  autovla_agent.py            # NAVSIM agent wrapper (checkpoint-load fix applied)
  nocot_sample_generation.py  # preprocessing script (waymo import removed)
  config/
    qwen2.5-vl-7B-nuplan-b2-mini.yaml   # dataset config for preprocessing
    b2_mini_subset.yaml                  # train_test_split config
    b2_mini_subset_scene_filter.yaml     # scene_filter (log_names for shard 0)
results/
  autovla_agent_138scenarios.csv   # per-scenario PDM-Score breakdown
  run_log.txt                      # trimmed evaluation run log
README.md
```

Raw sensor data, model checkpoints, and the metric cache are excluded from
this backup (large binary artifacts, not source-controlled).

## Reproduction

```bash
cd AutoVLA
export PYTHONPATH=$(pwd):$PYTHONPATH
conda activate navsim_eval

CUDA_VISIBLE_DEVICES=0 python navsim/navsim/planning/script/run_pdm_score_cot.py \
  train_test_split=b2_mini_subset \
  agent=autovla_agent \
  +agent.config_path="<path>/qwen2.5-vl-3B-nuplan-grpo-cot.yaml" \
  +agent.checkpoint_path="<path>/AutoVLA_PDMS_89.ckpt" \
  +agent.sensor_data_path="<path>/mini_sensor_blobs/mini" \
  +agent.lora_conf.use_lora=false \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  json_data_path="<path>/autovla_json_input" \
  output_dir=$NAVSIM_EXP_ROOT/autovla_agent \
  experiment_name=autovla_agent
```

Requires: `transformers==4.49.0` pinned (see Key Technical Issues #4).
