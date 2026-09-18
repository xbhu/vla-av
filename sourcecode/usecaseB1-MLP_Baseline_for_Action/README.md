# Use Case B1: Ego-State MLP Baseline (Exposing Open-Loop Evaluation Limitations)

## Motivation

Open-loop L2 has been the primary metric across the A-series (A1: OpenDriveVLA-0.5B
inference, A2: AutoVLA fast-eval, A3: LoRA memorization check), but a background
assumption never directly tested: **open-loop L2 may reward GT-mimicry rather than
genuine scene understanding** ("open-loop hallucination"). B1 tests this by comparing
VLA models against two zero/low-learning ego-state-only baselines on the same
evaluation set.

## Method

Two reference points, both blind to camera input:

- **Tier 0 (zero-learning)**: closed-form kinematic extrapolation from the current
  frame's scalar velocity/acceleration. Two variants: constant velocity, constant
  acceleration.
- **Tier 1 (light learning)**: a small MLP (3 hidden layers, mirrors navsim's
  `EgoStatusMLPAgent` architecture) trained on ego status only (velocity,
  acceleration, one-hot driving instruction — 5-dim input, no visual tokens, no
  multi-frame history — the preprocessed data has neither).

Compared against the existing A-series VLA models: A1 (OpenDriveVLA-0.5B) and A2
(AutoVLA).

## Data

- Train: 164 samples in `datasets/autovla_nusc_preprocessed_train` (162 used after
  dropping 2 flagged as degenerate GT by a proxy filter — the train split has no
  `future_mask` field, unlike val, so degenerate-GT detection uses a spread-based
  proxy instead of the official mask).
- Val: 126 samples in `datasets/autovla_nusc_preprocessed_val_filtered` (same set
  used by A2/A3, GT-degenerate samples already excluded).
- Eval: reuses `PlanningMetric` from `AutoVLA/tools/eval/planning_metrics.py`
  verbatim (not reimplemented), same coordinate transform and future_mask
  consistency check as `nusc_eval_a2.py`. Both STP-3 (cumulative average) and UniAD
  (per-timestep) conventions are reported.

## Important note on sample-count alignment across models

This comparison is **not perfectly apples-to-apples** across all five rows, and that
is stated explicitly rather than glossed over:

| Model | n | Alignment with our 126-sample val_filtered set |
|---|---|---|
| Tier0 (both variants) | 126 | Exact |
| Tier1 MLP | 126 | Exact |
| A1 OpenDriveVLA-0.5B | 122 | Re-scored from A1's cached `pred_trajs_dict.json`; 4/126 tokens are absent from that cache and excluded |
| A2 AutoVLA | 150 | Uses A2's original 150-sample eval run, which is a strict superset of our 126 (confirmed `val_filtered ⊂ val`, i.e. all 126 are among the 150) but was NOT re-scored on exactly 126 |

A1 and Tier0/Tier1 numbers are on close-to-identical, precisely stated sample sets
and are directly comparable. The A2 number is comparable in spirit but not in exact
sample composition — flagged so future readers don't over-index on small differences
against it.

## A pipeline bug worth documenting (and how it was caught)

Re-scoring A1's cached predictions initially produced implausible L2 values (3.9m at
0.5s, 23.7m at 3.0s — worse than a stationary-vehicle guess). Root cause: A1's cached
`pred_trajs_dict.json` already stores predictions in UniAD-coordinate frame
(`(x,y) -> (-y,x)`), not the raw ego-relative frame our eval pipeline expects and
transforms internally. Feeding it through `evaluate_predictions()` applied the
transform a second time. Fixed by inverse-transforming A1's cached predictions back
to raw frame before evaluation; verified via a GT-vs-GT smoke test (L2 must be ~0)
run before any real numbers were trusted, and via direct numeric comparison against
a known-good A2-series coordinate-frame diagnostic script.

## Results

### STP-3 convention (cumulative average L2 / collision %, by horizon)

| Model | n | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|---|---|---|---|---|---|---|---|
| Tier0 const-velocity | 126 | 0.136 | 0.259 | 0.419 | 0.614 | 0.844 | 1.106 |
| Tier0 const-accel | 126 | 0.129 | 0.238 | 0.386 | 0.574 | 0.805 | 1.077 |
| Tier1 Ego-Status MLP | 126 | 0.206 | 0.366 | 0.584 | 0.848 | 1.162 | 1.514 |
| A1 OpenDriveVLA-0.5B | 122 | 0.060 | 0.117 | 0.199 | 0.310 | 0.446 | 0.608 |
| A2 AutoVLA | 150 | 0.207 | 0.349 | 0.531 | 0.769 | 1.065 | 1.420 |

(Collision rates, UniAD per-timestep numbers, and full per-model JSON are in this
directory: `usecaseB1_tier0_*_eval.json`, `usecaseB1_tier1_mlp_eval.json`,
`usecaseB1_a1_rescored_on_filtered_eval_v2.json`.)

## Findings

1. **The open-loop-hallucination hypothesis is NOT supported by this experiment.**
   A1 clearly outperforms both zero-learning kinematic baselines (0.61m vs ~1.1m at
   3.0s) — nearly half the error. This suggests OpenDriveVLA-0.5B is extracting
   real predictive signal beyond naive physical extrapolation, at least on this
   evaluation set.

2. **The genuine negative finding is Tier1, not the VLA models.** The ego-status
   MLP underperforms both zero-learning baselines despite having strictly more
   information (learned weights vs a closed-form formula). Diagnosed and confirmed
   this is not underfitting: train-set L2 (0.99m) is in the same range as the
   kinematic baselines' val-set L2, and a sensitivity check (smaller hidden dim,
   higher LR, 5000 epochs) only marginally improved train fit (0.99m → 0.89m) with
   visible loss-curve plateauing. This is a real train→val generalization gap
   (~52%) under small-sample (162 examples), low-dimensional (5-feature) training —
   a finding about small-sample learning fragility, not about VLA models or
   open-loop evaluation per se.

3. **A2 underperforms A1 and sits in the same range as Tier0** (1.42m vs A1's
   0.61m), which is closer to the original hypothesis — but given the sample-count
   caveat above (150 vs 126), this comparison is weaker evidence and should not be
   over-interpreted.

## What would sharpen this further

The current comparison averages over all val samples regardless of maneuver
difficulty. A more precise test of the open-loop-hallucination hypothesis would
stratify by scenario type (e.g. straight/near-constant-velocity vs turning/braking
scenes) and check whether A1's advantage over Tier0 is concentrated in the harder,
more scene-dependent cases or spread uniformly — uniform spread would be weaker
evidence of genuine scene understanding than concentrated advantage in hard cases.
Not done in this pass; noted as a possible follow-up (B2 or a B1 extension) rather
than assumed.

## Files

- `sourcecode/clusterB_openloop_limitations/b1_setup/` — all scripts (common utils,
  smoke test, Tier0, Tier1 training, Tier1 diagnostics, A1 re-scoring + coordinate
  frame fix)
- `outputs/usecaseB1/` — predictions, eval JSON results, checkpoint, setup logs
