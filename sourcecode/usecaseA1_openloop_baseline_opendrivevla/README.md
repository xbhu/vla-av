# VLA Series — Use Case A1: OpenDriveVLA-0.5B Baseline Inference

## Purpose

Run baseline open-loop inference with OpenDriveVLA-0.5B on nuScenes data to:
1. Establish and verify the model's I/O contract (structured 3D perception tokens in, coordinate-serialized trajectory text out)
2. Obtain quantitative open-loop planning metrics (L2 distance, collision rate) as a reference point for later VLA use cases (A2, B1, B2)
3. Build hands-on understanding of why open-loop evaluation is structurally limited before using it to interpret any VLA/VLM planning results

## Model & Data

- **Model:** OpenDriveVLA-0.5B (Qwen2.5-0.5B-Instruct backbone), AAAI 2026, arXiv 2503.23463
- **Perception front-end:** mmdet3d/BEVFormer-based 3D structured tokens (scene/agent/map query tokens from 6-camera images), not raw image patches
- **Action representation:** coordinate serialization — 6 future waypoints (3s horizon, 0.5s interval) generated autoregressively as text, e.g. `[(0.12,4.17),(0.35,8.36),...]`
- **Dataset:** nuScenes-mini (10 scenes, 404 samples originally)
- **Effective evaluation set:** 146 val samples after two filtering passes:
  1. Filtered to scene/sample tokens present in nuScenes-mini (162 of 6019 full-trainval-val samples overlap)
  2. Further filtered to remove scene-start frames lacking sufficient temporal history — BEVFormer requires queue_length=5 (4 historical frames); frames with frame_idx < 4 were dropped, leaving 146 usable val samples (and 218 train samples, unused in A1)

## Experimental Design

- Full inference pipeline: 6-camera images → mmdet3d 3D perception tokens → Qwen2.5-0.5B LLM → coordinate-serialized trajectory output
- Evaluated under two independent scoring conventions found in the official eval_share codebase — UniAD and STP-3 — which differ in metric computation details (frame inclusion, normalization)
- Ground truth: human-driven trajectories from nuScenes (not any AV-controlled trajectory — none exists, since nuScenes is a pure human-driving recording dataset)

## Results

| Metric | UniAD convention | STP-3 convention |
|---|---|---|
| L2 @ 1s | 0.17m | 0.12m |
| L2 @ 2s | 0.58m | 0.29m |
| L2 @ 3s | 1.18m | 0.53m |
| L2 avg | 0.64m | 0.31m |
| Collision @ 3s | 2.74% | 0.80% |
| Collision avg | 1.14% | 0.32% |

STP-3 L2 avg (0.31m) closely matches the paper's reported 0.33m on the full nuScenes val set (150 scenes), despite this run using only 146 mini-subset samples — suggesting the mini subset's scene distribution is not meaningfully harder or easier than the full val set for this metric, or possibly that it's more homogeneous (worth a note, not a firm conclusion).

## Key Conclusions

1. **Pipeline correctness confirmed.** Successfully reproducing a metric value close to the published number validates the environment, model loading, and evaluation logic — not just "a number came out," but a number consistent with the literature.
2. **I/O contract confirmed empirically**, not just theoretically: input is a text-templated prompt combining ego kinematic state (velocity, acceleration, yaw rate, CAN bus signals), 2-second historical trajectory, and a mission goal string ("keep forward" etc.); output is coordinate-serialized text, matching the Phase 0/1 conceptual predictions.
3. **Open-loop L2's structural limitation directly discussed and internalized**, not just accepted as textbook fact: L2 measures similarity to one specific human driving instance, not decision quality. A model can score well by mimicking a human's idiosyncratic choice (e.g., slightly right-biased driving) without understanding why that choice was made, or whether it was the only reasonable one. This motivates both B1 (exposing the open-loop illusion with a trivial ego-state-only baseline) and B2 (NAVSIM pseudo-closed-loop evaluation).
4. **No comparison run yet** against a general-purpose VLM (e.g., directly prompting Qwen2.5-VL with raw images) — this remains an open, not-yet-executed question about whether OpenDriveVLA's structured 3D front-end outperforms a naive VLM baseline on the same metric.

## Infrastructure Notes (high signal-to-noise only)

- Required a dedicated conda env (`drivevla`, Python 3.10, PyTorch 2.1.2+cu121) separate from other VLA/VLM environments due to mmcv 1.7.2 / mmdet3d 1.0.0rc6 binary compatibility requiring an older PyTorch version
- Several Shapely 2.0 breaking changes required patching OpenDriveVLA's map processing code (`MultiPolygon`/`MultiLineString` no longer directly iterable — must use `.geoms`)
- The repo contains two `eval_share/` directories; only `drivevla/eval_share/` is actually used at runtime
- The official eval script's `subset` parameter was present but not wired into the evaluation loop — had to patch it to allow evaluating on the mini subset instead of assuming the full 6019-sample val set
