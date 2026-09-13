# Use Case A2 — AutoVLA Action Codebook: Learning Summary

Part of a self-directed VLA (Vision-Language-Action) learning series for
transportation/autonomous-driving research. Prior use cases: LLM series
(UC1-8, EV charging), VLM series (UC1-10, DriveLM-nuScenes), VLA A1
(OpenDriveVLA-0.5B baseline inference on nuScenes-mini).

**Goal of this write-up**: this was a hands-on learning exercise, not a
push for a publication-grade benchmark number. The point was to
understand AutoVLA's action-tokenization design well enough to run it
myself, and to build debugging fluency with VLA-specific failure modes.
The numbers below are informative but carry known caveats (documented
throughout) and were not pursued to full statistical rigor.

## 1. What A2 set out to do

A1 used OpenDriveVLA, which serializes future waypoints as text
(autoregressive coordinate generation). A2 switches to AutoVLA, which
instead discretizes future motion into a fixed vocabulary of 2048
"action tokens" (a *codebook*) built via K-disk clustering on real and
simulated driving trajectories. The goal was to run AutoVLA's
fast-thinking (non-CoT) inference on the same nuScenes-mini validation
scenes used in A1, and compare L2 / collision-rate results.

## 2. Core concept: what the action codebook actually does

Each action token represents one 0.5s segment of motion: a
`(Δx, Δy, Δθ)` triple. A 5-second planning horizon becomes a sequence
of 10 tokens. Instead of asking a language model to generate precise
floating-point coordinates character-by-character (fragile, slow, prone
to physically-infeasible outputs), the model just picks from a fixed
vocabulary of pre-clustered, physically-plausible motion primitives —
the same mechanism a language model uses to pick the next word. This is
why AutoVLA's paper reports roughly 2x faster inference and lower L2
than text-based waypoint generation on their own ablation.

AutoVLA also supports a fast/slow "dual thinking" mode (a fixed short
template vs. full chain-of-thought reasoning before emitting action
tokens), trained via SFT then further refined with GRPO-based
reinforcement fine-tuning (RFT) that penalizes unnecessary reasoning
length. A2 only exercised fast-thinking mode; slow-thinking (CoT) was
scoped out as lower priority for this pass.

## 3. Environment / infra notes worth remembering

- AutoVLA needs its own conda env (`autovla_codeclean`, Python 3.9,
  PyTorch 2.4.0), **not** the `vla-av` env used for other VLA work —
  the version mismatch wasn't obvious until actually reading
  `environment.yml`.
- `flash-attn` is listed in `install.sh` but is **not required** —
  the model loads Qwen2.5-VL without `attn_implementation="flash_attention_2"`
  set, so it silently falls back to `sdpa`. Not worth fighting a CUDA
  toolkit install for it on a workstation GPU.
- The official checkpoint (`AutoVLA_PDMS_89.ckpt`, HuggingFace
  `Zewei-Zhou/AutoVLA`) is the NAVSIM/PDMS-optimized, post-RFT
  checkpoint — evaluating it on nuScenes is an out-of-domain
  generalization test, not an apples-to-apples comparison with a
  nuScenes-finetuned model like A1's OpenDriveVLA-0.5B.

## 4. Two real bugs found in the official `nusc_eval.py`

These are documented in more detail in `results/usecaseA2_bugfix_notes.md`
and fixed in `patched_code/nusc_eval_a2.py` (kept as a separate copy;
the vendored repo file itself was left untouched).

**Bug 1 — checkpoint load OOM on a 12GB GPU.** The script did
`torch.load(checkpoint_path, map_location=args.device)`, loading the
full 16.3GB checkpoint file directly onto the GPU on top of the already-
loaded ~6GB bf16 model. Fix: load to CPU first, let `load_state_dict`
copy only the matching tensors onto the GPU-resident model.

**Bug 2 — checkpoint weights silently never loaded.** The script called
`model.autovla.load_state_dict(state_dict, strict=False)`, but the
released checkpoint's keys are all prefixed with `autovla.` (it was
saved from the outer LightningModule, not the inner submodule). This
produced 825/825 keys missing *and* 825/825 unexpected — every single
key mismatched — and because `strict=False` was set, this failed with
**no error at all**. The model ran with randomly-initialized
action-token embeddings and never emitted a single action token, only
prose. This is a genuinely dangerous class of bug: it doesn't crash, it
just quietly produces garbage. It only surfaced downstream as an
unrelated-looking `TypeError` when the (empty) action-token list hit a
tensor-indexing call. Fix: load onto `model` instead of `model.autovla`.

## 5. Final results (fast-thinking mode, nuScenes-mini val)

150 samples (10 val scenes filtered to 2 available in nuScenes-mini:
`scene-0103`, `scene-0916`; 404 raw frames → 150 after AutoVLA's own
history/future-frame sufficiency filtering).

| Metric | STP3 avg | UniAD avg |
|---|---|---|
| L2 (AutoVLA, this run) | 0.77 m | 1.50 m |
| L2 (A1 / OpenDriveVLA-0.5B, 146 samples) | 0.31 m | 0.64 m |
| Collision `obj_box_col` (AutoVLA) | 0.41% | 1.78% |
| Collision (A1) | 0.32% | 1.14% |

**Read this with the caveats below — do not cite these as clean,
final numbers.**

- **L2 is clearly and consistently worse than A1** (roughly 2.3-2.5x)
  across both scoring conventions. This is a stable, repeated finding,
  not a fluke of one run.
- **Collision rate is roughly comparable to A1**, not systematically
  worse — the gap is much smaller than for L2.
- Sample sets are not directly comparable: A1 (146 samples) used a
  model fine-tuned specifically on nuScenes; A2 (150 samples) used a
  checkpoint optimized for a different benchmark (NAVSIM/PDMS) and is
  being evaluated out-of-domain here.
- **A known, unfixed scoring artifact exists** in AutoVLA's own
  `PlanningMetric.compute()`: the denominator (`total`) counts samples,
  not valid-timesteps-per-horizon. Scenes ending early get some future
  timesteps zero-masked (correctly excluded from the numerator via
  `gt_trajs_mask`), but those samples still count fully in the
  denominator, which pulls the averaged long-horizon L2 *down*
  (optimistically) for samples near a scene boundary. This means the
  0.77m/1.50m numbers above are, if anything, **flattering** to
  AutoVLA — the true gap vs. A1 is likely somewhat larger, not smaller.
  This was not fixed (see Section 7).

## 6. Debugging journey — two mistakes worth remembering

**Mistake 1: proposed a coordinate-frame mismatch hypothesis that was
mathematically impossible.** When a handful of samples showed 15-35m
L2 errors, I guessed the predicted and ground-truth trajectories were
being compared in different coordinate frames. The "fix" was to check
L2 before/after applying the official axis-swap + sign-flip transform.
The transform is an isometry — it cannot change Euclidean distance by
definition — so this check could never have shown a difference, and in
fact didn't. Lesson: check whether a hypothesis is even mathematically
possible before spending a debugging cycle testing it.

**Mistake 2: over-aggressive filtering based on an incomplete
signal.** The real cause of those outliers was found by inspecting
`future_mask`: some val samples are near the end of a scene and have
their future trajectory padded by repeating the last known position
(AutoVLA's preprocessing only guards against this on the train split,
not val). The first fix filtered out every sample with
`future_mask_sum_3s < 6` — but this conflated two different cases:
vehicles that were genuinely moving and got their future incorrectly
frozen (real data defect), versus vehicles that were already stationary
and *correctly* show no future movement (valid, easy samples). Removing
both indiscriminately stripped out a disproportionate share of "easy"
samples, and the filtered re-run's average L2 got *worse*, not better —
the opposite of what a real data-quality fix should do. That surprising
result was the clue that led to finding the actual denominator bug
described in Section 5, rather than a flaw in the filter's precision.

Both mistakes share a pattern: a diagnostic script or a "fix" was
trusted before its own correctness was independently verified. The
useful habit going forward is to ask "could this tool itself be wrong?"
before accepting a surprising result as a finding about the model.

## 7. What was deliberately left undone

- **The `PlanningMetric` denominator bug (Section 5) was not fixed.**
  Doing so correctly requires per-timestep valid-sample counts rather
  than a single scalar `total`, which is a real code change to
  third-party evaluation logic, not a quick patch — better done
  carefully in a future session than rushed.
- **No formal A1-vs-A2 comparison writeup (planned "Phase 3") was
  produced.** The numbers above are the closest thing to it, with
  caveats spelled out rather than a clean side-by-side table.
- **Slow-thinking (CoT) mode was not evaluated** — descoped early as
  lower priority for this pass; the config flag (`use_cot`) and the
  fixed-template `<think>...</think>` marker for fast mode were
  observed and documented, but no CoT inference was run.
- **RT-1 / FAST tokenization baselines were not reproduced** (per
  earlier discussion, no official code exists for these in the
  AutoVLA repo; the paper's own ablation numbers are cited instead
  where relevant, not re-derived).

## 8. Files in this folder

- `patched_code/nusc_eval_a2.py` — official eval script with the two
  bugfixes applied (commented inline); the unfixed vendored copy is
  not included here (available in the upstream AutoVLA repo).
- `config/usecaseA2_nusc_fast_eval.yaml` — eval config actually used
  (fast-thinking, `use_cot: false`).
- `diagnostic_scripts/` — the scripts that found and characterized the
  real issues (per-sample L2 breakdown, outlier inspection, degenerate-
  GT detection, mask-handling check), plus the shell drivers for the
  smoke test and the two full inference runs.
- `results/` — inference logs, planning-metric tables (both the
  unfiltered 150-sample run and the filtered-but-flawed 126-sample
  run, kept for the record), and the standalone bugfix notes.

## 9. Source

Model: AutoVLA (Zhou, Cai, Zhao, Zhang, Huang, Zhou, Ma — UCLA Mobility
Lab), NeurIPS 2025. Paper: arXiv:2506.13757. Code:
github.com/ucla-mobility/AutoVLA. Checkpoint:
huggingface.co/Zewei-Zhou/AutoVLA.
