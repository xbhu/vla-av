# UC A3 — LoRA Fine-Tuning on AutoVLA: Memorization vs. Learning

## What question this answers

LLM UC6 and VLM UC1 both showed the same failure mode: fine-tuning a model on a
small, template-heavy dataset produces *memorization*, not *learning* — the
model's apparent gains don't generalize. A3 asks whether this holds for VLA's
**discrete action-codebook** representation, or whether a small (2,048-token)
action vocabulary behaves differently from a natural-language vocabulary of
tens of thousands of tokens.

This is a cross-validation of an existing finding on a new representation, not
a search for a brand-new phenomenon. The payoff is in extending the
methodology, not in a novel result: it forces a VLA-specific memorization test
(input-invariance, described below) that has no equivalent in the LLM/VLM
series, and it produces a general, reusable answer for *when small-sample
fine-tuning of a discrete generative model is likely pure memorization*.

**Bar for "understanding" this use case** (can be answered without looking at
code or numbers):
1. Is VLA's discrete codebook fine-tuning a memorization hot spot, and why?
2. Does this result reinforce the LLM/VLM finding, or does it show the finding
   needs to be qualified by representation type?
3. What concrete, actionable risk rule can a future student use when
   fine-tuning a VLA (or similar discrete generative model) on a small dataset?

## Model architecture (why the fine-tuning approach was designed this way)

AutoVLA (NeurIPS 2025, UCLA) is a Qwen2.5-VL-3B-Instruct backbone extended with
a fixed action-token vocabulary.

- **The action codebook is fixed and offline**, not an end-to-end VQ-VAE.
  It is built once via K-disk clustering over ~2 million sampled trajectory
  segments (far larger than any dataset used for A3's fine-tuning) and
  pickled to disk; it is never updated by gradient descent. Consequence: any
  memorization observed in A3 is attributable purely to the fine-tuning step,
  not inherited from codebook construction — this keeps the experiment's
  conclusion clean.
- **There is no separate action-prediction head.** The 2,048 action tokens are
  remapped onto the base LLM vocabulary's least-used token IDs, and both text
  and action tokens are predicted by the same shared embedding matrix and LM
  head via ordinary next-token prediction. This means LoRA on the standard
  attention projections (q/k/v/o_proj) is sufficient to affect action-token
  prediction — no separate module needs to be made trainable.
- **Why a small, discrete action vocabulary makes memorization easier to
  observe than in LLM/VLM**: a natural-language vocabulary has tens of
  thousands of tokens, so a small dataset's outputs almost never repeat
  literally — memorization can only be inferred indirectly, via a train/val
  gap. A 2,048-token action vocabulary, combined with driving's inherently
  low-entropy behavior distribution (mostly "go straight" / "slow follow"),
  means a memorizing model's output distribution can be measured directly:
  if the model's predictions barely change across very different visual
  inputs, that is direct evidence of memorization, not just an inference from
  a loss gap.

## Two failure modes that must be distinguished

A train/val loss gap alone conflates two different failures:

- **Surface memorization**: the model ignores the input almost entirely and
  maps toward whatever action tokens were frequent in training. Under an
  input-invariance test (see below), output would not change even under
  heavy, semantically-irrelevant perturbation.
- **Shallow shortcut learning**: the model learns a coarse statistical
  regularity that happens to hold on both train and val (e.g. "high current
  speed → output token cluster A"), without genuinely using fine-grained
  visual perception. This can produce a *small* train/val gap — looking like
  successful generalization — while the model still isn't grounded in vision.
  The distinguishing test is whether heavy/irrelevant visual perturbation
  breaks the output; if even that doesn't change predictions, it's true
  memorization, not a shortcut.

This distinction is the reason a plain train/val loss comparison is not
sufficient on its own, and is the motivation for the input-invariance test
below.

## Three independent memorization signals (as designed)

1. **Train/val loss gap** — reused directly from LLM UC6 / VLM UC1, not
   reinvented.
2. **Action-token exact-match accuracy** — a zero-tolerance metric (does the
   predicted token ID exactly equal the GT token ID), computed via the
   `action_mask` logic already present in AutoVLA's own eval code.
3. **Input-invariance test** (VLA-specific, no LLM/VLM equivalent) — run
   inference on the same val samples under (a) normal visual input, (b) mild
   pixel-level perturbation, (c) heavy/irrelevant perturbation (unrelated
   scene or pure noise), and compare the output action-token distribution
   (top-1 overlap rate, Shannon entropy) across the three conditions. A
   healthy, vision-grounded model's predictions should degrade sharply under
   (c); a memorizing model's predictions barely move.

Why three, not one: exact-match and L2 can disagree in an informative way — L2
tolerates a wrong-but-spatially-nearby token (because it decodes back to
continuous coordinates), so a memorizing model can look fine on L2 while
failing exact-match badly on val. Relying on L2 alone risks a false "good
generalization" conclusion.

**What was actually run**: only signal 1 (train/val loss gap) was completed.
Signals 2 and 3 were deliberately scoped out (see "What was cut" below) — this
is the main methodological gap left in this use case.

## Fine-tuning design decisions

- **LoRA-only, not full-parameter SFT.** The 3B backbone doesn't fit
  full-parameter fine-tuning on a single 12GB GPU, but more importantly: the
  question here is whether small-sample fine-tuning *itself* causes
  memorization, and full SFT introduces confounds (catastrophic forgetting,
  optimizer-state effects) that would muddy that signal. LoRA on
  q/k/v/o_proj (r=8, alpha=8), vision encoder fully frozen — this mirrors the
  VLM UC1 setup exactly, so the two experiments are comparable.
- **Continued from AutoVLA's PDMS-optimized checkpoint**, not trained from
  the base Qwen weights. Starting from a model that already has reasonable
  driving-action competence isolates "did small-sample fine-tuning add
  memorization" from "did the model learn basic competence at all" — starting
  from scratch would conflate the two.
- **Chain-of-thought (CoT) supervision was turned off** (`use_cot: false`).
  The official preprocessing only fills in real CoT text when a DriveLM
  annotation file is supplied; without it, CoT fields are silently empty, and
  training with `use_cot: true` against empty CoT would inject a second,
  uncontrolled learning objective. Turning it off keeps the experiment
  isolated to the one thing A3 cares about: vision → action-token prediction.
- **Real nuScenes-mini data used as-is, not an artificially inflated
  degenerate dataset.** A worse-case, hand-crafted "3-template" dataset (like
  the one that exposed memorization in VLM UC1) was considered but rejected
  for the first pass: a naturally small dataset is itself a realistic
  memorization-risk condition, and the resulting conclusion generalizes
  better to "should I fine-tune a VLA on a small real dataset" than a
  worst-case synthetic one would.

## Results

Full run: 164 train samples, 126 (degenerate-GT-filtered) val samples, 5
epochs, LoRA.

**Val loss decreased monotonically across all 5 epochs** (4.291 → 4.194 →
3.989 → 3.734 → 3.543), with no overfitting rebound. This is the main direct
evidence against pure surface memorization.

**Train loss was noisy and non-monotonic**, most likely because with only 164
samples and per-epoch shuffling, the epoch-average loss is sensitive to
whichever hard samples land in that epoch. Train-side loss alone is not
reliable evidence here; only val loss's trend is informative.

**Scene-diversity finding (the most substantive new discovery of this use
case):** cross-referencing image-path log identifiers against nuScenes'
official scene metadata showed that train and val samples come from
completely non-overlapping scenes (good — val loss improvement reflects real
generalization across scenes, not within-scene interpolation), but that the
164 train samples are drawn from only **2 of nuScenes-mini's 8 official train
scenes** — i.e., 2 continuous driving segments, not a diverse sample. This
raises memorization risk (high sample autocorrelation) in a way that pulls
against the reassuring val-loss trend; neither signal should be trusted
alone.

## Final judgment

Leaning toward **"this LoRA fine-tune is not purely surface-level
memorization"**, based on: monotonic val loss decrease + zero scene overlap
between train and val. This judgment has two explicit weaknesses:

1. Actual training-scene diversity (2 scenes) is much lower than "164 samples"
   suggests on its face, which itself raises memorization risk.
2. Loss-based evidence is coarse; the two more direct memorization signals
   (exact-match accuracy, input-invariance test) were never run — see below.

**Answering the three "understanding" questions from the top:**
1. VLA's discrete codebook fine-tuning leans toward *not* being a memorization
   hot spot — but the scene-diversity caveat above means this isn't fully
   settled.
2. This is a reinforcement of the LLM UC6/VLM UC1 finding, not a case where
   representation type changes the conclusion — no evidence emerged that VLA
   behaves fundamentally differently.
3. Actionable rule for future small-sample VLA fine-tuning: **scene/log
   diversity, not raw sample count, is the metric to watch.** Train/val loss
   trend is the cheapest first check, but is not sufficient on its own —
   pairing it with a more direct test (like input-invariance) is necessary
   for a confident conclusion, and that pairing was not completed here.

## What was cut, and why this matters for future work

Action-token exact-match accuracy and the input-invariance test were designed
specifically to catch "loss looks fine but the model actually learned a
shallow shortcut, not true vision-grounded reasoning" — a failure mode that a
bare loss comparison structurally cannot detect (see "Two failure modes"
above). They were **not run**, as a deliberate scope cut once the loss-based
evidence and scene-diversity check were in hand, not as an oversight.

This is recorded explicitly so a future revisit does not mistake "A3 is done"
for "A3 fully answered the memorization question." If a more rigorous answer
to "does VLA discrete-codebook fine-tuning follow the same memorization logic
as LLM/VLM continuous-token fine-tuning" is ever needed, these two tests are
the concrete next step, not a full redo.

## Repo layout

- `diagnostic_scripts/` — setup/diagnostic scripts (patch generator for the
  LoRA-enabled training script; the degenerate-GT val filter)
- `config/` — training configs (full run + smoke test)
- `patched_code/` — the LoRA-enabled training script, derived from AutoVLA's
  official `run_sft.py`
- `results/` — logs, the raw per-epoch metrics CSV, and this use case's
  detailed operational bugfix notes (`usecaseA3_bugfix_notes.md` — file-level
  patch/config specifics; this README is the conceptual companion, not a
  duplicate)
