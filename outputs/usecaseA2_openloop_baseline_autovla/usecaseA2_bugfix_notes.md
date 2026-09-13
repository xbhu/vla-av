# A2 — Two bugs found in AutoVLA's official nusc_eval.py, and fixes

Discovered while running Phase 0 smoke testing on nuScenes-mini. Both are
bugs in the upstream ucla-mobility/AutoVLA repo (2026/02 code release),
not caused by our environment or data setup.

## Bug 1: checkpoint loading causes CUDA OOM on 12GB GPUs
`torch.load(checkpoint_path, map_location=args.device)` loads the entire
16.3GB checkpoint file directly onto the GPU, on top of the ~6GB bf16
model already resident there -> CUDA OOM on a 12GB card.
Fix: load to CPU first (`map_location='cpu'`); `load_state_dict` then
copies only the matching tensors onto the already-GPU-resident model.

## Bug 2: state_dict key prefix mismatch -> weights silently never load
`model.autovla.load_state_dict(state_dict, strict=False)` — but the
released checkpoint's keys are all prefixed with "autovla." (it was
saved from the outer SFTAutoVLA LightningModule, not the inner AutoVLA
submodule). Loading onto `model.autovla` instead of `model` causes all
825 keys to mismatch (825 missing + 825 unexpected). Because
`strict=False` is set, this fails completely silently: the model runs
with randomly-initialized action-token embeddings and never emits a
single action token — it only ever outputs prose.
Fix: load onto `model` (the outer LightningModule) so the "autovla."
prefix matches. Verified missing=0 / unexpected=0 on the actual
checkpoint after this fix.

This bug is particularly dangerous because it does not raise any error
on its own — it only surfaces downstream, when a sample with zero action
tokens hits an unrelated-looking `TypeError: list indices must be
integers or slices, not tuple` in action_tokenizer.py. Easy to
misdiagnose as a data or environment issue instead of a weight-loading
issue.
