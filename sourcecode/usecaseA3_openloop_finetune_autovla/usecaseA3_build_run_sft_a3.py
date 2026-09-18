"""
Build tools/run_sft_a3.py from the official tools/run_sft.py by inserting:
  1. Checkpoint loading from AutoVLA_PDMS_89.ckpt (continuing from PDMS-optimized weights,
     not training from scratch), following the proven load order from run_rft.py.
  2. LoRA wrapping of model.autovla.vlm, applied AFTER checkpoint loading (order matters:
     wrapping before loading risks strict=False silently skipping base weights, same class
     of bug as A2's checkpoint key-prefix issue).
  3. Single-GPU strategy in place of FSDPStrategy, since A3 runs on a single RTX A2000 12GB,
     not a multi-node cluster.

Run this once from the AutoVLA repo root:
    python3 <path-to-this-file>
Output: tools/run_sft_a3.py
Does not modify tools/run_sft.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/home/xzh5180/Research/vla-av/sourcecode/AutoVLA")
SRC = REPO_ROOT / "tools" / "run_sft.py"
DST = REPO_ROOT / "tools" / "run_sft_a3.py"

content = SRC.read_text()

# --- Patch 1: add peft import ---
old_import = "import pytorch_lightning as pl\n"
new_import = "import pytorch_lightning as pl\nfrom peft import get_peft_model, LoraConfig, TaskType\n"
if content.count(old_import) != 1:
    print(f"ERROR: expected exactly 1 occurrence of import line, found {content.count(old_import)}. Aborting, no file written.")
    sys.exit(1)
content = content.replace(old_import, new_import)

# --- Patch 2: checkpoint loading + LoRA wrapping (gradient_checkpointing_enable line preserved) ---
old_ckpt_block = (
    "    model = SFTAutoVLA(config)\n"
    "    model.autovla.vlm.model.gradient_checkpointing_enable() # enable gradient checkpointing to save memory\n"
    "\n"
    "    # checkpoint_path = Path(\".ckpt\")\n"
    "    # state_dict = torch.load(checkpoint_path)['state_dict']\n"
    "    # model.load_state_dict(state_dict)\n"
)
new_ckpt_block = '''    model = SFTAutoVLA(config)
    model.autovla.vlm.model.gradient_checkpointing_enable() # enable gradient checkpointing to save memory

    # A3: continue fine-tuning from the PDMS-optimized checkpoint rather than training
    # from scratch. Loading BEFORE LoRA wrapping, matching the proven working order in
    # run_rft.py. strict=False is required because LoRA/PEFT bookkeeping keys are not in
    # this checkpoint, but the missing/unexpected key counts are printed explicitly so a
    # silent load failure (A2's original bug) cannot pass unnoticed here.
    print(f"Loading and remapping checkpoint from: {config['model']['sft_model_path']}")
    full_checkpoint = torch.load(config['model']['sft_model_path'], map_location="cpu")
    sd = full_checkpoint['state_dict']
    msg = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint load report: missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"First 10 missing keys: {msg.missing_keys[:10]}")
    if len(msg.unexpected_keys) > 0:
        print(f"First 10 unexpected keys: {msg.unexpected_keys[:10]}")
    assert len(msg.missing_keys) < 50, "Too many missing keys - checkpoint likely did not load correctly, aborting before wasting a training run."

    if config['model'].get('lora', {}).get("use", False):
        print("Using LoRA mode for A3 SFT fine-tuning.")
        lora_conf = config['model']['lora']
        lora_config = LoraConfig(
            task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
            target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            r=lora_conf.get("r", 8),
            lora_alpha=lora_conf.get("alpha", 8),
            lora_dropout=lora_conf.get("dropout", 0.1),
            bias=lora_conf.get("bias", "none")
        )
        model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
        trainable_params = sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.autovla.vlm.parameters())
        print(f"LoRA-enabled model trainable parameters: {trainable_params} / {total_params} ({100.0 * trainable_params / total_params:.4f}%)")
    model = model.to(torch.bfloat16)
'''
if content.count(old_ckpt_block) != 1:
    print(f"ERROR: expected exactly 1 occurrence of checkpoint block, found {content.count(old_ckpt_block)}. Aborting, no file written.")
    sys.exit(1)
content = content.replace(old_ckpt_block, new_ckpt_block)

# --- Patch 3: replace FSDPStrategy (multi-node) with single-GPU strategy ---
old_strategy_block = '''        devices='auto',
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy=FSDPStrategy(
            auto_wrap_policy=wrap_policy,
            cpu_offload=False,
            # Mixed precision training
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            ),
            # sharding strategy
            sharding_strategy='FULL_SHARD',
            # prefetching backward computation
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            # save state dict type
            state_dict_type="full", # can be full or sharded
            limit_all_gathers=True, # limit all_gathers to save memory
        ),'''
new_strategy_block = '''        devices=1,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy="auto",
        precision="bf16-true",'''
if content.count(old_strategy_block) != 1:
    print(f"ERROR: expected exactly 1 occurrence of strategy block, found {content.count(old_strategy_block)}. Aborting, no file written.")
    sys.exit(1)
content = content.replace(old_strategy_block, new_strategy_block)

DST.write_text(content)
print(f"Wrote {DST}")
print("Diff summary: added peft import, added checkpoint-load+LoRA block after model init (gradient checkpointing line preserved), replaced FSDPStrategy with single-GPU bf16-true strategy.")
