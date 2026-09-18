import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
import functools
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig, TaskType

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Model, dataset, and dataloader
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Get using_cot setting from config (default to True if not specified)
    using_cot = config['model']['use_cot']
    

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)
        
    # Randomly sample from training set if train_sample_size is specified
    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print("no sampling")
        
    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    model = SFTAutoVLA(config)
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
    
    # Create data collator with config parameters
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id']
    )
    
    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['inference']['num_workers'],
        shuffle=False,
    )    

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"runs/sft/{current_date}"
    
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        strategy="auto",
        precision="bf16-true",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                dirpath=f"{save_dir}",
                filename="epoch={epoch}-loss={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_weights_only=True,
                every_n_epochs=1,
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,

        logger=CSVLogger(save_dir=f"{save_dir}"),
        enable_model_summary=True,

        # limit_val_batches=0.001
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)