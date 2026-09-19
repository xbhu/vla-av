import os
import json
import argparse
import yaml
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
from transformers import AutoProcessor
from torch.utils.data import DataLoader
import shutil
from dataset_utils.preprocessing.nuplan_dataset import NuplanCoTAnnotationDataset, DataCollator as NuplanDataCollator


CAM_LIST = ['front', 'front_left', 'front_right', 
            'back', 'back_left', 'back_right', 'left', 'right']

def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def process_sample(sample, dataset_name):
    """
    Process a single sample from the dataset.
    """
    token = sample.get("token", "scene_unknown")
    
    gt_trajectory = sample.get("gt_trajectory", "")
    if torch.is_tensor(gt_trajectory):
        gt_trajectory = gt_trajectory.detach().cpu().tolist()
    his_trajectory = sample.get("his_trajectory", "")
    if torch.is_tensor(his_trajectory):
        his_trajectory = his_trajectory.detach().cpu().tolist()

    # Convert velocity and acceleration to lists if they are tensors.
    velocity = sample.get("velocity", "")
    if torch.is_tensor(velocity):
        velocity = velocity.detach().cpu().tolist()
    acceleration = sample.get("acceleration", "")
    if torch.is_tensor(acceleration):
        acceleration = acceleration.detach().cpu().tolist()
    preference_scores = sample.get("preference_scores", "")
    if torch.is_tensor(preference_scores):
        preference_scores = preference_scores.detach().cpu().tolist()
    preference_trajectories = sample.get("preference_trajectories", "") 
    if torch.is_tensor(preference_trajectories):
        preference_trajectories = preference_trajectories.detach().cpu().tolist()
    
    # Build the final output dictionary.
    result = {
        "token": token,
        "dataset_name": dataset_name,
        "cot_output": [],  # No CoT inference needed.
        "velocity": velocity,
        "acceleration": acceleration,
        "instruction": sample.get("instruction", ""),
        "gt_trajectory": gt_trajectory,
        "his_trajectory": his_trajectory
    }
    if dataset_name == "waymo":
            result["preference_scores"] = sample.get("preference_scores", "")
            result["preference_trajectories"] = sample.get("preference_trajectories", "")
    # Include camera paths.
    for side in CAM_LIST:
        key = f"{side}_camera_paths"
        result[key] = sample.get(key, [])
    
    return token, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset samples to JSON format using DataLoader for faster processing. "
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Name of the configuration file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the generated JSON files")
    parser.add_argument("--num_workers", type=int, default=32, 
                        help="Number of worker processes for the DataLoader")
    parser.add_argument("--pre_generated_dir", type=str, default=None,
                        help="Directory containing pre-generated scene JSON files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    # load pre generated tokens
    print("Collecting pre generated tokens from JSON sample files...")
    pre_generated_tokens = set()
    if args.pre_generated_dir is not None:
        if os.path.exists(args.pre_generated_dir):
            for file in os.listdir(args.pre_generated_dir):
                if file.endswith('.json'):
                    token = os.path.splitext(file)[0]
                    pre_generated_tokens.add(token)
            pregen_paths = {
                tok: os.path.join(args.pre_generated_dir, f"{tok}.json")
                for tok in pre_generated_tokens
            }
        else:
            print(f"Pre_generated_dir {args.pre_generated_dir} does not exist.")

    # Load configuration.
    config = load_config(f"./config/{args.config}.yaml")
    
    # Initialize the processor and dataset.
    processor = AutoProcessor.from_pretrained(config['pretrained_model_path'], use_fast=True)
    dataset_name = config.get("dataset_name", "")

    if dataset_name == "nuplan":
        dataset = NuplanCoTAnnotationDataset(config, processor)
        collator = NuplanDataCollator(processor)
    elif dataset_name == "waymo":
        dataset = WaymoE2ECoTAnnotationDataset(config, processor)
        collator = WaymoDataCollator(processor)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Create output directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Use DataLoader to load samples concurrently.
    data_loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, 
                             collate_fn=collator, shuffle=True)
    
    for batch in tqdm(data_loader, total=len(dataset), desc="Processing samples"):
        # Since batch_size=1, extract the single sample.
        sample = {key: batch[key][0] for key in batch}
        token, result = process_sample(sample, dataset_name)
        
        if token in pre_generated_tokens:
            continue

        output_path = os.path.join(args.output_dir, f"{token}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"All preprocessing data without CoT results have been saved in directory: {args.output_dir}")