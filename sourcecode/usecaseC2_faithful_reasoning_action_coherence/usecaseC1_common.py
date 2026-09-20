"""
Cluster C1 (Sensitivity/Perturbation Battery) shared utility module.
Does not modify sourcecode/AutoVLA/models/autovla.py itself -- instead,
predict_c1()/get_prompt_c1() are added via a subclass of AutoVLA, so the
B2 production code path is completely unaffected.
"""

import os
import json
import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from models.autovla import AutoVLA
from navsim.agents.autovla_agent import AutoVLAAgent, AutoVLAAgentFeatureBuilder
from qwen_vl_utils import process_vision_info

INSTRUCTION_CLASSES = ["keep forward", "turn left", "turn right"]

FORCED_NO_COT_PREFIX = (
    "<think>\n"
    "This is a straightforward scenario, and a direct decision can be made.\n"
    "</think>\n"
    "<answer>\n"
    "The final output action is: "
)


class AutoVLA_C1(AutoVLA):

    def get_prompt_c1(self, input_features, camera_mask=None):
        images = input_features['images']
        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        camera_mask = camera_mask or []

        camera_images = {}
        for camera_type in camera_types:
            camera_images[camera_type] = []
            for i in range(4):
                img = images[camera_type][i]
                if input_features['sensor_data_path']:
                    path = os.path.join(input_features['sensor_data_path'], img)
                else:
                    path = img

                if camera_type in camera_mask:
                    path = get_blank_frame_path(path)

                camera_images[camera_type].append(path)

        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']

        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, (list, np.ndarray)):
            velocity = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, (list, np.ndarray)):
            acceleration = np.sqrt(acceleration[0] ** 2 + acceleration[1] ** 2)

        instruction = input_features["driving_command"].lower()

        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
            {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_camera_1}", f"file://{front_camera_2}", f"file://{front_camera_3}", f"file://{front_camera_4}"]},
            {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_left_camera_1}", f"file://{front_left_camera_2}", f"file://{front_left_camera_3}", f"file://{front_left_camera_4}"]},
            {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
            {"type": "video", "min_pixels": min_pixels, "max_pixels": max_pixels,
             "video": [f"file://{front_right_camera_1}", f"file://{front_right_camera_2}", f"file://{front_right_camera_3}", f"file://{front_right_camera_4}"]},
            {"type": "text", "text": (
                f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s\u00b2. "
                f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
            )},
        ]

        if self.use_cot:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will receive visual observations from the ego vehicle\u2019s cameras and dynamic information about the vehicle\u2019s current state. "
                "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                "Structure your reasoning as follows:\n"
                "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                "Present the final action clearly after your reasoning steps."
            )
        else:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle\u2019s surrounding cameras, along with the vehicle\u2019s current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        return text, image_inputs, video_inputs

    def predict_c1(self, input_features, forced_cot_prefix=None, camera_mask=None, num_poses=10, greedy=False):
        text, image_inputs, video_inputs = self.get_prompt_c1(input_features, camera_mask=camera_mask)

        if forced_cot_prefix:
            text = text + forced_cot_prefix

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        if greedy:
            # Deterministic decoding: no temperature/top_k/top_p sampling.
            # Used by C2 (reasoning-action coherence) so that repeated calls
            # on the same input are reproducible, isolating "does the stated
            # decision match the decoded action" from sampling noise.
            generate_kwargs = dict(
                max_length=self.gen_conf['max_length'],
                do_sample=False,
            )
        else:
            generate_kwargs = dict(
                max_length=self.gen_conf['max_length'],
                do_sample=True,
                temperature=self.gen_conf['temperature'],
                top_k=self.gen_conf['top_k'],
                top_p=self.gen_conf['top_p'],
            )

        outputs = self.vlm.generate(
            **model_inputs,
            **generate_kwargs,
        )

        outputs_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        outputs_trimmed = outputs_trimmed[0][:-1].cpu()
        cot_results = self.processor.decode(outputs_trimmed)
        if forced_cot_prefix:
            cot_results = forced_cot_prefix + cot_results

        actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]

        if len(actions_tokens) > num_poses:
            actions_tokens = actions_tokens[:num_poses]
        elif len(actions_tokens) < num_poses:
            pad = torch.zeros(num_poses - len(actions_tokens), dtype=actions_tokens.dtype)
            actions_tokens = torch.cat([actions_tokens, pad])

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)[0, 1:]
        return trajectory, cot_results


_BLANK_FRAME_CACHE = {}

def get_blank_frame_path(reference_img_path, mode="black", cache_dir="/tmp/c1_blank_frames"):
    os.makedirs(cache_dir, exist_ok=True)
    key = (reference_img_path, mode)
    if key in _BLANK_FRAME_CACHE:
        return _BLANK_FRAME_CACHE[key]

    with Image.open(reference_img_path) as im:
        w, h = im.size

    out_path = os.path.join(cache_dir, f"{mode}_{w}x{h}.jpg")
    if not os.path.exists(out_path):
        if mode == "black":
            arr = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(out_path, quality=95)

    _BLANK_FRAME_CACHE[key] = out_path
    return out_path


def load_c1_scenes(json_data_path):
    scene_files = sorted(glob.glob(os.path.join(json_data_path, "*.json")))
    scenes = []
    for f in scene_files:
        with open(f, "r") as fh:
            scene_data = json.load(fh)
        scenes.append((Path(f).stem, scene_data))
    return scenes


def build_c1_agent(config_path, checkpoint_path, sensor_data_path, codebook_cache_path, device='cuda'):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    trajectory_sampling = TrajectorySampling(time_horizon=5, interval_length=0.5)

    agent = AutoVLAAgent(
        trajectory_sampling=trajectory_sampling,
        checkpoint_path=checkpoint_path,
        sensor_data_path=sensor_data_path,
        codebook_cache_path=codebook_cache_path,
        lora_conf={"use_lora": False},
        config_path=config_path,
        device=device,
    )
    agent.initialize()
    agent.autovla.__class__ = AutoVLA_C1
    return agent


def trajectory_to_xy(trajectory):
    if torch.is_tensor(trajectory):
        trajectory = trajectory.cpu().numpy()
    return np.asarray(trajectory)[:, :2]

def endpoint_l2(traj_a, traj_b):
    a, b = trajectory_to_xy(traj_a), trajectory_to_xy(traj_b)
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[n - 1] - b[n - 1]))

def mean_pointwise_l2(traj_a, traj_b):
    a, b = trajectory_to_xy(traj_a), trajectory_to_xy(traj_b)
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n], axis=-1).mean())
