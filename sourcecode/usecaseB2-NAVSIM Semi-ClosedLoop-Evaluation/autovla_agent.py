from typing import Any, List, Dict, Optional, Union
import torch
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Scene, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

import pickle
from typing import Dict, Tuple
import torch
import numpy as np
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

from models.autovla import AutoVLA
from transformers import AutoProcessor
from models.action_tokenizer import ActionTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import yaml

from navsim.agents.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)



class TokenProcessor(torch.nn.Module):
    def __init__(
        self,
        agent_token_file: str,
        agent_token_sampling: DictConfig,
    ) -> None:
        super(TokenProcessor, self).__init__()
  
        self.agent_token_sampling = agent_token_sampling

        self.init_agent_token(agent_token_file)
        self.n_token_agent = self.agent_token_all_veh.shape[0]

    def __call__(self, data: torch.Tensor) -> Tuple[Dict[str, Tensor]]:
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_agent

    def init_agent_token(self, agent_token_path: str) -> None:
        agent_token_data = pickle.load(open(agent_token_path, "rb"))

        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)
    
    def tokenize_agent(self, data: torch.Tensor) -> Dict[str, Tensor]:
        """
        Args: data["agent"]: Dict
            "position": [1, n_step, 2], float32
            "heading": [1, n_step], float32
            "shape": [1, 3], float32
        """
        # collate width/length, traj tokens for current batch
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj()
        
        # prepare output dict
        tokenized_agent = {
            "gt_pos_raw": data[:, :2],  # [n_step=16, 2]
            "gt_head_raw": data[:, -1],  # [n_step=16]
        }

        # get raw trajectory data
        heading = data[None, :, -1]  # [1, n_step]
        pos = data[None, :, :2]  # [1, n_step, 2]
        
        # match token with trajectory
        token_dict = self._match_agent_token(
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
        )

        # add token_dict to output dict
        tokenized_agent.update(token_dict)

        return tokenized_agent

    def _match_agent_token(
        self,
        pos: Tensor,  # [1, n_step, 2]
        heading: Tensor,  # [1, n_step]
        agent_shape: Tensor,  # [1, 2]
        token_traj: Tensor,  # [1, n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        """n_step_token=n_step//5
        n_step_token=16
 
        Returns: Dict
            # action that goes from [(0->5), (5->10), ..., (85->90)]
            "gt_idx": [1, n_step_token]
            "gt_pos": [1, n_step_token, 2]
            "gt_heading": [1, n_step_token]

            # noisy sampling for training data augmentation
            "sampled_idx": [1, n_step_token]
            "sampled_pos": [1, n_step_token, 2]
            "sampled_heading": [1, n_step_token]
        """
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_step = pos.shape[1]

        prev_pos, prev_head = torch.tensor([[0, 0]]), torch.tensor([0])
        prev_pos_sample, prev_head_sample = torch.tensor([[0, 0]]), torch.tensor([0])

        out_dict = {
            "gt_idx": [],
            "gt_pos": [],
            "gt_heading": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        for i in range(n_step): 
            # gt_contour: [1, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1) # [1, 1, 4, 2]
   
            # tokenize without sampling
            token_world_gt = transform_to_global(
                pos_local=token_traj.flatten(1, 2), # [1, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,  # [1, 2]
                head_now=prev_head, # [1]
            )[0].view(*token_traj.shape)

            token_idx_gt = torch.argmin(
                torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1), dim=-1
            )  # [1]

            # [1, 4, 2]
            token_contour_gt = token_world_gt[0, token_idx_gt]

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            prev_head = torch.arctan2(dxy[:, 1], dxy[:, 0])
            prev_pos = pos[:, i].clone()
            prev_pos = token_contour_gt.mean(1)
  
            # add to output dict
            out_dict["gt_idx"].append(token_idx_gt)
            out_dict["gt_pos"].append(prev_pos)
            out_dict["gt_heading"].append(prev_head)

            # tokenize from sampled rollout state
            if num_k == 1:  # K=1 means no sampling
                out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
                out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
                out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
            else:
                # contour: [n_agent, n_token, 4, 2], 2HZ, global coord
                token_world_sample = transform_to_global(
                    pos_local=token_traj.flatten(1, 2),  # [1, n_token*4, 2]
                    head_local=None,
                    pos_now=prev_pos_sample,  # [1, 2]
                    head_now=prev_head_sample,  # [1]
                )[0].view(*token_traj.shape)

                # dist: [1, n_token]
                dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
                topk_dists, topk_indices = torch.topk(
                    dist, num_k, dim=-1, largest=False, sorted=False
                )  # [1, K]

                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K
                token_idx_sample = topk_indices[0, _samples]
                token_contour_sample = token_world_sample[0, token_idx_sample]

                # udpate prev_pos_sample, prev_head_sample
                prev_head_sample = heading[:, i].clone()
                dxy = token_contour_sample[:, 0] - token_contour_sample[:, 3]
                prev_head_sample = torch.arctan2(dxy[:, 1], dxy[:, 0])

                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample = token_contour_sample.mean(1)

                # add to output dict
                out_dict["sampled_idx"].append(token_idx_sample)
                out_dict["sampled_pos"].append(prev_pos_sample)
                out_dict["sampled_heading"].append(prev_head_sample)

        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}

        return out_dict

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]

        return heading

    def _get_agent_shape_and_token_traj(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        agent_shape: [2]
        token_traj_all: [n_token, 6, 4, 2]
        token_traj: [n_token, 4, 2]
        """
        width = 2.0
        length = 4.8
         
        agent_shape = torch.tensor([[width, length]])
        token_traj_all = getattr(self, "agent_token_all_veh").unsqueeze(0)

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()

        return agent_shape, token_traj_all, token_traj


class AutoVLAAgentFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of AutoVLA Agent."""

    def __init__(self, sensor_data_path: Optional[str] = None):
        """Initializes the feature builder."""
        self.sensor_data_path = sensor_data_path

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vla_input_features"

    def compute_features(self, scene_data) -> Dict:
        """Inherited, see superclass."""
        command = scene_data['instruction']
        velocity = scene_data['velocity']
        acceleration = scene_data['acceleration']
        gt_trajectory = scene_data['gt_trajectory']
        history_trajectory = scene_data['gt_trajectory']
        dataset_name = scene_data['dataset_name']

        front_cam = scene_data['front_camera_paths']
        back_camera = scene_data.get('back_camera_paths')
        back_left_camera = scene_data.get('back_left_camera_paths')
        back_right_camera = scene_data.get('back_right_camera_paths')
        
        # Normalize camera names across datasets
        # NuPlan uses left/right_camera, others use front_left/right_camera
        if dataset_name == 'nuplan':
            front_left_cam = scene_data.get('left_camera_paths')
            front_right_cam = scene_data.get('right_camera_paths')
        else:
            front_left_cam = scene_data.get('front_left_camera_paths')
            front_right_cam = scene_data.get('front_right_camera_paths')

        images = {
            "front_camera": front_cam,
            "front_left_camera": front_left_cam,
            "front_right_camera": front_right_cam,
            "back_camera": back_camera,
            "back_left_camera": back_left_camera,
            "back_right_camera": back_right_camera,
        }

        features = {
            "vehicle_velocity": velocity,
            "vehicle_acceleration": acceleration,
            "driving_command": command,
            "images": images,
            "dataset_name": dataset_name,
            "gt_trajectory": gt_trajectory,
            "history_trajectory": history_trajectory,
            "sensor_data_path": self.sensor_data_path,
        }

        return features


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of AutoVLA Agent."""

    def __init__(self, trajectory_sampling: TrajectorySampling, codebook_cache_path: Optional[str] = None):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        :param codebook_cache_path: optional codebook cache path as string, defaults to None
        """
        agent_token_sampling = DictConfig({"num_k": 1, "temp": 1.0})

        self.token_processor = TokenProcessor(
            agent_token_file=codebook_cache_path,
            agent_token_sampling=agent_token_sampling,
        )
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene_data) -> Dict:
        """Inherited, see superclass."""
        gt_trajectory = np.array(scene_data['gt_trajectory'])
        future_trajectory = Trajectory(
            gt_trajectory, 
            TrajectorySampling(
                num_poses=len(gt_trajectory), 
                interval_length=self._trajectory_sampling.interval_length,
                )
            )
        traj = torch.tensor(future_trajectory.poses, dtype=torch.float32)
        tokenized_agent = self.token_processor(traj)
    
        return tokenized_agent


class AutoVLAAgent(AbstractAgent):
    """AutoVLA Agent interface."""

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        sensor_data_path: Optional[str] = None,
        codebook_cache_path: Optional[str] = None,
        lora_conf: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda',  # Default to CUDA if available
        skip_model_load: bool = False
    ):
        """
        Initializes the agent interface for AutoVLA.
        :param trajectory_sampling: trajectory sampling specification.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        :param sensor_data_path: optional sensor data path as string, defaults to None
        :param codebook_cache_path: optional codebook cache path as string, defaults to None
        :param lora_conf: optional lora configuration as string, defaults to None
        :param config_path: optional config path as string, defaults to None
        :param device: device to use, defaults to 'cuda'
        :param skip_model_load: whether to skip model loading, defaults to False
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        
        config = None
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self._checkpoint_path = checkpoint_path

        if not codebook_cache_path:
            self._codebook_cache_path = config['model']['codebook_cache_path']
        else:
            self._codebook_cache_path = codebook_cache_path

        if not skip_model_load:
            self.autovla = AutoVLA(config, device=device)
            self.autovla.eval()
        
        self.sensor_data_path = sensor_data_path
        self.lora_conf = lora_conf


    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self.lora_conf.get("use_lora", False):
                lora_config = LoraConfig(
                    task_type=TaskType[self.lora_conf.get("task_type", "CAUSAL_LM")],
                    target_modules=self.lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                    r=self.lora_conf.get("r", 8),
                    lora_alpha=self.lora_conf.get("lora_alpha", 8),
                    lora_dropout=self.lora_conf.get("lora_dropout", 0.1),
                    bias=self.lora_conf.get("bias", "none")
                )
                self.autovla.vlm = get_peft_model(self.autovla.vlm, lora_config)
        
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, 
                                                map_location="cpu")["state_dict"]
        self.autovla.load_state_dict( {k.replace("autovla.", ""): v for k, v in state_dict.items()}, strict=False)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(cam_f0=True, 
                            cam_l0=True, 
                            cam_l1=True, 
                            cam_l2=True, 
                            cam_r0=True, 
                            cam_r1=True, 
                            cam_r2=True,
                            cam_b0=True, 
                            lidar_pc=False)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(
            trajectory_sampling=self._trajectory_sampling,
            codebook_cache_path=self._codebook_cache_path
        )]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [AutoVLAAgentFeatureBuilder(sensor_data_path=self.sensor_data_path)]
    
    def compute_trajectory(self, scene_data) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.autovla.eval()
        features: Dict[str, torch.Tensor] = {}

        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))

        # integrate the sensor data path
        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})

        # forward pass
        with torch.no_grad():
            poses, cot_results = self.autovla.predict(features)

        submission = False
        if submission:
            poses_sub = self.upsample_trajectory(poses)
            return Trajectory(poses_sub, self._trajectory_sampling)
        else:        
            # extract trajectory
            return Trajectory(poses[: self._trajectory_sampling.num_poses,:], self._trajectory_sampling), cot_results
    
    def upsample_trajectory(self, traj, old_total_time=5.0, new_total_time=4.0, new_interval=0.1):
        # Generate original time points uniformly from 0.5s to old_total_time (excluding current time)
        N = traj.shape[0]
        dims = traj.shape[1]

        # Add initial zero point at time 0
        start_traj = torch.zeros((1, dims), device=traj.device, dtype=traj.dtype)
        traj = torch.cat([start_traj, traj], dim=0)
        original_times = torch.linspace(0.0, old_total_time, steps=N + 1, device=traj.device, dtype=traj.dtype)

        # Keep only times <= new_total_time
        valid_mask = original_times <= new_total_time
        valid_times = original_times[valid_mask]
        valid_traj = traj[valid_mask]

        # New interpolated timestamps
        new_times = torch.arange(new_interval, new_total_time + new_interval / 10,
                                step=new_interval, device=traj.device, dtype=traj.dtype)

        new_traj = torch.empty((len(new_times), dims), device=traj.device, dtype=traj.dtype)
        for dim in range(dims):
            new_traj[:, dim] = self.torch_interp(new_times, valid_times, valid_traj[:, dim])
        return new_traj
        
    @staticmethod
    def torch_interp(x, xp, fp):
        """
        Linear interpolation function similar to np.interp.
        
        Parameters:
        x (1D tensor): New sample points.
        xp (1D tensor): Known x-coordinates (must be sorted in ascending order).
        fp (1D tensor): Known y-values corresponding to xp.
        
        Returns:
        A tensor containing the interpolated values. For x < xp[0] it returns fp[0],
        and for x > xp[-1] it returns fp[-1].
        """
        # Find the indices where elements of x should be inserted into xp
        idx = torch.searchsorted(xp, x, right=False)
        
        # Clamp indices for left and right bounds
        left_idx = torch.clamp(idx - 1, min=0)
        right_idx = torch.clamp(idx, max=len(xp) - 1)
        
        xp_left = xp[left_idx]
        xp_right = xp[right_idx]
        fp_left = fp[left_idx]
        fp_right = fp[right_idx]
        
        # Calculate the denominator and avoid division by zero
        denom = xp_right - xp_left
        denom[denom == 0] = 1.0  # Prevent division by zero
        t = (x - xp_left) / denom
        result = fp_left + t * (fp_right - fp_left)
        
        # For x values outside xp's range, return the boundary values
        result = torch.where(x < xp[0], fp[0].expand_as(result), result)
        result = torch.where(x > xp[-1], fp[-1].expand_as(result), result)
        
        return result