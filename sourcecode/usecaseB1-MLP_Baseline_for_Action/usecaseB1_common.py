"""
Shared utilities for Use Case B1: Ego-State MLP Baseline vs kinematic
extrapolation, evaluated on the same val_filtered set (126 samples) and
with the same L2/collision pipeline as usecaseA1/usecaseA2 (STP-3 and
UniAD conventions, reusing PlanningMetric from
AutoVLA/tools/eval/planning_metrics.py verbatim, not reimplemented).

Design notes (locked in after inspecting the actual preprocessed JSON
and the generation script tools/preprocessing/nusc_sample_generation.py):
- 'velocity' and 'acceleration' in the JSON are SCALARS (speed
  magnitude and its finite difference), NOT 2D vectors like the
  official navsim EgoStatusFeatureBuilder assumes. Confirmed from
  source: ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
- No multi-frame ego-state history exists in the data; 'acceleration'
  already encodes one step of temporal history (v(t) - v(t-1)) / 0.5
- 'gt_trajectory' is a (10, 3) array [x, y, yaw] in an ego-relative,
  x-forward frame anchored at the current pose (t0). Numerically
  confirmed identical to 'gt_pos_raw' used in nusc_eval_a2.py for at
  least one shared token.
- The eval pipeline (nusc_eval_a2.py) only ever uses the xy columns
  (gt_pos_raw / pred_trajectory[:, :2]) for L2/collision -- yaw is
  dropped before metric computation. We mirror this: only xy enters
  PlanningMetric.

All code/comments/print statements are English only, per project
convention.
"""
import os
import sys
import json

import numpy as np
import torch

AUTOVLA_ROOT = "/home/xzh5180/Research/vla-av/sourcecode/AutoVLA"
sys.path.insert(0, AUTOVLA_ROOT)

from tools.eval.planning_metrics import PlanningMetric  # noqa: E402

TRAIN_DIR = "/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_train"
VAL_DIR = "/home/xzh5180/Research/vla-av/datasets/autovla_nusc_preprocessed_val_filtered"
SEG_DIR = "/home/xzh5180/Research/vla-av/datasets/autovla_nusc_seg/nusc_eval_seg_6s"

INSTRUCTION_CLASSES = ["Go Straight", "Turn Left", "Turn Right"]
N_FUTURE_EVAL = 6     # matches PlanningMetric(n_future=6) used in nusc_eval_a2.py
N_FUTURE_FULL = 10    # length of stored gt_trajectory (5.0s at 0.5s steps)
FEATURE_DIM = 2 + len(INSTRUCTION_CLASSES)  # velocity, acceleration, onehot(instruction)


def list_json_files(d):
    return sorted([f for f in os.listdir(d) if f.endswith(".json")])


def load_sample(d, fname):
    with open(os.path.join(d, fname), "r") as f:
        return json.load(f)


def load_all_samples(d):
    """Returns dict: token -> sample dict."""
    out = {}
    for fn in list_json_files(d):
        s = load_sample(d, fn)
        out[s["token"]] = s
    return out


def instruction_to_onehot(instruction):
    vec = [0.0, 0.0, 0.0]
    if instruction not in INSTRUCTION_CLASSES:
        raise ValueError(f"Unknown instruction: {instruction}")
    vec[INSTRUCTION_CLASSES.index(instruction)] = 1.0
    return vec


def build_ego_status_feature(sample):
    """5-dim feature: [velocity, acceleration, onehot(instruction)]."""
    feat = [sample["velocity"], sample["acceleration"]] + instruction_to_onehot(sample["instruction"])
    return np.array(feat, dtype=np.float32)


def is_train_sample_degenerate(sample, spread_thresh=0.15, vel_thresh=0.5):
    """
    Proxy filter for degenerate GT in the train split, which has no
    future_mask field (unlike val). Flags a sample if the first 6 future
    xy positions barely move while velocity says it should be moving --
    this is the same pattern confirmed in A2's worst-L2 tokens (padded
    GT is constant across all future steps).

    Note: nusc_sample_generation.py already applies a stricter filter at
    generation time for split == 'train' (skip if
    np.sum(gt_ego_fut_masks) < 10, over the full 16-frame future before
    trimming to 10). This proxy is a sanity backstop, not the primary
    filter -- only 2/164 train samples are flagged by it.
    """
    traj = np.array(sample["gt_trajectory"][:6])
    spread = (traj[:, 0].max() - traj[:, 0].min()) + (traj[:, 1].max() - traj[:, 1].min())
    return spread < spread_thresh and abs(sample["velocity"]) > vel_thresh


def kinematic_extrapolation(sample, use_acceleration):
    """
    Tier 0 baseline: zero-learning kinematic extrapolation in the
    ego-relative frame. Current heading is 0 by construction of the
    frame (x-forward, anchored at t0).

    use_acceleration=False -> constant velocity:  x(t) = v0*t,             y=0, yaw=0
    use_acceleration=True  -> constant accel:      x(t) = v0*t + 0.5*a0*t^2, y=0, yaw=0
    Returns np.ndarray, shape (N_FUTURE_FULL, 3).
    """
    v0 = sample["velocity"]
    a0 = sample["acceleration"]
    ts = np.arange(1, N_FUTURE_FULL + 1) * 0.5  # 0.5s .. 5.0s
    x = v0 * ts + (0.5 * a0 * (ts ** 2) if use_acceleration else 0.0)
    y = np.zeros_like(x)
    yaw = np.zeros_like(x)
    return np.stack([x, y, yaw], axis=-1).astype(np.float32)


def to_uniad_coord_and_double_unsqueeze(xy):
    """
    Reproduces, exactly, the transform + shape sequence nusc_eval_a2.py
    applies to a single sample's (10, 2) xy tensor before calling
    PlanningMetric:
        1. unsqueeze(0)                      -> (1, 10, 2)
        2. swap x/y, then negate new x        (ego-relative -> UniAD coord)
        3. unsqueeze(0) again                 -> (1, 1, 10, 2)
    Do not simplify this to a single unsqueeze -- the double-unsqueeze
    is load-bearing for how nusc_eval_a2.py indexes the result
    ([0, :, :6, :]) and must match exactly for numbers to be comparable.
    xy: torch.Tensor, shape (10, 2)
    Returns torch.Tensor, shape (1, 1, 10, 2)
    """
    out = xy.unsqueeze(0).clone()
    out[:, :, [0, 1]] = out[:, :, [1, 0]]
    out[:, :, 0] = -out[:, :, 0]
    out = out.unsqueeze(0)
    return out


def evaluate_predictions(pred_trajs_by_token, val_samples_by_token, label):
    """
    Run predictions through the same PlanningMetric used by A1/A2, with
    identical coordinate transform, shape handling, and future_mask
    consistency check as nusc_eval_a2.py.

    pred_trajs_by_token: dict token -> np.ndarray, shape (N_FUTURE_FULL, 2) or (N_FUTURE_FULL, 3)
        (only the first 2 columns are used, matching the official pipeline)
    Returns (eval_result_dict, n_used, n_skipped)
    """
    planning_metrics = PlanningMetric(n_future=N_FUTURE_EVAL)
    n_used = 0
    n_skipped = 0

    for token, sample in val_samples_by_token.items():
        if token not in pred_trajs_by_token:
            n_skipped += 1
            continue

        pred = np.asarray(pred_trajs_by_token[token])[:, :2]
        gt = np.asarray(sample["gt_trajectory"])[:, :2]

        pred_xy = torch.tensor(pred, dtype=torch.float32)
        gt_xy = torch.tensor(gt, dtype=torch.float32)

        seg_path = os.path.join(SEG_DIR, f"{token}.pt")
        if not os.path.exists(seg_path):
            print(f"[WARNING] {label}: no segmentation file for token {token}, skipping")
            n_skipped += 1
            continue
        uniad_data = torch.load(seg_path, map_location="cpu")
        sdc_planning_mask = uniad_data["sdc_planning_mask"].to(gt_xy.dtype)
        segmentation = uniad_data["segmentation"].to(gt_xy.dtype)

        gt_traj_uniadcoord = to_uniad_coord_and_double_unsqueeze(gt_xy)
        pred_traj_uniadcoord = to_uniad_coord_and_double_unsqueeze(pred_xy)

        cache_future_mask = torch.tensor(sample["future_mask"][:N_FUTURE_EVAL])
        sdc_mask = sdc_planning_mask[0, 0, :, 0]
        if not torch.allclose(cache_future_mask, sdc_mask):
            print(f"[WARNING] {label}: future_mask mismatch for token {token}, skipping")
            n_skipped += 1
            continue

        planning_metrics(
            pred_traj_uniadcoord[0, :, :N_FUTURE_EVAL, :],
            gt_traj_uniadcoord[0, :, :N_FUTURE_EVAL, :],
            sdc_planning_mask[0, :, :N_FUTURE_EVAL, :2],
            segmentation[:, [1, 2, 3, 4, 5, 6]],
        )
        n_used += 1

    print(f"[{label}] evaluated {n_used} samples, skipped {n_skipped}")
    return planning_metrics.compute(), n_used, n_skipped


def print_planning_tables(eval_result, title_prefix):
    from prettytable import PrettyTable

    tab_stp3 = PrettyTable()
    tab_stp3.title = f"{title_prefix} - STP3's Definition (Cumulative Average)"
    tab_stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for key, value in eval_result.items():
        row = [key]
        for i in range(min(len(value), 6)):
            row.append("%.4f" % float(value[: i + 1].mean()))
        tab_stp3.add_row(row)
    print(tab_stp3)

    tab_uniad = PrettyTable()
    tab_uniad.title = f"{title_prefix} - UniAD's Definition (Per-Timestep)"
    tab_uniad.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    for key, value in eval_result.items():
        row = [key]
        for i in range(min(len(value), 6)):
            row.append("%.4f" % float(value[i]))
        tab_uniad.add_row(row)
    print(tab_uniad)
    return tab_stp3, tab_uniad
