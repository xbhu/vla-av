"""
Tier 1 baseline: a small MLP trained on ego status only (velocity,
acceleration, one-hot instruction -- 5 dims), no visual input, no
history window (none available in the preprocessed data; acceleration
already encodes one step of temporal difference). Architecture mirrors
navsim's official EgoStatusMLPAgent (3 hidden ReLU layers + linear
output reshaped to (n_poses, 3)), adjusted for our 5-dim input instead
of its 8-dim [vx,vy,ax,ay,driving_command(4)] input, since our data
only has scalar velocity/acceleration, not 2D components (confirmed
from tools/preprocessing/nusc_sample_generation.py source).

Trained on all available train data: 164 samples, minus 2 flagged as
degenerate by the proxy filter (162 usable). This is the full local
dataset, not a deliberately restricted subset -- there is no larger
preprocessed nuScenes split available without re-running the original
extraction pipeline, which is out of scope per the task instructions.
"""
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import (
    TRAIN_DIR, VAL_DIR, load_all_samples, build_ego_status_feature,
    is_train_sample_degenerate, evaluate_predictions, print_planning_tables,
    FEATURE_DIM, N_FUTURE_FULL,
)

OUT_DIR = "/home/xzh5180/Research/vla-av/outputs/usecaseB1"
SEED = 42
HIDDEN_DIM = 64
LR = 1e-3
N_EPOCHS = 2000
LOG_EVERY = 200

torch.manual_seed(SEED)
np.random.seed(SEED)


class EgoStatusMLP(nn.Module):
    """Same depth/activation pattern as navsim's EgoStatusMLPAgent, adjusted input dim."""

    def __init__(self, in_dim, hidden_dim, n_future):
        super().__init__()
        self.n_future = n_future
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_future * 3),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.n_future, 3)


def main():
    train_samples = load_all_samples(TRAIN_DIR)
    val_samples = load_all_samples(VAL_DIR)
    print(f"Loaded {len(train_samples)} train samples, {len(val_samples)} val samples")

    usable_train = {
        tok: s for tok, s in train_samples.items() if not is_train_sample_degenerate(s)
    }
    n_dropped = len(train_samples) - len(usable_train)
    print(f"Dropped {n_dropped} degenerate train samples, {len(usable_train)} usable")

    tokens = list(usable_train.keys())
    X = np.stack([build_ego_status_feature(usable_train[t]) for t in tokens])  # (N, 5)
    Y = np.stack([np.array(usable_train[t]["gt_trajectory"], dtype=np.float32) for t in tokens])  # (N, 10, 3)

    # Standardize features (velocity/acceleration scales differ from one-hot 0/1);
    # targets are kept in raw meters/radians since eval must compare in raw units.
    feat_mean = X.mean(axis=0, keepdims=True)
    feat_std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - feat_mean) / feat_std

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    model = EgoStatusMLP(FEATURE_DIM, HIDDEN_DIM, N_FUTURE_FULL)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = torch.nn.functional.l1_loss(pred, Y_t)
        loss.backward()
        optimizer.step()
        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:5d}  train_L1_loss={loss.item():.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "feat_mean": feat_mean, "feat_std": feat_std},
        f"{OUT_DIR}/usecaseB1_tier1_mlp_checkpoint.pt",
    )

    # Inference on val set
    model.eval()
    pred_trajs = {}
    with torch.no_grad():
        for tok, s in val_samples.items():
            feat = build_ego_status_feature(s)
            feat_norm = (feat - feat_mean[0]) / feat_std[0]
            feat_t = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)
            pred = model(feat_t)[0].numpy()
            pred_trajs[tok] = pred.tolist()

    with open(f"{OUT_DIR}/usecaseB1_tier1_mlp_preds.json", "w") as f:
        json.dump(pred_trajs, f, indent=2)

    result, n_used, n_skipped = evaluate_predictions(pred_trajs, val_samples, label="TIER1_MLP")
    print_planning_tables(result, "Tier1 (Ego-Status MLP)")

    serializable = {k: v.tolist() for k, v in result.items()}
    with open(f"{OUT_DIR}/usecaseB1_tier1_mlp_eval.json", "w") as f:
        json.dump(
            {
                "n_train_usable": len(usable_train),
                "n_train_dropped_degenerate": n_dropped,
                "n_used": n_used,
                "n_skipped": n_skipped,
                "final_train_l1_loss": loss.item(),
                "metrics": serializable,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
