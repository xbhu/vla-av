"""
Diagnostic: check whether the Tier1 MLP is underfitting, before trusting
the conclusion that a trained MLP under-performs kinematic extrapolation.
Loads the saved checkpoint, evaluates on the TRAIN set (not val) using
the same L2 metric style (simple mean L2 over 6 future steps in raw
ego-relative coords, no UniAD-coord transform needed for a same-frame
train-fit check), and also tries a longer training run + a smaller
hidden dim as a quick sensitivity check.
"""
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/xzh5180/Research/vla-av/sourcecode/clusterB_openloop_limitations/b1_setup")
from usecaseB1_common import (
    TRAIN_DIR, load_all_samples, build_ego_status_feature,
    is_train_sample_degenerate, FEATURE_DIM, N_FUTURE_FULL,
)
from usecaseB1_tier1_train_mlp import EgoStatusMLP

OUT_DIR = "/home/xzh5180/Research/vla-av/outputs/usecaseB1"

train_samples = load_all_samples(TRAIN_DIR)
usable_train = {t: s for t, s in train_samples.items() if not is_train_sample_degenerate(s)}
tokens = list(usable_train.keys())
X = np.stack([build_ego_status_feature(usable_train[t]) for t in tokens])
Y = np.stack([np.array(usable_train[t]["gt_trajectory"], dtype=np.float32) for t in tokens])

ckpt = torch.load(f"{OUT_DIR}/usecaseB1_tier1_mlp_checkpoint.pt", map_location="cpu")
feat_mean, feat_std = ckpt["feat_mean"], ckpt["feat_std"]
model = EgoStatusMLP(FEATURE_DIM, 64, N_FUTURE_FULL)
model.load_state_dict(ckpt["state_dict"])
model.eval()

X_norm = (X - feat_mean) / feat_std
with torch.no_grad():
    pred = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()

l2_per_sample_6s = np.linalg.norm(pred[:, 5, :2] - Y[:, 5, :2], axis=-1)  # index 5 = 3.0s step (t6 in 1-indexed)
print(f"[TRAIN-SET FIT CHECK, original checkpoint]")
print(f"  mean L2 at 3.0s on TRAINING data: {l2_per_sample_6s.mean():.4f} m")
print(f"  (compare: Tier0 const_velocity val L2 at 3.0s was 1.1064m)")
print(f"  if this train-set number is also >1m, the MLP is UNDERFITTING its own training data")

# Quick sensitivity: retrain longer with smaller hidden dim + higher lr, see if it helps
print("\n[SENSITIVITY CHECK] retraining with hidden_dim=32, lr=3e-3, epochs=5000")
torch.manual_seed(42)
model2 = EgoStatusMLP(FEATURE_DIM, 32, N_FUTURE_FULL)
opt2 = torch.optim.Adam(model2.parameters(), lr=3e-3)
X_t = torch.tensor(X_norm, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)
for epoch in range(5000):
    opt2.zero_grad()
    p = model2(X_t)
    loss = torch.nn.functional.l1_loss(p, Y_t)
    loss.backward()
    opt2.step()
    if epoch % 1000 == 0 or epoch == 4999:
        print(f"  epoch {epoch:5d} train_L1_loss={loss.item():.4f}")

model2.eval()
with torch.no_grad():
    pred2 = model2(X_t).numpy()
l2_2 = np.linalg.norm(pred2[:, 5, :2] - Y[:, 5, :2], axis=-1)
print(f"  retrained mean L2 at 3.0s on TRAINING data: {l2_2.mean():.4f} m")
