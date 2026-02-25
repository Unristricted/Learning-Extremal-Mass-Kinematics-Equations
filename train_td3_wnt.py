import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise


# -----------------------------
# 1) Load dataset
# -----------------------------
def load_wnt_csv(path: str, target: str = "n_real"):
    df = pd.read_csv(path)
    # Expect columns: k1..k31, c1..c5, plus n_real, ok, etc.
    feature_cols = [f"k{i}" for i in range(1, 32)] + [f"c{i}" for i in range(1, 6)]
    missing = [c for c in feature_cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Optionally filter out failed solves
    if "ok" in df.columns:
        df = df[df["ok"] == True].copy()
        df.reset_index(drop=True, inplace=True)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y, df, feature_cols


# -----------------------------
# 2) Reward surrogate model
# -----------------------------
class RewardNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_reward_model(X, y, device="cpu", epochs=50, batch_size=256, lr=1e-3):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    model = RewardNet(d_in=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def batches(Xb, yb, bs):
        idx = np.arange(Xb.shape[0])
        np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            j = idx[i:i+bs]
            yield Xb[j], yb[j]

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in batches(X_train_s, y_train_s, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            xv_t = torch.from_numpy(X_val_s).to(device)
            yv_t = torch.from_numpy(y_val_s).to(device)
            pv = model(xv_t)
            val_loss = loss_fn(pv, yv_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[RewardNet] epoch {ep:03d} | train_mse={np.mean(tr_losses):.4f} | val_mse={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_scaler, y_scaler


# -----------------------------
# 3) One-step bandit env (TD3-friendly)
# -----------------------------
class SurrogateBanditEnv(gym.Env):
    """
    Observation: dummy 1D (constant 0). We don't need state for bandit.
    Action: 36-D continuous vector in [-1, 1].
    We map it to parameter bounds (same as your sampling box).
    Reward: surrogate_predicted n_real (or n_pos, etc.)
    Episode length: 1 step.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_model: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        param_lo: float,
        param_hi: float,
        device="cpu",
        reward_clip=(0.0, 200.0),
        action_l2_penalty=0.0,
    ):
        super().__init__()
        self.reward_model = reward_model.to(device)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device

        self.param_lo = float(param_lo)
        self.param_hi = float(param_hi)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(36,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

        self.reward_clip = reward_clip
        self.action_l2_penalty = float(action_l2_penalty)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros((1,), dtype=np.float32)
        return obs, {}

    def _action_to_params(self, a: np.ndarray) -> np.ndarray:
        # Map from [-1,1] -> [param_lo, param_hi]
        # p = lo + (a+1)/2 * (hi-lo)
        return self.param_lo + (a + 1.0) * 0.5 * (self.param_hi - self.param_lo)

    @torch.no_grad()
    def _predict_reward(self, params: np.ndarray) -> float:
        x = params.reshape(1, -1).astype(np.float32)
        x_s = self.x_scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x_s).to(self.device)
        y_s = self.reward_model(xt).cpu().numpy()  # scaled reward
        y = self.y_scaler.inverse_transform(y_s).item()
        return float(y)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        params = self._action_to_params(action)
        r = self._predict_reward(params)

        # Optional regularization: discourages extreme actions if you want
        if self.action_l2_penalty > 0:
            r -= self.action_l2_penalty * float(np.sum(action**2))

        if self.reward_clip is not None:
            lo, hi = self.reward_clip
            r = float(np.clip(r, lo, hi))

        obs = np.zeros((1,), dtype=np.float32)
        terminated = True   # 1-step episode
        truncated = False
        info = {"pred_reward": r, "params": params}
        return obs, r, terminated, truncated, info


# -----------------------------
# 4) Train TD3 + export proposals
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to wnt_samples.csv")
    ap.add_argument("--target", default="n_real", choices=["n_real", "n_pos"], help="Which target column to optimize")
    ap.add_argument("--param_lo", type=float, default=0.0, help="Lower bound for parameters (match your sampling)")
    ap.add_argument("--param_hi", type=float, default=100.0, help="Upper bound for parameters (match your sampling)")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="td3_out")
    ap.add_argument("--topk", type=int, default=50, help="How many candidate params to export")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    X, y, df, feature_cols = load_wnt_csv(args.csv, target=args.target)
    print(f"Loaded {len(df)} rows (after ok-filter if present). Target={args.target}")

    # Train surrogate
    model, x_scaler, y_scaler = train_reward_model(X, y, device=args.device, epochs=60, batch_size=256, lr=1e-3)
    torch.save(model.state_dict(), os.path.join(args.outdir, "reward_net.pt"))

    # Build env
    env = SurrogateBanditEnv(
        reward_model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        param_lo=args.param_lo,
        param_hi=args.param_hi,
        device=args.device,
        reward_clip=(0.0, 200.0),
        action_l2_penalty=0.0,
    )

    # TD3 config
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    td3 = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,      # irrelevant-ish for 1-step but fine
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    print(f"Training TD3 for {args.timesteps} timesteps on surrogate bandit...")
    td3.learn(total_timesteps=args.timesteps)
    td3.save(os.path.join(args.outdir, "td3_policy.zip"))

    # Generate candidate proposals by sampling many actions from policy + noise
    rng = np.random.default_rng(args.seed)
    proposals = []

    obs, _ = env.reset()
    for _ in range(5000):
        action, _ = td3.predict(obs, deterministic=False)
        # add extra exploration when collecting proposals
        action = np.clip(action + rng.normal(0, 0.1, size=action.shape).astype(np.float32), -1.0, 1.0)
        _, r, *_ , info = env.step(action)
        proposals.append((r, info["params"]))

    proposals.sort(key=lambda t: t[0], reverse=True)
    top = proposals[: args.topk]

    # Save proposals
    rows = []
    for rank, (r, params) in enumerate(top, start=1):
        d = {"rank": rank, "pred_reward": float(r)}
        for i in range(31):
            d[f"k{i+1}"] = float(params[i])
        for i in range(5):
            d[f"c{i+1}"] = float(params[31+i])
        rows.append(d)

    out_csv = os.path.join(args.outdir, "top_proposals.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote top-{args.topk} proposals to: {out_csv}")
    print("Next step: verify these proposals with your Julia oracle and append them to your dataset (active learning).")


if __name__ == "__main__":
    main()