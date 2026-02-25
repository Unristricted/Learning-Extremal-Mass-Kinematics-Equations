import os
import time
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

# Julia bridge
from juliacall import Main as jl


# -----------------------------
# Utilities
# -----------------------------
FEATURE_COLS = [f"k{i}" for i in range(1, 32)] + [f"c{i}" for i in range(1, 6)]
D = 36


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist; if not, raise.
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_cols(df)
    # If ok exists, keep both ok and failed, but training uses ok==True by default.
    return df


def append_rows(csv_path: str, new_rows: pd.DataFrame):
    # Safely merge new rows with existing CSV to avoid malformed lines
    # If the CSV exists, read it fully, concat, and rewrite with a consistent header/order.
    if os.path.exists(csv_path):
        # Read existing CSV fully (keeps types and header order)
        df_existing = pd.read_csv(csv_path)

        # Make a copy to avoid mutating caller's DataFrame
        df_new = new_rows.copy()

        # Ensure both frames have the same columns (add missing cols as NaN)
        for c in df_existing.columns:
            if c not in df_new.columns:
                df_new[c] = np.nan
        for c in df_new.columns:
            if c not in df_existing.columns:
                # preserve extra new columns by adding them to existing frame (as NaN)
                df_existing[c] = np.nan

        # Concatenate and preserve existing column order, then any extras
        combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        ordered_cols = list(df_existing.columns) + [c for c in combined.columns if c not in df_existing.columns]
        combined = combined[ordered_cols]

        # Write entire CSV atomically (overwrite) to ensure consistent quoting/formatting
        combined.to_csv(csv_path, index=False)
    else:
        # If file doesn't exist, write as-is (new_rows column order will become header)
        new_rows.to_csv(csv_path, index=False)


# -----------------------------
# Surrogate reward model
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


def train_reward_model(df: pd.DataFrame, target: str, device: str, epochs=60, batch_size=256, lr=1e-3):
    # Train only on successful rows if ok column exists
    if "ok" in df.columns:
        df_train = df[df["ok"] == True].copy()
    else:
        df_train = df.copy()

    X = df_train[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df_train[target].to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(df_train) < 200:
        raise ValueError(f"Not enough training rows (got {len(df_train)}). Need more oracle-labeled samples.")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_tr_s = x_scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = x_scaler.transform(X_va).astype(np.float32)

    y_tr_s = y_scaler.fit_transform(y_tr).astype(np.float32)
    y_va_s = y_scaler.transform(y_va).astype(np.float32)

    model = RewardNet(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def iter_batches(Xb, yb):
        idx = np.arange(Xb.shape[0])
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = idx[i:i + batch_size]
            yield Xb[j], yb[j]

    best = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in iter_batches(X_tr_s, y_tr_s):
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
            xv = torch.from_numpy(X_va_s).to(device)
            yv = torch.from_numpy(y_va_s).to(device)
            pv = model(xv)
            val = loss_fn(pv, yv).item()

        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[RewardNet] ep={ep:03d} train_mse={np.mean(tr_losses):.4f} val_mse={val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_scaler, y_scaler, len(df_train)


# -----------------------------
# Surrogate bandit env (1-step)
# -----------------------------
class SurrogateBanditEnv(gym.Env):
    """
    Observation: constant 0
    Action: 36D in [-1,1]
    Map action -> params in [param_lo, param_hi]
    Reward: surrogate predicted target
    Episode: 1 step
    """
    metadata = {"render_modes": []}

    def __init__(self, model, x_scaler, y_scaler, device, param_lo, param_hi, reward_clip=None, action_l2_penalty=0.0):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device

        self.param_lo = float(param_lo)
        self.param_hi = float(param_hi)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(D,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

        self.reward_clip = reward_clip
        self.action_l2_penalty = float(action_l2_penalty)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1,), dtype=np.float32), {}

    def action_to_params(self, a: np.ndarray) -> np.ndarray:
        # [-1,1] -> [lo,hi]
        return self.param_lo + (a + 1.0) * 0.5 * (self.param_hi - self.param_lo)

    @torch.no_grad()
    def predict_reward(self, params: np.ndarray) -> float:
        x = params.reshape(1, -1).astype(np.float32)
        x_s = self.x_scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x_s).to(self.device)
        y_s = self.model(xt).cpu().numpy()               # scaled
        y = self.y_scaler.inverse_transform(y_s).item()  # unscaled
        return float(y)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        p = self.action_to_params(a)
        r = self.predict_reward(p)

        if self.action_l2_penalty > 0:
            r -= self.action_l2_penalty * float(np.sum(a * a))

        if self.reward_clip is not None:
            r = float(np.clip(r, self.reward_clip[0], self.reward_clip[1]))

        obs = np.zeros((1,), dtype=np.float32)
        terminated = True
        truncated = False
        info = {"pred_reward": r, "params": p}
        return obs, r, terminated, truncated, info


# -----------------------------
# Julia oracle wrapper
# -----------------------------
class JuliaOracle:
    def __init__(self, wnt_oracle_path: str):
        jl.include(wnt_oracle_path)
        self.fn = jl.WNTOracle.evaluate_params
        self.Symbol = jl.Symbol  # <-- add

    def eval_params(self, params: np.ndarray, start_system="polyhedral", tol_real=1e-7, tol_pos=1e-9):
        p = np.asarray(params, dtype=np.float64)

        # Convert "polyhedral" -> :polyhedral, "total_degree" -> :total_degree
        start_system_sym = self.Symbol(start_system)

        out = self.fn(p, start_system=start_system_sym, tol_real=tol_real, tol_pos=tol_pos)

        return {
            "ok": bool(out["ok"]),
            "n_solutions": int(out["n_solutions"]),
            "n_real": int(out["n_real"]),
            "n_pos": int(out["n_pos"]),
            "err": str(out["err"]),
        }


# -----------------------------
# Candidate generation
# -----------------------------
def propose_candidates(td3: TD3, env: SurrogateBanditEnv, n: int, noise_std: float, seed: int):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    props = []
    for _ in range(n):
        a, _ = td3.predict(obs, deterministic=False)
        a = np.asarray(a, dtype=np.float32)
        if noise_std > 0:
            a = np.clip(a + rng.normal(0, noise_std, size=a.shape).astype(np.float32), -1.0, 1.0)
        _, r, *_ , info = env.step(a)
        props.append((float(r), info["params"]))
    props.sort(key=lambda t: t[0], reverse=True)
    return props  # list of (pred_reward, params)


def dedup_by_rounding(params_list, decimals=4):
    seen = set()
    out = []
    for p in params_list:
        key = tuple(np.round(p, decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# -----------------------------
# Active search loop
# -----------------------------
def active_search(
    csv_path: str,
    wnt_oracle_path: str,
    target: str,
    param_lo: float,
    param_hi: float,
    rounds: int,
    td3_timesteps: int,
    proposals_per_round: int,
    eval_batch_size: int,
    device: str,
    seed: int,
    start_system: str,
    tol_real: float,
    tol_pos: float,
    proposal_pool: int,
    proposal_noise: float,
):
    oracle = JuliaOracle(wnt_oracle_path)

    for r in range(1, rounds + 1):
        print(f"\n==================== ROUND {r}/{rounds} ====================")
        df = load_dataset(csv_path)

        # Choose target
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in CSV columns. Available: {list(df.columns)[:20]} ...")

        # Train surrogate
        model, x_scaler, y_scaler, n_train = train_reward_model(df, target=target, device=device)
        print(f"[Round {r}] Surrogate trained on {n_train} successful rows.")

        # Train TD3
        env = SurrogateBanditEnv(model, x_scaler, y_scaler, device=device,
                                 param_lo=param_lo, param_hi=param_hi,
                                 reward_clip=None, action_l2_penalty=0.0)

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
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1,
            seed=seed + r,
            device=device,
        )

        
        print(f"[Round {r}] Training TD3 for {td3_timesteps} steps (surrogate bandit)...")
        td3.learn(total_timesteps=td3_timesteps)
        
        # td3_path = os.path.join("td3_out", "td3_policy.zip")
        # td3 = TD3.load(
        #     td3_path,
        #     env=env,          # IMPORTANT: attach current env
        #     device=device
        # )
        # Propose a pool then pick top proposals_per_round after de-dup
        print(f"[Round {r}] Proposing candidate pool of {proposal_pool}...")
        pool = propose_candidates(td3, env, n=proposal_pool, noise_std=proposal_noise, seed=seed + 10 * r)
        pool_params = [p for _, p in pool]
        pool_params = dedup_by_rounding(pool_params, decimals=4)

        # Avoid re-evaluating exact points already in dataset (approx check)
        existing = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        existing_keys = set(tuple(np.round(row, 4)) for row in existing)

        filtered = []
        for p in pool_params:
            key = tuple(np.round(p, 4))
            if key in existing_keys:
                continue
            filtered.append(p)
            if len(filtered) >= proposals_per_round:
                break

        if not filtered:
            print(f"[Round {r}] No novel proposals found after de-dup; stopping.")
            return

        print(f"[Round {r}] Selected {len(filtered)} novel proposals for Julia evaluation.")

        # Evaluate with Julia oracle in batches
        new_rows = []
        t_eval0 = time.time()
        for i in range(0, len(filtered), eval_batch_size):
            batch = filtered[i:i + eval_batch_size]
            for p in batch:
                out = oracle.eval_params(
                    p,
                    start_system=start_system,
                    tol_real=tol_real,
                    tol_pos=tol_pos,
                )
                row = {"ok": out["ok"], "n_solutions": out["n_solutions"], "n_real": out["n_real"], "n_pos": out["n_pos"], "err": out["err"]}
                for j in range(31):
                    row[f"k{j+1}"] = float(p[j])
                for j in range(5):
                    row[f"c{j+1}"] = float(p[31 + j])
                new_rows.append(row)

            print(f"[Round {r}] Evaluated {min(i+eval_batch_size, len(filtered))}/{len(filtered)} proposals...")

        dt = time.time() - t_eval0
        new_df = pd.DataFrame(new_rows)

        # Append to dataset CSV
        append_rows(csv_path, new_df)
        print(f"[Round {r}] Appended {len(new_df)} new oracle-labeled rows to {csv_path}.")
        print(f"[Round {r}] Oracle eval time: {dt:.1f}s")

        # Simple progress stats
        ok_rate = new_df["ok"].mean()
        best_real = new_df["n_real"].max()
        best_pos = new_df["n_pos"].max()
        print(f"[Round {r}] New points: ok_rate={ok_rate:.2f}, best_n_real={best_real}, best_n_pos={best_pos}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to wnt_samples.csv (will be appended to)")
    ap.add_argument("--oracle", required=True, help="Path to wnt_oracle.jl")
    ap.add_argument("--target", default="n_real", choices=["n_real", "n_pos"], help="Objective for surrogate/TD3")
    ap.add_argument("--param_lo", type=float, default=0.0, help="Must match your intended search box")
    ap.add_argument("--param_hi", type=float, default=100.0, help="Must match your intended search box")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--td3_steps", type=int, default=200_000)
    ap.add_argument("--proposals_per_round", type=int, default=50)
    ap.add_argument("--proposal_pool", type=int, default=5000, help="How many proposals to sample before taking top")
    ap.add_argument("--proposal_noise", type=float, default=0.10, help="Extra noise when sampling proposals")
    ap.add_argument("--eval_batch_size", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_system", default="polyhedral", choices=["polyhedral", "total_degree"])
    ap.add_argument("--tol_real", type=float, default=1e-7)
    ap.add_argument("--tol_pos", type=float, default=1e-9)
    args = ap.parse_args()

    active_search(
        csv_path=args.csv,
        wnt_oracle_path=args.oracle,
        target=args.target,
        param_lo=args.param_lo,
        param_hi=args.param_hi,
        rounds=args.rounds,
        td3_timesteps=args.td3_steps,
        proposals_per_round=args.proposals_per_round,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        seed=args.seed,
        start_system=args.start_system,
        tol_real=args.tol_real,
        tol_pos=args.tol_pos,
        proposal_pool=args.proposal_pool,
        proposal_noise=args.proposal_noise,
    )


if __name__ == "__main__":
    main()