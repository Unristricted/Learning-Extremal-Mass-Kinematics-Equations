# active_td3_wnt_with_graphs.py
#
# Adds:
#  - RewardNet training history + saved CSV + saved PNG plot
#  - Stable-Baselines3 TensorBoard logging for TD3
#
# Usage example:
#   python active_td3_wnt_with_graphs.py --csv wnt_samples.csv --oracle wnt_oracle.jl \
#       --target n_real --device cpu --td3_steps 200000 --tb_dir tb_logs --run_name wnt_td3_run1
#
# Then launch TensorBoard:
#   tensorboard --logdir tb_logs
#
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

# RewardNet plot
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
FEATURE_COLS = [f"k{i}" for i in range(1, 32)] + [f"c{i}" for i in range(1, 6)]
D = 36


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_cols(df)
    return df


def append_rows(csv_path: str, new_rows: pd.DataFrame):
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_new = new_rows.copy()

        for c in df_existing.columns:
            if c not in df_new.columns:
                df_new[c] = np.nan
        for c in df_new.columns:
            if c not in df_existing.columns:
                df_existing[c] = np.nan

        combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
        ordered_cols = list(df_existing.columns) + [c for c in combined.columns if c not in df_existing.columns]
        combined = combined[ordered_cols]
        combined.to_csv(csv_path, index=False)
    else:
        new_rows.to_csv(csv_path, index=False)


def plot_rewardnet_history(history: dict, out_png: str = "rewardnet_training.png"):
    """
    history: {"epoch":[...], "train_mse":[...], "val_mse":[...]}
    Saves a PNG plot.
    """
    dfh = pd.DataFrame(history)
    plt.figure()
    plt.plot(dfh["epoch"], dfh["train_mse"], label="train_mse")
    plt.plot(dfh["epoch"], dfh["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (scaled target space)")
    plt.title("RewardNet Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Plot] Saved {out_png}")


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


def train_reward_model(
    df: pd.DataFrame,
    target: str,
    device: str,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    log_csv_path: str | None = None,
):
    """
    Trains RewardNet on successful rows (ok==True if present).
    Returns:
      model, x_scaler, y_scaler, n_train_rows, history_dict
    """
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

    history = {"epoch": [], "train_mse": [], "val_mse": []}

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

        tr = float(np.mean(tr_losses))
        history["epoch"].append(ep)
        history["train_mse"].append(tr)
        history["val_mse"].append(float(val))

        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[RewardNet] ep={ep:03d} train_mse={tr:.4f} val_mse={val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if log_csv_path is not None:
        pd.DataFrame(history).to_csv(log_csv_path, index=False)
        print(f"[RewardNet] Wrote history CSV to {log_csv_path}")

    return model, x_scaler, y_scaler, len(df_train), history


# -----------------------------
# Surrogate bandit env (1-step)
# -----------------------------
class SurrogateBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model, x_scaler, y_scaler, device, param_lo, param_hi,
                 reward_clip=None, action_l2_penalty=0.0):
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
        return self.param_lo + (a + 1.0) * 0.5 * (self.param_hi - self.param_lo)

    @torch.no_grad()
    def predict_reward(self, params: np.ndarray) -> float:
        x = params.reshape(1, -1).astype(np.float32)
        x_s = self.x_scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x_s).to(self.device)
        y_s = self.model(xt).cpu().numpy()
        y = self.y_scaler.inverse_transform(y_s).item()
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
        self.Symbol = jl.Symbol

    def eval_params(self, params: np.ndarray, start_system="polyhedral", tol_real=1e-7, tol_pos=1e-9):
        p = np.asarray(params, dtype=np.float64)
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
# Active Search Agent
# -----------------------------
class ActiveSearchAgent:
    """
    After training TD3 on the surrogate env, this agent:
      - proposes candidates from the policy (with optional noise)
      - de-dups and avoids re-evaluating existing points
      - queries the Julia oracle
      - appends new rows to the dataset

    Optionally: retrain surrogate + TD3 every `retrain_every` new evaluations.
    """
    def __init__(
        self,
        csv_path: str,
        oracle: JuliaOracle,
        target: str,
        param_lo: float,
        param_hi: float,
        device: str,
        start_system: str,
        tol_real: float,
        tol_pos: float,
        seed: int,
        policy: TD3,
        surrogate_env: SurrogateBanditEnv,
        proposal_noise: float = 0.10,
        dedup_decimals: int = 4,
    ):
        self.csv_path = csv_path
        self.oracle = oracle
        self.target = target
        self.param_lo = float(param_lo)
        self.param_hi = float(param_hi)
        self.device = device
        self.start_system = start_system
        self.tol_real = float(tol_real)
        self.tol_pos = float(tol_pos)
        self.seed = int(seed)

        self.policy = policy
        self.env = surrogate_env

        self.proposal_noise = float(proposal_noise)
        self.dedup_decimals = int(dedup_decimals)

        self.rng = np.random.default_rng(self.seed)

    def propose_pool(self, n: int):
        obs, _ = self.env.reset()
        props = []
        for _ in range(n):
            a, _ = self.policy.predict(obs, deterministic=False)
            a = np.asarray(a, dtype=np.float32)
            if self.proposal_noise > 0:
                a = np.clip(
                    a + self.rng.normal(0, self.proposal_noise, size=a.shape).astype(np.float32),
                    -1.0, 1.0
                )
            _, r, *_ , info = self.env.step(a)
            props.append((float(r), info["params"]))

        props.sort(key=lambda t: t[0], reverse=True)
        return props  # list of (pred_reward, params)

    def filter_novel(self, df: pd.DataFrame, pool_params, k: int):
        pool_params = dedup_by_rounding(pool_params, decimals=self.dedup_decimals)

        existing = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        existing_keys = set(tuple(np.round(row, self.dedup_decimals)) for row in existing)

        filtered = []
        for p in pool_params:
            key = tuple(np.round(p, self.dedup_decimals))
            if key in existing_keys:
                continue
            filtered.append(p)
            if len(filtered) >= k:
                break
        return filtered

    def oracle_eval_and_append(self, params_list, eval_batch_size: int):
        new_rows = []
        t0 = time.time()

        for i in range(0, len(params_list), eval_batch_size):
            batch = params_list[i:i + eval_batch_size]
            for p in batch:
                out = self.oracle.eval_params(
                    p,
                    start_system=self.start_system,
                    tol_real=self.tol_real,
                    tol_pos=self.tol_pos,
                )
                row = {
                    "ok": out["ok"],
                    "n_solutions": out["n_solutions"],
                    "n_real": out["n_real"],
                    "n_pos": out["n_pos"],
                    "err": out["err"],
                }
                for j in range(31):
                    row[f"k{j+1}"] = float(p[j])
                for j in range(5):
                    row[f"c{j+1}"] = float(p[31 + j])
                new_rows.append(row)

            print(f"[Search] Oracle-evaluated {min(i+eval_batch_size, len(params_list))}/{len(params_list)}")

        dt = time.time() - t0
        new_df = pd.DataFrame(new_rows)
        append_rows(self.csv_path, new_df)

        ok_rate = float(new_df["ok"].mean()) if len(new_df) else 0.0
        best_real = int(new_df["n_real"].max()) if len(new_df) else -1
        best_pos = int(new_df["n_pos"].max()) if len(new_df) else -1

        print(f"[Search] Appended {len(new_df)} rows to {self.csv_path} | ok_rate={ok_rate:.2f} "
              f"| best_n_real={best_real} | best_n_pos={best_pos} | dt={dt:.1f}s")

        return new_df

    def run(
        self,
        total_evals: int,
        proposal_pool: int,
        proposals_per_iter: int,
        eval_batch_size: int,
        retrain_every: int,
        retrain_td3_steps: int,
        # NEW: log & plot paths for surrogate retrains
        reward_hist_csv: str = "rewardnet_history_retrain.csv",
        reward_hist_png: str = "rewardnet_training_retrain.png",
    ):
        """
        Start searching immediately using current policy + surrogate env.
        If retrain_every > 0: every retrain_every new evaluations, refit surrogate & TD3.
        """
        done = 0
        iter_id = 0

        while done < total_evals:
            iter_id += 1
            print(f"\n========== SEARCH ITER {iter_id} | done {done}/{total_evals} ==========")

            df = load_dataset(self.csv_path)

            # Propose
            print(f"[Search] Proposing pool={proposal_pool} ...")
            pool = self.propose_pool(proposal_pool)
            if pool:
                print(f"[Search] Best predicted reward in pool: {pool[0][0]:.4f}")
            pool_params = [p for _, p in pool]

            # Filter novel & select
            k = min(proposals_per_iter, total_evals - done)
            selected = self.filter_novel(df, pool_params, k=k)
            if not selected:
                print("[Search] No novel proposals found after de-dup; stopping.")
                break

            print(f"[Search] Selected {len(selected)} novel points for oracle eval.")
            new_df = self.oracle_eval_and_append(selected, eval_batch_size=eval_batch_size)
            done += len(new_df)

            # Optional retrain loop
            if retrain_every > 0 and done % retrain_every == 0 and done < total_evals:
                print(f"\n[Search] Retraining surrogate + TD3 (trigger: {done} new evals) ...")

                df2 = load_dataset(self.csv_path)
                model, x_scaler, y_scaler, n_train, hist = train_reward_model(
                    df2, target=self.target, device=self.device, log_csv_path=reward_hist_csv
                )
                print(f"[Search] Surrogate retrained on {n_train} successful rows.")

                # Save plot for retrain history
                plot_rewardnet_history(hist, out_png=reward_hist_png)

                # Rebuild env with new surrogate
                self.env = SurrogateBanditEnv(
                    model, x_scaler, y_scaler, device=self.device,
                    param_lo=self.param_lo, param_hi=self.param_hi
                )

                # Re-attach env to policy and continue training
                self.policy.set_env(self.env)
                self.policy.learn(total_timesteps=retrain_td3_steps)
                print("[Search] TD3 updated.")


# -----------------------------
# Build & train policy
# -----------------------------
def build_and_train_policy(
    csv_path: str,
    target: str,
    device: str,
    param_lo: float,
    param_hi: float,
    seed: int,
    td3_timesteps: int,
    tb_dir: str = "tb_logs",
    run_name: str = "wnt_td3",
    reward_hist_csv: str = "rewardnet_history.csv",
    reward_hist_png: str = "rewardnet_training.png",
):
    df = load_dataset(csv_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in CSV columns.")

    model, x_scaler, y_scaler, n_train, hist = train_reward_model(
        df, target=target, device=device, log_csv_path=reward_hist_csv
    )
    print(f"[Train] Surrogate trained on {n_train} successful rows.")

    # Save RewardNet plot
    plot_rewardnet_history(hist, out_png=reward_hist_png)

    env = SurrogateBanditEnv(
        model, x_scaler, y_scaler,
        device=device, param_lo=param_lo, param_hi=param_hi
    )

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
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,   # NEW
    )

    print(f"[Train] Training TD3 for {td3_timesteps} steps (surrogate bandit)...")
    td3.learn(total_timesteps=td3_timesteps, tb_log_name=run_name)  # NEW: tb_log_name

    return td3, env


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True, help="Path to wnt_samples.csv (will be appended to)")
    ap.add_argument("--oracle", required=True, help="Path to wnt_oracle.jl")

    ap.add_argument("--target", default="n_real", choices=["n_real", "n_pos"], help="Objective for surrogate/TD3")
    ap.add_argument("--param_lo", type=float, default=0.0)
    ap.add_argument("--param_hi", type=float, default=100.0)

    # TRAIN phase
    ap.add_argument("--td3_steps", type=int, default=200_000)
    ap.add_argument("--policy_path", default="td3_out/td3_policy.zip")

    # TensorBoard logging
    ap.add_argument("--tb_dir", default="tb_logs", help="TensorBoard log dir")
    ap.add_argument("--run_name", default="wnt_td3", help="TensorBoard run name")

    # RewardNet history outputs
    ap.add_argument("--reward_hist_csv", default="rewardnet_history.csv", help="RewardNet history CSV (train)")
    ap.add_argument("--reward_hist_png", default="rewardnet_training.png", help="RewardNet history plot PNG (train)")
    ap.add_argument("--reward_hist_csv_retrain", default="rewardnet_history_retrain.csv",
                    help="RewardNet history CSV (retrain during search)")
    ap.add_argument("--reward_hist_png_retrain", default="rewardnet_training_retrain.png",
                    help="RewardNet history plot PNG (retrain during search)")

    # SEARCH phase
    ap.add_argument("--total_evals", type=int, default=250, help="Total oracle eval budget for the search agent")
    ap.add_argument("--proposal_pool", type=int, default=5000)
    ap.add_argument("--proposals_per_iter", type=int, default=50)
    ap.add_argument("--proposal_noise", type=float, default=0.10)
    ap.add_argument("--eval_batch_size", type=int, default=10)

    # optional retrain during search
    ap.add_argument("--retrain_every", type=int, default=0,
                    help="If >0, retrain surrogate+TD3 every this many new evals during search")
    ap.add_argument("--retrain_td3_steps", type=int, default=50_000)

    # misc
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_system", default="polyhedral", choices=["polyhedral", "total_degree"])
    ap.add_argument("--tol_real", type=float, default=1e-7)
    ap.add_argument("--tol_pos", type=float, default=1e-9)

    # mode
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--search_only", action="store_true")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.policy_path), exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)

    oracle = JuliaOracle(args.oracle)

    # -------- TRAIN (unless search_only) --------
    td3 = None
    env = None

    if not args.search_only:
        td3, env = build_and_train_policy(
            csv_path=args.csv,
            target=args.target,
            device=args.device,
            param_lo=args.param_lo,
            param_hi=args.param_hi,
            seed=args.seed,
            td3_timesteps=args.td3_steps,
            tb_dir=args.tb_dir,
            run_name=args.run_name,
            reward_hist_csv=args.reward_hist_csv,
            reward_hist_png=args.reward_hist_png,
        )
        td3.save(args.policy_path)
        print(f"[Train] Saved policy to {args.policy_path}")

        if args.train_only:
            return

    # -------- SEARCH (unless train_only) --------
    if not args.train_only:
        if td3 is None or env is None:
            # Build env from current surrogate, then load policy weights
            # (we use td3_timesteps=1 as a dummy fit to instantiate TD3 object properly before load)
            td3, env = build_and_train_policy(
                csv_path=args.csv,
                target=args.target,
                device=args.device,
                param_lo=args.param_lo,
                param_hi=args.param_hi,
                seed=args.seed,
                td3_timesteps=1,
                tb_dir=args.tb_dir,
                run_name=args.run_name,
                reward_hist_csv=args.reward_hist_csv,
                reward_hist_png=args.reward_hist_png,
            )
            td3 = TD3.load(args.policy_path, env=env, device=args.device)
            print(f"[Search] Loaded policy from {args.policy_path}")

        agent = ActiveSearchAgent(
            csv_path=args.csv,
            oracle=oracle,
            target=args.target,
            param_lo=args.param_lo,
            param_hi=args.param_hi,
            device=args.device,
            start_system=args.start_system,
            tol_real=args.tol_real,
            tol_pos=args.tol_pos,
            seed=args.seed,
            policy=td3,
            surrogate_env=env,
            proposal_noise=args.proposal_noise,
            dedup_decimals=4,
        )

        agent.run(
            total_evals=args.total_evals,
            proposal_pool=args.proposal_pool,
            proposals_per_iter=args.proposals_per_iter,
            eval_batch_size=args.eval_batch_size,
            retrain_every=args.retrain_every,
            retrain_td3_steps=args.retrain_td3_steps,
            reward_hist_csv=args.reward_hist_csv_retrain,
            reward_hist_png=args.reward_hist_png_retrain,
        )

        td3.save(args.policy_path)
        print(f"[Search] Saved policy checkpoint to {args.policy_path}")


if __name__ == "__main__":
    main()