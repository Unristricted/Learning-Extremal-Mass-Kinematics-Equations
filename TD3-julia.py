# td3_lstm.py
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import time
from functools import lru_cache

from juliacall import Main as jl

# ==============================
# Utils
# ==============================
def fanin_init(tensor, fanin=None):
    fanin = fanin or tensor.size(0)
    bound = 1. / math.sqrt(fanin)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ==============================
# Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6), device="cpu"):
        self.capacity = capacity
        self.device = device

        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = to_tensor(self.states[idxs], self.device)
        actions = to_tensor(self.actions[idxs], self.device)
        rewards = to_tensor(self.rewards[idxs], self.device)
        next_states = to_tensor(self.next_states[idxs], self.device)
        dones = to_tensor(self.dones[idxs], self.device)
        return states, actions, rewards, next_states, dones


# ==============================
# Actor (LSTM) and Critics (MLPs)
# ==============================
class ActorLSTM(nn.Module):
    """
    LSTM-based policy. We keep a hidden state between environment steps.
    For training from replay, we use zero-init hidden state (sequence length = 1).
    """
    def __init__(self, state_dim, action_dim, hidden_size=128, lstm_layers=1, out_hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, out_hidden)
        self.fc2 = nn.Linear(out_hidden, action_dim)

        # Init
        fanin_init(self.fc1.weight)
        fanin_init(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x: (B, state_dim) or (B, 1, state_dim)
        Returns: action (B, action_dim), new_hidden
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,state_dim)
        out, new_hidden = self.lstm(x, hidden)        # (B,1,H)
        h = F.relu(self.fc1(out[:, -1, :]))
        act = torch.tanh(self.fc2(h))                 # actions in [-1,1]
        return act, new_hidden

    @torch.no_grad()
    def act(self, x, hidden=None):
        a, new_hidden = self.forward(x, hidden)
        return a, new_hidden


class Critic(nn.Module):
    """
    Twin Q networks: Q1(s,a) and Q2(s,a)
    """
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fanin_init(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


# ==============================
# TD3 Agent (with LSTM Actor)
# ==============================
@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    actor_lstm_hidden: int = 128
    actor_lstm_layers: int = 1
    actor_head_hidden: int = 128
    critic_hidden: int = 256

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2

    batch_size: int = 256
    buffer_capacity: int = int(1e6)

    # exploration
    start_timesteps: int = 25_000
    expl_noise: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TD3Agent:
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.device = cfg.device

        self.actor = ActorLSTM(cfg.state_dim, cfg.action_dim, cfg.actor_lstm_hidden,
                               cfg.actor_lstm_layers, cfg.actor_head_hidden).to(self.device)
        self.actor_target = ActorLSTM(cfg.state_dim, cfg.action_dim, cfg.actor_lstm_hidden,
                                      cfg.actor_lstm_layers, cfg.actor_head_hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.critic_hidden).to(self.device)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim, cfg.critic_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_capacity, cfg.device)

        # hidden state carried during environment interaction (not stored)
        self.actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.total_it = 0

    def reset_hidden(self, batch_size=1):
        # Let PyTorch allocate zeros automatically by passing None; we keep this for clarity.
        self.actor_hidden = None

    @torch.no_grad()
    def select_action(self, state: np.ndarray, add_noise=True):
        state_t = to_tensor(state[None, :], self.device)
        action_t, self.actor_hidden = self.actor.act(state_t, self.actor_hidden)
        action = action_t.cpu().numpy()[0]
        if add_noise:
            action = np.clip(action + np.random.normal(0, self.cfg.expl_noise, size=action.shape), -1.0, 1.0)
        return action

    def train_step(self):
        if self.replay.size < self.cfg.batch_size:
            return {}

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            # Target policy smoothing
            # Use zero-init hidden state for target actor; sequence len = 1
            target_actions, _ = self.actor_target(next_states, hidden=None)
            noise = (torch.randn_like(target_actions) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            target_actions = (target_actions + noise).clamp(-1.0, 1.0)

            # Compute target Q
            target_q1, target_q2 = self.critic_target(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.cfg.gamma * target_q

        # Critic update
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item()}

        # Delayed policy updates
        if self.total_it % self.cfg.policy_delay == 0:
            # Actor update (maximize Q1)
            # Zero-init hidden state for training step
            pi_actions, _ = self.actor(states, hidden=None)
            actor_loss = -self.critic.q1_only(states, pi_actions).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()

            # Polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_targ.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return info


# ==============================
# Minimal Training Loop
# ==============================
def train_td3(env, eval_env=None, seed=0, steps=300_000, cfg: TD3Config = None, eval_interval=10_000):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    assert cfg is not None
    agent = TD3Agent(cfg)

    state, _ = env.reset(seed=seed)
    agent.reset_hidden()
    episode_return = 0.0
    episode_len = 0

    for t in range(1, steps + 1):
        if t < cfg.start_timesteps:
            action = env.action_space_sample()
        else:
            action = agent.select_action(state, add_noise=True)

        next_state, reward, done, truncated, info = env.step(action)
        agent.replay.add(state, action, reward, next_state, done or truncated)

        state = next_state
        episode_return += reward
        episode_len += 1

        # Train
        train_info = agent.train_step()

        if done or truncated:
            # Reset episode and hidden state
            state, _ = env.reset()
            agent.reset_hidden()
            episode_return = 0.0
            episode_len = 0

        if eval_env is not None and (t % eval_interval == 0):
            eval_ret = evaluate_policy(agent, eval_env)
            print(f"[step {t}] eval_return={eval_ret:.2f} | {train_info}")

    return agent


@torch.no_grad()
def evaluate_policy(agent: TD3Agent, env, episodes: int = 5):
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        agent.reset_hidden()
        done = False
        truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            a = agent.select_action(s, add_noise=False)
            s, r, done, truncated, _ = env.step(a)
            ep_ret += r
        returns.append(ep_ret)
    return float(np.mean(returns))

class JuliaHomotopyOracle:
    """
    Boots Julia once, defines the steady-state system F (ODE ss + 5 conservation laws),
    and provides count_pos(params) where params = [k1..k31, c1..c5] (length 36).
    Reward = # of positive real solutions.
    """
    def __init__(self, timeout_s=30.0, tol_im=1e-8, tol_res=1e-8, tol_pos=1e-10, use_cache=True):
        if jl is None:
            raise ImportError(
                "juliacall is not installed or failed to import. "
                "Install with: pip install juliacall"
            )

        self.timeout_s = float(timeout_s)
        self.tol_im = float(tol_im)
        self.tol_res = float(tol_res)
        self.tol_pos = float(tol_pos)
        self.use_cache = bool(use_cache)

        # Boot Julia and load packages
        jl.seval("using Pkg")
        jl.seval('Pkg.add("HomotopyContinuation")')  # ok if already installed
        jl.seval("using HomotopyContinuation, LinearAlgebra")

        # Define the system and the counter in Julia space (once)
        jl.seval(r"""
        # Variables
        @var x[1:19]
        @var k[1:31]
        @var c[1:5]

        # ODE RHS (Eq. (1))
        f1  = -k[1]*x[1] + k[2]*x[2]
        f2  =  k[1]*x[1] - (k[2] + k[26])*x[2] + k[27]*x[3] - k[3]*x[2]*x[4] + (k[4] + k[5])*x[14]
        f3  =  k[26]*x[2] - k[27]*x[3] - k[14]*x[3]*x[6] + (k[15] + k[16])*x[15]
        f4  = -k[3]*x[2]*x[4] - k[9]*x[4]*x[10] + k[4]*x[14] + k[8]*x[16] + (k[10] + k[11])*x[18]
        f5  = -k[28]*x[5] + k[29]*x[7] - k[6]*x[5]*x[8] + k[5]*x[14] + k[7]*x[16]
        f6  = -k[14]*x[3]*x[6] - k[20]*x[6]*x[11] + k[15]*x[15] + k[19]*x[17] + (k[21] + k[22])*x[19]
        f7  =  k[28]*x[5] - k[29]*x[7] - k[17]*x[7]*x[9] + k[16]*x[15] + k[18]*x[17]
        f8  = -k[6]*x[5]*x[8] + (k[7] + k[8])*x[16]
        f9  = -k[17]*x[7]*x[9] + (k[18] + k[19])*x[17]
        f10 =  k[12] - (k[13] + k[30])*x[10] - k[9]*x[4]*x[10] + k[31]*x[11] + k[10]*x[18]
        f11 = -k[23]*x[11] + k[30]*x[10] - k[31]*x[11] - k[20]*x[6]*x[11] - k[24]*x[11]*x[12] + k[25]*x[13] + k[21]*x[19]
        f12 = -k[24]*x[11]*x[12] + k[25]*x[13]
        f14 =  k[3]*x[2]*x[4] - (k[4] + k[5])*x[14]
        f18 =  k[9]*x[4]*x[10] - (k[10] + k[11])*x[18]

        # Conservation laws (Eq. (2))
        g1 = (x[1] + x[2] + x[3] + x[14] + x[15]) - c[1]
        g2 = (x[4] + x[5] + x[6] + x[7] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19]) - c[2]
        g3 = (x[8] + x[16])  - c[3]
        g4 = (x[9] + x[17])  - c[4]
        g5 = (x[12] + x[13]) - c[5]

        eqs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f18,g1,g2,g3,g4,g5]
        F = System(eqs; variables=x, parameters=vcat(k,c))

        function count_pos_from_params_safe(params_in;
                                   tol_im=1e-8, tol_res=1e-8, tol_pos=1e-10,
                                   start_system=:total_degree)

    # Force a concrete Float64 vector (prevents type/Any issues)
        params = Vector{Float64}(params_in)

        t = @elapsed begin
            try
                res = solve(F, params; start_system=start_system)
                sols = solutions(res)
                good = 0
                for s in sols
                    if maximum(abs.(imag.(s))) > tol_im
                        continue
                    end
                    xr = real.(s)
                    if minimum(xr) <= tol_pos
                        continue
                    end
                    r = evaluate(F, xr, params)
                    if maximum(abs.(r)) > tol_res
                        continue
                    end
                    good += 1
                end
                return good, true, "", t
            catch e
                return 0, false, sprint(showerror, e), t
            end
        end
    end
        """)

    def _key(self, params: np.ndarray):
        # tuple of python floats -> hashable
        return tuple(np.round(params.astype(np.float64), 4).tolist())

    def count_pos(self, params: np.ndarray) -> float:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (36,):
            raise ValueError(f"Expected params shape (36,), got {params.shape}")

        if self.use_cache:
            return self._count_pos_cached(self._key(params))

        return self._count_pos_uncached(params)

    @lru_cache(maxsize=50_000)
    def _count_pos_cached(self, key):
        # reconstruct params from the rounded key
        params = np.array(key, dtype=np.float64)
        return self._count_pos_uncached(params)

    def _count_pos_uncached(self, params: np.ndarray) -> float:
        t0 = time.time()
        try:
            good, ok, errmsg, elapsed = jl.count_pos_from_params_safe(
                params.tolist(),
                tol_im=self.tol_im,
                tol_res=self.tol_res,
                tol_pos=self.tol_pos,
                start_system="total_degree",   # juliacall passes Symbols via strings OK
            )
            self.last_elapsed = float(elapsed)
            self.last_error = None if bool(ok) else str(errmsg)

            if not bool(ok):
                return -1.0

            if self.last_elapsed > self.timeout_s:
                self.last_error = f"timeout>{self.timeout_s}s"
                return -1.0

            return float(good)
        except Exception as e:
            self.last_elapsed = time.time() - t0
            self.last_error = repr(e)
            return -1.0

class HomotopyParamEnv:
    """
    RL environment for parameter search.
    State: theta (log-parameters) for a chosen subset of params
    Action: delta_theta in [-1,1]^d scaled by step_size
    Reward: # positive real steady states from HomotopyContinuation
    """
    def __init__(self, oracle: JuliaHomotopyOracle, baseline_params: np.ndarray,
                 control_idx: np.ndarray, step_size=0.10, horizon=50):
        """
        baseline_params: shape (36,) for [k1..k31,c1..c5]
        control_idx: indices into the 36-vector that the agent controls (e.g., 10 dims)
        """
        self.oracle = oracle
        self.base = np.asarray(baseline_params, dtype=np.float64)
        assert self.base.shape == (36,)
        self.control_idx = np.asarray(control_idx, dtype=np.int64)
        self.d = int(self.control_idx.size)

        self.step_size = float(step_size)
        self.horizon = int(horizon)

        self.t = 0
        self.theta = None  # shape (d,)

    # Your training loop expects these:
    def action_space_sample(self):
        return np.random.uniform(-1.0, 1.0, size=(self.d,)).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0

        # Initialize theta around log(baseline) with small noise
        base_sub = self.base[self.control_idx]
        base_sub = np.clip(base_sub, 1e-12, None)
        self.theta = np.log(base_sub) + 0.01 * np.random.randn(self.d)

        return self.theta.astype(np.float32).copy(), {}

    def step(self, action):
        self.t += 1
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # Update theta in log space
        self.theta = self.theta + self.step_size * action

        # Build full params vector
        params = self.base.copy()
        params[self.control_idx] = np.exp(self.theta)  # keep positive

        # Reward: count positive real solutions
        reward = self.oracle.count_pos(params)
        info = {"reward": reward, "elapsed": getattr(self.oracle, "last_elapsed", None), "err": getattr(self.oracle, "last_error", None)}

        done = self.t >= self.horizon
        truncated = False
        info = {"count_pos": reward}
        return self.theta.astype(np.float32).copy(), float(reward), bool(done), bool(truncated), info


if __name__ == "__main__":
    # ---- baseline params (k1..k31,c1..c5) ----
    # You MUST set these. Start with paper nominal values or something reasonable.
    baseline_params = np.ones(36, dtype=np.float64)  # <-- TODO: replace with real baseline

    # Choose which parameters the agent controls (start small, e.g. 8-12 dims)
    # Indices are 0-based in Python.
    control_idx = np.array([0,1,2,3,4,8,9,10,13,19,30,31,32,33,34,35], dtype=np.int64)  # example

    assert baseline_params.shape == (36,)
    print("params len:", len(baseline_params), "min/max:", baseline_params.min(), baseline_params.max())
    
    oracle = JuliaHomotopyOracle(timeout_s=30.0, use_cache=True)
    env = HomotopyParamEnv(oracle, baseline_params, control_idx, step_size=0.10, horizon=30)
    eval_env = HomotopyParamEnv(oracle, baseline_params, control_idx, step_size=0.10, horizon=30)

    n = env.d
    m = env.d

    cfg = TD3Config(
        state_dim=n,
        action_dim=m,
        actor_lstm_hidden=128,
        actor_head_hidden=128,
        critic_hidden=256,
        start_timesteps=2_000,   # shorter initially
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    r0 = oracle.count_pos(baseline_params)
    print("baseline reward:", r0)
    print("last_error:", oracle.last_error)
    print("last_elapsed:", oracle.last_elapsed)

    # random perturbation test
    for i in range(10):
        p = baseline_params.copy()
        p[control_idx] *= np.exp(0.5*np.random.randn(len(control_idx)))  # big-ish perturb
        print(i, oracle.count_pos(p))

    agent = train_td3(env, eval_env=eval_env, steps=50_000, cfg=cfg, eval_interval=5_000)