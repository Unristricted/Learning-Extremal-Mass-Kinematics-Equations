# td3_lstm.py
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ==============================
# Tiny Toy Environment (continuous, state in [0,1]^n, actions in [-1,1]^m)
# Replace with your real env (e.g., gymnasium)
# ==============================
class ToyEnv:
    """
    State: n-dim vector with each element in [0,1].
    Action: m-dim in [-1,1]. Reward encourages driving state toward a target vector.
    Dynamics: s_{t+1} = clip(s_t + 0.05 * action[:n], [0,1])
    """
    def __init__(self, initial_state, reward_fn, n_state=6, m_action=6, horizon=200, lambda_=0.5, target=None):
        self.n = n_state
        self.m = m_action
        self.horizon = horizon
        self.lambda_ = lambda_
        self.t = 0
        self.state = initial_state
        self.target = np.ones(self.n, dtype=np.float32) * 0.85 if target is None else np.asarray(target, dtype=np.float32)
        self.reward_fn = reward_fn #Our MC implementation

        # scale step
        self.alpha = 0.05

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        self.state = np.random.rand(self.n).astype(np.float32)  # in [0,1]
        return self.state.copy(), {}

    def step(self, delta_action):
        self.t += 1
        delta_action = np.asarray(delta_action, dtype=np.float32)
        delta_action = np.clip(delta_action, -1.0, 1.0)  # Ensure delta is in [-1, 1]

        # Compute the weighted action
        # Is this a matrix operation or scalar operation? 🤔
        # It is not we can probably use torch.matmul or some np function. Torch is better because we will need to convert to tensor to do anything
        # Actually I think it does!
        action = self.lambda_ * self.state + (1 - self.lambda_) * delta_action

        # Use first n dims to move state
        delta = self.alpha * action[: self.n]
        next_state = np.clip(action, -1.0, 1.0)

        # Reward: our reward is calculated using
        # TODO: track previous MC scores and calculate reward based on distance
        reward = self.reward_fn(next_state)

        #Potential Idea
        '''
        reward_ratio = reward/prev_reward
        if reward_ratio > 1:
            continue with this action
        else:
            select new action
        '''

        self.state = next_state
        done = self.t >= self.horizon
        truncated = False
        info = {}
        return self.state.copy(), float(reward), bool(done), bool(truncated), info


# ==============================
# Run a quick smoke test
# ==============================
if __name__ == "__main__":
    n = 6              # state dimension (each element max 1.0)
    m = 6              # action dimension
    env = ToyEnv(n_state=n, m_action=m, horizon=150)
    eval_env = ToyEnv(n_state=n, m_action=m, horizon=150)

    cfg = TD3Config(
        state_dim=n,
        action_dim=m,
        actor_lstm_hidden=128,
        actor_head_hidden=128,
        critic_hidden=256,
        start_timesteps=2_000,   # shorter for the toy demo
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = train_td3(env, eval_env=eval_env, steps=50_000, cfg=cfg, eval_interval=10_000)
