import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_ACT = 36
D_STATE = 1  # dummy

# ---- Networks ----
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self, s):
        return self.net(s)

class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

# ---- Replay buffer built from dataset ----
class OfflineBuffer:
    def __init__(self, s, a, r, sp, done):
        self.s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        self.a = torch.tensor(a, dtype=torch.float32, device=DEVICE)
        self.r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        self.sp = torch.tensor(sp, dtype=torch.float32, device=DEVICE)
        self.done = torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        self.n = self.s.shape[0]

    def sample(self, batch_size):
        idx = torch.randint(0, self.n, (batch_size,), device=DEVICE)
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # keep only successful solves
    df = df[df["status"] == "ok"].copy()

    a_cols = [f"a_{i}" for i in range(1, 37)]
    A = df[a_cols].values.astype(np.float32)

    # reward target (choose one)
    R = df["n_real"].values.astype(np.float32)  # or df["n_pos"]

    # Optional: normalize rewards for stability (keeps ranking)
    R = (R - R.mean()) / (R.std() + 1e-8)

    # dummy state
    S = np.zeros((len(df), 1), dtype=np.float32)
    SP = np.zeros((len(df), 1), dtype=np.float32)
    DONE = np.ones((len(df),), dtype=np.float32)  # 1-step

    # Normalize actions (strongly recommended)
    a_mean = A.mean(axis=0, keepdims=True)
    a_std = A.std(axis=0, keepdims=True) + 1e-6
    A_norm = (A - a_mean) / a_std

    norms = {"a_mean": a_mean, "a_std": a_std}
    return S, A_norm, R, SP, DONE, norms

# ---- TD3 training (offline) ----
def train_td3_offline(csv_path, steps=200_000, batch=256, gamma=0.0,
                      policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                      tau=0.005, lr=3e-4):

    S, A, R, SP, DONE, norms = load_dataset(csv_path)
    buf = OfflineBuffer(S, A, R, SP, DONE)

    actor = Actor(D_STATE, D_ACT).to(DEVICE)
    actor_t = Actor(D_STATE, D_ACT).to(DEVICE)
    actor_t.load_state_dict(actor.state_dict())

    c1 = Critic(D_STATE, D_ACT).to(DEVICE)
    c2 = Critic(D_STATE, D_ACT).to(DEVICE)
    c1_t = Critic(D_STATE, D_ACT).to(DEVICE)
    c2_t = Critic(D_STATE, D_ACT).to(DEVICE)
    c1_t.load_state_dict(c1.state_dict())
    c2_t.load_state_dict(c2.state_dict())

    opt_a = optim.Adam(actor.parameters(), lr=lr)
    opt_c = optim.Adam(list(c1.parameters()) + list(c2.parameters()), lr=lr)

    # gamma=0.0 is correct for 1-step bandit; keep it explicit
    for t in range(steps):
        s, a, r, sp, done = buf.sample(batch)

        with torch.no_grad():
            # target policy smoothing
            noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = actor_t(sp) + noise

            # TD target
            q1_t = c1_t(sp, a2)
            q2_t = c2_t(sp, a2)
            q_t = torch.min(q1_t, q2_t)

            y = r + (1.0 - done) * gamma * q_t

        q1 = c1(s, a)
        q2 = c2(s, a)
        critic_loss = ((q1 - y).pow(2).mean() + (q2 - y).pow(2).mean())

        opt_c.zero_grad()
        critic_loss.backward()
        opt_c.step()

        # delayed policy update
        if t % policy_delay == 0:
            actor_loss = -c1(s, actor(s)).mean()

            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            # soft update targets
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)
                for p, pt in zip(c1.parameters(), c1_t.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)
                for p, pt in zip(c2.parameters(), c2_t.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)

        if (t + 1) % 5000 == 0:
            print(f"step {t+1}: critic_loss={critic_loss.item():.4f}")

    return actor, norms

def propose_action(actor, norms, n=1):
    # dummy state
    s = torch.zeros((n, 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        a_norm = actor(s).cpu().numpy()
    # unnormalize back to original log-parameter scale
    a = a_norm * norms["a_std"] + norms["a_mean"]
    return a