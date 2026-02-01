from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gazebo_rl.gazebo_env import GazeboEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.size = size
        self.ptr = 0
        self.len = 0
        self.device = device

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.len, size=batch_size)
        obs = to_torch(self.obs[idx], self.device)
        acts = to_torch(self.acts[idx], self.device)
        rews = to_torch(self.rews[idx], self.device)
        next_obs = to_torch(self.next_obs[idx], self.device)
        done = to_torch(self.done[idx], self.device)
        return obs, acts, rews, next_obs, done


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.backbone = MLP(obs_dim, 256, hidden=hidden[:1] if len(hidden) > 0 else (256,))
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.backbone(obs))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)

        log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        mu_action = torch.tanh(mu)
        return a, log_prob, mu_action


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden=hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


@dataclass
class SACConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_steps: int = 120_000
    start_steps: int = 2_000
    batch_size: int = 256
    replay_size: int = 300_000

    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4

    update_after: int = 1_000
    updates_per_step: int = 1

    auto_alpha: bool = True
    target_entropy: Optional[float] = None

    log_every: int = 1_000
    eval_every: int = 5_000
    eval_episodes: int = 3
    save_every: int = 10_000
    save_dir: str = "sac_checkpoints_xyyaw"


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.act_low = act_low.astype(np.float32)
        self.act_high = act_high.astype(np.float32)

        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_bias = (self.act_high + self.act_low) / 2.0

        self.pi = GaussianPolicy(obs_dim, act_dim).to(self.device)
        self.q1 = QFunction(obs_dim, act_dim).to(self.device)
        self.q2 = QFunction(obs_dim, act_dim).to(self.device)
        self.q1_targ = QFunction(obs_dim, act_dim).to(self.device)
        self.q2_targ = QFunction(obs_dim, act_dim).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.policy_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.q_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.q_lr)

        if cfg.target_entropy is None:
            self.target_entropy = -float(act_dim)
        else:
            self.target_entropy = float(cfg.target_entropy)

        if cfg.auto_alpha:
            self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(0.2), device=self.device)
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _squash_to_env(self, a_tanh: np.ndarray) -> np.ndarray:
        return (a_tanh * self.act_scale + self.act_bias).astype(np.float32)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = to_torch(obs[None, :], self.device)
        with torch.no_grad():
            a_tanh, _, mu_tanh = self.pi.sample(obs_t)
        a = mu_tanh[0].cpu().numpy() if deterministic else a_tanh[0].cpu().numpy()
        return self._squash_to_env(a)

    def update(self, rb: ReplayBuffer) -> Dict[str, float]:
        cfg = self.cfg
        obs, act, rew, next_obs, done = rb.sample(cfg.batch_size)

        act_norm = (act - to_torch(self.act_bias, self.device)) / (to_torch(self.act_scale, self.device) + 1e-8)
        act_norm = torch.clamp(act_norm, -1.0, 1.0)

        with torch.no_grad():
            next_a, next_logp, _ = self.pi.sample(next_obs)
            q1_next = self.q1_targ(next_obs, next_a)
            q2_next = self.q2_targ(next_obs, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            backup = rew + cfg.gamma * (1.0 - done) * q_next

        q1 = self.q1(obs, act_norm)
        q2 = self.q2(obs, act_norm)
        q1_loss = ((q1 - backup) ** 2).mean()
        q2_loss = ((q2 - backup) ** 2).mean()

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        a_new, logp, _ = self.pi.sample(obs)
        q1_new = self.q1(obs, a_new)
        q2_new = self.q2(obs, a_new)
        q_new = torch.min(q1_new, q2_new)
        pi_loss = (self.alpha * logp - q_new).mean()

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.auto_alpha and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_targ.parameters()):
                tp.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_targ.parameters()):
                tp.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "q1": float(q1.mean().item()),
            "q2": float(q2.mean().item()),
            "logp": float(logp.mean().item()),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "pi": self.pi.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_targ": self.q1_targ.state_dict(),
                "q2_targ": self.q2_targ.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )


def run_eval(env: GazeboEnv, agent: SACAgent, episodes: int = 3, max_steps: int = 1500) -> Dict[str, float]:
    rets = []
    lens = []
    goals = 0
    for _ in range(episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        for t in range(max_steps):
            act = agent.act(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(act)
            ep_ret += float(r)
            if terminated:
                if info.get("reached_goal", False):
                    goals += 1
                break
            if truncated:
                break
        rets.append(ep_ret)
        lens.append(t + 1)
    return {
        "eval_return_mean": float(np.mean(rets)),
        "eval_return_std": float(np.std(rets)),
        "eval_len_mean": float(np.mean(lens)),
        "eval_goal_rate": float(goals) / float(episodes),
    }


def main():
    cfg = SACConfig(seed=0)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = GazeboEnv(
        world_name="depot",
        cmd_topic="/cmd_vel",
        amcl_pose_topic="/amcl_pose",
        path_topic="/planned_path",
        robot_name_in_gz="robot",
        reset_pose_xyzyaw=(-4.0, -2.0, 0.9, 0.0),
        ign_bin="ign",
        dt=0.0,
        real_time_sleep=False,
        max_episode_steps=1500,
        initialpose_xyyaw=(11.0, 6.0, 0.0),
        lock_first_path_forever=True,
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    act_low = env.action_space.low
    act_high = env.action_space.high

    agent = SACAgent(obs_dim, act_dim, act_low, act_high, cfg)
    rb = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device=device)

    os.makedirs(cfg.save_dir, exist_ok=True)

    obs, info = env.reset()
    ep_ret = 0.0
    ep_len = 0
    ep_num = 1
    t0 = time.time()
    metrics = None

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.start_steps:
            act = env.action_space.sample()
        else:
            act = agent.act(obs, deterministic=False)

        next_obs, r, terminated, truncated, info = env.step(act)

        done = float(terminated or truncated)
        rb.add(obs, act, [r], next_obs, [done])

        obs = next_obs
        ep_ret += float(r)
        ep_len += 1

        if terminated or truncated:
            print(
                f"[train] ep={ep_num:04d} steps={ep_len:4d} return={ep_ret:+.3f} "
                f"term={terminated} trunc={truncated} missing={info.get('missing', False)} "
                f"wall={(time.time()-t0):.1f}s"
            )
            obs, info = env.reset()
            ep_ret = 0.0
            ep_len = 0
            ep_num += 1

        if step >= cfg.update_after and rb.len >= cfg.batch_size:
            metrics = agent.update(rb)

        if (step % cfg.log_every) == 0 and metrics is not None:
            print(
                f"[log] step={step:06d} "
                f"q1_loss={metrics['q1_loss']:.3f} q2_loss={metrics['q2_loss']:.3f} "
                f"pi_loss={metrics['pi_loss']:.3f} alpha={metrics['alpha']:.3f} "
                f"logp={metrics['logp']:.3f}"
            )

        if (step % cfg.eval_every) == 0 and step >= cfg.update_after:
            ev = run_eval(env, agent, episodes=cfg.eval_episodes, max_steps=env._max_episode_steps)
            print(
                f"[eval] step={step:06d} "
                f"R={ev['eval_return_mean']:+.2f}±{ev['eval_return_std']:.2f} "
                f"len={ev['eval_len_mean']:.1f} goal_rate={ev['eval_goal_rate']:.2f}"
            )

        if (step % cfg.save_every) == 0 and step >= cfg.update_after:
            ckpt = os.path.join(cfg.save_dir, f"sac_step_{step:06d}.pt")
            agent.save(ckpt)
            print(f"[save] {ckpt}")

    env.step(np.array([0.0, 0.0], dtype=np.float32))
    env.close()


if __name__ == "__main__":
    main()
