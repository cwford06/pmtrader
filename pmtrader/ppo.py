"""Proximal Policy Optimization (clipped objective, GAE) in pure NumPy.

Actor: MLP -> softmax over discrete target-position levels.
Critic: MLP -> state value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np

from .env import N_ACTIONS, OBS_DIM, TradingEnv
from .nn import MLP, Adam


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@dataclass
class PPOConfig:
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    ent_coef: float = 0.10          # start of linear entropy schedule
    ent_coef_final: float = 0.002   # end of schedule
    lr_final: float = 5e-5
    vf_coef: float = 0.5
    n_envs: int = 256


@dataclass
class TrainHistory:
    iteration: list = field(default_factory=list)
    mean_pnl: list = field(default_factory=list)       # $ per market (max 100 contracts)
    entropy: list = field(default_factory=list)
    value_loss: list = field(default_factory=list)
    clip_frac: list = field(default_factory=list)


class PPOAgent:
    def __init__(self, cfg: PPOConfig | None = None, seed: int = 0):
        self.cfg = cfg or PPOConfig()
        self.rng = np.random.default_rng(seed)
        h = self.cfg.hidden
        self.actor = MLP([OBS_DIM, h, h, N_ACTIONS], self.rng, out_scale=0.01)
        self.critic = MLP([OBS_DIM, h, h, 1], self.rng, out_scale=1.0)
        self._make_opts()
        self.history = TrainHistory()

    def _make_opts(self):
        self.opt_a = Adam(self.actor.params, lr=self.cfg.lr)
        self.opt_c = Adam(self.critic.params, lr=self.cfg.lr)

    # -- inference ----------------------------------------------------------
    def policy(self, obs):
        return softmax(self.actor.forward(obs))

    def value(self, obs):
        return self.critic.forward(obs)[:, 0]

    def act(self, obs, greedy=False):
        pi = self.policy(obs)
        if greedy:
            return pi.argmax(-1), pi
        u = self.rng.random((len(pi), 1))
        a = (pi.cumsum(-1) < u).sum(-1)
        return np.minimum(a, pi.shape[1] - 1), pi

    # -- training -----------------------------------------------------------
    def collect(self, env: TradingEnv):
        obs_l, act_l, logp_l, rew_l, val_l = [], [], [], [], []
        obs = env.reset()
        done = False
        pnl = np.zeros(env.n)
        while not done:
            a, pi = self.act(obs)
            v = self.value(obs)
            nobs, r, done, info = env.step(a)
            obs_l.append(obs); act_l.append(a); rew_l.append(r); val_l.append(v)
            logp_l.append(np.log(pi[np.arange(len(a)), a] + 1e-12))
            pnl += info["pnl"]
            obs = nobs
        # GAE (terminal value = 0 at settlement)
        T = len(rew_l)
        R, V = np.array(rew_l), np.array(val_l)              # (T, N)
        adv = np.zeros_like(R)
        last = np.zeros(env.n)
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else 0.0
            delta = R[t] + self.cfg.gamma * nv - V[t]
            last = delta + self.cfg.gamma * self.cfg.lam * last
            adv[t] = last
        ret = adv + V
        flat = lambda x: np.asarray(x).reshape(T * env.n, *np.asarray(x).shape[2:])
        return dict(obs=flat(obs_l), act=flat(act_l), logp=flat(logp_l),
                    adv=flat(adv), ret=flat(ret)), pnl

    def update(self, data, ent_coef=None):
        c = self.cfg
        ent = c.ent_coef if ent_coef is None else ent_coef
        n = len(data["act"])
        adv = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
        stats = {"entropy": [], "vloss": [], "clipfrac": []}
        for _ in range(c.epochs):
            idx = self.rng.permutation(n)
            for i in range(0, n, c.minibatch):
                j = idx[i:i + c.minibatch]
                o, a, A = data["obs"][j], data["act"][j], adv[j]
                m = len(j)
                # actor
                pi = self.policy(o)
                logpi = np.log(pi + 1e-12)
                ratio = np.exp(logpi[np.arange(m), a] - data["logp"][j])
                clipped = ((A > 0) & (ratio > 1 + c.clip)) | ((A < 0) & (ratio < 1 - c.clip))
                dlogp = np.where(clipped, 0.0, -ratio * A) / m     # d loss / d log pi(a)
                onehot = np.zeros_like(pi); onehot[np.arange(m), a] = 1
                dz = dlogp[:, None] * (onehot - pi)
                H = -(pi * logpi).sum(-1)
                dz += ent * pi * (logpi + H[:, None]) / m  # grad of -ent_coef*H
                self.opt_a.step(self.actor.backward(dz))
                # critic
                v = self.value(o)
                dv = c.vf_coef * 2 * (v - data["ret"][j]) / m
                self.opt_c.step(self.critic.backward(dv[:, None]))
                stats["entropy"].append(H.mean())
                stats["vloss"].append(np.mean((v - data["ret"][j]) ** 2))
                stats["clipfrac"].append(clipped.mean())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(self, env: TradingEnv, iterations: int, callback=None):
        c = self.cfg
        for it in range(iterations):
            frac = it / max(iterations - 1, 1)
            lr = c.lr + (c.lr_final - c.lr) * frac
            self.opt_a.lr = self.opt_c.lr = lr
            data, pnl = self.collect(env)
            s = self.update(data, ent_coef=c.ent_coef + (c.ent_coef_final - c.ent_coef) * frac)
            h = self.history
            h.iteration.append(len(h.iteration) + 1)
            h.mean_pnl.append(float(pnl.mean()))
            h.entropy.append(s["entropy"]); h.value_loss.append(s["vloss"]); h.clip_frac.append(s["clipfrac"])
            if callback and callback(it, h) is False:
                break
        return self.history

    # -- persistence --------------------------------------------------------
    def state(self):
        return {"cfg": asdict(self.cfg), "actor": self.actor.state(), "critic": self.critic.state(),
                "history": asdict(self.history)}

    def load(self, s):
        self.cfg = PPOConfig(**s["cfg"])
        self.actor.load(s["actor"]); self.critic.load(s["critic"])
        self.history = TrainHistory(**s.get("history", {}))
        self._make_opts()
        return self

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.state(), f)
