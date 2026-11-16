"""Synthetic prediction-market generator.

Each market has a latent "truth" process L_t (Brownian motion). The event
resolves YES iff L_T > 0, so the true probability at time t is

    p*(t) = Phi( L_t / (sigma * sqrt(T - t)) ).

The traded price is a *biased, noisy* view of p*:
  * favorite-longshot bias  - prices compressed toward 0.5 (k < 1 in logit space)
  * mean-reverting mispricing - AR(1) noise in logit space (crowd over/under-reaction)
  * a bid/ask spread around the mid

An external "signal" (think polls, a sportsbook line, a weather model) is a
noisy, independent read on p*. The ML probability model learns to combine
price + signal; the RL agent learns when and how much to trade on that edge.

This is a stylised testbed. Real markets have different (and changing)
inefficiencies, so anything trained here must be re-validated on real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from math import erf, sqrt

_vphi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class SimConfig:
    T: int = 40                    # decision steps per market
    sigma: float = 1.0             # truth volatility per step
    flb_k: float = 0.80            # favorite-longshot compression (1 = none)
    mispricing_sd: float = 0.35    # stationary sd of logit mispricing
    mispricing_ar: float = 0.85    # persistence of mispricing
    signal_sd: float = 0.45        # noise of external signal (logit space)
    spread: float = 0.02           # bid/ask spread in $ (payout = 1)
    fee_rate: float = 0.07


@dataclass
class MarketBatch:
    """N markets x (T+1) time points. Index t = 0..T; t = T is resolution."""
    p_true: np.ndarray     # (N, T+1)
    mid: np.ndarray        # (N, T+1)
    signal: np.ndarray     # (N, T+1)
    outcome: np.ndarray    # (N,)
    cfg: SimConfig = field(repr=False)

    @property
    def n(self):
        return self.p_true.shape[0]


def simulate(n: int, cfg: SimConfig, rng: np.random.Generator) -> MarketBatch:
    T, s = cfg.T, cfg.sigma
    # random starting point gives a spread of initial probabilities
    L0 = rng.normal(0, s * np.sqrt(T) * 0.9, n)
    steps = rng.normal(0, s, (n, T))
    L = np.concatenate([L0[:, None], L0[:, None] + np.cumsum(steps, 1)], 1)
    remaining = np.arange(T, -1, -1, dtype=float)                   # T..0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = L / (s * np.sqrt(np.maximum(remaining, 1e-9)))
    p_true = _vphi(np.clip(z, -8, 8))
    outcome = (L[:, -1] > 0).astype(float)
    p_true[:, -1] = outcome

    # AR(1) mispricing in logit space, fading near expiry
    ar, sd = cfg.mispricing_ar, cfg.mispricing_sd
    eps = np.zeros((n, T + 1))
    eps[:, 0] = rng.normal(0, sd, n)
    innov_sd = sd * np.sqrt(1 - ar ** 2)
    for t in range(1, T + 1):
        eps[:, t] = ar * eps[:, t - 1] + rng.normal(0, innov_sd, n)
    fade = np.sqrt(remaining / T)
    mid = sigmoid(cfg.flb_k * logit(p_true) + eps * fade)
    half = cfg.spread / 2
    mid = np.clip(mid, half + 0.005, 1 - half - 0.005)
    mid[:, -1] = outcome                                            # settles
    signal = sigmoid(logit(p_true) + rng.normal(0, cfg.signal_sd, (n, T + 1)))
    return MarketBatch(p_true, mid, signal, outcome, cfg)
