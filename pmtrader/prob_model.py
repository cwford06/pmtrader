"""ML probability model: estimates P(YES) from market + signal features.

A small MLP classifier (binary cross-entropy) trained on resolved markets.
Swap `featurize` / training data for real historical markets to use it live.
"""
from __future__ import annotations

import numpy as np

from .nn import MLP, Adam
from .simulator import MarketBatch, logit, sigmoid

FEATURES = ["logit_mid", "logit_signal", "signal_minus_mid", "time_frac", "momentum"]


def featurize(mid, signal, time_frac, momentum) -> np.ndarray:
    """All inputs broadcastable arrays; prices are probabilities (payout = 1)."""
    lm, ls = logit(np.asarray(mid, float)), logit(np.asarray(signal, float))
    tf = np.asarray(time_frac, float) * np.ones_like(lm)
    mo = np.asarray(momentum, float) * np.ones_like(lm)
    return np.stack([lm, ls, ls - lm, tf, mo], -1)


def batch_features(b: MarketBatch, lookback: int = 3) -> np.ndarray:
    """(N, T, F) features for decision times t = 0..T-1."""
    T = b.cfg.T
    lm = logit(b.mid[:, :T])
    mom = np.zeros_like(lm)
    mom[:, lookback:] = lm[:, lookback:] - lm[:, :-lookback]
    tf = np.broadcast_to(1 - np.arange(T) / T, lm.shape)   # fraction of time remaining
    return featurize(b.mid[:, :T], b.signal[:, :T], tf, mom)


class ProbModel:
    def __init__(self, seed: int = 0, hidden: int = 32):
        self.rng = np.random.default_rng(seed)
        self.net = MLP([len(FEATURES), hidden, hidden, 1], self.rng, out_scale=0.1)
        self.mu = np.zeros(len(FEATURES))
        self.sd = np.ones(len(FEATURES))

    def _norm(self, X):
        return (X - self.mu) / self.sd

    def predict(self, X: np.ndarray) -> np.ndarray:
        shape = X.shape[:-1]
        z = self.net.forward(self._norm(X.reshape(-1, X.shape[-1])))[:, 0]
        return sigmoid(z).reshape(shape)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=8, batch=1024, lr=2e-3, verbose=False):
        X = X.reshape(-1, X.shape[-1])
        y = y.reshape(-1)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xn = self._norm(X)
        opt = Adam(self.net.params, lr=lr, max_norm=5.0)
        n = len(y)
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            for i in range(0, n, batch):
                j = idx[i:i + batch]
                z = self.net.forward(Xn[j])[:, 0]
                dz = (sigmoid(z) - y[j]) / len(j)       # d BCE / d logit
                opt.step(self.net.backward(dz[:, None]))
            if verbose:
                p = self.predict(X[:20000])
                print(f"  prob-model epoch {ep+1}: brier={np.mean((p - y[:20000])**2):.4f}")
        return self

    def state(self):
        return {"net": self.net.state(), "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    def load(self, s):
        self.net.load(s["net"])
        self.mu, self.sd = np.array(s["mu"]), np.array(s["sd"])
        return self


def train_prob_model(cfg, n_markets=6000, seed=1, verbose=False):
    from .simulator import simulate
    rng = np.random.default_rng(seed)
    b = simulate(n_markets, cfg, rng)
    X = batch_features(b)
    y = np.broadcast_to(b.outcome[:, None], X.shape[:2])
    return ProbModel(seed).fit(X, y, verbose=verbose)
