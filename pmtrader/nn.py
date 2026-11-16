"""Tiny dependency-free neural network: MLP with manual backprop and Adam.

Kept in pure NumPy so the project installs anywhere without PyTorch.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator, out_scale: float = 1.0):
        self.sizes = sizes
        self.params: list[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            # orthogonal-ish init; small final layer helps policy start near-uniform
            W = rng.normal(0, 1, (a, b))
            u, _, vt = np.linalg.svd(W, full_matrices=False)
            W = (u if u.shape == (a, b) else vt) * (out_scale if last else np.sqrt(2))
            self.params += [W.astype(np.float64), np.zeros(b)]
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        acts = [x]
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            h = h @ W + b
            if i < n_layers - 1:
                h = np.tanh(h)
            acts.append(h)
        self._cache = acts
        return h

    def backward(self, dout: np.ndarray) -> list[np.ndarray]:
        """Gradients of sum(dout * output) w.r.t. params (call after forward)."""
        acts = self._cache
        grads = [None] * len(self.params)
        n_layers = len(self.params) // 2
        g = dout
        for i in reversed(range(n_layers)):
            W = self.params[2 * i]
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            if i > 0:
                g = (g @ W.T) * (1 - acts[i] ** 2)   # tanh'
        return grads

    def state(self) -> list[list]:
        return [p.tolist() for p in self.params]

    def load(self, state: list[list]):
        self.params = [np.array(p, dtype=np.float64) for p in state]


class Adam:
    def __init__(self, params: list[np.ndarray], lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, max_norm=0.5):
        self.params, self.lr, self.b1, self.b2, self.eps, self.max_norm = params, lr, b1, b2, eps, max_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        if self.max_norm and norm > self.max_norm:
            grads = [g * self.max_norm / (norm + 1e-12) for g in grads]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
