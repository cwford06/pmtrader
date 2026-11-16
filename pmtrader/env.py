"""Vectorised trading environment over N simulated binary markets.

Position q is in YES-equivalent contracts: q > 0 holds YES, q < 0 holds NO.
Increasing q fills at the ask, decreasing q fills at the bid; every contract
traded pays a P(1-P)-shaped fee. Reward = change in mark-to-market equity
(at mid), so the episode's summed reward equals realised PnL at settlement.

reward_mode="ev" (default for training) uses a lower-variance, unbiased reward:
realised PnL decomposes exactly into
    sum_t dq_t * (p*_t - fill_t) - fees      (edge captured on each trade)
  + sum_t q_t * (p*_{t+1} - p*_t)            (zero-mean martingale noise)
so dropping the second term keeps the expected reward identical while
removing the coin-flip settlement noise. p* is only known inside the
simulator; it is never part of the agent's observation. `info["pnl"]` is
always the real mark-to-market PnL, and backtests report only that.
"""
from __future__ import annotations

import numpy as np

from .pricing import kalshi_style_fee
from .prob_model import ProbModel, batch_features
from .simulator import MarketBatch, SimConfig, simulate

# discrete action -> target position as fraction of max position.
# Action 0 ("Hold") keeps the current position (NaN sentinel) so the agent can
# sit on a trade without paying to re-trade it.
ACTION_LEVELS = np.array([np.nan, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
ACTION_NAMES = ["Hold", "Max NO", "Half NO", "Quarter NO", "Flat", "Quarter YES", "Half YES", "Max YES"]
N_ACTIONS = len(ACTION_LEVELS)


def targets_from_actions(actions, current_q, max_pos):
    lv = ACTION_LEVELS[actions]
    return np.where(np.isnan(lv), current_q, lv * max_pos)
OBS_NAMES = ["edge", "p_model", "mid", "time_frac", "position", "signal_minus_mid",
             "momentum", "uncertainty", "ev_yes", "ev_no"]
OBS_DIM = len(OBS_NAMES)


def make_obs(p_model, mid, time_frac, position, signal, momentum, spread, fee_rate):
    """Build the agent's observation. All arrays broadcast to (N,)."""
    half = spread / 2
    ask, bid = mid + half, mid - half
    ev_yes = p_model - ask - kalshi_style_fee(ask, 1.0, fee_rate)
    ev_no = (1 - p_model) - (1 - bid) - kalshi_style_fee(1 - bid, 1.0, fee_rate)
    return np.stack([
        (p_model - mid) * 10,
        p_model,
        mid,
        np.broadcast_to(time_frac, np.shape(mid)),
        position,
        (signal - mid) * 10,
        momentum,
        p_model * (1 - p_model) * 4,
        ev_yes * 10,
        ev_no * 10,
    ], -1).astype(np.float64)


class TradingEnv:
    def __init__(self, n_envs: int, prob_model: ProbModel, cfg: SimConfig | None = None,
                 max_pos: float = 100.0, risk_aversion: float = 0.0, reward_scale: float = 0.1,
                 reward_mode: str = "ev", seed: int = 0):
        self.n, self.pm, self.cfg = n_envs, prob_model, cfg or SimConfig()
        self.max_pos, self.risk, self.scale = max_pos, risk_aversion, reward_scale
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)

    # -- episode management -------------------------------------------------
    def reset(self, batch: MarketBatch | None = None):
        self.b = batch or simulate(self.n, self.cfg, self.rng)
        self.n = self.b.n
        feats = batch_features(self.b)                    # (N, T, F)
        self.p_model = self.pm.predict(feats)             # (N, T)
        self.momentum = feats[..., 4]
        self.t = 0
        self.q = np.zeros(self.n)
        self.cash = np.zeros(self.n)
        self.fees_paid = np.zeros(self.n)
        self.volume = np.zeros(self.n)
        return self.obs()

    def obs(self):
        t, T = self.t, self.cfg.T
        return make_obs(self.p_model[:, t], self.b.mid[:, t], 1 - t / T, self.q / self.max_pos,
                        self.b.signal[:, t], self.momentum[:, t], self.cfg.spread, self.cfg.fee_rate)

    def equity(self, t=None):
        t = self.t if t is None else t
        return self.cash + self.q * self.b.mid[:, t]

    # -- dynamics -----------------------------------------------------------
    def step_targets(self, target_q: np.ndarray):
        """Trade to target positions (contracts), advance one step."""
        t, half = self.t, self.cfg.spread / 2
        mid = self.b.mid[:, t]
        eq_before = self.equity()
        target_q = np.clip(target_q, -self.max_pos, self.max_pos)
        dq = target_q - self.q
        px = np.where(dq > 0, mid + half, mid - half)
        fee_px = np.where(dq > 0, mid + half, 1 - (mid - half))   # YES ask or NO ask
        fees = np.abs(dq) * kalshi_style_fee(fee_px, 1.0, self.cfg.fee_rate)
        self.cash -= dq * px + fees
        self.fees_paid += fees
        self.volume += np.abs(dq)
        self.q = target_q
        self.t += 1
        pnl = self.equity() - eq_before
        if self.reward_mode == "ev":
            shaped = dq * (self.b.p_true[:, t] - px) - fees
        else:
            shaped = pnl
        risk_pen = self.risk * (self.q / self.max_pos) ** 2 * mid * (1 - mid)
        reward = shaped / self.max_pos / self.scale - risk_pen
        done = self.t >= self.cfg.T
        info = {"pnl": pnl}
        return (None if done else self.obs()), reward, done, info

    def step(self, actions: np.ndarray):
        return self.step_targets(targets_from_actions(actions, self.q, self.max_pos))
