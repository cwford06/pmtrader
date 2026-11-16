"""Backtest the RL agent against rule-based EV baselines on identical markets."""
from __future__ import annotations

import numpy as np

from .env import TradingEnv
from .pricing import kalshi_style_fee
from .simulator import SimConfig, simulate


def _evs(env: TradingEnv):
    t, half, fr = env.t, env.cfg.spread / 2, env.cfg.fee_rate
    p, mid = env.p_model[:, t], env.b.mid[:, t]
    ask, bid = mid + half, mid - half
    ev_yes = p - ask - kalshi_style_fee(ask, 1, fr)
    ev_no = (1 - p) - (1 - bid) - kalshi_style_fee(1 - bid, 1, fr)
    return p, ask, bid, ev_yes, ev_no


def ev_threshold_policy(threshold=0.02):
    """Go max size on whichever side has EV > threshold; otherwise hold."""
    def f(env):
        _, _, _, ey, en = _evs(env)
        tgt = env.q.copy()
        tgt[ey > threshold] = env.max_pos
        tgt[en > threshold] = -env.max_pos
        return tgt
    return f


def kelly_policy(fraction=0.5, min_ev=0.01, band=0.10):
    """Fractional Kelly sizing (bankroll = max_pos dollars per market).

    Re-sizes only when a side clears `min_ev` and the new size differs from the
    current one by more than `band` of max position; otherwise holds.
    """
    def f(env):
        p, ask, bid, ey, en = _evs(env)
        cost_y, cost_n = ask, 1 - bid
        ky = np.clip((p - cost_y) / (1 - cost_y), 0, None)
        kn = np.clip(((1 - p) - cost_n) / (1 - cost_n), 0, None)
        bank = env.max_pos
        tgt = np.where(ey > min_ev, fraction * ky * bank / cost_y,
                       np.where(en > min_ev, -fraction * kn * bank / cost_n, env.q))
        tgt = np.clip(tgt, -env.max_pos, env.max_pos)
        keep = np.abs(tgt - env.q) < band * env.max_pos
        return np.where(keep, env.q, tgt)
    return f


def rl_policy(agent):
    from .env import targets_from_actions

    def f(env):
        a, _ = agent.act(env.obs(), greedy=True)
        return targets_from_actions(a, env.q, env.max_pos)
    return f


def run_policy(env: TradingEnv, batch, policy):
    env.reset(batch)
    done = False
    pnl = np.zeros(env.n)
    while not done:
        _, _, done, info = env.step_targets(policy(env))
        pnl += info["pnl"]
    return pnl, env.fees_paid.copy(), env.volume.copy()


def summarize(pnl, fees, volume, max_pos, max_points=300):
    n = len(pnl)
    curve = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(np.concatenate([[0], curve])) - np.concatenate([[0], curve])).max())
    traded = volume > 0
    idx = np.linspace(0, n - 1, min(n, max_points)).astype(int)
    return {
        "total_pnl": float(pnl.sum()),
        "mean_pnl_per_market": float(pnl.mean()),
        "roi_on_capital": float(pnl.sum() / (n * max_pos)),
        "sharpe_per_market": float(pnl.mean() / (pnl.std() + 1e-9)),
        "t_stat": float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(n)),
        "hit_rate": float((pnl[traded] > 0).mean()) if traded.any() else 0.0,
        "pct_markets_traded": float(traded.mean()),
        "fees_paid": float(fees.sum()),
        "contracts_traded": float(volume.sum()),
        "max_drawdown": dd,
        "equity_curve": {"x": (idx + 1).tolist(), "y": curve[idx].round(2).tolist()},
    }


def backtest(agent, prob_model, n_markets=2000, seed=12345, cfg: SimConfig | None = None, max_pos=100.0):
    cfg = cfg or SimConfig()
    batch = simulate(n_markets, cfg, np.random.default_rng(seed))
    env = TradingEnv(n_markets, prob_model, cfg, max_pos=max_pos)
    strategies = {
        "PPO agent (RL)": rl_policy(agent),
        "EV threshold": ev_threshold_policy(0.02),
        "Half Kelly": kelly_policy(0.5),
    }
    out = {}
    for name, pol in strategies.items():
        out[name] = summarize(*run_policy(env, batch, pol), max_pos=max_pos)
    return {"n_markets": n_markets, "seed": seed, "max_pos": max_pos, "results": out}
