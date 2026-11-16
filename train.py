"""Train the probability model + PPO agent, then backtest out of sample.

    python train.py --iterations 150
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from pmtrader.backtest import backtest
from pmtrader.env import TradingEnv
from pmtrader.ppo import PPOAgent, PPOConfig
from pmtrader.prob_model import train_prob_model
from pmtrader.simulator import SimConfig

MODELS = Path(__file__).parent / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--risk-aversion", type=float, default=0.0)
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    cfg = SimConfig()
    t0 = time.time()
    print("Training probability model...")
    pm = train_prob_model(cfg, seed=args.seed + 1, verbose=True)
    (MODELS / "prob_model.json").write_text(json.dumps(pm.state()))

    print("Training PPO agent...")
    agent = PPOAgent(PPOConfig(n_envs=args.n_envs), seed=args.seed)
    env = TradingEnv(args.n_envs, pm, cfg, risk_aversion=args.risk_aversion, seed=args.seed + 2)

    def cb(it, h):
        if (it + 1) % 10 == 0:
            recent = np.mean(h.mean_pnl[-10:])
            print(f"  iter {it+1:4d}  pnl/market ${recent:6.2f}  entropy {h.entropy[-1]:.3f}  "
                  f"vloss {h.value_loss[-1]:.4f}  ({time.time()-t0:.0f}s)")

    agent.train(env, args.iterations, cb)
    agent.save(MODELS / "ppo_agent.json")

    print("Backtesting on unseen markets...")
    res = backtest(agent, pm, n_markets=4000, seed=999_999)
    for name, r in res["results"].items():
        print(f"  {name:16s} total ${r['total_pnl']:>9.0f}  per-mkt ${r['mean_pnl_per_market']:6.2f}  "
              f"ROI {r['roi_on_capital']*100:5.2f}%  t={r['t_stat']:5.1f}  traded {r['pct_markets_traded']*100:4.0f}%  "
              f"fees ${r['fees_paid']:.0f}")
    print(f"Done in {time.time()-t0:.0f}s. Models saved to {MODELS}/")


if __name__ == "__main__":
    main()
