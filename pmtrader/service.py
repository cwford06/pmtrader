"""Model loading, inference, and background training shared by the API."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .backtest import backtest
from .env import ACTION_LEVELS, ACTION_NAMES, TradingEnv, make_obs
from .ppo import PPOAgent, PPOConfig
from .pricing import Quote, price_contract
from .prob_model import FEATURES, ProbModel, featurize, train_prob_model
from .simulator import SimConfig, logit, simulate

MODELS = Path(__file__).resolve().parent.parent / "models"


def describe_order(pos: float, target: float) -> str:
    """Translate a YES-equivalent position change into exchange orders."""
    legs = []
    if pos > 0 and target < pos:
        legs.append(f"Sell {pos - max(target, 0):.0f} YES")
    if pos < 0 and target > pos:
        legs.append(f"Sell {-pos - max(-target, 0):.0f} NO")
    if target > 0 and target > max(pos, 0):
        legs.append(f"Buy {target - max(pos, 0):.0f} YES")
    if target < 0 and -target > max(-pos, 0):
        legs.append(f"Buy {-target - max(-pos, 0):.0f} NO")
    return ", then ".join(legs) or "No trade"


class Service:
    def __init__(self, models_dir: Path = MODELS):
        self.dir = Path(models_dir)
        self.dir.mkdir(exist_ok=True)
        self.cfg = SimConfig()
        self.lock = threading.Lock()
        self.train_state = {"running": False, "iteration": 0, "total": 0, "history": None,
                            "started": None, "finished": None, "error": None}
        self._stop = threading.Event()
        self.last_backtest = None
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        pm_path, ag_path = self.dir / "prob_model.json", self.dir / "ppo_agent.json"
        if pm_path.exists():
            self.pm = ProbModel().load(json.loads(pm_path.read_text()))
        else:
            self.pm = train_prob_model(self.cfg)
            pm_path.write_text(json.dumps(self.pm.state()))
        self.agent = PPOAgent()
        if ag_path.exists():
            self.agent.load(json.loads(ag_path.read_text()))
        self.trained = ag_path.exists()

    # -- inference -----------------------------------------------------------
    def probability(self, mid: float, signal: float | None, time_frac: float, momentum: float) -> float:
        sig = mid if signal is None else signal
        X = featurize(np.array([mid]), np.array([sig]), time_frac, momentum)
        return float(self.pm.predict(X)[0])

    def price(self, req: dict) -> dict:
        payout = req.get("payout", 1.0)
        bid, ask = req["yes_bid"] / payout, req["yes_ask"] / payout
        mid = (bid + ask) / 2
        momentum = 0.0
        if req.get("prev_mid") is not None:
            momentum = float(logit(mid) - logit(req["prev_mid"] / payout))
        p_model = self.probability(mid, req.get("signal_prob"), req.get("time_frac", 0.5), momentum)
        p = req["prob_override"] if req.get("prob_override") is not None else p_model
        res = price_contract(p, Quote(bid, ask), 1.0, req.get("fee_rate", self.cfg.fee_rate),
                             req.get("min_edge", 0.0)).to_dict()
        for k in ("fair_value_yes", "fair_value_no", "ev_yes", "ev_no"):
            res[k] *= payout
        res.update(model_prob=p_model, market_mid=mid * payout, momentum=momentum, payout=payout)
        return res

    def recommend(self, req: dict) -> dict:
        priced = self.price(req)
        payout = req.get("payout", 1.0)
        max_pos = float(req.get("max_position", 100))
        pos = float(req.get("position", 0))
        bid, ask = req["yes_bid"] / payout, req["yes_ask"] / payout
        mid, spread = (bid + ask) / 2, ask - bid
        sig = req.get("signal_prob")
        obs = make_obs(np.array([priced["prob_yes"]]), np.array([mid]), req.get("time_frac", 0.5),
                       np.array([pos / max_pos]), np.array([mid if sig is None else sig]),
                       np.array([priced["momentum"]]), spread, req.get("fee_rate", self.cfg.fee_rate))
        with self.lock:
            pi = self.agent.policy(obs)[0]
            value = float(self.agent.value(obs)[0])
        a = int(pi.argmax())
        lvl = ACTION_LEVELS[a]
        target = pos if np.isnan(lvl) else float(lvl * max_pos)
        delta = target - pos
        order = describe_order(pos, target)
        return {
            "pricing": priced,
            "action": ACTION_NAMES[a],
            "action_probs": {n: float(p) for n, p in zip(ACTION_NAMES, pi)},
            "current_position": pos,
            "target_position": target,
            "order": order,
            "state_value": value,
            "agent_trained": self.trained,
        }

    def simulate_market(self, seed: int) -> dict:
        b = simulate(1, self.cfg, np.random.default_rng(seed))
        env = TradingEnv(1, self.pm, self.cfg)
        obs = env.reset(b)
        pos, pnl, acts = [], [], []
        cum, done = 0.0, False
        while not done:
            with self.lock:
                a, _ = self.agent.act(obs, greedy=True)
            acts.append(ACTION_NAMES[int(a[0])])
            obs, _, done, info = env.step(a)
            pos.append(float(env.q[0])); cum += float(info["pnl"][0]); pnl.append(cum)
        T = self.cfg.T
        return {"t": list(range(T + 1)), "p_true": b.p_true[0].tolist(), "mid": b.mid[0].tolist(),
                "signal": b.signal[0].tolist(), "p_model": env.p_model[0].tolist() + [float(b.outcome[0])],
                "position": [0.0] + pos, "pnl": [0.0] + pnl, "actions": acts,
                "outcome": "YES" if b.outcome[0] else "NO", "final_pnl": cum}

    def run_backtest(self, n_markets: int, seed: int) -> dict:
        with self.lock:
            agent = self.agent
        self.last_backtest = backtest(agent, self.pm, n_markets=n_markets, seed=seed, cfg=self.cfg)
        return self.last_backtest

    def info(self) -> dict:
        return {"features": FEATURES, "actions": ACTION_NAMES, "agent_trained": self.trained,
                "ppo_config": self.agent.state()["cfg"], "history": self.agent.state()["history"],
                "sim_config": self.cfg.__dict__}

    # -- training ------------------------------------------------------------
    def start_training(self, iterations: int, fresh: bool, seed: int = 0) -> dict:
        if self.train_state["running"]:
            raise RuntimeError("Training already running")
        self._stop.clear()
        self.train_state.update(running=True, iteration=0, total=iterations, started=time.time(),
                                finished=None, error=None, history=None)

        def job():
            try:
                agent = PPOAgent(PPOConfig(), seed=seed)
                if not fresh:
                    agent.load(self.agent.state())
                env = TradingEnv(agent.cfg.n_envs, self.pm, self.cfg, seed=seed + 2)

                def cb(it, h):
                    self.train_state["iteration"] = it + 1
                    self.train_state["history"] = {"iteration": h.iteration, "mean_pnl": h.mean_pnl,
                                                   "entropy": h.entropy}
                    return not self._stop.is_set()

                agent.train(env, iterations, cb)
                agent.save(self.dir / "ppo_agent.json")
                with self.lock:
                    self.agent, self.trained = agent, True
            except Exception as e:  # surfaced via /api/train/status
                self.train_state["error"] = repr(e)
            finally:
                self.train_state.update(running=False, finished=time.time())

        threading.Thread(target=job, daemon=True).start()
        return self.train_state

    def stop_training(self):
        self._stop.set()
