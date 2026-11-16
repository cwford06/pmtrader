"""Pull a live Kalshi game market and run pmtrader's models on it.

Run from the project root:  .venv/bin/python live_kalshi.py TICKER SERIES START_UTC
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")
from pmtrader.pricing import kalshi_style_fee
from pmtrader.service import Service
from pmtrader.simulator import logit

TICKER, SERIES, START = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
BASE = "https://api.elections.kalshi.com/trade-api/v2"
STEPS = 40          # decision steps per market, as in training (SimConfig.T)
MAX_POS = 100       # contracts
FEE = 0.07


def get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "pmtrader-example/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


m = get(f"/markets/{TICKER}")["market"]
ev = get(f"/events/{m['event_ticker']}")["event"]
now = time.time()
t0, t1 = ts(START), ts(m["expected_expiration_time"])
cs = get(f"/series/{SERIES}/markets/{TICKER}/candlesticks?start_ts={int(t0) - 600}&end_ts={int(now)}&period_interval=1")["candlesticks"]
cand = [(c["end_period_ts"], float(c["yes_bid"]["close_dollars"]), float(c["yes_ask"]["close_dollars"])) for c in cs
        if c["yes_bid"].get("close_dollars") and c["yes_ask"].get("close_dollars")]


def quote_at(t):
    """Last 1-minute close at or before time t."""
    prior = [c for c in cand if c[0] <= t]
    return (prior[-1][1], prior[-1][2]) if prior else (cand[0][1], cand[0][2])


svc = Service()
bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
step_len = (t1 - t0) / STEPS
time_frac = max(0.0, min(1.0, (t1 - now) / (t1 - t0)))
pb, pa = quote_at(now - 3 * step_len)
req = {"yes_bid": bid, "yes_ask": ask, "payout": 1.0, "prev_mid": (pb + pa) / 2, "time_frac": time_frac,
       "fee_rate": FEE, "position": 0, "max_position": MAX_POS}
rec = svc.recommend(req)
pr = rec["pricing"]


def ev_rule(p, b, a, q):          # EV threshold baseline (2c)
    ey = p - a - kalshi_style_fee(a, 1, FEE)
    en = (1 - p) - (1 - b) - kalshi_style_fee(1 - b, 1, FEE)
    return MAX_POS if ey > 0.02 else -MAX_POS if en > 0.02 else q


def kelly_rule(p, b, a, q):       # Half Kelly baseline
    ey = p - a - kalshi_style_fee(a, 1, FEE)
    en = (1 - p) - (1 - b) - kalshi_style_fee(1 - b, 1, FEE)
    cy, cn = a, 1 - b
    # a side priced at $1.00 can't pay out more than it costs: Kelly size 0
    ky = max(0, (p - cy) / (1 - cy)) if cy < 1 else 0.0
    kn = max(0, ((1 - p) - cn) / (1 - cn)) if cn < 1 else 0.0
    tgt = 0.5 * ky * MAX_POS / cy if ey > 0.01 else -0.5 * kn * MAX_POS / cn if en > 0.01 else q
    tgt = float(np.clip(tgt, -MAX_POS, MAX_POS))
    return q if abs(tgt - q) < 0.10 * MAX_POS else tgt


snapshot = {
    "PPO agent (RL)": {"action": rec["action"], "order": rec["order"], "target": rec["target_position"]},
    "EV threshold": {"target": ev_rule(pr["prob_yes"], bid, ask, 0)},
    "Half Kelly": {"target": round(kelly_rule(pr["prob_yes"], bid, ask, 0), 1)},
}

# Replay the game so far on the training decision clock
n_done = int((now - t0) // step_len) + 1
mids = []
replay = {k: {"q": 0.0, "cash": 0.0, "fees": 0.0, "trades": []} for k in snapshot}
path = []
for k in range(n_done):
    t = t0 + k * step_len
    b, a = quote_at(t)
    mid = (b + a) / 2
    mids.append(mid)
    prev = mids[max(0, k - 3)]
    tf = 1 - k / STEPS
    r = svc.recommend({"yes_bid": b, "yes_ask": a, "prev_mid": prev, "time_frac": tf, "fee_rate": FEE,
                       "position": replay["PPO agent (RL)"]["q"], "max_position": MAX_POS})
    p = r["pricing"]["prob_yes"]
    targets = {"PPO agent (RL)": r["target_position"],
               "EV threshold": ev_rule(p, b, a, replay["EV threshold"]["q"]),
               "Half Kelly": kelly_rule(p, b, a, replay["Half Kelly"]["q"])}
    row = {"step": k, "utc": datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M"), "bid": b, "ask": a,
           "p_model": round(p, 4), "ppo_action": r["action"]}
    for name, tgt in targets.items():
        st = replay[name]
        dq = tgt - st["q"]
        if abs(dq) > 1e-9:
            # YES-equivalent: buying YES (or selling NO) fills at the ask, selling at the bid
            px = a if dq > 0 else b
            fee = kalshi_style_fee(px, 1, FEE) * abs(dq)
            st["cash"] -= dq * px + fee
            st["fees"] += fee
            st["q"] = tgt
            st["trades"].append(f"{row['utc']} {'+' if dq > 0 else ''}{dq:.0f} @ {px:.2f}")
        row[name] = st["q"]
    path.append(row)

final = {}
for name, st in replay.items():
    q = st["q"]
    liq = bid if q > 0 else ask          # close out now: sell YES at bid / buy back at ask
    final[name] = {"position": q, "fees": round(st["fees"], 2), "trades": st["trades"],
                   "pnl_mark_mid": round(st["cash"] + q * (bid + ask) / 2, 2),
                   "pnl_if_closed_now": round(st["cash"] + q * liq - kalshi_style_fee(liq, 1, FEE) * abs(q), 2),
                   "pnl_if_YES": round(st["cash"] + q * 1.0, 2), "pnl_if_NO": round(st["cash"], 2)}

res = {"pulled_utc": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
       "event": ev["title"], "market": m["title"], "ticker": TICKER, "yes_bid": bid, "yes_ask": ask,
       "last": m["last_price_dollars"], "pregame_yes": m.get("previous_price_dollars"),
       "volume": m.get("volume_fp"), "open_interest": m.get("open_interest_fp"),
       "time_frac": round(time_frac, 3), "prev_mid": req["prev_mid"], "pricing": pr,
       "ppo_probs": rec["action_probs"], "state_value": rec["state_value"], "snapshot": snapshot,
       "replay_steps": n_done, "replay": final, "path": path}
if OUT:
    json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "path"}, indent=1, default=float))
for r in path:
    print(r)
