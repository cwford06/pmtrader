"""Scan every Kalshi NBA game-winner market for one date and rank trades with pmtrader's models.

Run from the project root:  .venv/bin/python scan_nba.py 26NOV16 <out_dir>

Pregame inputs: time_frac = 1.0 (the game hasn't started, so all of its uncertainty is
still ahead), momentum from the mid price 1 hour ago, no outside forecast.
Each game can be bet four ways (YES or NO on either team's market). Each way is priced
with the model, and the best after-fee EV per game is kept.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from pmtrader.service import Service

DATE, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://api.elections.kalshi.com/trade-api/v2"
FEE, MAX_POS, MIN_EV = 0.07, 100, 0.0


def get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "pmtrader-example/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


now = time.time()
pulled = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
markets = [m for m in get("/markets?series_ticker=KXNBAGAME&status=open&limit=500")["markets"] if f"-{DATE}" in m["ticker"]]
svc = Service()

rows, options, history = [], [], []
for m in markets:
    bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
    cs = get(f"/series/KXNBAGAME/markets/{m['ticker']}/candlesticks?start_ts={int(now) - 6 * 3600}&end_ts={int(now)}&period_interval=60")["candlesticks"]
    hist = [(c["end_period_ts"], float(c["yes_bid"]["close_dollars"]), float(c["yes_ask"]["close_dollars"]))
            for c in cs if c["yes_bid"].get("close_dollars") and c["yes_ask"].get("close_dollars")]
    prev_mid = (hist[-2][1] + hist[-2][2]) / 2 if len(hist) >= 2 else (bid + ask) / 2
    for t, b, a in hist:
        history.append({"ticker": m["ticker"], "team": m["yes_sub_title"], "utc": datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M"), "mid": round((b + a) / 2, 4)})
    rec = svc.recommend({"yes_bid": bid, "yes_ask": ask, "prev_mid": prev_mid, "time_frac": 1.0, "fee_rate": FEE,
                         "position": 0, "max_position": MAX_POS})
    p = rec["pricing"]
    rows.append({"ticker": m["ticker"], "event": m["event_ticker"], "team": m["yes_sub_title"], "bid": bid, "ask": ask,
                 "mid": (bid + ask) / 2, "p_model": p["prob_yes"], "ev_yes": p["ev_yes"], "ev_no": p["ev_no"],
                 "ppo_action": rec["action"], "ppo_order": rec["order"], "ppo_conf": rec["action_probs"][rec["action"]],
                 "ask_size": float(m.get("yes_ask_size_fp") or 0), "bid_size": float(m.get("yes_bid_size_fp") or 0),
                 "volume": float(m.get("volume_fp") or 0), "tipoff_est": m["expected_expiration_time"]})

games = []
by_event = {}
for r in rows:
    by_event.setdefault(r["event"], []).append(r)
for ev, rs in sorted(by_event.items()):
    if len(rs) != 2:
        continue
    a, b = rs
    # P(team A wins): average the model's read of both markets (they should mirror)
    pa = (a["p_model"] + (1 - b["p_model"])) / 2
    ways = [  # (team that must win, how, cost per contract)
        (a["team"], f"Buy YES on {a['team']}", a["ask"], a["ticker"], "YES", a),
        (b["team"], f"Buy NO on {a['team']}", 1 - a["bid"], a["ticker"], "NO", a),
        (b["team"], f"Buy YES on {b['team']}", b["ask"], b["ticker"], "YES", b),
        (a["team"], f"Buy NO on {b['team']}", 1 - b["bid"], b["ticker"], "NO", b),
    ]
    best = None
    for team, how, cost, tk, side, src in ways:
        pw = pa if team == a["team"] else 1 - pa
        fee = FEE * cost * (1 - cost)
        evc = pw - cost - fee
        o = {"game": f"{a['team']} vs {b['team']}", "event": ev, "bet_on": team, "how": how, "ticker": tk, "side": side,
             "cost": round(cost, 2), "p_win_model": round(pw, 4), "p_win_market": round(cost, 4),
             "fee": round(fee, 4), "ev_c": round(evc * 100, 2), "roi_pct": round(evc / cost * 100, 2),
             "kelly_pct": round(max(0, (pw - cost) / (1 - cost)) * 100, 2) if cost < 1 else 0.0}
        options.append(o)
        if best is None or o["ev_c"] > best["ev_c"]:
            best = o
    sa = max(a["ppo_conf"], 0)
    games.append({**best, "tipoff_est": a["tipoff_est"],
                  "fav": a["team"] if a["mid"] >= b["mid"] else b["team"],
                  "ppo_a": f"{a['team']}: {a['ppo_action']}", "ppo_b": f"{b['team']}: {b['ppo_action']}",
                  "volume": round(a["volume"] + b["volume"])})

games.sort(key=lambda g: -g["ev_c"])
for i, g in enumerate(games):
    g["rank"] = i + 1
top = games[0]
decision = {
    "pulled_utc": pulled, "date": DATE, "n_games": len(games), "n_markets": len(rows),
    "decision": "TRADE" if top["ev_c"] > MIN_EV else "NO TRADE",
    **{k: top[k] for k in ("game", "bet_on", "how", "ticker", "side", "cost", "p_win_model", "p_win_market", "ev_c", "roi_pct", "kelly_pct")},
    "contracts": MAX_POS if top["ev_c"] > MIN_EV else 0,
    "outlay": round(MAX_POS * top["cost"], 2), "win_pnl": round(MAX_POS * (1 - top["cost"] - top["fee"]), 2),
    "loss_pnl": round(-MAX_POS * (top["cost"] + top["fee"]), 2), "exp_pnl": round(MAX_POS * top["ev_c"] / 100, 2),
    "n_positive": sum(g["ev_c"] > 0 for g in games),
}
# Sensitivity: best single-market EV under three readings of "time left", no momentum
sens = []
for r in rows:
    for tf in (1.0, 0.5, 0.23):
        p = svc.price({"yes_bid": r["bid"], "yes_ask": r["ask"], "time_frac": tf, "fee_rate": FEE})
        side = "YES" if p["ev_yes"] >= p["ev_no"] else "NO"
        sens.append({"ticker": r["ticker"], "team": r["team"], "time_frac": tf, "p_model": round(p["prob_yes"], 4),
                     "mid": round(r["mid"], 4), "best_side": side, "best_ev_c": round(max(p["ev_yes"], p["ev_no"]) * 100, 2)})
decision["max_ev_any_clock_c"] = max(s["best_ev_c"] for s in sens)
decision["bar_c"] = 2.0
code = Path(__file__).read_text()
meta = [{"label": "scan_nba.py", "value": code, "type": "code", "lang": "python"},
        {"label": "Input", "value": f"Kalshi public API, series KXNBAGAME, games dated {DATE}, pulled {pulled} UTC; models/prob_model.json, models/ppo_agent.json"}]
for n, data in [("nba_decision", [decision]), ("nba_games", games), ("nba_options", options),
                ("nba_markets", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]),
                ("nba_history", history), ("nba_sensitivity", sens)]:
    (OUT / f"{n}.json").write_text(json.dumps({"rows": data, "meta": meta}))
print(json.dumps(decision, indent=1))
for g in games:
    print(g["rank"], g["game"], "|", g["how"], "@", g["cost"], "| P model", g["p_win_model"], "| EV", g["ev_c"], "c | ROI", g["roi_pct"], "%", "|", g["ppo_a"], "/", g["ppo_b"])
