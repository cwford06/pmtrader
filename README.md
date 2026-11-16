# PM·RL Trader

A trading system for binary prediction-market contracts (Kalshi- and Polymarket-style). It has three layers:

1. **An EV pricer.** Fair value is `p × payout`. EV per contract is worked out after spread and fees for both YES and NO, with Kelly sizing.
2. **An ML probability model.** This is an MLP classifier that estimates P(YES) from the market price, an external signal, time left and momentum.
3. **A PPO reinforcement-learning agent.** It decides when to trade, which side to take, how big to go and when to exit, based on the model's edge.

It comes with a FastAPI REST API and a web dashboard. The code is pure NumPy (no PyTorch), so it installs anywhere.

## Quick start

```bash
pip install -r requirements.txt
python train.py --iterations 400        # optional: pretrained models are already in models/
uvicorn pmtrader.api:app --reload       # dashboard → http://127.0.0.1:8000
                                        # API docs  → http://127.0.0.1:8000/docs
pytest -q                               # 8 tests, incl. gradient checks
```

## Results (out-of-sample simulated markets, 100-contract max per market)

| Strategy | PnL / market | ROI on capital | Sharpe / market | t-stat |
|---|---|---|---|---|
| **PPO agent (RL)** | **$8.17** | **8.2%** | **0.215** | 11.8 |
| PPO + EV guardrail | $8.06 | 8.1% | 0.213 | 11.7 |
| EV threshold (tuned at 2¢) | $6.24 | 6.2% | 0.154 | 8.4 |
| Half Kelly | $0.67 | 0.7% | 0.155 | 8.5 |

These numbers come from 3,000 unseen markets (seed 12345). The EV-threshold baseline was tuned on a separate seed, and 2¢ was its best threshold. **These are simulator results.** The simulator has edge built into it on purpose: favorite-longshot bias, mean-reverting mispricing and an informative signal. Real markets may have less edge, or none. Validate on real data before you risk money.

## How it works

**Simulator (`simulator.py`).** A hidden Brownian process decides each outcome, so the true probability is `Φ(L_t / σ√(T−t))`. The market price is a distorted view of that probability: it's compressed toward 50% (favorite-longshot bias) and carries AR(1) mispricing. The external signal is a noisy but independent read on the true probability.

**Probability model (`prob_model.py`).** An MLP with BCE loss, trained on resolved markets. Its Brier score is 0.1177, against 0.1190 for the market and 0.1153 for the true probability. So it captures about half the information the market leaves on the table. With a real forecast that you trust, check **"Price on my forecast directly"** in the UI, or pass `prob_override` to the API.

**RL agent (`ppo.py`, `env.py`).**
- **Actions:** Hold, Flat, or ¼, ½ or full max position in YES or NO.
- **Observations:** edge, model probability, price, time left, current position, the signal gap, momentum, uncertainty, and EV for each side.
- **Fills:** buys fill at the ask and sells fill at the bid, with Kalshi-shaped fees of `0.07·P(1−P)`.
- **Reward trick:** realised PnL splits exactly into *edge captured on each fill* plus a *zero-mean martingale term* (the coin flip at settlement). Training on just the first part has the same expected reward with much lower variance. Training on raw PnL made the agent collapse to "never trade". `tests/test_core.py` checks this split exactly. The true probability is used only to compute the training reward and is never part of the agent's inputs. Backtests always report real mark-to-market PnL.
- **PPO setup:** clipped surrogate, GAE, an entropy schedule from 0.10 to 0.002, and an LR schedule. Actor and critic gradients are derived by hand and checked against finite differences.

**EV guardrail (`pricing.ev_guardrail`).** The agent's decision passes through a hard check before it becomes an order: it may only add YES contracts when buying YES has positive after-fee EV, and only add NO contracts when buying NO does (or more than `min_edge`, if you set one). Selling what it already holds, back toward flat, is always allowed, because that reduces risk rather than placing a new bet. A flip from YES to NO is split: the sale goes through, and the new NO purchase only if NO has edge.

- **Why it exists:** the agent was trained only on simulated prices between 1.5¢ and 98.5¢ with a 2¢ spread. On a live Kalshi market quoted 99¢ / $1.00, it picked "Buy 25 YES", a trade that can't make money (see [What real runs looked like](#what-real-runs-looked-like-nov-2026)). The guardrail turns that into "No trade".
- **What it costs:** in the simulator the agent sometimes buys without edge at the current price, expecting the mispricing to correct, and that pays slightly. With the guardrail, PnL per market is $8.06 instead of $8.17, and the agent trades 553,100 contracts instead of 598,475. It still beats both rule-based baselines by a wide margin.
- **Where it applies:** `/api/recommend` and both examples, on by default. Pass `"ev_guardrail": false` to see the raw agent. The backtest reports both versions.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/price` | Model probability, fair value, EV, ROI and Kelly for YES and NO |
| POST | `/api/recommend` | `/api/price` plus the RL agent's action, its action probabilities and the concrete order (e.g. "Sell 100 YES, then Buy 50 NO"), after the EV guardrail. `agent_action` and `agent_target_position` show the raw pick, and `guardrail.reason` says what was blocked. |
| POST | `/api/backtest` | RL agent (raw and guarded) vs. baselines on N fresh markets |
| GET | `/api/simulate?seed=` | One market path with the agent's trades |
| POST | `/api/train`, `/api/train/stop`; GET `/api/train/status` | Background training, with a live learning curve |
| GET | `/api/model` | Configs and training history |

```bash
curl -X POST localhost:8000/api/recommend -H 'Content-Type: application/json' \
  -d '{"yes_bid":0.41,"yes_ask":0.43,"signal_prob":0.58,"time_frac":0.5,"position":0,"max_position":100}'
```

Prices can be in any unit as long as `payout` matches. For example, Kalshi cents work with `payout: 100`.

## Using it on real markets

- **Retrain the model on real history.** Replace `train_prob_model`'s simulated data with resolved Kalshi or Polymarket markets, using the same `featurize` features plus your own signal, then retrain.
- **Fit the simulator to real data**, or train the agent on replayed historical price paths, so that the agent learns real dynamics.
- **Check the fee setting.** `fee_rate` is a smooth approximation. Kalshi rounds fees up to the cent, which matters a lot for small orders. Polymarket is mostly fee-free, so set `fee_rate: 0` there.
- **Things not modeled:** order-book depth and slippage beyond the quoted spread, correlated markets and portfolio-level risk.

## Examples: live Kalshi data

Two scripts in `examples/` run the models on real markets from Kalshi's public market-data API. They need no account or API key and **place no orders**. Run them from the repo root with the project's environment active.

| Script | What it does |
|---|---|
| `live_kalshi.py` | Prices one market right now and replays the game so far, showing what each strategy would have done |
| `scan_nba.py` | Prices every NBA game on a date, ranks the best bet in each game by after-fee EV, and checks how sensitive the answer is |

Both scripts give the model only the market's own prices and no outside forecast, so expect few or no trades. The models were trained on simulated markets (see [Results](#results-out-of-sample-simulated-markets-100-contract-max-per-market)), so treat the output as a demonstration of the pipeline, not a trading signal.

### Finding a market ticker

Kalshi game-winner markets live in a series per league: `KXNBAGAME`, `KXNFLGAME`, `KXNHLGAME`, `KXNCAAFGAME` and so on. Each game is an *event* with one market per team, for example `KXNBAGAME-26NOV16PHXDEN-DEN` ("Denver wins" in Phoenix vs Denver on Nov 16, 2026). To list a league's open markets:

```bash
curl -s "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNBAGAME&status=open&limit=200" \
  | python -c "import json,sys; [print(m['ticker'], m['yes_bid_dollars'], m['yes_ask_dollars']) for m in json.load(sys.stdin)['markets']]"
```

### `live_kalshi.py`: one market, now and since tip-off

```bash
python examples/live_kalshi.py <MARKET_TICKER> <SERIES_TICKER> <GAME_START_UTC> [out.json]
python examples/live_kalshi.py KXNBAGAME-26NOV15OKCSAC-OKC KXNBAGAME 2026-11-16T02:10:00Z okc.json
```

- **`GAME_START_UTC`** is when the game began. The script measures "time left" from it, up to Kalshi's expected settlement time. Kalshi doesn't publish the start, so take it from the schedule, or from the first big price and volume jump in the market's history.
- **Live snapshot:** pulls the current bid and ask, plus the price about three decision steps ago for momentum. It then runs:
  - the ML probability model: P(YES), edge, and after-fee EV for buying YES and for buying NO;
  - the PPO agent: its action, its probability for each action and the concrete order;
  - the EV-threshold rule (more than 2¢ of EV) and Half Kelly.
- **Replay:** steps through the game on the same 40-decision clock the models were trained on, using Kalshi's 1-minute candles. Each strategy trades the real bid and ask, paying Kalshi-style fees, up to 100 contracts. It reports each strategy's trades and its PnL if closed now, if YES wins and if NO wins.
- **Output:** the summary is printed as JSON, followed by one line per decision step. If you give `out.json`, the full result is also saved there, including the step-by-step `path`.

### `scan_nba.py`: every NBA game on a date

```bash
python examples/scan_nba.py <YYMONDD> <out_dir>
python examples/scan_nba.py 26NOV16 scan_out/
```

- **What it covers:** every open `KXNBAGAME` market dated `YYMONDD`. Each game can be bet four ways: YES or NO on either team's market. A team's win probability is the average of the model's read of both markets, since they should mirror each other.
- **How it ranks:** it keeps each game's best bet by after-fee EV and ranks the games. The pick is `TRADE` only if the best EV is positive. The PPO agent's pick is also shown for every market.
- **Pregame inputs:** time left = 1.0 (none of the game has been played), and momentum is the price move over the last hour.
- **Sensitivity check:** "time left" has no exact pregame meaning, so the scan also re-prices every market at 0.5 and 0.23 and reports the largest EV under any setting.
- **Output:** the decision and the ranked games are printed. JSON files written to `out_dir`, each as `{rows, meta}`:

| File | Contents |
|---|---|
| `nba_decision.json` | The best bet on the slate, its EV, payoff if right or wrong, and `TRADE` / `NO TRADE` |
| `nba_games.json` | The best bet in each game, ranked |
| `nba_options.json` | All four ways to bet every game, with model win probability and EV |
| `nba_markets.json` | Each market's quote, model probability, EV for both sides and the PPO agent's pick |
| `nba_history.json` | Hourly mid prices over the last 6 hours |
| `nba_sensitivity.json` | Best EV per market under each time-left setting |

### What real runs looked like (Nov 2026)

| Run | Result |
|---|---|
| NHL, Vancouver at Vegas, mid-game, Vancouver YES 73¢ / 74¢ | Model 76.0%, EV of buying YES +0.7¢. No strategy traded, either at that moment or anywhere in the replay. |
| NBA, Oklahoma City at Sacramento, late in the game, OKC YES 99¢ / $1.00 | EV negative on both sides (−0.72¢ YES, −0.35¢ NO), yet the raw PPO agent picked **Buy 25 YES** at $1.00, a trade that can't make money. This run led to the [EV guardrail](#how-it-works), which now returns "No trade" here. `tests/test_guardrail.py` checks this case. |
| NBA slate scan for Nov 16, 8 games and 16 markets | No game had a positive-EV bet. The best was −0.96¢ per contract, and no time-left setting gave more than +1.12¢. |

The guardrail stops trades with no edge at the current price. It can't make the agent's picks reliable outside the conditions it was trained on: very wide spreads, prices near 0 or 1, or real-market dynamics that differ from the simulator.

## Layout

```
examples/        live Kalshi runner and NBA slate scanner
pmtrader/
  pricing.py     EV, fees, Kelly
  simulator.py   synthetic markets
  prob_model.py  ML probability model
  nn.py          NumPy MLP + Adam
  env.py         vectorised trading environment
  ppo.py         PPO agent
  backtest.py    RL vs baselines
  service.py     inference/training service
  api.py         FastAPI app
  static/index.html  dashboard
models/          trained weights (JSON)
train.py         CLI training
tests/
```
