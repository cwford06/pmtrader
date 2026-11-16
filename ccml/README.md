# ccml: ML covered calls on the S&P 500

A machine-learning model that decides, for every S&P 500 stock each month, whether to sell a 1-month covered call and at which strike: at the money (ATM), 2%, 5%, 8% or 12% out of the money (OTM), or no call.

## Quick start

From the project root (`path/to/pmtrader`), using the project's virtual environment:

```bash
.venv/bin/pip install -r ccml/requirements.txt
.venv/bin/python -m ccml.run all              # download data, walk-forward backtest, train, recommend for all stocks (~2-3 min)
.venv/bin/python -m ccml.run simulate AAPL    # one stock: backtest detail + pick on its live option chain
.venv/bin/python -m ccml.run recommend        # refresh live picks with the saved model
.venv/bin/python -m ccml.calibrate --n 80     # check the volatility estimate against live option prices
```

Add `--refresh` to `all` or `recommend` to download fresh prices; otherwise the cache in `ccml/data/` is used.

## How it works

1. **Data** (`data.py`): current S&P 500 members from Wikipedia, plus daily dividend-adjusted closes from Yahoo Finance since 2012, SPY, VIX and the 13-week T-bill rate.
2. **Features** (`features.py`): 19 per stock-month, including momentum over 1, 3, 6 and 12 months, realized volatility over 1, 3 and 12 months, distance from the 52-week high and the 50- and 200-day averages, beta, the S&P 500's return and volatility, VIX, VIX's premium over realized volatility, and rates.
3. **Labels:** for each strike, the call's net contribution over the next month: `(premium - max(S_end - K, 0)) / S_start`.
4. **Option prices** (`options.py`): free historical option prices aren't available, so past premiums are Black-Scholes values with a volatility estimate, less a haircut for selling at the bid. Both are calibrated against live chains (`calibrate.py`).
5. **Model** (`model.py`): one scikit-learn `HistGradientBoostingRegressor` per strike. Each month it writes the call with the highest predicted contribution, or none if no strike's prediction is above zero.
6. **Backtest** (`backtest.py`): walk-forward from 2019. Each year's model sees only outcomes known before that year. It's compared against buy-and-hold, always-ATM and always-5%-OTM, per stock and as an equal-weight portfolio.

## Outputs (`ccml/output/`)

| File | Contents |
|---|---|
| `summary.json` | Portfolio results per strategy and how often ML beats each baseline |
| `per_ticker.csv` | Return and Sharpe per stock per strategy |
| `recommendations.csv` | This month's pick and predicted edge for every stock |
| `simulate_<TICKER>.json` | Monthly detail, stats, live option chain and the live pick |
| `feature_importance.csv`, `iv_calibration.*` | Model diagnostics |

## Results and caveats

Walk-forward results, 2019 to Nov 2026, for an equal-weight S&P 500 portfolio:

| Strategy | Annual return | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold | 19.2% | 18.7% | 1.04 | −28.0% |
| Covered call ATM | 14.2% | 11.0% | 1.28 | −22.0% |
| Covered call 5% OTM | 20.3% | 14.0% | 1.40 | −24.3% |
| ML covered call | 21.1% | 15.4% | 1.33 | −23.8% |

- **ML isn't clearly better than a simple rule.** It had the highest return, but always selling 5% OTM had a better Sharpe ratio, both for the portfolio and on about 77% of individual stocks.
- **Backtest premiums are estimates.** On the median stock the volatility estimate matches live option prices, but individual stocks differ. For Apple, the real premium was about a third below the estimate. Always check a pick against the real chain: `simulate` does this for you.
- **Survivorship bias:** only today's index members are used.
- **Not modeled:** early assignment, taxes on called-away shares, commissions and lost dividends.

This is a research tool, not investment advice.
