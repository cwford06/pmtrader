"""Check the historical IV proxy against real option markets.

Pulls live ~30-day at-the-money call implied volatility from Yahoo for a random sample
of S&P 500 stocks and compares it with (a) 63-day realized vol and (b) the proxy.
The fitted ratio real_iv / rv63 sets IV_SCALE in options.py.

    python -m ccml.calibrate --n 80
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import ROOT, load_prices, sp500_constituents
from .features import daily_features


def atm_iv(ticker, days=30):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    exps = [e for e in tk.options if (pd.Timestamp(e) - today).days >= 14]
    if not exps:
        return None
    exp = min(exps, key=lambda e: abs((pd.Timestamp(e) - today).days - days))
    calls = tk.option_chain(exp).calls
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
    if calls.empty:
        return None
    row = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    return {"ticker": ticker, "expiry": exp, "spot": spot, "strike": float(row["strike"]),
            "market_iv": float(row["impliedVolatility"]), "spread_pct": float((row["ask"] - row["bid"]) / ((row["ask"] + row["bid"]) / 2) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cons = sp500_constituents()
    px = load_prices(cons["yahoo"].tolist())
    tickers = [t for t in cons["yahoo"] if t in px.columns]
    close, f, m = daily_features(px, tickers)
    last = close.index[-1]
    sample = pd.Series(tickers).sample(args.n, random_state=args.seed).tolist()
    rows = []
    for t in sample:
        try:
            r = atm_iv(t)
        except Exception:
            r = None
        if r and 0.03 < r["market_iv"] < 3:
            r.update(rv63=float(f["rv_63"].loc[last, t]), proxy_iv=float(f["iv"].loc[last, t]))
            rows.append(r)
    df = pd.DataFrame(rows).dropna(subset=["rv63", "proxy_iv"])
    ratio_rv = float(np.median(df["market_iv"] / df["rv63"]))
    ratio_proxy = float(np.median(df["market_iv"] / df["proxy_iv"]))
    vp = float(m["vix_premium"].loc[last])
    out = {"as_of": str(last.date()), "n": len(df), "vix": float(m["vix"].loc[last]),
           "spy_rv63": float(m["spy_rv_63"].loc[last]), "vix_over_spy_rv": round(vp, 3),
           "median_market_iv_over_rv63": round(ratio_rv, 3), "median_market_iv_over_proxy": round(ratio_proxy, 3),
           "suggested_IV_BETA": round((ratio_rv - 1) / (vp - 1), 2) if vp > 1 else None,
           "median_spread_pct": round(float(df["spread_pct"].median()), 2),
           "suggested_HAIRCUT": round(float(df["spread_pct"].median()) / 200, 4),
           "corr_market_iv_rv63": round(float(np.corrcoef(df["market_iv"], df["rv63"])[0, 1]), 3)}
    (ROOT / "output").mkdir(exist_ok=True)
    df.to_csv(ROOT / "output" / "iv_calibration.csv", index=False)
    (ROOT / "output" / "iv_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
