"""Build the monthly decision panel: features at each month's first trading day and
the realized covered-call outcome at each candidate strike over the following month.

Each row is (date, ticker). The label for strike choice k is the call overlay return

    overlay_k = (premium_k - max(S_end - K_k, 0)) / S_start

i.e. what writing one month of calls adds to (or takes from) simply holding the stock.
Covered call return = stock return + overlay_k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .options import HAIRCUT, bs_call, bs_delta, iv_proxy

STRIKES = {"ATM": 0.00, "2% OTM": 0.02, "5% OTM": 0.05, "8% OTM": 0.08, "12% OTM": 0.12}
STOCK_FEATURES = ["ret_21", "ret_63", "ret_126", "ret_252", "rv_21", "rv_63", "rv_252", "rv_ratio",
                  "dd_252", "ma50_gap", "ma200_gap", "beta_126", "iv"]
MARKET_FEATURES = ["spy_ret_21", "spy_rv_21", "spy_rv_63", "vix", "vix_premium", "r"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES


def _rv(lr, n):
    return lr.rolling(n, min_periods=int(n * 0.8)).std() * np.sqrt(252)


def daily_features(px: pd.DataFrame, tickers: list[str]):
    close = px[tickers].astype(float)
    spy = px["SPY"].astype(float)
    vix = px["^VIX"].astype(float).ffill() / 100
    r = (px["^IRX"].astype(float).ffill() / 100).fillna(0.0)
    lr, spy_lr = np.log(close).diff(), np.log(spy).diff()
    spy_rv63 = _rv(spy_lr, 63)
    f = {
        "ret_21": close.pct_change(21, fill_method=None), "ret_63": close.pct_change(63, fill_method=None),
        "ret_126": close.pct_change(126, fill_method=None), "ret_252": close.pct_change(252, fill_method=None),
        "rv_21": _rv(lr, 21), "rv_63": _rv(lr, 63), "rv_252": _rv(lr, 252),
        "dd_252": close / close.rolling(252, min_periods=200).max() - 1,
        "ma50_gap": close / close.rolling(50, min_periods=40).mean() - 1,
        "ma200_gap": close / close.rolling(200, min_periods=160).mean() - 1,
        "beta_126": lr.rolling(126, min_periods=100).cov(spy_lr).div(spy_lr.rolling(126, min_periods=100).var(), axis=0),
    }
    f["rv_ratio"] = f["rv_21"] / f["rv_63"]
    f["iv"] = pd.DataFrame(iv_proxy(f["rv_63"].values, vix.values[:, None], spy_rv63.values[:, None]),
                           index=close.index, columns=close.columns)
    m = {"spy_ret_21": spy.pct_change(21, fill_method=None), "spy_rv_21": _rv(spy_lr, 21), "spy_rv_63": spy_rv63,
         "vix": vix, "vix_premium": vix / spy_rv63, "r": r}
    return close, f, m


def _snapshot(close, f, m, d):
    rec = pd.DataFrame({k: v.loc[d] for k, v in f.items()})
    for k, s in m.items():
        rec[k] = s.loc[d]
    rec["S0"] = close.loc[d]
    rec["date"] = d
    rec["ticker"] = rec.index
    return rec


def build_panel(px: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    close, f, m = daily_features(px, tickers)
    idx = close.index
    starts = list(pd.Series(idx, index=idx).groupby(idx.to_period("M")).first())
    rows = []
    for i, d0 in enumerate(starts[:-1]):
        d1 = starts[i + 1]
        rec = _snapshot(close, f, m, d0)
        T = (idx.get_loc(d1) - idx.get_loc(d0)) / 252
        S0, S1 = rec["S0"].values, close.loc[d1].values
        rec["end_date"], rec["T"], rec["S1"] = d1, T, S1
        rec["stock_ret"] = S1 / S0 - 1
        for name, mny in STRIKES.items():
            K = S0 * (1 + mny)
            prem = bs_call(S0, K, T, rec["iv"].values, rec["r"].values) * (1 - HAIRCUT)
            rec[f"prem_{name}"] = prem / S0
            rec[f"delta_{name}"] = bs_delta(S0, K, T, rec["iv"].values, rec["r"].values)
            rec[f"called_{name}"] = S1 > K
            rec[f"ov_{name}"] = (prem - np.maximum(S1 - K, 0)) / S0
        rows.append(rec)
    panel = pd.concat(rows, ignore_index=True)
    return panel.dropna(subset=FEATURES + ["stock_ret"]).reset_index(drop=True)


def live_snapshot(px: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Features at the latest trading day, for live recommendations."""
    close, f, m = daily_features(px, tickers)
    return _snapshot(close, f, m, close.index[-1]).dropna(subset=FEATURES).reset_index(drop=True)
