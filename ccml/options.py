"""Black-Scholes call pricing and the implied-volatility proxy used for history.

Free historical option prices aren't available, so past premiums are priced with
Black-Scholes using an IV proxy:

    scale = clip(1 + IV_BETA x (VIX / SPY 63-day realized vol - 1), 0.9, 1.6)
    iv    = stock 63-day realized vol x scale

Single-stock options carry a smaller volatility premium than the index (VIX), so only
part of the index premium is passed through. IV_BETA and HAIRCUT were calibrated
against live Yahoo option chains for 79 random S&P 500 stocks (python -m ccml.calibrate,
2026-11-13 closes): real ATM ~30-day IV was a median 1.17x realized vol while
VIX / SPY realized was 1.30, giving IV_BETA = 0.17 / 0.30 = 0.57; the median ATM
bid-ask spread was 17.5% of mid, so selling at the bid gives up about half of that.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

HAIRCUT = 0.0875    # sell at the bid: half of the measured 17.5% median spread
IV_BETA = 0.57      # share of the index volatility premium passed to single stocks
IV_FLOOR, IV_CAP = 0.10, 2.0


def _d1(S, K, T, sigma, r):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, sigma, r=0.0):
    S, K, T, sigma, r = (np.asarray(x, float) for x in (S, K, T, sigma, r))
    d1 = _d1(S, K, T, sigma, r)
    d2 = d1 - sigma * np.sqrt(T)
    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def bs_delta(S, K, T, sigma, r=0.0):
    return ndtr(_d1(*(np.asarray(x, float) for x in (S, K, T, sigma, r))))


def iv_proxy(rv63, vix, spy_rv63):
    scale = np.clip(1 + IV_BETA * (vix / spy_rv63 - 1), 0.9, 1.6)
    return np.clip(rv63 * scale, IV_FLOOR, IV_CAP)
