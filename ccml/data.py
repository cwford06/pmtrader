"""Download and cache S&P 500 constituents and daily prices.

Prices come from Yahoo Finance via yfinance (dividend/split-adjusted closes).
Market series: SPY (S&P 500 ETF), ^VIX (implied volatility index), ^IRX (13-week T-bill yield).
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
UA = {"User-Agent": "ccml-research/1.0"}
MARKET = ["SPY", "^VIX", "^IRX"]


def sp500_constituents(refresh: bool = False) -> pd.DataFrame:
    """Current S&P 500 members: symbol, name, sector, yahoo (Yahoo ticker)."""
    f = CACHE / "constituents.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f)
    try:
        html = urllib.request.urlopen(urllib.request.Request(WIKI, headers=UA), timeout=30).read().decode()
        t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    except Exception:
        t = pd.read_csv(GITHUB)
    df = t.rename(columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"})[["symbol", "name", "sector"]]
    df["yahoo"] = df["symbol"].str.replace(".", "-", regex=False)
    CACHE.mkdir(exist_ok=True)
    df.to_csv(f, index=False)
    return df


def load_prices(tickers: list[str], start: str = "2012-01-01", refresh: bool = False) -> pd.DataFrame:
    """Wide frame of adjusted closes (dates x tickers), including the MARKET series."""
    f = CACHE / "prices.pkl"
    want = sorted(set(tickers) | set(MARKET))
    if f.exists() and not refresh:
        px = pd.read_pickle(f)
        if set(want) <= set(px.columns):
            return px
    import yfinance as yf

    frames = []
    for i in range(0, len(want), 100):
        d = yf.download(want[i:i + 100], start=start, auto_adjust=True, progress=False, threads=True)["Close"]
        frames.append(d if isinstance(d, pd.DataFrame) else d.to_frame())
    px = pd.concat(frames, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()].dropna(how="all")
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    CACHE.mkdir(exist_ok=True)
    px.to_pickle(f)
    return px
