"""Walk-forward backtest: each test year uses a model trained only on months whose
outcome was known before that year began. Compares the ML strategy to buy-and-hold
and to fixed-strike covered calls, per stock and as an equal-weight S&P portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import STRIKES
from .model import NO_CALL, CoveredCallModel

STRATS = {"Buy & hold": None, "Covered call ATM": "ATM", "Covered call 5% OTM": "5% OTM", "ML covered call": "ml"}


def walk_forward(panel: pd.DataFrame, test_start: int = 2019, seed: int = 0, log=print) -> pd.DataFrame:
    out = []
    for year in range(test_start, panel["date"].dt.year.max() + 1):
        y0 = pd.Timestamp(f"{year}-01-01")
        train = panel[panel["end_date"] < y0]
        test = panel[(panel["date"] >= y0) & (panel["date"] < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty:
            continue
        model = CoveredCallModel(seed).fit(train)
        pred = model.predict(test)
        t = test.copy()
        for name in STRIKES:
            t[f"pred_{name}"] = pred[name].values
        t["ml_choice"] = CoveredCallModel.choose(pred).values
        out.append(t)
        log(f"  {year}: trained on {len(train):,} stock-months, tested {len(test):,}")
    res = pd.concat(out, ignore_index=True)
    ov = np.zeros(len(res))
    for name in STRIKES:
        ov = np.where(res["ml_choice"] == name, res[f"ov_{name}"], ov)
    res["ov_ml"] = ov
    res["ret_Buy & hold"] = res["stock_ret"]
    res["ret_Covered call ATM"] = res["stock_ret"] + res["ov_ATM"]
    res["ret_Covered call 5% OTM"] = res["stock_ret"] + res["ov_5% OTM"]
    res["ret_ML covered call"] = res["stock_ret"] + res["ov_ml"]
    return res


def metrics(monthly: pd.Series) -> dict:
    m = monthly.dropna()
    if len(m) < 2:
        return {}
    eq = (1 + m).cumprod()
    years = len(m) / 12
    return {"cagr_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
            "vol_pct": round(m.std() * np.sqrt(12) * 100, 2),
            "sharpe": round(m.mean() / m.std() * np.sqrt(12), 3) if m.std() > 0 else None,
            "max_dd_pct": round(((eq / eq.cummax()) - 1).min() * 100, 2),
            "total_pct": round((eq.iloc[-1] - 1) * 100, 1), "months": int(len(m))}


def portfolio(res: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of every stock's monthly return, per strategy."""
    return res.groupby("date")[[f"ret_{s}" for s in STRATS]].mean().rename(columns=lambda c: c[4:])


def per_ticker(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in res.groupby("ticker"):
        g = g.sort_values("date")
        row = {"ticker": t, "months": len(g)}
        for s in STRATS:
            mt = metrics(g[f"ret_{s}"])
            row[f"{s} CAGR %"] = mt.get("cagr_pct")
            row[f"{s} Sharpe"] = mt.get("sharpe")
        row["ML no-call %"] = round((g["ml_choice"] == NO_CALL).mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)
