"""Command line for the covered-call ML model. Run from the project root:

    python -m ccml.run all                 # download S&P 500 data, backtest, train, recommend for every stock
    python -m ccml.run simulate AAPL       # one stock: backtest detail + recommendation on its live option chain
    python -m ccml.run recommend           # refresh live recommendations only (uses the saved model)

Outputs go to ccml/output/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import STRATS, metrics, per_ticker, portfolio, walk_forward
from .data import ROOT, load_prices, sp500_constituents
from .features import FEATURES, STRIKES, build_panel, live_snapshot
from .model import NO_CALL, CoveredCallModel
from .options import HAIRCUT, bs_call, bs_delta

OUT = ROOT / "output"
MODELS = ROOT / "models"


def _universe(refresh=False):
    cons = sp500_constituents(refresh)
    px = load_prices(cons["yahoo"].tolist(), refresh=refresh)
    tickers = [t for t in cons["yahoo"] if t in px.columns and px[t].notna().sum() > 300]
    return cons, px, tickers


def recommend(model, px, tickers, cons):
    snap = live_snapshot(px, tickers)
    pred = model.predict(snap)
    snap["choice"] = CoveredCallModel.choose(pred).values
    for name in STRIKES:
        snap[f"pred_{name}"] = pred[name].values
    snap["pred_best_pct"] = (pred.max(axis=1).clip(lower=0) * 100).round(3).values
    rec = snap.merge(cons.rename(columns={"yahoo": "ticker"}), on="ticker", how="left")
    cols = ["ticker", "name", "sector", "date", "S0", "iv", "choice", "pred_best_pct"] + [f"pred_{n}" for n in STRIKES]
    rec = rec[cols].sort_values("pred_best_pct", ascending=False)
    rec.to_csv(OUT / "recommendations.csv", index=False)
    return rec


def cmd_all(args):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    print("Loading S&P 500 constituents and prices...")
    cons, px, tickers = _universe(args.refresh)
    print(f"  {len(tickers)} stocks, {px.index[0].date()} to {px.index[-1].date()}")
    print("Building monthly panel...")
    panel = build_panel(px, tickers)
    panel.to_pickle(OUT / "panel.pkl")
    print(f"  {len(panel):,} stock-months")
    print("Walk-forward backtest...")
    res = walk_forward(panel, args.test_start)
    res.to_pickle(OUT / "walkforward.pkl")
    port = portfolio(res)
    summary = {s: metrics(port[s]) for s in STRATS}
    pt = per_ticker(res)
    pt.to_csv(OUT / "per_ticker.csv", index=False)
    summary["_stocks"] = int(res["ticker"].nunique())
    summary["_ml_beats_bh_sharpe_pct"] = round((pt["ML covered call Sharpe"] > pt["Buy & hold Sharpe"]).mean() * 100, 1)
    summary["_ml_beats_otm5_sharpe_pct"] = round((pt["ML covered call Sharpe"] > pt["Covered call 5% OTM Sharpe"]).mean() * 100, 1)
    summary["_ml_choices_pct"] = (res["ml_choice"].value_counts(normalize=True) * 100).round(1).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("Training final model on all history...")
    model = CoveredCallModel().fit(panel)
    model.save(MODELS / "covered_call_model.joblib")
    imp = model.importances(panel)
    imp.round(6).to_csv(OUT / "feature_importance.csv", header=["importance"])
    print("Live recommendations for every stock...")
    rec = recommend(model, px, tickers, cons)
    print("\nEqual-weight S&P 500 portfolio, out of sample:")
    for s in STRATS:
        m = summary[s]
        print(f"  {s:22s} CAGR {m['cagr_pct']:6.2f}%  vol {m['vol_pct']:5.2f}%  Sharpe {m['sharpe']:.3f}  maxDD {m['max_dd_pct']:6.2f}%")
    print(f"  ML Sharpe beats buy & hold on {summary['_ml_beats_bh_sharpe_pct']}% of stocks, 5% OTM on {summary['_ml_beats_otm5_sharpe_pct']}%")
    print(f"  ML choices: {summary['_ml_choices_pct']}")
    print("\nLive recommendations:", rec["choice"].value_counts().to_dict())
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {OUT}/")


def live_chain(ticker: str, days: int = 30):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    exps = [e for e in tk.options if (pd.Timestamp(e) - today).days >= 14]
    exp = min(exps, key=lambda e: abs((pd.Timestamp(e) - today).days - days))
    calls = tk.option_chain(exp).calls
    return spot, exp, (pd.Timestamp(exp) - today).days, calls


def cmd_simulate(args):
    t = args.ticker.upper().replace(".", "-")
    res = pd.read_pickle(OUT / "walkforward.pkl")
    g = res[res["ticker"] == t].sort_values("date").reset_index(drop=True)
    if g.empty:
        raise SystemExit(f"{t} not in the backtest. Run `python -m ccml.run all` first.")
    model = CoveredCallModel.load(MODELS / "covered_call_model.joblib")
    cons, px, tickers = _universe()
    name = cons.set_index("yahoo").loc[t, "name"] if t in set(cons["yahoo"]) else t

    monthly = []
    eq = {s: 1.0 for s in STRATS}
    for _, r in g.iterrows():
        ch = r["ml_choice"]
        mny = STRIKES.get(ch)
        for s in STRATS:
            eq[s] *= 1 + r[f"ret_{s}"]
        monthly.append({
            "date": r["date"].strftime("%Y-%m-%d"), "end_date": r["end_date"].strftime("%Y-%m-%d"),
            "price": round(r["S0"], 2), "price_end": round(r["S1"], 2), "iv_pct": round(r["iv"] * 100, 1),
            "ml_choice": ch, "strike": round(r["S0"] * (1 + mny), 2) if mny is not None else None,
            "premium_pct": round(r[f"prem_{ch}"] * 100, 2) if mny is not None else 0.0,
            "called": bool(r[f"called_{ch}"]) if mny is not None else False,
            "overlay_pct": round(r["ov_ml"] * 100, 2), "stock_ret_pct": round(r["stock_ret"] * 100, 2),
            "pred_best_pct": round(max(r[f"pred_{n}"] for n in STRIKES) * 100, 3),
            **{f"eq_{k}": round(v, 4) for k, v in zip(["bh", "atm", "otm5", "ml"], eq.values())},
        })
    stats = {s: metrics(g[f"ret_{s}"]) for s in STRATS}
    ml = g["ml_choice"]
    written = ml != NO_CALL
    stats["_ml"] = {"months": int(len(g)), "months_written": int(written.sum()),
                    "times_called": int(sum(m["called"] for m in monthly)),
                    "premium_collected_pct": round(sum(m["premium_pct"] for m in monthly), 1),
                    "choices": ml.value_counts().to_dict()}

    # Live: model recommendation applied to the real option chain
    snap = live_snapshot(px, [t])
    pred = model.predict(snap).iloc[0]
    choice = CoveredCallModel.choose(pred.to_frame().T).iloc[0]
    spot, exp, dte, calls = live_chain(t, args.days)
    T = dte / 365
    iv0, r0 = float(snap["iv"].iloc[0]), float(snap["r"].iloc[0])
    chain = calls[(calls["strike"] >= spot * 0.95) & (calls["strike"] <= spot * 1.20)].copy()
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    chain["model_price"] = bs_call(spot, chain["strike"], T, iv0, r0)
    chain["delta"] = bs_delta(spot, chain["strike"], T, chain["impliedVolatility"].clip(lower=0.05), r0)
    chain["moneyness_pct"] = (chain["strike"] / spot - 1) * 100
    chain["premium_yield_pct"] = chain["bid"] / spot * 100
    chain["annualized_pct"] = chain["premium_yield_pct"] * 365 / dte
    chain["max_return_pct"] = (chain["strike"] - spot + chain["bid"]) / spot * 100
    chain["breakeven"] = spot - chain["bid"]
    pick = None
    if choice != NO_CALL:
        target = spot * (1 + STRIKES[choice])
        above = chain[chain["strike"] >= target]
        pick = (above.iloc[0] if len(above) else chain.iloc[-1]).to_dict()
    live = {"ticker": t, "name": name, "spot": round(spot, 2), "expiry": exp, "dte": dte,
            "pulled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "iv_proxy_pct": round(iv0 * 100, 1), "choice": choice,
            "pred_pct": {k: round(v * 100, 3) for k, v in pred.items()}}
    if pick:
        live.update({"strike": pick["strike"], "bid": pick["bid"], "ask": pick["ask"],
                     "market_iv_pct": round(pick["impliedVolatility"] * 100, 1), "delta": round(pick["delta"], 3),
                     "premium_yield_pct": round(pick["premium_yield_pct"], 2), "annualized_pct": round(pick["annualized_pct"], 1),
                     "max_return_pct": round(pick["max_return_pct"], 2), "breakeven": round(pick["breakeven"], 2),
                     "premium_100sh": round(pick["bid"] * 100, 2), "contract": pick["contractSymbol"]})
    chain_rows = chain[["contractSymbol", "strike", "bid", "ask", "mid", "lastPrice", "volume", "openInterest",
                        "impliedVolatility", "model_price", "delta", "moneyness_pct", "premium_yield_pct",
                        "annualized_pct", "max_return_pct", "breakeven"]].replace({np.nan: None}).round(4)
    out = {"monthly": monthly, "stats": stats, "live": live, "chain": chain_rows.to_dict("records")}
    (OUT / f"simulate_{t}.json").write_text(json.dumps(out, indent=1, default=str))

    print(f"{name} ({t}) out-of-sample {monthly[0]['date']} to {monthly[-1]['end_date']}:")
    for s in STRATS:
        m = stats[s]
        print(f"  {s:22s} CAGR {m['cagr_pct']:6.2f}%  Sharpe {m['sharpe']:.3f}  maxDD {m['max_dd_pct']:6.2f}%  total {m['total_pct']}%")
    print("  ML:", stats["_ml"])
    print("Live:", json.dumps(live, indent=1, default=str))


def cmd_recommend(args):
    model = CoveredCallModel.load(MODELS / "covered_call_model.joblib")
    cons, px, tickers = _universe(args.refresh)
    rec = recommend(model, px, tickers, cons)
    print(rec.head(25).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all"); a.add_argument("--refresh", action="store_true"); a.add_argument("--test-start", type=int, default=2019)
    s = sub.add_parser("simulate"); s.add_argument("ticker"); s.add_argument("--days", type=int, default=30)
    r = sub.add_parser("recommend"); r.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    {"all": cmd_all, "simulate": cmd_simulate, "recommend": cmd_recommend}[args.cmd](args)


if __name__ == "__main__":
    main()
