"""Export ccml outputs to JSON dataset files for the dashboard.

    python cc_export.py <project_root> <ticker> <out_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

root, T, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(root))
from ccml.backtest import STRATS, metrics, portfolio
from ccml.options import HAIRCUT

O = root / "ccml" / "output"
out.mkdir(parents=True, exist_ok=True)
summary = json.loads((O / "summary.json").read_text())
res = pd.read_pickle(O / "walkforward.pkl")
sim = json.loads((O / f"simulate_{T}.json").read_text())
recs = pd.read_csv(O / "recommendations.csv")
imp = pd.read_csv(O / "feature_importance.csv", index_col=0)["importance"]
cal = pd.read_csv(O / "iv_calibration.csv").dropna(subset=["rv63", "proxy_iv"])
keys = {"Buy & hold": "bh", "Covered call ATM": "atm", "Covered call 5% OTM": "otm5", "ML covered call": "ml"}

port = portfolio(res)
eq = (1 + port).cumprod()
port_rows = [{"date": d.strftime("%Y-%m-%d"), **{keys[s]: round(eq.loc[d, s], 4) for s in STRATS}} for d in eq.index]
pt = pd.read_csv(O / "per_ticker.csv")
summ_rows = [{"key": keys[s], "strategy": s, **summary[s]} for s in STRATS]

stats_rows = [{"key": keys[s], "strategy": s, **sim["stats"][s]} for s in STRATS]
mlst = sim["stats"]["_ml"]

live = dict(sim["live"])
pred = live.pop("pred_pct")
chain = pd.DataFrame(sim["chain"])
if live["choice"] != "No call":
    row = chain[chain["strike"] == live["strike"]].iloc[0]
    proxy_prem = row["model_price"] * (1 - HAIRCUT)
    live["proxy_premium"] = round(proxy_prem, 2)
    live["model_edge_pct"] = pred[live["choice"]]
    live["real_edge_pct"] = round(pred[live["choice"]] + (live["bid"] - proxy_prem) / live["spot"] * 100, 3)
    # An earnings report before expiry inflates the market premium for a reason the model
    # can't see (jump risk), so the extra premium isn't counted as edge in that case.
    import yfinance as yf
    try:
        ycal = yf.Ticker(T).calendar
        ed = ycal.get("Earnings Date") if isinstance(ycal, dict) else None
        ed = str(ed[0]) if ed else None
    except Exception:
        ed = None
    live["earnings_date"] = ed
    live["earnings_in_window"] = bool(ed and ed <= live["expiry"])
    live["edge_used_pct"] = round(min(live["real_edge_pct"], live["model_edge_pct"]) if live["earnings_in_window"] else live["real_edge_pct"], 3)
    live["decision"] = "SELL CALL" if live["edge_used_pct"] > 0 else "NO TRADE"
    live["contracts_per_100"] = 1
else:
    live["decision"] = "NO TRADE"
live.update({"months": mlst["months"], "months_written": mlst["months_written"], "times_called": mlst["times_called"],
             "premium_collected_pct": mlst["premium_collected_pct"],
             "sp_stocks": summary["_stocks"], "ml_beats_bh_pct": summary["_ml_beats_bh_sharpe_pct"],
             "ml_beats_otm5_pct": summary["_ml_beats_otm5_sharpe_pct"]})
pred_rows = [{"strike_choice": k, "pred_overlay_pct": v} for k, v in pred.items()]

choice_order = ["No call", "ATM", "2% OTM", "5% OTM", "8% OTM", "12% OTM"]
rec_rows = recs[["ticker", "name", "sector", "S0", "iv", "choice", "pred_best_pct"]].rename(columns={"S0": "price"})
rec_rows["iv_pct"] = (rec_rows.pop("iv") * 100).round(1)
rec_rows["price"] = rec_rows["price"].round(2)
mix_live = recs["choice"].value_counts()
mix_bt = res["ml_choice"].value_counts(normalize=True) * 100
mix_rows = [{"choice": c, "live_stocks": int(mix_live.get(c, 0)), "backtest_pct": round(float(mix_bt.get(c, 0)), 1)} for c in choice_order]
imp_rows = [{"feature": k, "importance": round(float(v), 6)} for k, v in imp.items()]
cal_rows = cal[["ticker", "rv63", "proxy_iv", "market_iv", "spread_pct"]].round(4).to_dict("records")
calj = json.loads((O / "iv_calibration.json").read_text())

code = Path(__file__).read_text()
srcs = {p: (root / "ccml" / p).read_text() for p in ["features.py", "options.py", "model.py", "backtest.py", "run.py"]}
meta = [{"label": "cc_export.py", "value": code, "type": "code", "lang": "python"}] + \
       [{"label": f"ccml/{p}", "value": v, "type": "code", "lang": "python"} for p, v in srcs.items()] + \
       [{"label": "Input", "value": f"Yahoo Finance daily adjusted closes for {summary['_stocks']} S&P 500 stocks, SPY, ^VIX, ^IRX (2012 on); live {T} option chain pulled {sim['live']['pulled_utc']} UTC"}]
files = {"cc_summary": summ_rows, "cc_portfolio": port_rows, "cc_monthly": sim["monthly"], "cc_stats": stats_rows,
         "cc_live": [live], "cc_pred": pred_rows, "cc_chain": sim["chain"], "cc_recs": rec_rows.to_dict("records"),
         "cc_mix": mix_rows, "cc_importance": imp_rows, "cc_calibration": cal_rows,
         "cc_calib_summary": [calj]}
for n, rows in files.items():
    (out / f"{n}.json").write_text(json.dumps({"rows": rows, "meta": meta}, default=str))
    print(n, len(rows))
print(json.dumps(live, indent=1))
