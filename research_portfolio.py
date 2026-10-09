#!/usr/bin/env python
"""Compare portfolio rules on real prices (run in GitHub Actions: Research workflow).

  python research_portfolio.py --years 11          # real data
  python research_portfolio.py --demo               # synthetic, offline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from scout import data, universe as uni
from scout.portfolio_bt import Rules, score_panel, simulate

VARIANTS = [
    Rules("A naive: monthly top5 replace-all, no gate", check="monthly", replace_all=True, gate="none",
          max_trades_month=99),
    Rules("B weekly 10/30, no gate", gate="none"),
    Rules("C weekly 10/30, no new buys in downtrend", gate="nobuy"),
    Rules("D weekly 10/30, cash in downtrend", gate="cash"),
    Rules("E daily 10/30, cash in downtrend", check="daily", gate="cash"),
    Rules("F weekly 5/20, cash in downtrend", entry_pct=0.05, exit_pct=0.20, gate="cash"),
    Rules("G weekly 10/50, cash in downtrend", exit_pct=0.50, gate="cash"),
    Rules("H weekly 10/30, cash in downtrend, n=8", n_hold=8, gate="cash"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=11)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fee", type=float, default=0.10)
    a = ap.parse_args()
    if a.demo:
        _, close, _, bench, _ = data.demo_data(n=120, years=8)
        spy = bench["SPY"]
    else:
        cfg = yaml.safe_load(open("config.yaml"))
        u, src = uni.load_universe(cfg, ".")
        close_all, _ = data.fetch_prices(u["ticker"].tolist() + ["SPY"], a.years)
        spy = close_all["SPY"]
        close = close_all.drop(columns=["SPY"])
        print(f"universe {close.shape[1]} ({src}), {close.index[0].date()} → {close.index[-1].date()}")

    score = score_panel(close)
    rows = []
    for r in VARIANTS:
        r.fee_per_side_pct = a.fee
        res = simulate(close, spy, r, score=score)
        rows.append({"rules": r.name, **res["stats"]})
        res["trades"].to_csv(Path("output") / f"trades_{r.name[0]}.csv", index=False)
    df = pd.DataFrame(rows).set_index("rules")
    Path("output").mkdir(exist_ok=True)
    df.to_csv("output/portfolio_research.csv")
    fmt = df.copy()
    for c in fmt.columns:
        if c.startswith(("cagr", "maxdd", "spy", "worst")):
            fmt[c] = (fmt[c] * 100).map(lambda v: f"{v:+.1f}%")
    fmt["trades_per_month"] = df["trades_per_month"].map(lambda v: f"{v:.1f}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"fee {a.fee:.2f}%/side · today's index members (survivorship bias) · rank = price momentum only")
    print(fmt.to_string())


if __name__ == "__main__":
    main()
