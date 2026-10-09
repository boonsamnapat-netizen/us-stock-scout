#!/usr/bin/env python
"""Do the scout's chart stages (🚀 breakout / 🔄 pullback) beat the market afterwards?
Price-only -> testable. Samples every 21 trading days; forward 3/6/12-month returns vs SPY.
Fundamental filters are NOT included (no point-in-time data) — this tests the timing part only.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from scout import data, stages, universe as uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=11)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _, close, volume, bench, _ = data.demo_data(n=120, years=6)
        spy = bench["SPY"]
    else:
        u, src = uni.load_universe(yaml.safe_load(open("config.yaml")), ".")
        close_all, volume = data.fetch_prices(u["ticker"].tolist() + ["SPY"], a.years)
        spy, close = close_all["SPY"], close_all.drop(columns=["SPY"])
        volume = volume.drop(columns=["SPY"], errors="ignore")
        print(f"universe {close.shape[1]} ({src}) {close.index[0].date()} → {close.index[-1].date()}")
    p = stages.stage_panels(close, volume)
    rows = []
    for h, name in ((63, "3m"), (126, "6m"), (252, "12m")):
        fwd = close.shift(-h) / close - 1
        spy_fwd = spy.reindex(close.index).shift(-h) / spy.reindex(close.index) - 1
        idx = range(300, len(close) - h, 21)
        for label, mask in (("🚀 breakout", p["breakout"]), ("🔄 pullback", p["pullback"]),
                            ("📈 uptrend", p["uptrend"]), ("all stocks", close.notna())):
            r, ex, n = [], [], 0
            for i in idx:
                m = mask.iloc[i].fillna(False).astype(bool)
                v = fwd.iloc[i][m].dropna()
                if len(v):
                    r.append(v.mean())
                    ex.append(v.mean() - spy_fwd.iloc[i])
                    n += len(v)
            if r:
                rows.append({"horizon": name, "group": label, "signals": n,
                             "avg_return": np.mean(r), "avg_vs_spy": np.mean(ex),
                             "beat_spy_pct_of_dates": np.mean(np.array(ex) > 0)})
    df = pd.DataFrame(rows)
    out = df.copy()
    for c in ("avg_return", "avg_vs_spy", "beat_spy_pct_of_dates"):
        out[c] = (df[c] * 100).map(lambda v: f"{v:+.1f}%" if c != "beat_spy_pct_of_dates" else f"{v:.0f}%")
    print("equal-weight per sampling date · today's index members (survivorship bias) · no costs")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
