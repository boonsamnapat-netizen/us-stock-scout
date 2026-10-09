#!/usr/bin/env python
"""Do the scout's chart stages beat comparable stocks afterwards?  (price-only -> testable)

Method (after methodology audit, 2026-10-09):
- Entry at the NEXT day's close after the signal; round-trip cost subtracted (assumption 2 x 0.10%).
- Headline = excess vs the equal-weighted universe on the same date ("all stocks"), not vs SPY
  (avoids equal- vs cap-weight mismatch). Also excess vs the 📈 uptrend group.
- Overlapping windows (monthly samples, 3–12m horizons) -> Newey-West t-stat with lags = h/21 - 1.
- Episodes counted once: a stock signalling on consecutive sample dates counts as one episode.
- Known, NOT fixed: survivorship (today's index members; delisted losers missing) — inflates
  every row, pullback/throwback most. Results = "historically associated with", not expected returns.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from scout import data, stages, universe as uni

COST = 0.002


def nw_tstat(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def evaluate(close, masks: dict, h: int, step: int = 21, start: int = 300) -> pd.DataFrame:
    fwd = close.shift(-h - 1) / close.shift(-1) - 1 - COST          # enter next day, pay costs
    idx = list(range(start, len(close) - h - 1, step))
    univ = pd.Series([fwd.iloc[i][close.iloc[i].notna()].mean() for i in idx], index=idx)
    up_mask = masks["📈 v1 uptrend"]
    upt = pd.Series([fwd.iloc[i][up_mask.iloc[i].fillna(False).astype(bool)].mean() for i in idx], index=idx)
    rows = []
    for label, mask in masks.items():
        ex_u, ex_up, per_sig, episodes, prev = [], [], [], 0, set()
        for i in idx:
            m = mask.iloc[i].fillna(False).astype(bool)
            names = set(m[m].index)
            v = fwd.iloc[i][list(names)].dropna() if names else pd.Series(dtype=float)
            episodes += len(names - prev)
            prev = names
            if len(v):
                ex_u.append(v.mean() - univ[i])
                ex_up.append(v.mean() - upt[i])
                per_sig += list(v.values - univ[i])
            else:
                ex_u.append(np.nan)
                ex_up.append(np.nan)
        ex_u, ex_up, per_sig = np.array(ex_u), np.array(ex_up), np.array(per_sig)
        lags = max(h // step - 1, 0)
        rows.append({
            "group": label, "episodes": episodes,
            "ex_vs_universe": np.nanmean(ex_u), "t_NW": nw_tstat(ex_u, lags),
            "ex_vs_uptrend": np.nanmean(ex_up),
            "median_signal_ex": np.median(per_sig) if len(per_sig) else np.nan,
            "hit_rate": (per_sig > 0).mean() if len(per_sig) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "horizon", f"{h // 21}m")
    out.attrs["universe_avg"] = univ.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=11)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--universe", choices=["base", "extended"], default="extended")
    a = ap.parse_args()
    if a.demo:
        _, close, volume, bench, _ = data.demo_data(n=120, years=6)
    else:
        cfg = yaml.safe_load(open("config.yaml"))
        u, src = uni.load_extended(cfg, ".") if a.universe == "extended" else uni.load_universe(cfg, ".")
        close_all, volume = data.fetch_prices(u["ticker"].tolist(), a.years)
        close = close_all
        print(f"universe {close.shape[1]} ({src}) {close.index[0].date()} → {close.index[-1].date()}")
    p = stages.stage_panels(close, volume)
    p2 = stages.stage_panels_v2(close, volume)
    masks = {"🚀 v1 breakout": p["breakout"], "🚀 v2 breakout (book)": p2["breakout2"],
             "🔄 v1 pullback": p["pullback"], "🔄 v2 throwback (book)": p2["throwback"],
             "Trend Template": p2["template"], "📈 v1 uptrend": p["uptrend"]}
    res = pd.concat([evaluate(close, masks, h) for h in (63, 126, 252)], ignore_index=True)
    fmt = res.copy()
    for c in ("ex_vs_universe", "ex_vs_uptrend", "median_signal_ex"):
        fmt[c] = (res[c] * 100).map(lambda v: f"{v:+.1f}%")
    fmt["hit_rate"] = (res["hit_rate"] * 100).map(lambda v: f"{v:.0f}%")
    fmt["t_NW"] = res["t_NW"].map(lambda v: f"{v:+.2f}")
    pd.set_option("display.width", 200)
    print("excess vs equal-weight universe · entry next day · cost 0.2% round trip · "
          "survivorship NOT removed (today's members) · |t_NW| < 2 = not distinguishable from noise")
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
