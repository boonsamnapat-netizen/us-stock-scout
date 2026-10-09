"""Chart stages (price/volume only -> backtestable) and industry-group strength.

🚀 breakout   "เพิ่งเริ่มวิ่ง": within the last 10 days the close broke above its prior 6-month high,
              after a base (prior 6-month range <= 40%), 200-day SMA not falling, volume picking up,
              and price not yet extended (< 10% above the breakout level).
🔄 pullback   "ย่อในขาขึ้น": long-term uptrend intact (close > rising 200-day SMA, 50-day > 200-day)
              but 10–30% below the 52-week high.
📈 uptrend    trend template met, near highs (neither of the above).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKOUT_WINDOW = 10      # breakout happened within the last N days
BASE_DAYS = 126           # ~6 months
MAX_BASE_RANGE = 0.40
MAX_EXTENSION = 0.10


def stage_panels(close: pd.DataFrame, volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    sma200_up = sma200 >= sma200.shift(21)
    hi252 = close.rolling(252, min_periods=200).max()
    dd = close / hi252 - 1

    # base = the 6 months *before* the breakout window
    base_hi = close.shift(BREAKOUT_WINDOW).rolling(BASE_DAYS, min_periods=BASE_DAYS).max()
    base_lo = close.shift(BREAKOUT_WINDOW).rolling(BASE_DAYS, min_periods=BASE_DAYS).min()
    broke = (close.rolling(BREAKOUT_WINDOW).max() > base_hi)
    tight_base = (base_hi / base_lo - 1) <= MAX_BASE_RANGE
    not_extended = close <= base_hi * (1 + MAX_EXTENSION)
    held = close > base_hi * 0.97                       # still holding near/above the breakout level
    if volume is not None:
        vol = volume.reindex_like(close)
        vol_up = vol.rolling(20).mean() >= 1.2 * vol.rolling(100).mean()
    else:
        vol_up = pd.DataFrame(True, index=close.index, columns=close.columns)
    breakout = broke & tight_base & not_extended & held & sma200_up & vol_up

    trend = (close > sma200) & sma200_up & (sma50 > sma200)
    pullback = trend & (dd <= -0.10) & (dd >= -0.30) & ~breakout
    uptrend = trend & (dd > -0.10) & ~breakout
    return {"breakout": breakout, "pullback": pullback, "uptrend": uptrend, "dd": dd,
            "sma200_up": sma200_up, "base_hi": base_hi}


def stage_today(panels: dict[str, pd.DataFrame]) -> pd.Series:
    last = {k: v.iloc[-1] for k, v in panels.items()}
    s = pd.Series("none", index=last["breakout"].index)
    s[last["uptrend"].fillna(False).astype(bool)] = "uptrend"
    s[last["pullback"].fillna(False).astype(bool)] = "pullback"
    s[last["breakout"].fillna(False).astype(bool)] = "breakout"
    return s


def group_strength(close: pd.DataFrame, industry: pd.Series, spy: pd.Series, min_members: int = 3,
                   lookback: int = 20) -> pd.DataFrame:
    """Industry groups ranked by: median 3-month return vs SPY + share of members above a rising
    50-day SMA. 'rank_change' = improvement vs `lookback` days ago (positive = heating up)."""
    def snapshot(i: int) -> pd.DataFrame:
        c = close.iloc[: len(close) + i] if i < 0 else close
        s = spy.reindex(close.index).iloc[: len(close) + i] if i < 0 else spy.reindex(close.index)
        rel3m = (c.iloc[-1] / c.iloc[-64] - 1) - (s.iloc[-1] / s.iloc[-64] - 1)
        sma50 = c.rolling(50).mean()
        strong = (c.iloc[-1] > sma50.iloc[-1]) & (sma50.iloc[-1] > sma50.iloc[-11])
        df = pd.DataFrame({"industry": industry.reindex(c.columns), "rel3m": rel3m, "strong": strong})
        g = df.groupby("industry").agg(n=("rel3m", "size"), rel3m=("rel3m", "median"),
                                       breadth=("strong", "mean"))
        g = g[g["n"] >= min_members]
        g["score"] = g["rel3m"].rank(pct=True) * 0.5 + g["breadth"].rank(pct=True) * 0.5
        g["rank"] = g["score"].rank(ascending=False, method="min")
        return g

    now = snapshot(0)
    before = snapshot(-lookback)
    now["rank_change"] = (before["rank"].reindex(now.index) - now["rank"])
    return now.sort_values("rank")
