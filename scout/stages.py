"""Chart stages (price/volume only -> backtestable) and industry-group strength.

breakout (not shown as a buy section — underperformed in research_stages): within the last 10 days the close broke above its prior 6-month high,
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
    close = close.ffill(limit=3)          # one missing bar must not blank 200-day SMAs for 200 days
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
        df = df[df["rel3m"].notna()]
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


# ---------------------------------------------------------------- v2: book-based definitions
# Sources (secondary; see research notes): Minervini Trend Template; IBD pivot/buy range/volume;
# O'Neil min base length & depth; Weinstein throwback. Windows marked (choice) are ours, not the books'.
def trend_template(close: pd.DataFrame, rs_min: float = 0.70) -> pd.DataFrame:
    close = close.ffill(limit=3)
    s50, s150, s200 = (close.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    lo52 = close.rolling(252, min_periods=200).min()
    hi52 = close.rolling(252, min_periods=200).max()
    rs = (close / close.shift(252) - 1).rank(axis=1, pct=True)        # proxy for IBD RS rating
    return ((close > s150) & (close > s200) & (s150 > s200) & (s200 > s200.shift(22))
            & (s50 > s150) & (s50 > s200) & (close > s50)
            & (close >= 1.30 * lo52) & (close >= 0.75 * hi52) & (rs >= rs_min))


def stage_panels_v2(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = close.ffill(limit=3)
    vol = volume.reindex_like(close)
    vol50 = vol.rolling(50, min_periods=50).mean()
    tt = trend_template(close)
    base_hi = close.shift(1).rolling(35, min_periods=35).max()           # 7-week base (O'Neil)
    base_lo = close.shift(1).rolling(35, min_periods=35).min()
    depth_ok = (base_hi - base_lo) / base_hi <= 0.33                     # cup depth limit (O'Neil)
    based = close.shift(1).rolling(10, min_periods=10).max() < base_hi   # high set >= 10 days ago (choice)
    depth_ok = depth_ok & based
    pivot = base_hi + 0.10                                               # IBD pivot
    sig = (tt & depth_ok & (close > pivot) & (close <= pivot * 1.05)    # IBD 5% buy range
           & (vol >= 1.4 * vol50))                                       # IBD +40% volume
    piv_recent = pivot.where(sig).ffill(limit=4)                         # signal in last 5 days (choice)
    recent = sig.astype(float).rolling(5, min_periods=1).max() > 0
    breakout = recent & (close <= piv_recent * 1.05) & (close >= piv_recent * 0.97)

    s50 = close.rolling(50, min_periods=50).mean()
    s150 = close.rolling(150, min_periods=150).mean()
    s200 = close.rolling(200, min_periods=200).mean()
    piv40 = pivot.where(sig).ffill(limit=40)                             # breakout in last 40 days (choice)
    support = np.maximum(piv40, s50)
    vol5 = vol.rolling(5, min_periods=5).mean()
    advanced = close.rolling(40, min_periods=1).max() >= piv40 * 1.05    # moved up after breakout (choice)
    throwback = (piv40.notna() & ~recent & advanced & (close > s150) & (s150 > s200)
                 & (close <= support * 1.03) & (close >= piv40 * 0.97)    # 3% tolerance (choice)
                 & (vol5 < vol50))                                        # lighter volume
    return {"breakout2": breakout.fillna(False), "throwback": throwback.fillna(False),
            "template": tt.fillna(False), "breakout2_day": sig.fillna(False)}
