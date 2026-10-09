"""Scores for 3 horizons. All scores are 0..1 percentile-rank averages (higher = better).

short (~4 weeks)  : price momentum, skipping the latest month (short-term reversal),
                    + proximity to 52-week high; only stocks above their 200-day SMA.
                    Price-only -> can be (and is) backtested honestly.
mid   (6-18 mo)   : earnings/revenue growth, analyst target upside & rating, 6-1 momentum.
long  (1-5 y)     : quality/valuation vs. the stock's own sector: ROE, margins, low debt,
                    revenue growth, FCF yield, forward P/E.
mid/long use today's fundamentals only (no point-in-time history) -> NOT backtested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_COMPONENTS = 3


def price_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Date x ticker panels. Every value at date t uses data up to t only."""
    hi = close.rolling(252, min_periods=200).max()
    return {
        "mom_1m": close / close.shift(21) - 1,
        "mom_3m": close / close.shift(63) - 1,
        "mom_6_1": close.shift(21) / close.shift(126) - 1,
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "dist_high": close / hi - 1,
        "sma50": close.rolling(50, min_periods=50).mean(),
        "sma200": close.rolling(200, min_periods=200).mean(),
        "close": close,
    }


def snapshot(feats: dict[str, pd.DataFrame], i: int = -1) -> pd.DataFrame:
    """Cross-section (ticker x feature) at row i."""
    return pd.DataFrame({k: v.iloc[i] for k, v in feats.items()})


def _avg_ranks(parts: dict[str, pd.Series], min_count: int = MIN_COMPONENTS) -> pd.Series:
    ranks = pd.DataFrame({k: s.rank(pct=True) for k, s in parts.items()})
    score = ranks.mean(axis=1)
    return score.where(ranks.notna().sum(axis=1) >= min_count)


def short_score(snap: pd.DataFrame) -> pd.Series:
    s = _avg_ranks({
        "mom_6_1": snap["mom_6_1"],
        "mom_12_1": snap["mom_12_1"],
        "dist_high": snap["dist_high"],      # closer to high (less negative) = better
    })
    uptrend = snap["close"] > snap["sma200"]
    return s.where(uptrend)


def mid_score(snap: pd.DataFrame, fund: pd.DataFrame, min_analysts: int = 5) -> pd.Series:
    f = fund.reindex(snap.index)
    price = snap["close"]
    enough = f["numberOfAnalystOpinions"] >= min_analysts
    upside = (f["targetMeanPrice"] / price - 1).where(enough)
    rating = (-f["recommendationMean"]).where(enough)   # 1 = strong buy, 5 = sell
    return _avg_ranks({
        "eps_q_growth": f["earningsQuarterlyGrowth"],
        "eps_growth": f["earningsGrowth"],
        "rev_growth": f["revenueGrowth"],
        "upside": upside,
        "rating": rating,
        "mom_6_1": snap["mom_6_1"],
    })


def _sector_rank(s: pd.Series, sector: pd.Series) -> pd.Series:
    return s.groupby(sector).rank(pct=True)


def long_score(snap: pd.DataFrame, fund: pd.DataFrame, sector: pd.Series) -> pd.Series:
    f = fund.reindex(snap.index)
    sec = sector.reindex(snap.index).fillna("Unknown")
    fpe = f["forwardPE"].where(f["forwardPE"] > 0, np.inf)   # losses -> worst valuation
    fpe = fpe.where(f["forwardPE"].notna())
    fcf_yield = f["freeCashflow"] / f["marketCap"]
    parts = {
        "roe": f["returnOnEquity"],
        "gross_margin": f["grossMargins"],
        "op_margin": f["operatingMargins"],
        "low_debt": -f["debtToEquity"],
        "rev_growth": f["revenueGrowth"],
        "fcf_yield": fcf_yield,
        "cheap_fpe": -fpe,
    }
    ranks = pd.DataFrame({k: _sector_rank(v, sec) for k, v in parts.items()})
    score = ranks.mean(axis=1)
    return score.where(ranks.notna().sum(axis=1) >= 4)


def sector_medians(fund: pd.DataFrame, sector: pd.Series, col: str) -> pd.Series:
    vals = fund[col].where(fund[col] > 0) if col == "forwardPE" else fund[col]
    return vals.groupby(sector.reindex(fund.index)).median()


def score_all(close: pd.DataFrame, fund: pd.DataFrame, universe: pd.DataFrame,
              min_analysts: int = 5) -> pd.DataFrame:
    feats = price_features(close)
    snap = snapshot(feats)
    sector = universe.set_index("ticker")["sector"]
    out = snap.copy()
    out["short"] = short_score(snap)
    out["mid"] = mid_score(snap, fund, min_analysts)
    out["long"] = long_score(snap, fund, sector)
    out["overall"] = out[["short", "mid", "long"]].mean(axis=1, skipna=True)
    out["sector"] = sector.reindex(out.index).fillna("Unknown")
    return out
