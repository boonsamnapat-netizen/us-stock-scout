"""Walk-forward test of the short-term (4-week) screen + per-stock history stats.

Rules are fixed in advance (no parameter fitting), so every period is out-of-sample
with respect to the rules. Known bias: the universe is TODAY's index members
(survivorship bias) -> results are optimistic vs. what was investable back then.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import price_features, short_score, snapshot


def backtest_short(close: pd.DataFrame, bench: pd.Series, horizon: int = 20, top_n: int = 10,
                   fee_per_side_pct: float = 0.10, min_history: int = 273) -> dict:
    feats = price_features(close)
    bench = bench.reindex(close.index).ffill()
    bench_sma = bench.rolling(200, min_periods=200).mean()
    fee = 2 * fee_per_side_pct / 100
    rows = []
    for i in range(min_history, len(close) - horizon, horizon):
        snap = snapshot(feats, i)
        score = short_score(snap).dropna()
        if len(score) < top_n:
            continue
        picks = score.nlargest(top_n).index
        fwd = close.iloc[i + horizon] / close.iloc[i] - 1
        rows.append({
            "date": close.index[i],
            "port": fwd[picks].mean() - fee,
            "bench": bench.iloc[i + horizon] / bench.iloc[i] - 1,
            "universe": fwd.mean(),
            "pick_win": (fwd[picks] > 0).mean(),
            "regime_up": bool(bench.iloc[i] > bench_sma.iloc[i]),
        })
    df = pd.DataFrame(rows)
    return {"periods": df, "stats": summarize(df, horizon), "horizon": horizon, "top_n": top_n,
            "fee_per_side_pct": fee_per_side_pct}


def _max_dd(rets: pd.Series) -> float:
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")


def summarize(df: pd.DataFrame, horizon: int = 20) -> dict:
    if df.empty:
        return {"n": 0}
    ex = df["port"] - df["bench"]
    half = len(df) // 2
    years = len(df) * horizon / 252
    out = {
        "n": len(df),
        "start": df["date"].iloc[0].date(), "end": df["date"].iloc[-1].date(),
        "avg_port": df["port"].mean(), "avg_bench": df["bench"].mean(),
        "avg_excess": ex.mean(), "beat_bench": (ex > 0).mean(),
        "pick_win": df["pick_win"].mean(),
        "cagr_port": (1 + df["port"]).prod() ** (1 / years) - 1 if years > 0 else np.nan,
        "cagr_bench": (1 + df["bench"]).prod() ** (1 / years) - 1 if years > 0 else np.nan,
        "maxdd_port": _max_dd(df["port"]), "maxdd_bench": _max_dd(df["bench"]),
        "excess_first_half": ex.iloc[:half].mean(), "excess_second_half": ex.iloc[half:].mean(),
    }
    up, down = df[df["regime_up"]], df[~df["regime_up"]]
    out["excess_regime_up"] = (up["port"] - up["bench"]).mean() if len(up) else np.nan
    out["excess_regime_down"] = (down["port"] - down["bench"]).mean() if len(down) else np.nan
    out["n_regime_down"] = len(down)
    return out


def stock_history(close: pd.Series, horizon: int = 20, lookback_days: int = 756) -> dict:
    """Real past behaviour of one stock (last ~3 years)."""
    c = close.dropna()
    if len(c) < 60:
        return {}
    def ret(n):
        return c.iloc[-1] / c.iloc[-n - 1] - 1 if len(c) > n else np.nan
    recent = c.iloc[-lookback_days:]
    fwd = (recent.shift(-horizon) / recent - 1).dropna()
    daily = c.pct_change().dropna()
    last_year = c.iloc[-252:]
    return {
        "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(252), "ret_3y": ret(756),
        "vol_1y": daily.iloc[-252:].std() * np.sqrt(252),
        "maxdd_1y": float((last_year / last_year.cummax() - 1).min()),
        "w4_win": (fwd > 0).mean() if len(fwd) else np.nan,
        "w4_avg": fwd.mean() if len(fwd) else np.nan,
        "w4_p10": fwd.quantile(0.10) if len(fwd) else np.nan,
        "w4_p90": fwd.quantile(0.90) if len(fwd) else np.nan,
        "w4_years": len(recent) / 252,
    }
