"""Analyst estimates + recent quarterly results from yfinance (today's values only — not point-in-time).

Per ticker we keep a flat dict of numbers used by the scout:
  eps_g_cy / eps_g_ny   expected EPS growth this / next fiscal year (analyst consensus, usually non-GAAP)
  rev_g_cy / rev_g_ny   expected revenue growth this / next fiscal year
  eps_rev_30 / eps_rev_90   % change of next-year EPS estimate vs 30 / 90 days ago (estimate revisions)
  up30 / down30         analysts revising next-year EPS up / down in the last 30 days
  n_analysts            analysts covering next-year EPS
  eps_ny                next-year EPS estimate (>0 = analysts expect a profit)
  ni_q0 / ni_q4         net income latest quarter / same quarter a year earlier (GAAP)
  ni_ttm                net income, last 4 quarters (GAAP)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

NI_ROWS = ["Net Income From Continuing Operation Net Minority Interest", "Net Income",
           "Net Income Common Stockholders"]
FIELDS = ["eps_g_cy", "eps_g_ny", "rev_g_cy", "rev_g_ny", "eps_rev_30", "eps_rev_90", "up30", "down30",
          "n_analysts", "eps_ny", "ni_q0", "ni_q4", "ni_ttm"]


def _get(df: pd.DataFrame | None, row: str, col: str):
    try:
        v = df.loc[row, col]
        return float(v) if v is not None and not pd.isna(v) else np.nan
    except Exception:
        return np.nan


def parse(earn: pd.DataFrame | None, rev: pd.DataFrame | None, trend: pd.DataFrame | None,
          revisions: pd.DataFrame | None, qis: pd.DataFrame | None) -> dict:
    out = {
        "eps_g_cy": _get(earn, "0y", "growth"), "eps_g_ny": _get(earn, "+1y", "growth"),
        "rev_g_cy": _get(rev, "0y", "growth"), "rev_g_ny": _get(rev, "+1y", "growth"),
        "n_analysts": _get(earn, "+1y", "numberOfAnalysts"), "eps_ny": _get(earn, "+1y", "avg"),
        "up30": _get(revisions, "+1y", "upLast30days"), "down30": _get(revisions, "+1y", "downLast30days"),
    }
    cur, d30, d90 = (_get(trend, "+1y", c) for c in ("current", "30daysAgo", "90daysAgo"))
    out["eps_rev_30"] = (cur / d30 - 1) if d30 and d30 > 0 and not np.isnan(cur) else np.nan
    out["eps_rev_90"] = (cur / d90 - 1) if d90 and d90 > 0 and not np.isnan(cur) else np.nan
    ni = pd.Series(dtype=float)
    if qis is not None and not qis.empty:
        for r in NI_ROWS:
            if r in qis.index:
                ni = pd.to_numeric(qis.loc[r], errors="coerce")
                ni.index = pd.to_datetime(ni.index)
                ni = ni.sort_index().dropna()
                break
    out["ni_q0"] = float(ni.iloc[-1]) if len(ni) else np.nan
    out["ni_q4"] = float(ni.iloc[-5]) if len(ni) >= 5 else np.nan
    out["ni_ttm"] = float(ni.iloc[-4:].sum()) if len(ni) >= 4 else np.nan
    return out


def fetch(tickers: list[str], pause: float = 0.1) -> pd.DataFrame:
    import yfinance as yf

    rows = {}
    for t in tickers:
        tk = yf.Ticker(t)
        parts = {}
        for name in ("earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions", "quarterly_income_stmt"):
            try:
                parts[name] = getattr(tk, name)
            except Exception:
                parts[name] = None
        rows[t] = parse(parts["earnings_estimate"], parts["revenue_estimate"], parts["eps_trend"],
                        parts["eps_revisions"], parts["quarterly_income_stmt"])
        time.sleep(pause)
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FIELDS)
    print(f"[estimates] next-year EPS growth for {df['eps_g_ny'].notna().sum()}/{len(df)} · "
          f"quarterly NI for {df['ni_q0'].notna().sum()}/{len(df)}")
    return df


def demo(tickers, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    q0 = rng.normal(2e8, 4e8, n)
    return pd.DataFrame({
        "eps_g_cy": rng.normal(0.12, 0.2, n), "eps_g_ny": rng.normal(0.12, 0.15, n),
        "rev_g_cy": rng.normal(0.08, 0.12, n), "rev_g_ny": rng.normal(0.08, 0.1, n),
        "eps_rev_30": rng.normal(0.0, 0.03, n), "eps_rev_90": rng.normal(0.01, 0.06, n),
        "up30": rng.integers(0, 20, n), "down30": rng.integers(0, 10, n), "n_analysts": rng.integers(3, 40, n),
        "eps_ny": rng.normal(3, 2, n), "ni_q0": q0, "ni_q4": q0 - rng.normal(5e7, 1e8, n), "ni_ttm": q0 * 4,
    }, index=list(tickers))
