"""Portfolio-rule simulator: hold N stocks, buy/sell on rank rules, monthly trade cap, regime gate.

Rank = price-only momentum score (same as scoring.short_score, computed for every day), so it can
be backtested honestly. Known bias: today's index members (survivorship) -> optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Rules:
    name: str = "base"
    n_hold: int = 5
    entry_pct: float = 0.10      # buy only from top 10% by score
    exit_pct: float = 0.30       # sell when no longer in top 30% (or below 200-day SMA)
    gate: str = "cash"           # "none" | "nobuy" (no new buys when SPY < SMA200) | "cash" (also sell all)
    check: str = "weekly"        # "daily" | "weekly" (Fridays) | "monthly" (first day of month)
    max_trades_month: int = 5    # Dime free package
    fee_per_side_pct: float = 0.10
    replace_all: bool = False    # True = classic "rebalance to top N" each check (ignores bands)
    gate_sma: int = 200          # regime: SPY above its N-day SMA
    stop_loss_pct: float = 0.0   # sell a holding that falls this much below its entry (0 = off)
    max_vol: float = 0.0         # skip buys with 60-day annualised vol above this (0 = off)
    invest_frac: float = 1.0     # fraction of equity in stocks; rest stays cash


def score_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Momentum score for every day: mean pct-rank of 6-1m, 12-1m momentum and 52w-high proximity,
    NaN when price is below its 200-day SMA. Uses data up to each day only."""
    hi = close.rolling(252, min_periods=200).max()
    parts = [
        (close.shift(21) / close.shift(126) - 1).rank(axis=1, pct=True),
        (close.shift(21) / close.shift(252) - 1).rank(axis=1, pct=True),
        (close / hi - 1).rank(axis=1, pct=True),
    ]
    score = sum(parts) / 3
    sma200 = close.rolling(200, min_periods=200).mean()
    return score.where(close > sma200)


def simulate(close: pd.DataFrame, bench: pd.Series, rules: Rules, start: int = 273,
             score: pd.DataFrame | None = None) -> dict:
    score = score_panel(close) if score is None else score
    pct_rank = score.rank(axis=1, ascending=False, pct=True)   # 0 = best
    bench = bench.reindex(close.index).ffill()
    regime_up = bench > bench.rolling(rules.gate_sma, min_periods=rules.gate_sma).mean()
    vol60 = close.pct_change().rolling(60).std() * np.sqrt(252)
    fee = rules.fee_per_side_pct / 100
    dates = close.index
    px = close.to_numpy()

    cash, units, entry = 1.0, {}, {}   # ticker col index -> units / entry price
    equity, trades_log = [], []
    month, trades_this_month = None, 0

    def is_check(i):
        d = dates[i]
        if rules.check == "daily":
            return True
        if rules.check == "weekly":
            return i + 1 >= len(dates) or dates[i + 1].isocalendar().week != d.isocalendar().week
        return i == start or dates[i - 1].month != d.month

    cols = {c: j for j, c in enumerate(close.columns)}
    for i in range(start, len(dates)):
        d = dates[i]
        if d.month != month:
            month, trades_this_month = d.month, 0
        value = cash + sum(u * px[i, j] for j, u in units.items() if not np.isnan(px[i, j]))
        if is_check(i):
            rk = pct_rank.iloc[i]
            up = bool(regime_up.iloc[i])

            def sell(j):
                nonlocal cash, trades_this_month
                p = px[i, j]
                if np.isnan(p):
                    return
                cash += units.pop(j) * p * (1 - fee)
                entry.pop(j, None)
                trades_this_month += 1
                trades_log.append((d, close.columns[j], "sell"))

            # 1) exits
            held = list(units)
            if rules.gate == "cash" and not up:
                to_sell = held
            elif rules.replace_all:
                top = set(cols[t] for t in rk.nsmallest(rules.n_hold).dropna().index)
                to_sell = [j for j in held if j not in top]
            else:
                to_sell = [j for j in held
                           if not (rk.iloc[j] <= rules.exit_pct)   # NaN (below SMA200) -> sell
                           or (rules.stop_loss_pct and px[i, j] < entry[j] * (1 - rules.stop_loss_pct))]
                to_sell.sort(key=lambda j: -(rk.iloc[j] if not np.isnan(rk.iloc[j]) else 9))
            for j in to_sell:
                if trades_this_month >= rules.max_trades_month and not (rules.gate == "cash" and not up):
                    break
                sell(j)

            # 2) entries
            can_buy = up or rules.gate == "none"
            if can_buy and len(units) < rules.n_hold:
                cands = rk[rk <= rules.entry_pct].sort_values().index
                value = cash + sum(u * px[i, j] for j, u in units.items() if not np.isnan(px[i, j]))
                for t in cands:
                    if len(units) >= rules.n_hold or trades_this_month >= rules.max_trades_month:
                        break
                    j = cols[t]
                    if j in units or np.isnan(px[i, j]):
                        continue
                    if rules.max_vol and vol60.iat[i, j] > rules.max_vol:
                        continue
                    alloc = min(cash, value * rules.invest_frac / rules.n_hold)
                    if alloc <= 0:
                        break
                    units[j] = alloc * (1 - fee) / px[i, j]
                    entry[j] = px[i, j]
                    cash -= alloc
                    trades_this_month += 1
                    trades_log.append((d, t, "buy"))
            value = cash + sum(u * px[i, j] for j, u in units.items() if not np.isnan(px[i, j]))
        equity.append(value)

    eq = pd.Series(equity, index=dates[start:])
    return {"equity": eq, "trades": pd.DataFrame(trades_log, columns=["date", "ticker", "side"]),
            "stats": stats(eq, bench.iloc[start:], len(trades_log))}


def _cagr(eq: pd.Series) -> float:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan


def _mdd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def stats(eq: pd.Series, bench: pd.Series, n_trades: int) -> dict:
    b = bench / bench.iloc[0]
    half = len(eq) // 2
    months = max(1, (eq.index[-1] - eq.index[0]).days / 30.44)
    return {
        "start": eq.index[0].date(), "end": eq.index[-1].date(),
        "cagr": _cagr(eq), "maxdd": _mdd(eq),
        "cagr_spy": _cagr(b), "maxdd_spy": _mdd(b),
        "cagr_1st_half": _cagr(eq.iloc[:half]), "cagr_2nd_half": _cagr(eq.iloc[half:]),
        "spy_1st_half": _cagr(b.iloc[:half]), "spy_2nd_half": _cagr(b.iloc[half:]),
        "trades_per_month": n_trades / months,
        "worst_year": eq.resample("YE").last().pct_change().min(),
    }
