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
    max_per_sector: int = 0      # max holdings per GICS sector (0 = no limit)


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


def new_state(cash: float = 1.0) -> dict:
    return {"cash": cash, "units": {}, "entry": {}, "month": None, "trades_this_month": 0}


def value_of(state: dict, price: pd.Series) -> float:
    return state["cash"] + sum(u * price.get(t, np.nan) for t, u in state["units"].items()
                               if not np.isnan(price.get(t, np.nan)))


def decide_day(state: dict, date: pd.Timestamp, rank: pd.Series, price: pd.Series, vol: pd.Series,
               up: bool, rules: Rules, sectors: pd.Series | None = None) -> list[dict]:
    """Apply the rules for one check day. Mutates `state`; returns the trades made.
    Shared by the backtest and the live model portfolio so both follow identical rules.
    rank: 0 = best (pct), NaN = below 200-day SMA / no score."""
    fee = rules.fee_per_side_pct / 100
    ym = f"{date.year}-{date.month:02d}"
    if state["month"] != ym:
        state["month"], state["trades_this_month"] = ym, 0
    trades = []
    units, entry = state["units"], state["entry"]

    def sell(t, why):
        p = price.get(t, np.nan)
        if np.isnan(p):
            return
        state["cash"] += units.pop(t) * p * (1 - fee)
        entry.pop(t, None)
        state["trades_this_month"] += 1
        trades.append({"date": date, "ticker": t, "side": "sell", "price": p, "why": why})

    # 1) exits
    forced = rules.gate == "cash" and not up
    to_sell = []
    for t in list(units):
        rk = rank.get(t, np.nan)
        if forced:
            to_sell.append((t, "ตลาดขาลง ถือเงินสด"))
        elif rules.replace_all:
            if t not in set(rank.nsmallest(rules.n_hold).dropna().index):
                to_sell.append((t, "หลุด Top"))
        elif np.isnan(rk):
            to_sell.append((t, "ราคาหลุดเส้น 200 วัน"))
        elif rk > rules.exit_pct:
            to_sell.append((t, f"หลุดกลุ่ม {rules.exit_pct:.0%} บน"))
        elif rules.stop_loss_pct and price.get(t, np.inf) < entry[t] * (1 - rules.stop_loss_pct):
            to_sell.append((t, "stop-loss"))
    to_sell.sort(key=lambda x: -(rank.get(x[0], np.nan) if not np.isnan(rank.get(x[0], np.nan)) else 9))
    for t, why in to_sell:
        if state["trades_this_month"] >= rules.max_trades_month and not forced:
            break
        sell(t, why)

    # 2) entries
    if (up or rules.gate == "none") and len(units) < rules.n_hold:
        value = value_of(state, price)
        for t in rank[rank <= rules.entry_pct].sort_values().index:
            if len(units) >= rules.n_hold or state["trades_this_month"] >= rules.max_trades_month:
                break
            p = price.get(t, np.nan)
            if t in units or np.isnan(p):
                continue
            if rules.max_vol and vol.get(t, np.nan) > rules.max_vol:
                continue
            if rules.max_per_sector and sectors is not None:
                sec = sectors.get(t, "Unknown")
                if sum(sectors.get(h, "Unknown") == sec for h in units) >= rules.max_per_sector:
                    continue
            alloc = min(state["cash"], value * rules.invest_frac / rules.n_hold)
            if alloc <= 0:
                break
            units[t] = alloc * (1 - fee) / p
            entry[t] = p
            state["cash"] -= alloc
            state["trades_this_month"] += 1
            trades.append({"date": date, "ticker": t, "side": "buy", "price": p,
                           "why": f"อันดับ {int((rank < rank[t]).sum()) + 1}", "weight": alloc / value})
    return trades


def signals(close: pd.DataFrame, bench: pd.Series, rules: Rules):
    """Daily panels used by decide_day: pct rank (0 best), 60-day vol, regime flag."""
    rank = score_panel(close).rank(axis=1, ascending=False, pct=True)
    vol = close.pct_change().rolling(60).std() * np.sqrt(252)
    bench = bench.reindex(close.index).ffill()
    up = bench > bench.rolling(rules.gate_sma, min_periods=rules.gate_sma).mean()
    return rank, vol, up


def simulate(close: pd.DataFrame, bench: pd.Series, rules: Rules, start: int = 273,
             score: pd.DataFrame | None = None, sectors: pd.Series | None = None) -> dict:
    if score is None:
        rank, vol, up = signals(close, bench, rules)
    else:
        rank = score.rank(axis=1, ascending=False, pct=True)
        vol = close.pct_change().rolling(60).std() * np.sqrt(252)
        b = bench.reindex(close.index).ffill()
        up = b > b.rolling(rules.gate_sma, min_periods=rules.gate_sma).mean()
    bench = bench.reindex(close.index).ffill()
    dates = close.index

    def is_check(i):
        d = dates[i]
        if rules.check == "daily":
            return True
        if rules.check == "weekly":
            return i + 1 >= len(dates) or dates[i + 1].isocalendar().week != d.isocalendar().week
        return i == start or dates[i - 1].month != d.month

    state, equity, log = new_state(), [], []
    for i in range(start, len(dates)):
        price = close.iloc[i]
        if is_check(i):
            log += decide_day(state, dates[i], rank.iloc[i], price, vol.iloc[i], bool(up.iloc[i]), rules, sectors)
        equity.append(value_of(state, price))

    eq = pd.Series(equity, index=dates[start:])
    trades = pd.DataFrame(log, columns=["date", "ticker", "side", "price", "why", "weight"])
    return {"equity": eq, "trades": trades, "state": state,
            "stats": stats(eq, bench.iloc[start:], len(trades))}


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
