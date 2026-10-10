"""Price-history risk for a card: how deep it has fallen before and how much it moves per day.

Historical facts from the last ~3 years of daily closes (what we fetch) — not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stats(p: pd.Series) -> dict | None:
    p = p.dropna()
    if len(p) < 120:
        return None
    r = p.pct_change().dropna().iloc[-252:]
    dd = p / p.cummax() - 1
    trough = dd.idxmin()
    return {
        "max_dd": float(dd.min()),                              # worst fall from a previous high
        "max_dd_date": trough,
        "day_move": float(r.abs().median()),                    # a typical day (median absolute move)
        "bad_day": float(r.quantile(0.05)),                     # 1 day in 20 is this bad or worse
        "vol": float(r.std() * np.sqrt(252)),
    }


def lines(p: pd.Series, money: float = 10_000) -> list[str]:
    s = stats(p)
    if s is None:
        return []
    start = p.dropna().index[0]
    out = ["<b>ความเสี่ยง (จากราคาในอดีต ไม่ใช่การคาดการณ์)</b>"]
    if s["max_dd"] < -0.005:
        out.append(f"• ตั้งแต่ {start:%m/%Y} เคยร่วงหนักสุด {s['max_dd'] * 100:.0f}% จากจุดสูงสุด "
                   f"(ต่ำสุด {s['max_dd_date']:%m/%Y})")
    day = f"• 1 ปีล่าสุด ปกติขยับวันละ ±{s['day_move'] * 100:.1f}%"
    if s["bad_day"] < 0:
        day += f" · วันแย่ 1 ใน 20 วัน {s['bad_day'] * 100:.1f}% หรือแย่กว่า"
    out.append(day)
    parts = []
    if s["bad_day"] < 0:
        parts.append(f"วันแย่ ๆ ≈ {s['bad_day'] * money:,.0f} บาท")
    if s["max_dd"] < -0.005:
        parts.append(f"ถ้าร่วงเท่าครั้งหนักสุด ≈ {s['max_dd'] * money:,.0f} บาท")
    if parts:
        out.append(f"• ถ้าถือ {money:,.0f} บาท: " + " · ".join(parts))
    return out
