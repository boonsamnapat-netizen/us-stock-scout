"""Forward track record of the scout's own weekly lists (the honest test of the parts we can't backtest).

Each weekly report appends its picks to data/scout_history.csv. Performance is measured from the
FIRST close after the report date (the owner reads it on Saturday and can act on Monday at the
earliest) to the latest close, against SPY and the equal-weighted universe over the same dates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECTION_TH = {"pullback": "🔄 ย่อในขาขึ้น", "top": "✅ พื้นฐานดีที่สุด", "emerging": "🌱 กำลังจะกำไร"}
COLUMNS = ["date", "section", "ticker"]


def record(path: str, asof: pd.Timestamp, picks: dict[str, list[str]]) -> pd.DataFrame:
    p = Path(path)
    hist = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLUMNS)
    day = str(asof.date())
    hist = hist[hist["date"] != day]                       # re-running a week replaces it
    new = pd.DataFrame([{"date": day, "section": s, "ticker": t} for s, ts in picks.items() for t in ts],
                       columns=COLUMNS)
    hist = pd.concat([hist, new], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(p, index=False)
    return hist


def performance(hist: pd.DataFrame, close: pd.DataFrame, spy: pd.Series, min_days: int = 20) -> pd.DataFrame:
    """One row per pick old enough (>= min_days trading days since entry)."""
    if hist.empty:
        return pd.DataFrame()
    px = close.ffill()
    spy = spy.reindex(close.index).ffill()
    univ = px.pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()   # equal-weight index
    rows = []
    for (day, section), g in hist.groupby(["date", "section"]):
        after = close.index[close.index > pd.Timestamp(day)]
        if len(after) < min_days + 1:
            continue
        entry, last = after[0], close.index[-1]
        for t in g["ticker"]:
            if t not in px.columns or pd.isna(px.at[entry, t]):
                continue
            rows.append({"date": day, "section": section, "ticker": t, "days": len(after) - 1,
                         "ret": px.at[last, t] / px.at[entry, t] - 1,
                         "spy": spy[last] / spy[entry] - 1,
                         "univ": univ[last] / univ[entry] - 1})
    return pd.DataFrame(rows)


def report_text(hist: pd.DataFrame, close: pd.DataFrame, spy: pd.Series) -> str:
    started = hist["date"].min() if not hist.empty else None
    perf = performance(hist, close, spy)
    lines = ["<b>📒 ผลงานจริงของ Scout</b> <i>(นับจากราคาปิดวันทำการแรกหลังรายงาน · ไม่หักค่าธรรมเนียม)</i>"]
    if perf.empty:
        n_weeks = hist["date"].nunique() if not hist.empty else 0
        lines.append(f"เริ่มเก็บ {started or 'สัปดาห์นี้'} · บันทึกแล้ว {n_weeks} สัปดาห์ — "
                     "จะเริ่มแสดงผลเมื่อรายชื่อมีอายุครบ 1 เดือน")
        return "\n".join(lines)
    for section, g in perf.groupby("section"):
        ex_spy = (g["ret"] - g["spy"])
        ex_u = (g["ret"] - g["univ"])
        lines.append(f"• {SECTION_TH.get(section, section)}: {len(g)} รายการ จาก {g['date'].nunique()} สัปดาห์ · "
                     f"เฉลี่ย {g['ret'].mean() * 100:+.1f}% · เทียบ SPY {ex_spy.mean() * 100:+.1f}% · "
                     f"เทียบหุ้นเฉลี่ย {ex_u.mean() * 100:+.1f}% · ชนะ SPY {(ex_spy > 0).mean() * 100:.0f}%")
    weeks = perf["date"].nunique()
    if weeks < 12:
        lines.append(f"<i>ข้อมูลยังน้อย ({weeks} สัปดาห์) — อย่าเพิ่งสรุป ตัวเลขช่วงแรกแกว่งมาก</i>")
    return "\n".join(lines)
