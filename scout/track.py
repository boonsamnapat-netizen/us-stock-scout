"""Forward track record of the scout's own weekly lists (the honest test of the parts we can't backtest).

Each AUTOMATIC weekly report appends its picks to data/scout_history.csv (one cohort = one week x
section). Performance per cohort at FIXED horizons (4 / 13 / 26 weeks) from the first close AFTER the
report date (the owner reads it Saturday, can act Monday at the earliest), equal-weighted inside the
cohort, vs SPY and vs the equal-weighted universe over the same window. Cohorts are the unit of N.
Picks without prices (delisted / dropped by Yahoo) are counted and shown — never silently dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SECTION_TH = {"pullback": "🔄 ย่อในขาขึ้น", "top": "✅ พื้นฐานดีที่สุด", "emerging": "🌱 กำลังจะกำไร"}
COLUMNS = ["date", "section", "ticker"]
HORIZONS = {"4 สัปดาห์": 20, "13 สัปดาห์": 63, "26 สัปดาห์": 126}


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, dtype=str, keep_default_na=False)      # "NA" is a valid ticker


def record(path: str, asof: pd.Timestamp, picks: dict[str, list[str]]) -> pd.DataFrame:
    hist = load(path)
    day = str(asof.date())
    hist = hist[hist["date"] != day]                       # re-running the same week replaces it
    new = pd.DataFrame([{"date": day, "section": s, "ticker": t} for s, ts in picks.items() for t in ts],
                       columns=COLUMNS)
    hist = pd.concat([hist, new], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(path, index=False)
    return hist


def cohorts(hist: pd.DataFrame, close: pd.DataFrame, spy: pd.Series) -> tuple[pd.DataFrame, int]:
    """Rows = (cohort, horizon) with equal-weight cohort return and benchmarks. Also #picks missing."""
    if hist.empty:
        return pd.DataFrame(), 0
    px = close.ffill()
    spy = spy.reindex(close.index).ffill()
    univ = px.pct_change(fill_method=None).mean(axis=1).fillna(0).add(1).cumprod()
    rows, missing = [], 0
    for (day, section), g in hist.groupby(["date", "section"]):
        after = close.index[close.index > pd.Timestamp(day)]
        if len(after) == 0:
            continue
        entry_i = close.index.get_loc(after[0])
        tick = [t for t in g["ticker"] if t in px.columns and not pd.isna(px.iloc[entry_i][t])]
        missing += len(g) - len(tick)
        for label, h in HORIZONS.items():
            if entry_i + h >= len(close.index) or not tick:
                continue
            a, b = close.index[entry_i], close.index[entry_i + h]
            r = (px.loc[b, tick] / px.loc[a, tick] - 1).mean()
            rows.append({"date": day, "section": section, "horizon": label, "n": len(tick), "ret": r,
                         "spy": spy[b] / spy[a] - 1, "univ": univ[b] / univ[a] - 1})
    return pd.DataFrame(rows), missing


def report_text(hist: pd.DataFrame, close: pd.DataFrame, spy: pd.Series) -> str:
    lines = ["<b>📒 ผลงานจริงของ Scout</b> <i>(ซื้อเท่ากันทุกตัวที่ราคาปิดวันทำการแรกหลังรายงาน · ถือคงที่ · "
             "ไม่หักค่าธรรมเนียม · 1 สัปดาห์ = 1 ตัวอย่าง)</i>"]
    if hist.empty:
        lines.append("เริ่มเก็บจากรายงานอัตโนมัติฉบับถัดไป")
        return "\n".join(lines)
    co, missing = cohorts(hist, close, spy)
    weeks = hist["date"].nunique()
    if co.empty:
        lines.append(f"เริ่มเก็บ {hist['date'].min()} · บันทึกแล้ว {weeks} สัปดาห์ — "
                     "จะเริ่มแสดงผลเมื่อรายชื่อมีอายุครบ 4 สัปดาห์")
        return "\n".join(lines)
    for label in HORIZONS:
        sub = co[co["horizon"] == label]
        if sub.empty:
            continue
        lines.append(f"<b>ถือ {label}</b>")
        for section, g in sub.groupby("section"):
            ex_s, ex_u = g["ret"] - g["spy"], g["ret"] - g["univ"]
            lines.append(f"• {SECTION_TH.get(section, section)}: {len(g)} สัปดาห์ · เฉลี่ย {g['ret'].mean() * 100:+.1f}% · "
                         f"เทียบ SPY {ex_s.mean() * 100:+.1f}% · เทียบหุ้นเฉลี่ย {ex_u.mean() * 100:+.1f}% · "
                         f"ชนะ SPY {(ex_s > 0).mean() * 100:.0f}% ของสัปดาห์")
    if missing:
        lines.append(f"⚠️ {missing} รายการไม่มีราคา ณ วันซื้อ (เช่น หลุดตลาด/ข้อมูลขาด) — ไม่ได้นับรวม")
    n4 = (co["horizon"] == "4 สัปดาห์").groupby(co["date"]).any().sum()
    if n4 < 12:
        lines.append(f"<i>ข้อมูลยังน้อย ({n4} สัปดาห์) — อย่าเพิ่งสรุป ช่วงแรกตัวเลขแกว่งมาก</i>")
    return "\n".join(lines)
