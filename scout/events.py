"""Per-ticker events from yfinance: earnings (dates, EPS vs estimate, price reaction) and insider trades.

Fetched only for a handful of tickers (weekly cards, /check, earnings recaps) — not the whole universe.
Field names verified with probe_estimates.py in Actions (yfinance 1.7, 2026-10-10):
  earnings_dates: index = tz-aware datetime, columns 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
  insider_transactions: columns 'Shares', 'Value', 'Text', 'Insider', 'Position', 'Start Date', ...
    Text like 'Sale at price 772.14 per share.' / 'Purchase at price ...' / 'Stock Award(Grant) ...'
"""
from __future__ import annotations

import math
import time
from html import escape

import numpy as np
import pandas as pd

from .brief import thdate


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ------------------------------------------------------------------ parsing (offline-testable)
def parse_earnings(ed: pd.DataFrame | None, today: pd.Timestamp) -> dict:
    """next date + reported quarters (newest first) with EPS actual/estimate."""
    out = {"next": None, "reported": []}
    if ed is None or len(ed) == 0 or "Reported EPS" not in ed.columns:
        return out
    idx = pd.DatetimeIndex(ed.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df = ed.copy()
    df.index = idx
    df = df.sort_index()
    fut = df[(df.index.normalize() >= today) & df["Reported EPS"].isna()]
    if len(fut):
        out["next"] = fut.index[0]
    rep = df[df["Reported EPS"].notna() & (df.index.normalize() <= today)]
    for d, r in rep.iloc[::-1].iterrows():
        out["reported"].append({"date": d, "est": float(r["EPS Estimate"]) if _ok(r["EPS Estimate"]) else np.nan,
                                "act": float(r["Reported EPS"])})
    return out


def reaction(close: pd.Series, when: pd.Timestamp) -> float:
    """Price move caused by a report. After the close (>= 16:00 ET): report-day close -> next close.
    Before the open (< 10:00 ET): previous close -> report-day close. Unknown time (e.g. 00:00):
    previous close -> next close (2 days). NaN if the needed closes don't exist yet."""
    p = close.dropna()
    day = when.normalize()
    hour = when.hour + when.minute / 60
    before = p[p.index < day]
    on = p[p.index == day]
    after = p[p.index > day]
    if hour >= 16:
        a, b = on, after
    elif 0 < hour < 10:
        a, b = before, on
    else:
        a, b = before, after
    if a.empty or b.empty:
        return np.nan
    return float(b.iloc[0] / a.iloc[-1] - 1)


def parse_insiders(tx: pd.DataFrame | None, today: pd.Timestamp, days: int = 180) -> dict | None:
    """Open-market buys vs sales in the last `days`. Awards, gifts and option exercises are ignored."""
    if tx is None or len(tx) == 0 or "Text" not in tx.columns:
        return None
    df = tx.copy()
    df["date"] = pd.to_datetime(df.get("Start Date"), errors="coerce")
    if df["date"].isna().all():
        return None                                          # can't tell what happened when -> say nothing
    df = df[df["date"] >= today - pd.Timedelta(days=days)]
    text = df["Text"].fillna("").astype(str).str.strip()
    val = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0)
    buy, sell = text.str.startswith("Purchase"), text.str.startswith("Sale")
    who = df["Insider"] if "Insider" in df.columns else pd.Series(index=df.index, dtype=object)
    return {"buy_n": int(buy.sum()), "buy_people": max(int(who[buy].nunique()), int(buy.any())),
            "buy_val": float(val[buy].sum()), "sell_n": int(sell.sum()),
            "sell_people": max(int(who[sell].nunique()), int(sell.any())), "sell_val": float(val[sell].sum()),
            "days": days}


# ------------------------------------------------------------------ fetching
def fetch(tickers: list[str], today: pd.Timestamp, pause: float = 0.2, insiders: bool = True) -> dict[str, dict]:
    import yfinance as yf

    out = {}
    for t in tickers:
        tk = yf.Ticker(t)
        ev = {}
        try:
            ev["earn"] = parse_earnings(tk.earnings_dates, today)
        except Exception as e:
            print(f"[events] {t} earnings: {type(e).__name__}")
            ev["earn"] = {"next": None, "reported": []}
        try:
            ev["insider"] = parse_insiders(tk.insider_transactions, today) if insiders else None
        except Exception as e:
            print(f"[events] {t} insiders: {type(e).__name__}")
            ev["insider"] = None
        out[t] = ev
        time.sleep(pause)
    return out


# ------------------------------------------------------------------ text
def _money(x: float) -> str:
    return f"${x / 1e9:,.1f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def beat_text(q: dict) -> str:
    if not _ok(q["est"]):
        return f"EPS จริง {q['act']:.2f}"
    act, est = round(q["act"], 2), round(q["est"], 2)            # compare what is shown
    tag = "ชนะคาด" if act > est else "ตรงคาด" if act == est else "แพ้คาด"
    return f"EPS จริง {q['act']:.2f} vs คาด {q['est']:.2f} ({tag})"


def card_lines(ev: dict | None, close: pd.Series) -> list[str]:
    if not ev:
        return []
    lines = ["<b>งบ &amp; คนในบริษัท</b>"]
    e = ev.get("earn") or {}
    if e.get("next") is not None:
        lines.append(f"• 📅 งบครั้งถัดไป {thdate(e['next'], True)} <i>(ตาม Yahoo อาจเลื่อน)</i>")
    rep = e.get("reported", [])[:4]
    rep_est = [q for q in rep if _ok(q["est"])]
    if rep:
        q = rep[0]
        r = reaction(close, q["date"])
        lines.append(f"• งบล่าสุด {thdate(q['date'])}: {beat_text(q)}"
                     + (f" · หุ้น {r * 100:+.0f}% หลังงบ" if _ok(r) else ""))
    if len(rep_est) >= 2:
        wins = sum(round(q["act"], 2) > round(q["est"], 2) for q in rep_est)
        lines.append(f"• ชนะคาด {wins} จาก {len(rep_est)} งบล่าสุด")
    ins = ev.get("insider")
    if ins is not None:
        if ins["buy_n"]:
            lines.append(f"• 🟢 คนในซื้อหุ้นด้วยเงินตัวเอง {ins['buy_people']} คน ({_money(ins['buy_val'])}) ใน 6 เดือน")
        else:
            lines.append("• ไม่พบคนในซื้อหุ้นด้วยเงินตัวเองใน 6 เดือน (ตามข้อมูล Yahoo)")
        if ins["sell_n"]:
            lines.append(f"• คนในขาย {ins['sell_people']} คน ({_money(ins['sell_val'])}) "
                         "<i>— การขายมักเป็นเรื่องภาษี/ส่วนตัว การซื้อมีความหมายกว่า</i>")
    return lines if len(lines) > 1 else []


def recap(asof: pd.Timestamp, items: list[tuple[str, dict, float, pd.Series]]) -> dict | None:
    """items: (ticker, reported quarter, price reaction, estimates row)."""
    if not items:
        return None
    lines = [f"<b>📊 สรุปงบหุ้นในรายชื่อเฝ้าดู · {thdate(asof)}</b>", ""]
    for t, q, r, e in items:
        lines.append(f"<b>{escape(t)}</b> ({thdate(q['date'])}) — {beat_text(q)}"
                     + (f" · หุ้น {r * 100:+.0f}% หลังงบ" if _ok(r) else ""))
        if e is not None and _ok(e.get("eps_rev_30")):
            ud = (f" · ขึ้น {e['up30']:.0f} / ลง {e['down30']:.0f} คน"
                  if _ok(e.get("up30")) and _ok(e.get("down30")) else "")
            lines.append(f"   ประมาณการกำไรปีหน้าเปลี่ยนใน 30 วัน {e['eps_rev_30'] * 100:+.1f}%{ud}")
    lines += ["", "<i>EPS ตามนิยามนักวิเคราะห์ (มักไม่ใช่ GAAP) · 'หลังงบ' = ราคาปิดก่อนงบ → ปิดแรกหลังงบ · "
              "นักวิเคราะห์มักปรับประมาณการภายในไม่กี่วันหลังงบ ดูต่อได้ด้วย /check</i>"]
    return {"text": "\n".join(lines)}


def calendar_lines(watch: list[str], fund: pd.DataFrame, today: pd.Timestamp, days: int = 7,
                   limit: int = 15) -> list[str]:
    """Watchlist names reporting within `days` (dates from yfinance .info)."""
    from .report import next_earnings

    rows = []
    for t in watch:
        if t in fund.index:
            d = next_earnings(fund.loc[t], today)
            if d is not None and (d - today).days <= days:
                est = fund.loc[t].get("isEarningsDateEstimate")
                rows.append((d, t + ("*" if _ok(est) and est else "")))
    if not rows:
        return []
    rows.sort()
    by_day: dict = {}
    for d, t in rows[:limit]:
        by_day.setdefault(d, []).append(t)
    out = ["", f"<b>📅 งบออกใน {days} วัน (รายชื่อเฝ้าดู)</b>"]
    out += [f"• {thdate(d)}: {', '.join(escape(t) for t in ts)}" for d, ts in by_day.items()]
    if len(rows) > limit:
        out.append(f"<i>…และอีก {len(rows) - limit} ตัว</i>")
    out.append("<i>วันที่ตาม Yahoo อาจเลื่อน · * = วันที่ยังเป็นการคาด</i>")
    return out
