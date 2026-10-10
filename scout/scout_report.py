"""Stock Scout: good fundamentals + 1–3y outlook + chart timing + emerging industry groups.

Owner's goal (2026-10-09): find stocks with good fundamentals and a future 1–3 years out, pullbacks
inside an uptrend, and industries heating up. (Breakout section removed after backtest — see STAGE notes.)
The owner decides trades; this is a watchlist, not buy signals.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd

from .brief import SECTOR_SHORT, thdate
from . import events, risk
from .report import next_earnings

STAGE_TH = {"breakout": "⬆️ ทำจุดสูงใหม่", "pullback": "🔄 ย่อในขาขึ้น", "uptrend": "📈 ขาขึ้น", "none": "–"}
# Backtest 2015–2026 (research_stages.py, vs equal-weight universe, costs, next-day entry):
# breakouts UNDERPERFORMED (12m ≈ −7…−9%, t≈−2); pullbacks ≈ +0.7% (t≈0.3, noise) but ≈ +7% vs chasing
# uptrends. So no 🚀 buy section; 🔄 is shown as "not chasing", not as a proven edge.
HONESTY = ("<i>จังหวะกราฟ: ทดสอบย้อนหลัง 2015–26 แล้ว 'ไม่ชนะหุ้นเฉลี่ยอย่างมีนัย' "
           "(🔄 ย่อ ≈ เท่าหุ้นเฉลี่ย · ในอดีตดีกว่ากลุ่มที่วิ่งใกล้จุดสูงสุด แต่ยังไม่ได้ทดสอบนัยสำคัญ) · "
           "พื้นฐาน/ประมาณการ: ทดสอบย้อนหลังไม่ได้</i>")


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, d: int = 0) -> str:
    return f"{x * 100:+,.{d}f}%" if _ok(x) else "–"


def upct(x) -> str:
    return f"{x * 100:.0f}%" if _ok(x) else "–"


# ------------------------------------------------------------------ classification
def classify(fund: pd.DataFrame, est: pd.DataFrame, tickers, good_top_pct: float = 0.30) -> pd.DataFrame:
    f = fund.reindex(tickers)
    e = est.reindex(tickers)
    ttm = e["ni_ttm"].where(e["ni_ttm"].notna(), np.where(f["profitMargins"] > 0, 1.0, np.nan))
    profitable = ttm > 0
    not_cut = ~(e["eps_rev_90"] < -0.02)                       # estimates not being cut
    good = (profitable & (e["eps_g_ny"] > 0) & ((e["rev_g_ny"] > 0) | (f["revenueGrowth"] > 0)) & not_cut)
    emerging = (~profitable & (f["revenueGrowth"] >= 0.25) & (e["ni_q0"] > e["ni_q4"]) & (e["eps_ny"] > 0)
                & not_cut)                                            # estimates must not be being cut

    sector = f["sector"].fillna("Unknown")
    parts = {
        "eps_g_ny": e["eps_g_ny"].rank(pct=True),
        "rev_g_ny": e["rev_g_ny"].rank(pct=True),
        "rev_g_now": f["revenueGrowth"].rank(pct=True),
        "revisions": e["eps_rev_90"].rank(pct=True),
        "op_margin": f["operatingMargins"].groupby(sector).rank(pct=True),
        "roe": f["returnOnEquity"].groupby(sector).rank(pct=True),
    }
    pr = pd.DataFrame(parts)
    score = pr.mean(axis=1).where(pr.notna().sum(axis=1) >= 3)
    # "good" = passes the gates AND fundamental score in the top `good_top_pct` of the whole universe
    good = good & (score.rank(pct=True) >= 1 - good_top_pct)
    out = pd.DataFrame({"good": good.fillna(False), "emerging": emerging.fillna(False), "fscore": score})
    return out.join(pr.add_prefix("r_"))


HIGHLIGHT = {
    "r_eps_g_ny": lambda f, e: f"กำไรปีหน้า {pct(e['eps_g_ny'])}",
    "r_rev_g_ny": lambda f, e: f"รายได้ปีหน้า {pct(e['rev_g_ny'])}",
    "r_rev_g_now": lambda f, e: f"รายได้โต {pct(f['revenueGrowth'])}",
    "r_revisions": lambda f, e: "นักวิเคราะห์ปรับประมาณการขึ้น" if _ok(e["eps_rev_90"]) and e["eps_rev_90"] > 0.02 else None,
    "r_op_margin": lambda f, e: f"อัตรากำไร {f['operatingMargins'] * 100:.0f}%" if _ok(f["operatingMargins"]) else None,
    "r_roe": lambda f, e: f"ROE {f['returnOnEquity'] * 100:.0f}%" if _ok(f["returnOnEquity"]) else None,
}


def highlights(t, cls, fund, est, n=2) -> list[str]:
    row = cls.loc[t, [c for c in cls.columns if c.startswith("r_")]].dropna().sort_values(ascending=False)
    out = []
    for k, _ in row.items():
        txt = HIGHLIGHT[k](fund.loc[t], est.loc[t])
        if txt and txt not in out:
            out.append(txt)
        if len(out) >= n:
            break
    return out


def warn_icons(t, fund, vol, today) -> str:
    ic = ""
    ed = next_earnings(fund.loc[t], today) if t in fund.index else None
    if ed is not None and (ed - today).days <= 14:
        ic += "📅"
    if _ok(vol.get(t)) and vol[t] > 0.6:
        ic += "🎢"
    return ic


def line(t, cls, fund, est, stage, dd, vol, today) -> str:
    hs = highlights(t, cls, fund, est)
    note = {"breakout": "ทำจุดสูงใหม่ (ระวังไล่ราคา)", "pullback": f"ย่อ {abs(dd.get(t, 0)) * 100:.0f}% จากจุดสูงสุด",
            "uptrend": "ขาขึ้น ใกล้จุดสูงสุด", "none": "ยังไม่มีขาขึ้น"}.get(stage.get(t), "")
    txt = " · ".join(hs + ([note] if note else []))
    ic = warn_icons(t, fund, vol, today)
    mc = fund.loc[t].get("marketCap") if t in fund.index else None
    size = " <i>(เล็ก)</i>" if _ok(mc) and mc < 3e9 else ""
    return f"<code>{escape(t):<5}</code> {escape(txt)}{(' ' + ic) if ic else ''}{size}"


# ------------------------------------------------------------------ messages
def weekly(asof, spy, close, fund, est, cls, stage, dd, groups, universe, n=8, new=frozenset(), calendar=()):
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    vol = close.pct_change().iloc[-60:].std() * np.sqrt(252)
    up = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
    head = [f"<b>🔭 Stock Scout · สัปดาห์ {thdate(asof, True)}</b>",
            f"ตลาด (SPY): {'🟢 ขาขึ้น' if up else '🔴 ขาลง'}",
            f"<i>พื้นฐานดี {int(cls['good'].sum())} ตัว · กำลังจะกำไร {int(cls['emerging'].sum())} ตัว "
            f"จาก {len(cls)} ตัว (S&amp;P 1500 + Nasdaq-100)</i>"]

    # 🔥 groups
    ind = fund["industry"]
    g_lines = ["", "<b>🔥 กลุ่มอุตสาหกรรมที่กำลังแรง</b>"]
    for name, g in groups.head(5).iterrows():
        members = [t for t in ind[ind == name].index if t in cls.index and (cls.loc[t, "good"] or cls.loc[t, "emerging"])]
        members = sorted(members, key=lambda t: -(cls.loc[t, "fscore"] if _ok(cls.loc[t, "fscore"]) else 0))[:3]
        arrow = "⬆️" if _ok(g["rank_change"]) and g["rank_change"] >= 3 else ""
        g_lines.append(f"• <b>{escape(str(name))}</b> {arrow} — ชนะตลาด {pct(g['rel3m'])} ใน 3 เดือน · "
                       f"แข็งแรง {g['breadth'] * 100:.0f}% ของ {int(g['n'])} ตัว"
                       + (f"\n   เด่น: {', '.join(escape(m) for m in members)}" if members else ""))

    def section(title, mask, limit, always=()):
        ranked = cls[mask].sort_values("fscore", ascending=False).index.tolist()
        picks = ranked[:limit] + [t for t in ranked[limit:] if t in always]     # never hide 🆕 names
        lines = ["", f"<b>{title}</b>"]
        lines += [line(t, cls, fund, est, stage, dd, vol, today) + (" 🆕" if t in new else "")
                  + (" 🌱" if cls.loc[t, "emerging"] else "") for t in picks] or ["– ไม่มีสัปดาห์นี้"]
        return lines, picks

    s1, p1 = section("🔄 พื้นฐานดี + กำลังย่อในขาขึ้น", cls["good"] & (stage.reindex(cls.index) == "pullback"), n,
                     always=new)
    rest = cls["good"] & ~cls.index.isin(p1)
    s2, p2 = section("✅ พื้นฐาน + ประมาณการดีที่สุด (ไม่ดูจังหวะกราฟ)", rest, n)
    s3, p3 = section("🌱 กำลังจะกำไร ⚠️เสี่ยงสูง (รายได้โตแรง ขาดทุนลดลง)", cls["emerging"], 5)
    foot = list(calendar) + ["", "🆕 เพิ่งเข้าโซนย่อสัปดาห์นี้ · 📅 งบออกใน 2 สัปดาห์ · 🎢 ผันผวนสูง · (เล็ก) มูลค่าบริษัท &lt; $3B",
            HONESTY,
            "<i>รายชื่อให้ศึกษาต่อ ไม่ใช่คำแนะนำซื้อ · ข้อมูล Yahoo Finance · ประมาณการนักวิเคราะห์อาจผิด</i>"]
    msgs = [{"text": "\n".join(head + g_lines)}, {"text": "\n".join(s1 + s2 + s3 + foot).strip()}]
    tops = [p[0] for p in (p1, p2, p3) if p]
    msgs[-1]["picks"] = {"pullback": p1, "top": p2, "emerging": p3}
    return msgs, tops


def card(t, cls, fund, est, stage, dd, close, sec_pe, ev=None) -> dict:
    f, e = fund.loc[t], est.loc[t]
    p = close[t].dropna()
    tgt = f.get("targetMeanPrice")
    kind = "🌱 กำลังจะกำไร" if cls.loc[t, "emerging"] else "✅ พื้นฐานดี"
    lines = [
        f"<b>🔎 {escape(t)} — {escape(str(f.get('longName')) if _ok(f.get('longName')) else t)}</b>",
        f"{escape(str(f.get('industry'))) if _ok(f.get('industry')) else '–'} · ราคา ${p.iloc[-1]:,.2f} · {STAGE_TH.get(stage.get(t), '–')}",
        f"{kind}",
        "",
        "<b>อนาคต (ประมาณการนักวิเคราะห์)</b>",
        f"• กำไรต่อหุ้น: ปีนี้ {pct(e['eps_g_cy'])} · ปีหน้า {pct(e['eps_g_ny'])}",
        f"• รายได้: ปีนี้ {pct(e['rev_g_cy'])} · ปีหน้า {pct(e['rev_g_ny'])}",
        f"• ประมาณการปีหน้าเปลี่ยนใน 90 วัน: {pct(e['eps_rev_90'], 1)} · 30 วัน ปรับขึ้น {e['up30']:.0f} / ลง {e['down30']:.0f} คน"
        if _ok(e["up30"]) and _ok(e["down30"]) else f"• ประมาณการปีหน้าเปลี่ยนใน 90 วัน: {pct(e['eps_rev_90'], 1)}",
        "",
        "<b>ผลงานจริงล่าสุด</b>",
        f"• รายได้ไตรมาสล่าสุด {pct(f.get('revenueGrowth'))} เทียบปีก่อน",
        f"• กำไรสุทธิไตรมาสล่าสุด ${e['ni_q0'] / 1e6:,.0f}M (ปีก่อน ${e['ni_q4'] / 1e6:,.0f}M)"
        if _ok(e["ni_q0"]) and _ok(e["ni_q4"]) else "• กำไรสุทธิรายไตรมาส: –",
        f"• อัตรากำไรจากธุรกิจ {upct(f.get('operatingMargins'))} · ROE {upct(f.get('returnOnEquity'))}",
        "",
        f"<b>ราคา</b> · ห่างจุดสูงสุด 1 ปี {pct(dd.get(t))} · Fwd P/E {f.get('forwardPE'):.1f}"
        + (f" (กลุ่ม {sec_pe.get(f.get('sector')):.1f})" if _ok(sec_pe.get(f.get('sector'))) else "")
        if _ok(f.get("forwardPE")) and f.get("forwardPE") > 0
        else f"<b>ราคา</b> · ห่างจุดสูงสุด 1 ปี {pct(dd.get(t))} · Fwd P/E – (คาดว่าขาดทุน)",
    ]
    if _ok(tgt):
        lines.append(f"🎯 เป้านักวิเคราะห์เฉลี่ย ${tgt:,.0f} ({pct(tgt / p.iloc[-1] - 1)})")
    for extra in (events.card_lines(ev, p), risk.lines(p)):
        if extra:
            lines += [""] + extra
    return {"text": "\n".join(lines),
            "buttons": [[{"text": f"📈 กราฟ {t}", "url": f"https://finance.yahoo.com/quote/{t}"}]]}


def daily_alerts(asof, cls, fund, est, stage, prev_stage: dict, dd, close,
                 last_alert: dict | None = None, cooldown_days: int = 20) -> dict | None:
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    vol = close.pct_change().iloc[-60:].std() * np.sqrt(252)
    watch = cls.index[cls["good"] | cls["emerging"]]
    last_alert = last_alert or {}

    def cooled(t):
        a = last_alert.get(t)
        return not a or a.get("stage") != stage.get(t) or (asof - pd.Timestamp(a["date"])).days > cooldown_days

    new = [t for t in watch if stage.get(t) == "pullback"
           and t in prev_stage and prev_stage[t] != stage.get(t) and cooled(t)]
    if not new:
        return None
    new = sorted(new, key=lambda t: -(cls.loc[t, "fscore"] if _ok(cls.loc[t, "fscore"]) else 0))[:10]
    lines = [f"<b>🔔 Scout · {thdate(asof)} — หุ้นในรายชื่อเฝ้าดูเพิ่งย่อเข้าโซน 🔄</b>", ""]
    for kind in ("pullback",):
        ts = [t for t in new if stage.get(t) == kind]
        if ts:
            lines.append(f"<b>{STAGE_TH[kind]}</b>")
            lines += [line(t, cls, fund, est, stage, dd, vol, today)
                      + (" 🌱" if cls.loc[t, "emerging"] else "") for t in ts]
    lines += ["", "<i>รายชื่อให้ศึกษาต่อ ไม่ใช่คำแนะนำซื้อ · จังหวะย่อไม่ได้ชนะหุ้นเฉลี่ยอย่างมีนัยในการทดสอบย้อนหลัง</i>"]
    return {"text": "\n".join(lines), "tickers": new}
