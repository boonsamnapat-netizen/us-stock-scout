"""Short, scannable Telegram report (HTML parse mode).

Layout follows patterns seen in Seeking Alpha / Zacks / TipRanks / Infoquest / Liberator:
one verdict per stock, short lists, one-line reason, letter grades instead of raw numbers,
market line on top, details split into separate cards.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from html import escape

import pandas as pd

from .backtest import stock_history
from .report import next_earnings
from .scoring import factor_grades, part_ranks, price_features, sector_medians, snapshot  # noqa: F401

TH_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
SECTOR_SHORT = {
    "Information Technology": "เทคโน", "Communication Services": "สื่อสาร",
    "Consumer Discretionary": "ฟุ่มเฟือย", "Consumer Staples": "ของจำเป็น",
    "Health Care": "สุขภาพ", "Financials": "การเงิน", "Industrials": "อุตสาหกรรม",
    "Energy": "พลังงาน", "Materials": "วัสดุ", "Real Estate": "อสังหาฯ",
    "Utilities": "สาธารณูปโภค", "Unknown": "–",
}
HORIZONS = [("short", "⚡ ระยะสั้น · ~4 สัปดาห์"), ("mid", "📈 ระยะกลาง · 6–18 เดือน"),
            ("long", "🏛 ระยะยาว · 1–5 ปี")]
GRADE_TH = [("growth", "เติบโต"), ("quality", "คุณภาพ"), ("value", "ราคา"),
            ("momentum", "โมเมนตัม"), ("stability", "ความนิ่ง")]


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, d: int = 0) -> str:
    return f"{x * 100:+,.{d}f}%" if _ok(x) else "–"


def thdate(d: pd.Timestamp, year: bool = False) -> str:
    return f"{d.day} {TH_MONTHS[d.month - 1]}" + (f" {d.year}" if year else "")


def grade(p) -> str:
    if not _ok(p):
        return "–"
    return "A" if p >= 0.8 else "B" if p >= 0.6 else "C" if p >= 0.4 else "D" if p >= 0.2 else "F"


# ------------------------------------------------------------------ reasons / warnings
def reason_text(key: str, t: str, r: pd.Series, f: pd.Series, sec_pe: pd.Series) -> str | None:
    g = f.get
    if key == "mom_6_1":
        return f"ขึ้น {pct(r['mom_6_1'])} ใน 6 เดือน"
    if key == "mom_12_1":
        return f"ขึ้น {pct(r['mom_12_1'])} ใน 1 ปี"
    if key == "dist_high":
        return "ใกล้จุดสูงสุด 1 ปี" if r["dist_high"] > -0.05 else None
    if key == "eps_q_growth":
        return f"กำไรไตรมาสโต {pct(g('earningsQuarterlyGrowth'))}"
    if key == "eps_growth":
        return f"กำไรโต {pct(g('earningsGrowth'))}"
    if key == "rev_growth":
        return f"รายได้โต {pct(g('revenueGrowth'))}"
    if key == "upside":
        return f"เป้านักวิเคราะห์ {pct(g('targetMeanPrice') / r['close'] - 1)}"
    if key == "rating":
        return "นักวิเคราะห์ส่วนใหญ่ให้ซื้อ" if _ok(g("recommendationMean")) and g("recommendationMean") <= 2 else None
    if key == "roe":
        return f"ROE {g('returnOnEquity') * 100:.0f}%" if _ok(g("returnOnEquity")) else None
    if key == "gross_margin":
        return f"อัตรากำไรขั้นต้น {g('grossMargins') * 100:.0f}%" if _ok(g("grossMargins")) else None
    if key == "op_margin":
        return f"อัตรากำไรจากธุรกิจ {g('operatingMargins') * 100:.0f}%" if _ok(g("operatingMargins")) else None
    if key == "low_debt":
        return "หนี้ต่ำ"
    if key == "fcf_yield":
        return "กระแสเงินสดดี"
    if key == "cheap_fpe":
        med = sec_pe.get(r["sector"]) if sec_pe is not None else None
        pe = g("forwardPE")
        if _ok(pe) and _ok(med) and 0 < pe < med:
            return f"P/E {pe:.0f} ถูกกว่ากลุ่ม ({med:.0f})"
    return None


SHORT_PHRASE = {
    "mom_6_1": "ขึ้นแรง", "mom_12_1": "ขึ้นแรง", "dist_high": "ใกล้จุดสูงสุด",
    "eps_q_growth": "กำไรโตแรง", "eps_growth": "กำไรโตแรง", "rev_growth": "รายได้โต",
    "upside": "เป้านักวิเคราะห์สูง", "rating": "นักวิเคราะห์ให้ซื้อ",
    "roe": "กำไรดี", "gross_margin": "กำไรดี", "op_margin": "กำไรดี",
    "low_debt": "หนี้ต่ำ", "fcf_yield": "เงินสดดี", "cheap_fpe": "ราคาไม่แพง",
}
ICONS = [("งบออก", "📅", "งบใกล้ออก"), ("ผันผวนสูง", "🎢", "ผันผวนสูง"), ("ยังขาดทุน", "💸", "ยังขาดทุน"),
         ("ราคาเกินเป้า", "🎯", "เกินเป้านักวิเคราะห์"), ("หนี้สูง", "💳", "หนี้สูง")]


def short_reason(h: str, t: str, ranks: dict, r, f, sec_pe, n: int = 2) -> str:
    """No-number phrase from the stock's strongest components, e.g. 'ขึ้นแรง ใกล้จุดสูงสุด'."""
    out = []
    for key, rk in ranks[h].loc[t].dropna().sort_values(ascending=False).items():
        if rk < 0.6 or len(out) >= n:
            break
        if key == "dist_high" and r["dist_high"] <= -0.05:
            continue
        if key == "cheap_fpe" and reason_text(key, t, r, f, sec_pe) is None:
            continue
        if key == "rating" and reason_text(key, t, r, f, sec_pe) is None:
            continue
        ph = SHORT_PHRASE.get(key)
        if ph and ph not in out:
            out.append(ph)
    return " ".join(out) or "คะแนนรวมสูง"


def icons(warns: list[str]) -> list[tuple[str, str]]:
    return [(ic, lbl) for key, ic, lbl in ICONS if any(w.startswith(key) for w in warns)]


def assign_horizons(scores: pd.DataFrame, n: int) -> dict[str, list[str]]:
    """Each stock appears once, in the horizon where it ranks best (greedy by score)."""
    triples = sorted(((sc, t, h) for h, _ in HORIZONS for t, sc in scores[h].dropna().items()), reverse=True)
    tops, used = {h: [] for h, _ in HORIZONS}, set()
    for sc, t, h in triples:
        if t not in used and len(tops[h]) < n:
            tops[h].append(t)
            used.add(t)
    return tops


def reasons(h: str, t: str, ranks: dict, r, f, sec_pe, n: int = 2) -> list[str]:
    row = ranks[h].loc[t].dropna().sort_values(ascending=False)
    out = []
    for key, rk in row.items():
        if rk < 0.6 or len(out) >= n:
            break
        txt = reason_text(key, t, r, f, sec_pe)
        if txt and txt not in out:
            out.append(txt)
    return out


def warnings(h: str, r, f, today: pd.Timestamp, vol: float | None) -> list[str]:
    out = []
    ed = next_earnings(f, today)
    if h in ("short", "mid") and ed is not None and (ed - today).days <= 28:
        out.append(f"งบออก {thdate(ed)}")
    pe = f.get("forwardPE")
    if _ok(pe) and pe <= 0:
        out.append("ยังขาดทุน")
    tgt = f.get("targetMeanPrice")
    if _ok(tgt) and tgt < r["close"]:
        out.append("ราคาเกินเป้านักวิเคราะห์")
    if _ok(vol) and vol > 0.6:
        out.append("ผันผวนสูง")
    if _ok(f.get("debtToEquity")) and f["debtToEquity"] > 200 and r["sector"] != "Financials":
        out.append("หนี้สูง")
    return out


# ------------------------------------------------------------------ messages
def build(scores, fund, universe, close, bench, bt, cfg, heatmap_path: str | None = None,
          weekly: bool = False) -> list[dict]:
    n = cfg.get("report", {}).get("top_n", 5)
    asof = close.index[-1]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    sector = universe.set_index("ticker")["sector"]
    snap = snapshot(price_features(close))
    ranks = part_ranks(snap, fund, sector, cfg.get("report", {}).get("min_analysts", 5))
    grades = factor_grades(snap, fund, sector, close)
    sec_pe = sector_medians(fund, sector, "forwardPE")
    vol = close.pct_change().iloc[-252:].std() * (252 ** 0.5)
    empty = pd.Series(dtype=object)

    # --- market line
    spy = bench[bench.columns[0]].dropna()
    up = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
    lines = [f"<b>🇺🇸 หุ้นเด่น {thdate(asof)}</b> · ตลาด {'🟢 ขาขึ้น' if up else '🔴 ขาลง'}"]
    if not up:
        lines.append("<i>ตลาดขาลง — ในอดีตเกณฑ์ระยะสั้นแพ้ SPY ช่วงแบบนี้</i>")

    tops = assign_horizons(scores, n)
    used_icons = {}
    for h, title in HORIZONS:
        lines += ["", f"<b>{title.replace('ระยะ', '').replace(' · ', ' ')}</b>"]
        for t in tops[h]:
            r, f = scores.loc[t], fund.loc[t] if t in fund.index else empty
            ics = icons(warnings(h, r, f, today, vol.get(t)))
            used_icons.update(dict(ics))
            mark = (" " + "".join(ic for ic, _ in ics)) if ics else ""
            lines.append(f"<code>{escape(t):<5}</code> {escape(short_reason(h, t, ranks, r, f, sec_pe))}{mark}")
    lines.append("")
    if used_icons:
        lines.append(" ".join(f"{ic}{lbl}" for ic, lbl in used_icons.items()))
    lines.append("<i>ไม่ใช่คำแนะนำการลงทุน · รายละเอียดตัวอันดับ 1 ด้านล่าง</i>")
    msgs = [{"text": "\n".join(lines)}]

    if heatmap_path:
        msgs.append({"photo": heatmap_path,
                     "caption": "🗺 แผนที่ตลาด 1 เดือน · ขนาด = มูลค่าบริษัท · เขียวขึ้น / แดงลง"})

    for h, title in HORIZONS:
        if tops[h]:
            msgs.append(card(tops[h][0], title, scores, fund, close, ranks, grades, sec_pe, today, vol, h))

    if weekly:
        msgs.append({"text": backtest_text(bt)})
    return msgs


def card(t, title, scores, fund, close, ranks, grades, sec_pe, today, vol, h) -> dict:
    r = scores.loc[t]
    f = fund.loc[t] if t in fund.index else pd.Series(dtype=object)
    name = f.get("longName") if _ok(f.get("longName")) else t
    hist = stock_history(close[t])
    g = grades.loc[t]
    gl = [f"{lbl} {grade(g[k])}" for k, lbl in GRADE_TH]
    lines = [
        f"<b>🔎 อันดับ 1 {title.split(' · ')[0]}: {escape(t)}</b>",
        f"{escape(str(name))} · {escape(str(f.get('industry') or r['sector']))}",
        f"ราคา ${r['close']:,.2f}",
        "",
        "<b>เกรด</b> <i>(A ดีสุด → F · เทียบหุ้นอื่นในลิสต์)</i>",
        f"<code>{'  '.join(gl[:3])}\n{'  '.join(gl[3:])}</code>",
        "",
    ]
    for x in reasons(h, t, ranks, r, f, sec_pe, n=3):
        lines.append(f"✅ {escape(x)}")
    for x in warnings(h, r, f, today, vol.get(t)):
        lines.append(f"⚠️ {escape(x)}")
    if _ok(hist.get("maxdd_1y")) and hist["maxdd_1y"] < -0.25:
        lines.append(f"⚠️ ใน 1 ปีเคยร่วง {pct(hist['maxdd_1y'])} จากจุดสูงสุด")
    if hist:
        lines += ["", f"📊 <b>อดีต {hist['w4_years']:.0f} ปี</b> ถ้าถือ 4 สัปดาห์: บวก {hist['w4_win'] * 100:.0f}% ของครั้ง · "
                      f"เฉลี่ย {pct(hist['w4_avg'], 1)} · แย่ 1 ใน 10 ครั้ง {pct(hist['w4_p10'], 1)}"]
    if _ok(f.get("targetMeanPrice")) and _ok(f.get("numberOfAnalystOpinions")):
        lines.append(f"🎯 นักวิเคราะห์ {f['numberOfAnalystOpinions']:.0f} คน เป้าเฉลี่ย ${f['targetMeanPrice']:,.0f} "
                     f"({pct(f['targetMeanPrice'] / r['close'] - 1)})")
    url_t = t.replace(".", "-")
    return {"text": "\n".join(lines),
            "buttons": [[{"text": f"📈 กราฟ {t}", "url": f"https://finance.yahoo.com/quote/{url_t}"}]]}


def backtest_text(bt: dict) -> str:
    s = bt["stats"]
    if not s.get("n"):
        return "🧪 ทดสอบย้อนหลัง: ข้อมูลไม่พอ"
    return "\n".join([
        "<b>🧪 สรุปผลย้อนหลังประจำสัปดาห์ — เกณฑ์ระยะสั้น</b>",
        f"ถ้าทุก 4 สัปดาห์ซื้อ Top {bt['top_n']} เท่ากัน ({s['start']:%m/%Y}–{s['end']:%m/%Y}, {s['n']} รอบ, "
        f"หักค่าธรรมเนียม {bt['fee_per_side_pct']:.2f}%/ขา*)",
        f"• ต่อปี: พอร์ต <b>{pct(s['cagr_port'], 1)}</b> vs SPY {pct(s['cagr_bench'], 1)}",
        f"• ชนะ SPY {s['beat_bench'] * 100:.0f}% ของรอบ",
        f"• ร่วงหนักสุด: พอร์ต {pct(s['maxdd_port'], 1)} vs SPY {pct(s['maxdd_bench'], 1)}",
        f"• ช่วงตลาดขาลง ({s['n_regime_down']} รอบ): ต่างจาก SPY เฉลี่ย {pct(s['excess_regime_down'], 2)} ต่อรอบ",
        "⚠️ ทดสอบด้วยรายชื่อหุ้นในดัชนีวันนี้ → ตัวเลขดูดีเกินจริง",
        "⚠️ ระยะกลาง/ยาวยังทดสอบย้อนหลังไม่ได้ (ไม่มีงบย้อนหลังแบบ ณ เวลานั้น)",
        "<i>* ค่าธรรมเนียมเป็นค่าสมมติ ยังไม่ใช่ของ Dime จริง</i>",
    ])


def cards_for(tickers: list[str], title: str, scores, fund, universe, close, cfg) -> list[dict]:
    """Detail cards (grades, reasons, history, chart button) for the given tickers."""
    if not tickers:
        return []
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    sector = universe.set_index("ticker")["sector"].reindex(scores.index).fillna("Unknown")
    snap = snapshot(price_features(close))
    ranks = part_ranks(snap, fund, sector, cfg.get("report", {}).get("min_analysts", 5))
    grades = factor_grades(snap, fund, sector, close)
    sec_pe = sector_medians(fund, sector, "forwardPE")
    vol = close.pct_change().iloc[-252:].std() * (252 ** 0.5)
    out = []
    for t in tickers:
        if t in scores.index:
            c = card(t, title, scores, fund, close, ranks, grades, sec_pe, today, vol, "short")
            first, rest = c["text"].split("\n", 1)
            c["text"] = f"<b>{escape(title)} {escape(t)}</b>\n" + rest
            out.append(c)
    return out


def model_backtest_text(stats: dict, rules) -> str:
    s = stats
    return "\n".join([
        "<b>🧪 สรุปประจำสัปดาห์ — ถ้าใช้กติกานี้ย้อนหลัง</b>",
        f"{s['start']:%m/%Y}–{s['end']:%m/%Y} · ถือ {rules.n_hold} ตัว · หักค่าธรรมเนียม {rules.fee_per_side_pct:.2f}%/ขา*",
        f"• ต่อปี: พอร์ต <b>{pct(s['cagr'], 1)}</b> vs SPY {pct(s['cagr_spy'], 1)}",
        f"• ร่วงหนักสุด: พอร์ต {pct(s['maxdd'], 1)} vs SPY {pct(s['maxdd_spy'], 1)}",
        f"• ซื้อขายเฉลี่ย {s['trades_per_month']:.1f} ครั้ง/เดือน",
        "⚠️ ใช้รายชื่อหุ้นในดัชนีวันนี้ทดสอบย้อนหลัง → ตัวเลขดูดีเกินจริง · ผลงานจริงดูที่ 'ตั้งแต่เริ่ม' ในข้อความทุกเช้า",
        "<i>* ค่าธรรมเนียมเป็นค่าสมมติ (ส่วนต่างค่าเงิน) ยังไม่ใช่ตัวเลขจริงของ Dime</i>",
    ])
