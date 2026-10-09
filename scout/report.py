"""Thai-language report (plain text, Telegram-friendly)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from .backtest import stock_history
from .universe import SECTOR_TH

HORIZON_TH = {"short": "ระยะสั้น (~4 สัปดาห์)", "mid": "ระยะกลาง (6–18 เดือน)", "long": "ระยะยาว (1–5 ปี+)"}
DISCLAIMER = ("ℹ️ ใช้ประกอบการตัดสินใจเท่านั้น ไม่ใช่คำแนะนำการลงทุน ไม่รับประกันกำไร "
              "ข้อมูลจาก Yahoo Finance อาจผิด/ล่าช้า — ตรวจสอบก่อนซื้อขาย และเช็กว่า Dime มีหุ้นตัวนั้น")


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, signed: bool = True, d: int = 1) -> str:
    if not _ok(x):
        return "–"
    return f"{x * 100:+.{d}f}%" if signed else f"{x * 100:.{d}f}%"


def num(x, d: int = 1) -> str:
    return f"{x:.{d}f}" if _ok(x) else "–"


def sec_th(sector: str) -> str:
    return SECTOR_TH.get(sector, sector)


def next_earnings(f: pd.Series, today: pd.Timestamp):
    for k in ("earningsTimestampStart", "earningsTimestamp"):
        ts = f.get(k)
        if _ok(ts):
            d = pd.Timestamp(int(ts), unit="s").normalize()
            if d >= today:
                return d
    return None


# --------------------------------------------------------------------- sections
def header(asof: pd.Timestamp, bench: pd.DataFrame, scores: pd.DataFrame, n_universe: int,
           n_fund: int, universe_src: str) -> str:
    lines = [f"🇺🇸 US Stock Scout — ข้อมูลปิดตลาด {asof:%d/%m/%Y}",
             f"จักรวาล: S&P 500 + Nasdaq-100 = {n_universe} ตัว (ราคา {len(scores)} · พื้นฐาน {n_fund}) [{universe_src}]",
             "", "📊 สภาพตลาด"]
    for col in bench.columns:
        s = bench[col].dropna()
        if len(s) < 200:
            continue
        sma = s.rolling(200).mean().iloc[-1]
        trend = "ขาขึ้น ✅" if s.iloc[-1] > sma else "ขาลง ⚠️"
        lines.append(f"• {col} {s.iloc[-1]:.2f} | เทียบเส้น 200 วัน {pct(s.iloc[-1] / sma - 1)} → {trend} | 1 เดือน {pct(s.iloc[-1] / s.iloc[-22] - 1)}")
    breadth = (scores["close"] > scores["sma200"]).mean()
    lines.append(f"• หุ้นที่อยู่เหนือเส้น 200 วัน: {pct(breadth, signed=False, d=0)} ของจักรวาล")
    return "\n".join(lines)


def backtest_section(bt: dict) -> str:
    s = bt["stats"]
    if not s.get("n"):
        return "🧪 ทดสอบย้อนหลัง: ข้อมูลไม่พอ"
    lines = [
        f"🧪 ผลทดสอบย้อนหลังของเกณฑ์ระยะสั้น (ข้อมูลจริง {s['start']:%m/%Y}–{s['end']:%m/%Y}, {s['n']} รอบ)",
        f"กติกา: ทุก {bt['horizon']} วันทำการ ซื้อ {bt['top_n']} อันดับแรกเท่ากัน ถือ 4 สัปดาห์ หักค่าธรรมเนียม {bt['fee_per_side_pct']:.2f}%/ขา",
        f"• เฉลี่ยต่อรอบ: พอร์ต {pct(s['avg_port'], d=2)} vs SPY {pct(s['avg_bench'], d=2)} (ส่วนต่าง {pct(s['avg_excess'], d=2)})",
        f"• ชนะ SPY {pct(s['beat_bench'], False, 0)} ของรอบ | หุ้นที่เลือกปิดบวก {pct(s['pick_win'], False, 0)}",
        f"• ต่อปี (ทบต้น): พอร์ต {pct(s['cagr_port'])} vs SPY {pct(s['cagr_bench'])}",
        f"• ขาดทุนสะสมสูงสุด: พอร์ต {pct(s['maxdd_port'])} vs SPY {pct(s['maxdd_bench'])}",
        f"• ครึ่งแรก/ครึ่งหลัง ส่วนต่างต่อรอบ: {pct(s['excess_first_half'], d=2)} / {pct(s['excess_second_half'], d=2)}",
        f"• ตอนตลาดขาลง ({s['n_regime_down']} รอบ) ส่วนต่างต่อรอบ: {pct(s['excess_regime_down'], d=2)}",
        "⚠️ ใช้รายชื่อหุ้นในดัชนี 'วันนี้' ย้อนไปทดสอบ (survivorship bias) → ผลจริงน่าจะแย่กว่านี้",
        "⚠️ ระยะกลาง/ยาวใช้งบการเงินล่าสุดเท่านั้น ยังไม่มีข้อมูลย้อนหลังแบบ ณ เวลานั้น → ยังทดสอบย้อนหลังไม่ได้",
    ]
    return "\n".join(lines)


def _line_short(t, r, f):
    return (f"{t} [{sec_th(r['sector'])}] ${num(r['close'], 2)} | 6ด.(ข้ามเดือนล่าสุด) {pct(r['mom_6_1'], d=0)} "
            f"| ห่าง high {pct(r['dist_high'], d=0)}")


def _line_mid(t, r, f):
    up = f.get("targetMeanPrice") / r["close"] - 1 if _ok(f.get("targetMeanPrice")) else None
    return (f"{t} [{sec_th(r['sector'])}] กำไรไตรมาส YoY {pct(f.get('earningsQuarterlyGrowth'), d=0)} "
            f"| รายได้ {pct(f.get('revenueGrowth'), d=0)} | เป้านักวิเคราะห์ {pct(up, d=0)} ({num(f.get('numberOfAnalystOpinions'), 0)} คน)")


def _line_long(t, r, f):
    return (f"{t} [{sec_th(r['sector'])}] ROE {pct(f.get('returnOnEquity'), False, 0)} "
            f"| GM {pct(f.get('grossMargins'), False, 0)} | D/E {num(f.get('debtToEquity'), 0)}% | Fwd P/E {num(f.get('forwardPE'))}")


LINE = {"short": _line_short, "mid": _line_mid, "long": _line_long}


def top_list(horizon: str, scores: pd.DataFrame, fund: pd.DataFrame, n: int, today: pd.Timestamp) -> str:
    top = scores[horizon].dropna().nlargest(n)
    lines = [f"🏁 {HORIZON_TH[horizon]} — Top {len(top)} (คะแนน 0–100)"]
    for k, (t, sc) in enumerate(top.items(), 1):
        f = fund.loc[t] if t in fund.index else pd.Series(dtype=object)
        warn = ""
        ed = next_earnings(f, today)
        if horizon == "short" and ed is not None and (ed - today).days <= 28:
            warn = f" ⚠️งบ {ed:%d/%m}"
        lines.append(f"{k}. {sc * 100:.0f} · {LINE[horizon](t, scores.loc[t], f)}{warn}")
    if horizon == "short":
        lines.append("เกณฑ์: โมเมนตัม 6 และ 12 เดือน (ข้ามเดือนล่าสุด) + ใกล้จุดสูงสุด 52 สัปดาห์ + ราคาเหนือเส้น 200 วัน")
    elif horizon == "mid":
        lines.append("เกณฑ์: การเติบโตกำไร/รายได้ + ราคาเป้า/เรตติ้งนักวิเคราะห์ + โมเมนตัม 6 เดือน")
    else:
        lines.append("เกณฑ์ (เทียบกับหุ้นอุตสาหกรรมเดียวกัน): ROE, margin, หนี้ต่ำ, รายได้โต, FCF yield, P/E ล่วงหน้าต่ำ")
    return "\n".join(lines)


def sector_section(scores: pd.DataFrame, per_sector: int) -> str:
    lines = ["🏭 หุ้นเด่นแยกตามอุตสาหกรรม (คะแนนรวม 3 ระยะ · S/M/L = สั้น/กลาง/ยาว)"]
    for sector, g in sorted(scores.dropna(subset=["overall"]).groupby("sector"), key=lambda x: x[0]):
        top = g["overall"].nlargest(per_sector)
        items = []
        for t in top.index:
            r = g.loc[t]
            sml = "/".join(f"{r[h] * 100:.0f}" if _ok(r[h]) else "–" for h in ("short", "mid", "long"))
            items.append(f"{t} ({sml})")
        lines.append(f"• {sec_th(sector)}: " + ", ".join(items))
    return "\n".join(lines)


def notes(t: str, r: pd.Series, f: pd.Series, h: dict, sec_pe: pd.Series, today: pd.Timestamp) -> list[str]:
    """Rule-based observations from the data itself (no predictions)."""
    out = []
    ed = next_earnings(f, today)
    if ed is not None and (ed - today).days <= 28:
        out.append(f"⚠️ ประกาศงบ {ed:%d/%m/%Y} (อีก {(ed - today).days} วัน) — ราคามักแกว่งแรงช่วงนี้")
    if _ok(r.get("sma200")) and r["close"] < r["sma200"]:
        out.append("⚠️ ราคาต่ำกว่าเส้น 200 วัน (แนวโน้มหลักยังไม่ขึ้น)")
    if _ok(r.get("dist_high")) and r["dist_high"] > -0.03:
        out.append("✅ อยู่ใกล้จุดสูงสุด 52 สัปดาห์")
    pe, sec = f.get("forwardPE"), r["sector"]
    med = sec_pe.get(sec) if sec_pe is not None else None
    if _ok(pe) and _ok(med) and pe > 0 and med > 0:
        ratio = pe / med
        if ratio > 1.5:
            out.append(f"⚠️ P/E ล่วงหน้า {pe:.1f} สูงกว่าค่ากลางกลุ่ม ({med:.1f}) {ratio:.1f} เท่า")
        elif ratio < 0.7:
            out.append(f"✅ P/E ล่วงหน้า {pe:.1f} ต่ำกว่าค่ากลางกลุ่ม ({med:.1f})")
    if _ok(pe) and pe <= 0:
        out.append("⚠️ คาดว่ายังขาดทุน (P/E ล่วงหน้าติดลบ)")
    tgt = f.get("targetMeanPrice")
    if _ok(tgt) and tgt < r["close"]:
        out.append("⚠️ ราคาสูงกว่าเป้าเฉลี่ยนักวิเคราะห์แล้ว")
    if _ok(f.get("debtToEquity")) and f["debtToEquity"] > 200:
        out.append(f"⚠️ หนี้/ทุนสูง ({f['debtToEquity']:.0f}%)")
    if _ok(f.get("beta")) and f["beta"] > 1.5:
        out.append(f"⚠️ ผันผวนกว่าตลาด (beta {f['beta']:.1f})")
    if _ok(h.get("maxdd_1y")) and h["maxdd_1y"] < -0.30:
        out.append(f"⚠️ ใน 1 ปีเคยร่วงจากจุดสูงสุด {pct(h['maxdd_1y'], d=0)}")
    return out


def detail(t: str, scores: pd.DataFrame, fund: pd.DataFrame, universe: pd.DataFrame, close: pd.DataFrame,
           sec_pe: pd.Series, today: pd.Timestamp, horizon_days: int) -> str:
    r = scores.loc[t]
    f = fund.loc[t] if t in fund.index else pd.Series(dtype=object)
    u = universe.set_index("ticker")
    name = f.get("longName") if _ok(f.get("longName")) else u.loc[t, "name"] if t in u.index else t
    h = stock_history(close[t], horizon_days)
    idx = [x for x, flag in (("S&P500", "in_sp500"), ("NDX100", "in_ndx")) if t in u.index and bool(u.loc[t, flag])]
    sml = " / ".join(f"{HORIZON_TH[k].split(' ')[0]} {r[k] * 100:.0f}" if _ok(r[k]) else f"{HORIZON_TH[k].split(' ')[0]} –"
                     for k in ("short", "mid", "long"))
    mc = f.get("marketCap")
    lines = [
        f"🔎 {t} — {name}",
        f"{sec_th(r['sector'])} · {f.get('industry') if _ok(f.get('industry')) else '–'} · {', '.join(idx)}"
        + (f" · มูลค่าตลาด ${mc / 1e9:,.0f}B" if _ok(mc) else ""),
        f"คะแนน: {sml}",
        f"ราคา ${num(r['close'], 2)} | 1ด. {pct(h.get('ret_1m'), d=0)} · 3ด. {pct(h.get('ret_3m'), d=0)} · "
        f"6ด. {pct(h.get('ret_6m'), d=0)} · 1ปี {pct(h.get('ret_1y'), d=0)} · 3ปี {pct(h.get('ret_3y'), d=0)}",
        f"ความผันผวน 1 ปี {pct(h.get('vol_1y'), False, 0)}/ปี | ร่วงสูงสุดใน 1 ปี {pct(h.get('maxdd_1y'), d=0)}",
        f"สถิติจริงถือ 4 สัปดาห์ (ย้อน {num(h.get('w4_years'), 1)} ปี): บวก {pct(h.get('w4_win'), False, 0)} ของเวลา · "
        f"เฉลี่ย {pct(h.get('w4_avg'))} · แย่ 10% {pct(h.get('w4_p10'))} · ดี 10% {pct(h.get('w4_p90'))}",
        f"พื้นฐาน: รายได้ YoY {pct(f.get('revenueGrowth'), d=0)} · กำไรไตรมาส YoY {pct(f.get('earningsQuarterlyGrowth'), d=0)} · "
        f"ROE {pct(f.get('returnOnEquity'), False, 0)} · GM {pct(f.get('grossMargins'), False, 0)} · "
        f"OPM {pct(f.get('operatingMargins'), False, 0)}",
        f"มูลค่า: P/E {num(f.get('trailingPE'))} · Fwd P/E {num(f.get('forwardPE'))} (กลุ่ม {num(sec_pe.get(r['sector']) if sec_pe is not None else None)}) · "
        f"P/S {num(f.get('priceToSalesTrailing12Months'))}",
    ]
    if _ok(f.get("targetMeanPrice")):
        lines.append(f"นักวิเคราะห์ {num(f.get('numberOfAnalystOpinions'), 0)} คน: เป้าเฉลี่ย ${num(f.get('targetMeanPrice'), 2)} "
                     f"({pct(f['targetMeanPrice'] / r['close'] - 1, d=0)}) ช่วง ${num(f.get('targetLowPrice'), 0)}–${num(f.get('targetHighPrice'), 0)} "
                     f"· เรตติ้ง {num(f.get('recommendationMean'))} ({f.get('recommendationKey') or '–'}; 1=ซื้อแรง 5=ขาย)")
    ns = notes(t, r, f, h, sec_pe, today)
    if ns:
        lines += ns
    return "\n".join(lines)


def build_report(scores, fund, universe, close, bench, bt, cfg, universe_src="") -> list[str]:
    """Return list of message sections (each sent as its own Telegram message)."""
    rcfg = cfg.get("report", {})
    n, per_sector, detail_n = rcfg.get("top_n", 10), rcfg.get("per_sector", 2), rcfg.get("detail_n", 3)
    horizon_days = cfg.get("short", {}).get("horizon_days", 20)
    asof = close.index[-1]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    sector = universe.set_index("ticker")["sector"]
    from .scoring import sector_medians
    sec_pe = sector_medians(fund, sector, "forwardPE") if "forwardPE" in fund else None
    n_fund = int(fund["marketCap"].notna().sum()) if "marketCap" in fund else 0

    sections = [header(asof, bench, scores, len(universe), n_fund, universe_src) + "\n\n" + backtest_section(bt)]
    for h in ("short", "mid", "long"):
        sections.append(top_list(h, scores, fund, n, today))
    sections.append(sector_section(scores, per_sector))

    picked, seen = [], set()
    for h in ("short", "mid", "long"):
        count = 0
        for t in scores[h].dropna().nlargest(n).index:
            if count >= detail_n:
                break
            if t not in seen:
                picked.append((t, h))
                seen.add(t)
                count += 1
    for t, h in picked:
        sections.append(f"[{HORIZON_TH[h]}]\n" + detail(t, scores, fund, universe, close, sec_pe, today, horizon_days))
    sections.append(DISCLAIMER)
    return sections
