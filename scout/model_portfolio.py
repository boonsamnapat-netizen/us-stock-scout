"""Live model portfolio (Alpha-Picks-style): the system holds a notional portfolio by fixed rules
and tells the owner what to buy/sell each morning. State lives in data/model_portfolio.json and is
committed back to the repo by the workflow. Rules = backtested variant E3 (see research_portfolio.py).
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .brief import SECTOR_SHORT, thdate
from .portfolio_bt import Rules, decide_day, new_state, signals, value_of

START_VALUE = 100.0


def rules_from_cfg(cfg: dict) -> Rules:
    m = cfg.get("model_portfolio", {})
    return Rules(name="E3", n_hold=m.get("n_hold", 5), entry_pct=m.get("entry_pct", 0.10),
                 exit_pct=m.get("exit_pct", 0.30), gate="cash", check="daily",
                 max_trades_month=m.get("max_trades_month", 5), max_vol=m.get("max_vol", 0.60),
                 gate_sma=m.get("gate_sma", 200),
                 fee_per_side_pct=cfg.get("fees", {}).get("per_side_pct", 0.10))


def load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def save(doc: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str))


def update(close: pd.DataFrame, spy: pd.Series, cfg: dict, path: str) -> dict:
    """Run today's rules once per closing date (idempotent on re-runs). Returns a view dict."""
    rules = rules_from_cfg(cfg)
    rank, vol, up = signals(close, spy, rules)
    asof = close.index[-1]
    price = close.iloc[-1]
    doc = load(path)
    if doc is None:
        doc = {"started": str(asof.date()), "spy_start": float(spy.iloc[-1]),
               "state": new_state(START_VALUE), "trades": [], "history": [], "last_date": None}
    state = doc["state"]
    today_trades = []
    if doc["last_date"] != str(asof.date()):
        today_trades = decide_day(state, asof, rank.iloc[-1], price, vol.iloc[-1], bool(up.iloc[-1]), rules)
        for t in today_trades:
            doc["trades"].append({**t, "date": str(asof.date()), "price": float(t["price"]),
                                  **({"weight": float(t["weight"])} if "weight" in t else {})})
        doc["last_date"] = str(asof.date())
        doc["history"].append({"date": str(asof.date()), "value": value_of(state, price),
                               "spy": float(spy.iloc[-1])})
        save(doc, path)
    else:
        today_trades = [t for t in doc["trades"] if t["date"] == str(asof.date())]

    # waiting list: best-ranked buyable names not held
    r = rank.iloc[-1]
    queue = [t for t in r[r <= rules.entry_pct].sort_values().index
             if t not in state["units"] and not (vol.iloc[-1].get(t, np.nan) > rules.max_vol)][:3]
    return {"doc": doc, "rules": rules, "asof": asof, "price": price, "up": bool(up.iloc[-1]),
            "today": today_trades, "queue": queue, "rank": r}


def message(view: dict, scores: pd.DataFrame, spy: pd.Series) -> str:
    doc, rules, price = view["doc"], view["rules"], view["price"]
    state = doc["state"]
    asof = view["asof"]
    value = value_of(state, price)
    lines = [f"<b>🇺🇸 Model Portfolio · {thdate(asof)}</b> · ตลาด {'🟢 ขาขึ้น' if view['up'] else '🔴 ขาลง'}", ""]

    lines.append("<b>🔔 วันนี้ต้องทำ</b>")
    if view["today"]:
        for t in view["today"]:
            if t["side"] == "sell":
                lines.append(f"🔴 ขาย <b>{escape(t['ticker'])}</b> — {escape(t['why'])}")
        for t in view["today"]:
            if t["side"] == "buy":
                w = t.get("weight", 1 / rules.n_hold)
                lines.append(f"🟢 ซื้อ <b>{escape(t['ticker'])}</b> ~{w * 100:.0f}% ของพอร์ต — {escape(t['why'])}")
    else:
        lines.append("ไม่ต้องทำอะไร ✋" + ("" if view["up"] else " (ตลาดขาลง ถือเงินสด)"))

    lines += ["", f"<b>💼 ถืออยู่ {len(state['units'])}/{rules.n_hold} ตัว</b>"]
    for t, u in sorted(state["units"].items(), key=lambda x: -x[1] * price.get(x[0], 0)):
        p, e = price.get(t, np.nan), state["entry"].get(t, np.nan)
        w = u * p / value if value else np.nan
        sec = SECTOR_SHORT.get(scores["sector"].get(t, "Unknown"), "") if "sector" in scores else ""
        lines.append(f"<code>{escape(t):<5}</code> {(p / e - 1) * 100:+.1f}% · {w * 100:.0f}% <i>{sec}</i>")
    cash_w = state["cash"] / value if value else 0
    if cash_w > 0.01:
        lines.append(f"<code>เงินสด</code> {cash_w * 100:.0f}%")

    spy_ret = spy.iloc[-1] / doc["spy_start"] - 1
    lines += ["", f"📈 ตั้งแต่เริ่ม ({doc['started']}): <b>{(value / START_VALUE - 1) * 100:+.1f}%</b> vs SPY {spy_ret * 100:+.1f}%",
              f"🎫 ซื้อขายเดือนนี้ {state['trades_this_month']}/{rules.max_trades_month} ครั้ง"]
    if view["queue"] and len(state["units"]) >= rules.n_hold:
        lines.append(f"⏳ คิวถัดไปถ้ามีที่ว่าง: {', '.join(escape(t) for t in view['queue'])}")
    lines += ["", "<i>กติกา: ซื้อหุ้นคะแนนโมเมนตัม 10% บน (ผันผวน ≤60%/ปี) · ขายเมื่อหลุด 30% บนหรือหลุดเส้น 200 วัน · "
                  "ตลาดขาลงถือเงินสด · ไม่ใช่คำแนะนำการลงทุน</i>"]
    return "\n".join(lines)
