#!/usr/bin/env python
"""Telegram command poller (GitHub Actions cron, ~every 15 min — replies are NOT instant).

  /check TSLA [NVDA ...]   full card for up to 3 tickers (any US ticker, not only the scout universe)
  /help                    usage

Security: only the owner's chat (TELEGRAM_CHAT_ID) is served; everything else is ignored.
State: data/bot_state.json keeps the Telegram update offset so each message is answered once.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from scout import data, estimates, scout_report as sr, stages, telegram

BASE = Path(__file__).resolve().parent
BOT_STATE = BASE / "data/bot_state.json"
SCOUT_STATE = BASE / "data/scout_state.json"
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
MAX_PER_MSG, MAX_PER_RUN = 3, 10
HELP = ("<b>🤖 คำสั่ง</b>\n/check TSLA — การ์ดข้อมูลหุ้น (ใส่ได้ถึง 3 ตัว เช่น /check NVDA AMD MU)\n"
        "<i>บอทเช็กข้อความทุก ~15 นาที จึงตอบไม่ทันที · ข้อมูล Yahoo Finance · ไม่ใช่คำแนะนำการลงทุน</i>")


def parse(text: str) -> tuple[str, list[str]]:
    parts = (text or "").strip().split()
    if not parts:
        return "none", []
    cmd = parts[0].split("@")[0].lower()
    if cmd == "/check":
        tickers = [p.upper().replace(",", "") for p in parts[1:]]
        tickers = [t.replace(".", "-") for t in tickers if TICKER_RE.match(t)]
        return ("check", tickers[:MAX_PER_MSG]) if tickers else ("help", [])
    if cmd in ("/help", "/start"):
        return "help", []
    return "none", []


def gate_lines(fund: pd.DataFrame, est: pd.DataFrame, t: str) -> list[str]:
    f, e = fund.loc[t], est.loc[t]
    def mark(ok):
        return "✅" if ok is True else ("❌" if ok is False else "❔")
    def cmp(x, fn):
        return None if x is None or pd.isna(x) else bool(fn(x))
    prof = cmp(e.get("ni_ttm"), lambda v: v > 0)
    if prof is None:
        prof = cmp(f.get("profitMargins"), lambda v: v > 0)
    return [
        "<b>เกณฑ์ 'พื้นฐานดี' ของ Scout</b>",
        f"{mark(prof)} มีกำไร 12 เดือนล่าสุด",
        f"{mark(cmp(e.get('eps_g_ny'), lambda v: v > 0))} นักวิเคราะห์คาดกำไรปีหน้าโต",
        f"{mark(cmp(e.get('rev_g_ny'), lambda v: v > 0))} นักวิเคราะห์คาดรายได้ปีหน้าโต",
        f"{mark(cmp(e.get('eps_rev_90'), lambda v: v >= -0.02))} ประมาณการไม่ได้ถูกปรับลด (90 วัน)",
        "<i>+ ต้องอยู่ในกลุ่มคะแนนพื้นฐาน 30% บนของหุ้น 1,500 ตัว (ดูได้จากรายงานสัปดาห์)</i>",
    ]


def check_card(t: str, watch: dict) -> dict:
    close_all, volume = data.fetch_prices([t, "SPY"], 3)
    if t not in close_all.columns or close_all[t].dropna().empty:
        return {"text": f"❓ ไม่พบข้อมูลราคาของ <b>{sr.escape(t)}</b> — ตรวจชื่อย่อหุ้นอีกครั้ง"}
    close = close_all[[t]]
    fund = data.fetch_fundamentals([t])
    est = estimates.fetch([t])
    panels = stages.stage_panels(close, volume.reindex(columns=[t]))
    stage = stages.stage_today(panels)
    dd = panels["dd"].iloc[-1]
    in_watch = t in watch
    row = watch.get(t, {})
    cls = pd.DataFrame({"good": [bool(row.get("good", False))], "emerging": [bool(row.get("emerging", False))]},
                       index=[t])
    card = sr.card(t, cls, fund, est, stage, dd, close, pd.Series(dtype=float))
    status = ("📌 <b>อยู่ในรายชื่อเฝ้าดูของ Scout</b>" if in_watch
              else "📌 ไม่อยู่ในรายชื่อเฝ้าดูของ Scout (ณ รายงานสัปดาห์ล่าสุด)")
    lines = card["text"].split("\n")
    lines[2] = status                                   # replace the ✅/🌱 label line
    card["text"] = "\n".join(lines + [""] + gate_lines(fund, est, t)
                             + ["", "<i>ข้อมูล Yahoo Finance · ไม่ใช่คำแนะนำการลงทุน</i>"])
    return card


def main() -> int:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[bot] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return 1
    st = json.loads(BOT_STATE.read_text()) if BOT_STATE.exists() else {"offset": 0}
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                     params={"offset": st["offset"] + 1, "timeout": 0}, timeout=30)
    r.raise_for_status()
    updates = r.json().get("result", [])
    if not updates:
        print("[bot] no new messages")
        return 0
    watch = json.loads(SCOUT_STATE.read_text()).get("watch", {}) if SCOUT_STATE.exists() else {}
    replies, budget = [], MAX_PER_RUN
    for u in updates:
        st["offset"] = max(st["offset"], u["update_id"])
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(chat):
            continue                                    # only the owner's chat
        kind, tickers = parse(msg.get("text", ""))
        if kind == "help":
            replies.append({"text": HELP})
        for t in tickers:
            if budget <= 0:
                replies.append({"text": f"⚠️ ขอได้ครั้งละไม่เกิน {MAX_PER_RUN} ตัวต่อรอบ — ส่ง /check {t} ใหม่อีกครั้ง"})
                break
            budget -= 1
            try:
                replies.append(check_card(t, watch))
            except Exception as e:                      # never lose the offset because of one ticker
                print(f"[bot] {t}: {e!r}")
                replies.append({"text": f"⚠️ ดึงข้อมูล {sr.escape(t)} ไม่สำเร็จ ลองใหม่ภายหลัง"})
    for rep in replies:                                 # one bad message must not block the others
        try:
            telegram.send(token, chat, [rep])
        except Exception as e:
            print(f"[bot] send failed: {e!r}")
            try:
                telegram.send(token, chat, [{"text": "⚠️ ส่งการ์ดไม่สำเร็จ (รูปแบบข้อความผิดพลาด)", "plain": True}])
            except Exception:
                pass
    BOT_STATE.parent.mkdir(parents=True, exist_ok=True)
    BOT_STATE.write_text(json.dumps(st))                # saved after replying -> crash = retry next run
    print(f"[bot] {len(updates)} update(s), {len(replies)} repl(ies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
