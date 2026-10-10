#!/usr/bin/env python
"""Stock Scout runner.

  python scout_main.py --demo --mode weekly        # offline synthetic data
  python scout_main.py --mode auto --notify        # real data: weekly report on Friday's close
                                                   # (Saturday morning TH), otherwise daily alerts only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from scout import data, estimates, events, heatmap, scout_report as sr, stages, telegram, track, universe as uni
from scout.report import next_earnings
from scout.scoring import sector_medians

BASE = Path(__file__).resolve().parent
STATE_FILE = "data/scout_state.json"
HISTORY_FILE = "data/scout_history.csv"


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:                         # extras must never cost us the report
        print(f"[scout] optional step failed: {e!r}")
        return default


def refresh_earn_next(state: dict, watch, fund: pd.DataFrame, asof: pd.Timestamp) -> None:
    """Store each watchlist name's next report date (from .info). An entry whose date has already
    passed is kept: it is still waiting for its recap (Yahoo may already show next quarter's date)."""
    day = pd.Timestamp(asof.date())
    cur = state.setdefault("earn_next", {})
    for t in watch:
        if t in fund.index and not (t in cur and pd.Timestamp(cur[t]) <= day):
            d = next_earnings(fund.loc[t], day)
            if d is not None:
                cur[t] = str(d.date())


def earnings_recap(state: dict, asof: pd.Timestamp, close: pd.DataFrame, window: int = 10) -> dict | None:
    """Watchlist names whose (stored) report date has passed: EPS vs estimate, price reaction, revisions.
    Each report is recapped once (state['earn_done']); dates older than `window` days are dropped."""
    day = pd.Timestamp(asof.date())
    nxt, done = state.setdefault("earn_next", {}), state.setdefault("earn_done", {})
    for t in [t for t in nxt if t not in state.get("watch", {})]:
        nxt.pop(t)                                 # left the watchlist
    due = [t for t, d in nxt.items() if pd.Timestamp(d) < day]
    if not due:
        return None
    ev = events.fetch(due, day, insiders=False)
    est = estimates.fetch(due)
    items, nxt2, done2 = [], dict(nxt), dict(done)
    for t in due:
        rep = ev[t]["earn"]["reported"]
        q = rep[0] if rep else None
        fresh = q is not None and 0 < (day - q["date"].normalize()).days <= window
        if fresh:
            if done2.get(t) != str(q["date"].date()):
                r = events.reaction(close[t], q["date"]) if t in close.columns else float("nan")
                items.append((t, q, r, est.loc[t] if t in est.index else None))
                done2[t] = str(q["date"].date())
            nxt2.pop(t, None)
        elif (day - pd.Timestamp(nxt2[t])).days > window:
            nxt2.pop(t, None)                      # never found the report -> stop looking
        nxt_new = ev[t]["earn"]["next"]
        if t not in nxt2 and nxt_new is not None and nxt_new.normalize() >= day:
            nxt2[t] = str(nxt_new.date())         # roll forward to the next report
    msg = events.recap(asof, items)                # build first: a failure here keeps the old state
    state["earn_next"], state["earn_done"] = nxt2, done2
    return msg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "weekly", "daily"], default="auto")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--skip-stale", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(BASE / "output"))
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(BASE / "config.yaml"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    state_path = BASE / STATE_FILE if not (a.demo or a.limit) else out / "scout_state_test.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if a.demo:
        universe, close, volume, bench, fund = data.demo_data(n=120, years=3)
        close_track = close
        fund["industry"] = [f"Industry {i % 15}" for i in range(len(fund))]
        est = estimates.demo(close.columns)
        spy = bench["SPY"]
    else:
        universe, src = (uni.load_extended(cfg, str(BASE))
                         if cfg.get("scout", {}).get("universe", "extended") == "extended"
                         else uni.load_universe(cfg, str(BASE)))
        print(f"[scout] universe {len(universe)} ({src})")
        tickers = universe["ticker"].tolist()[: a.limit or None]
        # also price every past pick (even if it left the index) so the track record has no survivorship
        past = [t for t in track.load(str(BASE / HISTORY_FILE))["ticker"].unique() if t not in tickers]
        close_all, volume = data.fetch_prices(tickers + past + ["SPY"], 3)
        spy = close_all["SPY"]
        close_track = close_all.drop(columns=["SPY"])
        close = close_track.drop(columns=[t for t in past if t in close_track.columns])
        volume = volume.reindex(columns=close.columns)
        if a.skip_stale and (datetime.now(timezone.utc).date() - close.index[-1].date()).days > 3:
            print(f"[scout] latest bar {close.index[-1].date()} is stale -> skip")
            return 0
        print(f"[scout] prices ok ({close.shape[1]} tickers, last {close.index[-1].date()})")

    asof = close.index[-1]
    # weekly = first run that sees a week's final bar (Friday close, or Thursday if Friday is a holiday
    # and the run date is already Fri/Sat UTC); once per ISO week. Robust to late cron starts.
    week = f"{asof.isocalendar().year}-W{asof.isocalendar().week:02d}"
    last_bar_of_week = asof.weekday() == 4 or (asof.weekday() == 3 and datetime.now(timezone.utc).weekday() in (4, 5))
    if a.mode != "auto":
        mode = a.mode
    else:
        mode = "weekly" if last_bar_of_week and state.get("weekly_week") != week else "daily"
    if mode == "daily" and not state.get("watch"):
        print("[scout] no cached watchlist yet -> running a full classification")
    full = mode == "weekly" or not state.get("watch")
    panels = stages.stage_panels(close, volume)
    stage = stages.stage_today(panels)
    dd = panels["dd"].iloc[-1]
    prev = state.get("stage", {})
    last_alert = state.get("last_alert", {})
    good_top = cfg.get("scout", {}).get("good_top_pct", 0.30)

    messages = []
    if full:
        if not a.demo:
            print("[scout] fundamentals (all)...")
            fund = data.fetch_fundamentals(list(close.columns))
            print("[scout] analyst estimates (all)...")
            est = estimates.fetch(list(close.columns))
        cls = sr.classify(fund, est, close.columns, good_top)
        watch = cls[cls["good"] | cls["emerging"]]
        coverage = est["eps_g_ny"].reindex(close.columns).notna().mean()
        print(f"[scout] estimate coverage {coverage:.0%} · watchlist {len(watch)}")
        if coverage >= 0.70 or not state.get("watch"):
            state["watch"] = json.loads(watch.to_json(orient="index"))
            state["watch_date"] = str(asof.date())
            _safe(lambda: refresh_earn_next(state, watch.index, fund, asof), None)
        else:
            print("[scout] coverage too low (Yahoo rate limit?) -> keeping the previous watchlist")
    else:
        cls = pd.DataFrame.from_dict(state["watch"], orient="index")
        cls[["good", "emerging"]] = cls[["good", "emerging"]].astype(bool)
        cand = [t for t in cls.index if stage.get(t) == "pullback" and prev.get(t) != stage.get(t)]
        print(f"[scout] daily: {len(cand)} watchlist stock(s) changed stage -> fetching their data")
        if a.demo:
            fund, est = fund.reindex(cand), est.reindex(cand)
        else:
            fund = data.fetch_fundamentals(cand) if cand else pd.DataFrame(columns=data.INFO_FIELDS)
            est = estimates.fetch(cand) if cand else pd.DataFrame(columns=estimates.FIELDS)
        cls = cls.loc[cand] if cand else cls.iloc[0:0]

    if mode == "weekly":
        groups = stages.group_strength(close, fund["industry"], spy)
        new_pb = {t for t in cls.index if stage.get(t) == "pullback" and t in prev and prev[t] != "pullback"}
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        cal = _safe(lambda: events.calendar_lines(list(cls.index[cls["good"] | cls["emerging"]]), fund, today), [])
        msgs, tops = sr.weekly(asof, spy, close, fund, est, cls, stage, dd, groups, universe, new=new_pb,
                               calendar=cal)
        for t in new_pb:
            last_alert[t] = {"stage": "pullback", "date": str(asof.date())}
        if a.mode == "auto":                       # manual weekly runs must not consume this week's report
            state["weekly_week"] = week
        messages += msgs
        hist_path = BASE / HISTORY_FILE if not (a.demo or a.limit) else out / "scout_history_test.csv"
        try:
            # only automatic weekly runs create cohorts (manual re-runs must not add extra "weeks")
            hist = (track.record(str(hist_path), asof, msgs[-1]["picks"]) if a.mode == "auto" or a.demo
                    else track.load(str(hist_path)))
            messages.append({"text": track.report_text(hist, close_track, spy)})
        except Exception as e:                       # never lose the weekly report because of the record
            print(f"[scout] track record failed: {e!r}")
            messages.append({"text": "📒 ผลงานจริงของ Scout: คำนวณไม่สำเร็จรอบนี้ (จะลองใหม่สัปดาห์หน้า)"})
        scores = pd.DataFrame({"sector": universe.set_index("ticker")["sector"].reindex(close.columns).fillna("Unknown"),
                               "mom_1m": close.iloc[-1] / close.iloc[-22] - 1})
        hm = heatmap.sector_heatmap(scores, fund, str(out / f"heatmap_{asof:%Y-%m-%d}.png"), asof)
        if hm:
            messages.append({"photo": hm, "caption": "🗺 แผนที่ตลาด 1 เดือน · ขนาด = มูลค่าบริษัท · เขียวขึ้น / แดงลง"})
        sec_pe = sector_medians(fund, fund["sector"].fillna("Unknown"), "forwardPE")
        state["sector_pe"] = {k: round(float(v), 2) for k, v in sec_pe.dropna().items()}   # for /check cards
        ev = {} if a.demo else _safe(lambda: events.fetch(tops, pd.Timestamp(asof.date())), {})
        messages += [sr.card(t, cls, fund, est, stage, dd, close, sec_pe, ev.get(t)) for t in tops]
    elif len(cls):
        alert = sr.daily_alerts(asof, cls, fund, est, stage, prev, dd, close, last_alert)
        if alert:
            messages.append(alert)
            for t in alert.get("tickers", []):
                last_alert[t] = {"stage": stage.get(t), "date": str(asof.date())}

    if not a.demo:
        rc = _safe(lambda: earnings_recap(state, asof, close), None)
        if rc:
            messages.append(rc)

    # merge (never drop tickers that failed to download today)
    state["stage"] = {**prev, **{k: v for k, v in stage.items() if k in close.columns and close[k].iloc[-1:].notna().all()}}
    state["date"] = str(asof.date())
    state["last_alert"] = last_alert
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=0, default=str))
    if full:
        table = cls.join(est).join(fund, rsuffix="_f")
        table.insert(0, "stage", stage)
        table.to_csv(out / f"scout_{asof:%Y-%m-%d}.csv")
    for m in messages:
        print(m.get("text") or f"[photo] {m['photo']}", end="\n\n=====\n\n")
    print(f"[scout] mode={mode} · {len(messages)} message(s)")

    if a.notify and messages:
        token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat:
            print("[scout] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
            return 1
        telegram.send(token, chat, messages)
        print(f"[scout] sent {len(messages)} Telegram message(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
