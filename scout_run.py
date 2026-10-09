#!/usr/bin/env python
"""Daily run: universe -> prices -> fundamentals -> scores -> backtest -> Thai report -> Telegram.

  python scout_run.py --demo                # offline synthetic data, prints report
  python scout_run.py                       # real data (needs internet), prints report
  python scout_run.py --notify              # + send to Telegram (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
  python scout_run.py --notify --skip-stale # skip if no new US trading day (holiday)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from scout import backtest, brief, data, heatmap, model_portfolio as mp, report, scoring, telegram, universe as uni
from scout.portfolio_bt import simulate

BASE = Path(__file__).resolve().parent
PORTFOLIO_FILE = "data/model_portfolio.json"
PORTFOLIO_F_FILE = "data/model_portfolio_f.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="synthetic offline data (not real)")
    ap.add_argument("--notify", action="store_true", help="send to Telegram")
    ap.add_argument("--skip-stale", action="store_true",
                    help="exit quietly if latest bar is not from today/yesterday (US holiday)")
    ap.add_argument("--weekly", action="store_true",
                    help="include the backtest summary (sent automatically on Fridays' data)")
    ap.add_argument("--limit", type=int, default=0, help="only first N tickers (debug)")
    ap.add_argument("--out", default=str(BASE / "output"))
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))

    bench_syms = [cfg.get("benchmark", "SPY")] + cfg.get("extra_benchmarks", [])
    if a.demo:
        universe, close, volume, bench, fund = data.demo_data()
        src = "DEMO ข้อมูลสังเคราะห์ ไม่ใช่ของจริง"
    else:
        universe, src = uni.load_universe(cfg, str(BASE))
        tickers = universe["ticker"].tolist()
        held = [t for f in (PORTFOLIO_FILE, PORTFOLIO_F_FILE)
                for t in (mp.load(str(BASE / f)) or {}).get("state", {}).get("units", {})]
        tickers += [t for t in held if t not in tickers]   # keep pricing names that left the index
        if a.limit:
            tickers = tickers[:a.limit]
            universe = universe[universe["ticker"].isin(tickers)]
        print(f"[run] universe {len(tickers)} from {src}; downloading prices...")
        close_all, volume = data.fetch_prices(tickers + bench_syms, cfg.get("history_years", 6))
        bench = close_all[[b for b in bench_syms if b in close_all.columns]]
        close = close_all.drop(columns=[b for b in bench_syms if b in close_all.columns and b not in tickers])
        if a.skip_stale:
            last = close.index[-1].date()
            age = (datetime.now(timezone.utc).date() - last).days
            if age > 1:
                print(f"[run] latest bar {last} is {age} days old (US market closed) -> skip")
                return 0
        print(f"[run] prices ok ({close.shape[1]} tickers, last {close.index[-1].date()}); fundamentals...")
        fund = data.fetch_fundamentals(list(close.columns))
        universe = uni.fill_sectors(universe, fund)

    scores = scoring.score_all(close, fund, universe, cfg.get("report", {}).get("min_analysts", 5))
    btc = cfg.get("backtest", {})
    bt = backtest.backtest_short(close, bench[bench_syms[0]], cfg.get("short", {}).get("horizon_days", 20),
                                 btc.get("top_n", 10), cfg.get("fees", {}).get("per_side_pct", 0.10),
                                 btc.get("min_history_days", 273))
    sections = report.build_report(scores, fund, universe, close, bench, bt, cfg, src)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = close.index[-1].strftime("%Y-%m-%d")
    (out / f"report_full_{stamp}.txt").write_text("\n\n".join(sections), encoding="utf-8")
    scores.join(fund, rsuffix="_f").to_csv(out / f"scores_{stamp}.csv")
    bt["periods"].to_csv(out / f"backtest_short_{stamp}.csv", index=False)
    hm = heatmap.sector_heatmap(scores, fund, str(out / f"heatmap_{stamp}.png"), close.index[-1])
    weekly = a.weekly or close.index[-1].weekday() == 4   # Friday close -> Saturday morning TH
    spy = bench[bench_syms[0]]

    # --- model portfolio (only full real runs may advance the real state file)
    pf_path = str(BASE / PORTFOLIO_FILE) if not (a.demo or a.limit) else str(out / "model_portfolio_test.json")
    view = mp.update(close, spy, cfg, pf_path)
    pf_f_path = str(BASE / PORTFOLIO_F_FILE) if not (a.demo or a.limit) else str(out / "model_portfolio_f_test.json")
    view_f = mp.update(close, spy, cfg, pf_f_path, eligible=mp.fundamental_filter(fund, close.columns))
    messages = [{"text": mp.message(view, scores, spy) + "\n\n" + mp.compare_line(view, view_f, spy)}]
    if hm:
        messages.append({"photo": hm, "caption": "🗺 แผนที่ตลาด 1 เดือน · ขนาด = มูลค่าบริษัท · เขียวขึ้น / แดงลง"})
    buys = [t["ticker"] for t in view["today"] if t["side"] == "buy"]
    messages += brief.cards_for(buys, "🟢 ซื้อใหม่:", scores, fund, universe, close, cfg)
    if weekly:
        res = simulate(close, spy, view["rules"])
        messages.append({"text": brief.model_backtest_text(res["stats"], view["rules"])})
    (out / f"telegram_{stamp}.txt").write_text(
        "\n\n=====\n\n".join(m.get("text") or f"[photo] {m['photo']}" for m in messages), encoding="utf-8")
    for m in messages:
        print(m.get("text") or f"[photo] {m['photo']} — {m.get('caption', '')}", end="\n\n=====\n\n")

    if a.notify:
        token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat:
            print("[run] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
            return 1
        telegram.send(token, chat, messages)
        print(f"[run] sent {len(messages)} Telegram messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
