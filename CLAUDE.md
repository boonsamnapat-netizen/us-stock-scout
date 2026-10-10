# CLAUDE.md — US Stock Scout

Daily screener for S&P 500 + Nasdaq-100: short (~4 weeks) / mid (6–18 mo) / long (1–5 y)
scores, per-sector leaders, real-history stats, Thai Telegram report via GitHub Actions.
Decision-support only; no profit guarantee — always quote honest, net-of-fees numbers and biases.

## Working rule — never guess (owner's standing instruction)
- Do NOT state guesses, predictions, or inferences as if they were findings. If it was not
  stated and cannot be verified, say "ไม่ได้บอกมา" / "ไม่รู้" and ask.
- Code in this repo is NOT evidence of what the owner knows (much was written by Claude).
- Need information to proceed? Ask for it.

## Facts / decisions from the owner
- Universe: S&P 500 + Nasdaq-100. Short term = 4 weeks. Broker: Dime (TH). Report: Telegram,
  every US trading day, its own bot/chat (secrets TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
- NOT yet provided: real Dime fee (config `fees.per_side_pct` is an ASSUMPTION), budget,
  risk tolerance, which tickers Dime actually lists.

- 2026-10-09: owner PAUSED everything (daily schedule commented out). Model portfolios E3/F are
  stopped — do NOT tell the owner to trade them. New direction = the original goal: a **stock scout**
  (good fundamentals, 1–3y outlook, early breakout / pullback-in-uptrend, emerging industry themes),
  owner decides trades himself.
- (Previous, paused) Product = **model portfolio** (owner chose, Alpha-Picks style): 5 stocks, rules E3, daily Telegram
  "what to do today". Owner: drawdown tolerance ≤ −25%, Dime free package = 5 trades/month.
  Real Dime FX/fee cost NOT known (owner confirmed 2026-10-09: keep 0.10%/side assumption).

## Stock Scout (current product, 2026-10-09)
- `scout_main.py` + `scout/{estimates,stages,scout_report}.py`, workflow `scout.yml` (weekday 23:15 UTC):
  First run seeing a week's final bar → weekly report (once per ISO week, `weekly_week`); other runs →
  alerts from the cached watchlist (state `data/scout_state.json`); watchlist not overwritten if estimate
  coverage < 70%.
- 2026-10-09 (owner approved): NO 🚀 breakout section — research_stages (1,518 stocks, vs equal-weight
  universe, costs, NW t) showed breakouts underperform (12m −7…−9%, t≈−2), pullbacks ≈ noise (+0.7%)
  but beat chasing uptrends (~+7%). Weekly = 🔥 groups, 🔄 good+pullback, ✅ top good, 🌱 emerging;
  daily alerts only for watchlist stocks newly entering 🔄. Every report carries the HONESTY note.
- Track record (owner asked, 2026-10-09): weekly picks appended to `data/scout_history.csv`
  (`scout/track.py`); weekly report shows returns from the first close AFTER the report vs SPY and the
  equal-weight universe. This forward record is the real test of the fundamentals/estimates parts.
- 2026-10-10 (owner chose "group 2"): cards (weekly top-3 + /check) show earnings (next date from
  Yahoo, last EPS vs estimate, price move after the report, beat count), insider open-market buys/sales
  (Form 4 'Purchase'/'Sale' only; awards/gifts/exercises ignored) and price-history risk (`scout/events.py`,
  `scout/risk.py`). Weekly report lists watchlist names reporting within 7 days. Every run recaps watchlist
  reports once (`earnings_recap`, state `earn_next`/`earn_done`, 10-day window). yfinance formats verified
  by probe_estimates.py in Actions.
- `/check TICKER` bot (`bot.py`, workflow `bot.yml`, cron */15): owner's chat only; offset in
  `data/bot_state.json` saved after replying; any US ticker; shows Scout gate checklist.
- "Good" = profitable TTM + next-year EPS & revenue growth expected + estimates not cut.
  "🌱 Emerging" (owner asked Claude to decide): unprofitable but revenue ≥ +25% YoY, losses narrowing,
  analysts expect profit next year — always flagged high risk.
- Analyst estimates come from yfinance (earnings_estimate, revenue_estimate, eps_trend, eps_revisions,
  quarterly_income_stmt) — today only, not backtestable. Chart stages are backtested in `research_stages.py`.

## Design notes
- `portfolio_bt.decide_day` is the single rule engine for BOTH backtest and live portfolio
  (test asserts live day-by-day == simulate). Change rules only there + config `model_portfolio`.
- Paper portfolio **F** (owner chose, forward test only): E3 + entry filter `model_portfolio.fundamental_filter`
  (profitable, quarterly NI & revenue YoY > 0, forwardEps > trailingEps; yfinance today). State
  `data/model_portfolio_f.json`. Owner trades E3 only. SEC EDGAR blocks GitHub Actions → no fundamental backtest.
- State: `data/model_portfolio.json`, committed by the daily workflow; `--limit`/`--demo` runs never touch it.
- Only the short score is backtested (price-only, rules fixed from literature, no fitting).
  Mid/long use today's yfinance fundamentals — no point-in-time history → not backtestable yet.
- Backtest uses today's constituents → survivorship bias (disclosed in every report).
- Claude sandbox may block Yahoo/Wikipedia; use `--demo` locally, real runs happen in Actions.
