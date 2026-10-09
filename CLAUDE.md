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

## Design notes
- Only the short score is backtested (price-only, rules fixed from literature, no fitting).
  Mid/long use today's yfinance fundamentals — no point-in-time history → not backtestable yet.
- Backtest uses today's constituents → survivorship bias (disclosed in every report).
- Claude sandbox may block Yahoo/Wikipedia; use `--demo` locally, real runs happen in Actions.
