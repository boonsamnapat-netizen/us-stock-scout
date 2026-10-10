#!/usr/bin/env python
"""Probe yfinance insider transaction wording across many tickers (run in Actions; Yahoo is blocked in the sandbox)."""
import collections
import re

import pandas as pd
import yfinance as yf

tickers = pd.read_csv("data/universe_ext.csv")["ticker"].tolist()[::6][:250]
kinds, examples = collections.Counter(), {}
for t in tickers:
    try:
        df = yf.Ticker(t).insider_transactions
    except Exception as e:
        print(t, "ERROR", repr(e)); continue
    if df is None or df.empty:
        continue
    for _, r in df.iterrows():
        txt = str(r.get("Text") or "")
        k = re.split(r" at price", txt)[0].strip() or f"<blank> Transaction={r.get('Transaction')!r}"
        kinds[k] += 1
        examples.setdefault(k, f"{t} {r.get('Start Date')} {r.get('Insider')} {r.get('Position')} sh={r.get('Shares')} val={r.get('Value')} | {txt}")
print(len(tickers), "tickers")
for k, n in kinds.most_common():
    print(f"{n:6d}  {k}  ||  {examples[k]}")
