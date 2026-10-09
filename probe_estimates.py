#!/usr/bin/env python
"""Probe which analyst-estimate tables yfinance returns (run in Actions; Yahoo is blocked in the sandbox)."""
import yfinance as yf

ATTRS = ["earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions", "growth_estimates",
         "analyst_price_targets", "quarterly_income_stmt"]
print("yfinance", yf.__version__)
for t in ["AAPL", "NVDA", "PLTR", "SNOW", "XOM"]:
    tk = yf.Ticker(t)
    print(f"\n===== {t}")
    for a in ATTRS:
        try:
            v = getattr(tk, a)
            shape = getattr(v, "shape", None)
            print(f"--- {a}: type={type(v).__name__} shape={shape}")
            if hasattr(v, "to_string"):
                print(v.iloc[:8, :6].to_string() if getattr(v, "ndim", 1) == 2 else v.to_string()[:600])
            else:
                print(str(v)[:400])
        except Exception as e:
            print(f"--- {a}: ERROR {e!r}")
