#!/usr/bin/env python
"""Probe which yfinance tables return data (run in Actions; Yahoo is blocked in the sandbox)."""
import time
import yfinance as yf

ATTRS = ["earnings_history", "earnings_dates", "calendar", "insider_purchases", "insider_transactions",
         "insider_roster_holders", "quarterly_income_stmt"]
print("yfinance", yf.__version__)
for t in ["NVDA", "STX", "GKOS", "TSLA", "MSEX"]:
    tk = yf.Ticker(t)
    print(f"\n===== {t}")
    for a in ATTRS:
        t0 = time.time()
        try:
            v = getattr(tk, a)
            shape = getattr(v, "shape", None)
            print(f"--- {a}: type={type(v).__name__} shape={shape} ({time.time() - t0:.1f}s)")
            if hasattr(v, "to_string"):
                print(v.iloc[:12, :9].to_string() if getattr(v, "ndim", 1) == 2 else v.to_string()[:800])
                if getattr(v, "ndim", 1) == 2:
                    print("columns:", list(v.columns), "index:", type(v.index).__name__, v.index[:3].tolist())
            else:
                print(str(v)[:600])
        except Exception as e:
            print(f"--- {a}: ERROR {e!r}")
    try:
        info = tk.info
        print("info earnings:", {k: info.get(k) for k in ("earningsTimestamp", "earningsTimestampStart",
              "earningsTimestampEnd", "isEarningsDateEstimate", "earningsCallTimestampStart")})
    except Exception as e:
        print("info ERROR", repr(e))
