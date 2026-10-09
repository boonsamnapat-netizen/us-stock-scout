"""Market data: daily prices + fundamentals/analyst fields (yfinance), and an offline demo."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

INFO_FIELDS = [
    "longName", "sector", "industry", "marketCap", "currentPrice",
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnEquity", "returnOnAssets", "debtToEquity", "freeCashflow",
    "targetMeanPrice", "targetLowPrice", "targetHighPrice",
    "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
    "dividendYield", "beta", "earningsTimestamp", "earningsTimestampStart",
]


def fetch_prices(tickers: list[str], years: int, chunk: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted daily close + volume, columns = tickers."""
    import yfinance as yf

    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        df = yf.download(part, period=f"{years}y", interval="1d", auto_adjust=True,
                         group_by="column", progress=False, threads=True)
        if df is None or df.empty:
            continue
        c, v = df["Close"], df["Volume"]
        if isinstance(c, pd.Series):  # single ticker
            c, v = c.to_frame(part[0]), v.to_frame(part[0])
        closes.append(c)
        vols.append(v)
    close = pd.concat(closes, axis=1).sort_index()
    vol = pd.concat(vols, axis=1).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    close = close.dropna(axis=1, how="all")
    return close, vol.reindex(columns=close.columns)


def fetch_fundamentals(tickers: list[str], pause: float = 0.25, retries: int = 2) -> pd.DataFrame:
    """yfinance .info subset per ticker. Missing tickers come back as NaN rows."""
    import yfinance as yf

    rows = {}
    for t in tickers:
        info = {}
        for attempt in range(retries + 1):
            try:
                info = yf.Ticker(t).info or {}
                break
            except Exception as e:
                if attempt == retries:
                    print(f"[fundamentals] {t}: {e!r}")
                time.sleep(2 * (attempt + 1))
        rows[t] = {k: info.get(k) for k in INFO_FIELDS}
        time.sleep(pause)
    df = pd.DataFrame.from_dict(rows, orient="index")
    num = [c for c in INFO_FIELDS if c not in ("longName", "sector", "industry", "recommendationKey")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------- offline demo
def demo_data(n: int = 60, years: int = 6, seed: int = 7):
    """Synthetic universe/prices/fundamentals for offline tests. NOT real data."""
    rng = np.random.default_rng(seed)
    sectors = ["Information Technology", "Health Care", "Financials", "Energy",
               "Consumer Discretionary", "Industrials"]
    tickers = [f"T{i:03d}" for i in range(n)]
    days = pd.bdate_range(end=pd.Timestamp("2026-10-08"), periods=252 * years)
    drift = rng.normal(0.0004, 0.0004, n)
    vol = rng.uniform(0.01, 0.03, n)
    rets = rng.normal(drift, vol, (len(days), n))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=tickers)
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (len(days), n)), index=days, columns=tickers)
    bench_r = rng.normal(0.0003, 0.01, len(days))
    bench = pd.DataFrame({"SPY": 400 * np.exp(np.cumsum(bench_r)),
                          "QQQ": 300 * np.exp(np.cumsum(bench_r * 1.2))}, index=days)
    universe = pd.DataFrame({
        "ticker": tickers, "name": [f"Demo Co {t}" for t in tickers],
        "sector": [sectors[i % len(sectors)] for i in range(n)], "industry": "Demo",
        "in_sp500": True, "in_ndx": [i % 5 == 0 for i in range(n)],
    })
    price = close.iloc[-1]
    fund = pd.DataFrame({
        "longName": universe["name"].values, "sector": universe["sector"].values,
        "industry": "Demo", "marketCap": rng.uniform(1e10, 1e12, n), "currentPrice": price.values,
        "trailingPE": rng.uniform(-10, 60, n), "forwardPE": rng.uniform(-5, 50, n),
        "priceToSalesTrailing12Months": rng.uniform(1, 15, n),
        "revenueGrowth": rng.normal(0.08, 0.1, n), "earningsGrowth": rng.normal(0.1, 0.3, n),
        "earningsQuarterlyGrowth": rng.normal(0.1, 0.3, n),
        "grossMargins": rng.uniform(0.2, 0.8, n), "operatingMargins": rng.uniform(0, 0.4, n),
        "profitMargins": rng.uniform(-0.05, 0.3, n), "returnOnEquity": rng.normal(0.15, 0.1, n),
        "returnOnAssets": rng.normal(0.06, 0.04, n), "debtToEquity": rng.uniform(0, 250, n),
        "freeCashflow": rng.uniform(-1e9, 2e10, n),
        "targetMeanPrice": price.values * rng.uniform(0.85, 1.3, n),
        "targetLowPrice": price.values * 0.8, "targetHighPrice": price.values * 1.5,
        "recommendationMean": rng.uniform(1.5, 3.2, n), "recommendationKey": "buy",
        "numberOfAnalystOpinions": rng.integers(2, 40, n),
        "dividendYield": rng.uniform(0, 3, n), "beta": rng.uniform(0.5, 2, n),
        "earningsTimestamp": (days[-1] + pd.to_timedelta(rng.integers(1, 90, n), "D")).astype("int64") // 10**9,
        "earningsTimestampStart": np.nan,
    }, index=tickers)
    return universe, close, volume, bench, fund
