"""Point-in-time quarterly fundamentals from SEC EDGAR XBRL (free, no key).

Each quarterly value is "known" only from the date its filing was submitted (earliest filing that
reported it; later restatements ignored) -> no look-ahead in backtests. Q4 is derived as
FY - (Q1+Q2+Q3) and becomes known when the 10-K is filed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "us-stock-scout research 283203097+boonsamnapat-netizen@users.noreply.github.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"}
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
}


def cik_map(session: requests.Session) -> dict[str, int]:
    js = session.get(TICKERS_URL, headers=UA, timeout=30).json()
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in js.values()}


def _rows(facts: dict, concept: str) -> pd.DataFrame:
    node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not node or "USD" not in node.get("units", {}):
        return pd.DataFrame()
    df = pd.DataFrame(node["units"]["USD"])
    if df.empty or "start" not in df:
        return pd.DataFrame()
    df = df[df["form"].isin(FORMS)].dropna(subset=["start", "end", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df[["start", "end", "val", "filed", "days"]]


def quarterly(facts: dict, concept: str) -> pd.DataFrame:
    """Quarterly values: columns end, val, avail (first filing date). Includes derived Q4."""
    df = _rows(facts, concept)
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "avail"])
    first = df.sort_values("filed").drop_duplicates(["start", "end"], keep="first")
    q = first[(first["days"] >= 80) & (first["days"] <= 100)]
    a = first[(first["days"] >= 350) & (first["days"] <= 380)]
    rows = [{"end": r.end, "val": r.val, "avail": r.filed} for r in q.itertuples()]
    have = set(q["end"])
    for r in a.itertuples():
        if r.end in have:
            continue
        inside = q[(q["start"] >= r.start - pd.Timedelta(days=7)) & (q["end"] < r.end)]
        inside = inside.sort_values("end").drop_duplicates("end").tail(3)
        if len(inside) == 3 and (r.end - inside["end"].max()).days < 110:
            rows.append({"end": r.end, "val": r.val - inside["val"].sum(), "avail": r.filed})
    out = pd.DataFrame(rows).sort_values(["end", "avail"]).drop_duplicates("end", keep="first")
    return out.reset_index(drop=True)


def combined(facts: dict, key: str) -> pd.DataFrame:
    """Merge alternative concept names (priority order) into one quarterly series."""
    parts = [quarterly(facts, c) for c in CONCEPTS[key]]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["end", "val", "avail"])
    out = pd.concat(parts).drop_duplicates("end", keep="first").sort_values("end")
    return out.reset_index(drop=True)


def signals_for(q: pd.DataFrame) -> pd.DataFrame:
    """Per quarter (as known on its avail date): YoY growth of the quarter and trailing-12m sum."""
    if q.empty:
        return pd.DataFrame(columns=["avail", "yoy", "ttm"])
    q = q.sort_values("end").reset_index(drop=True)
    yoy, ttm = [], []
    for i, r in q.iterrows():
        prev = q[(q["end"] >= r.end - pd.Timedelta(days=380)) & (q["end"] <= r.end - pd.Timedelta(days=350))]
        p = prev["val"].iloc[-1] if len(prev) else np.nan
        yoy.append((r.val - p) / abs(p) if p and not np.isnan(p) else np.nan)
        last4 = q[(q["end"] > r.end - pd.Timedelta(days=330)) & (q["end"] <= r.end)]   # 4 quarter-ends
        ttm.append(last4["val"].sum() if len(last4) == 4 else np.nan)
    # a quarter's signal is usable once both it and the data it needs were filed
    return pd.DataFrame({"avail": q["avail"], "end": q["end"], "yoy": yoy, "ttm": ttm})


def to_panel(per_ticker: dict[str, pd.DataFrame], col: str, dates: pd.DatetimeIndex,
             max_age_days: int = 200) -> pd.DataFrame:
    """Daily panel: latest known value (filed strictly before the day), NaN if stale."""
    out = {}
    for t, s in per_ticker.items():
        if s.empty:
            continue
        s = s.sort_values(["avail", "end"]).drop_duplicates("avail", keep="last")
        known = pd.Series(s[col].values, index=s["avail"] + pd.Timedelta(days=1))
        age = pd.Series(s["avail"].values, index=known.index)
        v = known.reindex(dates, method="ffill")
        a = age.reindex(dates, method="ffill")
        age_days = (pd.Series(dates, index=dates) - pd.to_datetime(a)).dt.days
        v[age_days > max_age_days] = np.nan
        out[t] = v
    return pd.DataFrame(out, index=dates)


def fetch_all(tickers: list[str], pause: float = 0.12) -> dict[str, dict[str, pd.DataFrame]]:
    """{ticker: {'revenue': signals, 'net_income': signals}} — run where sec.gov is reachable."""
    s = requests.Session()
    ciks = cik_map(s)
    out, missing = {}, []
    for t in tickers:
        cik = ciks.get(t)
        if cik is None:
            missing.append(t)
            continue
        try:
            r = s.get(FACTS_URL.format(cik=cik), headers=UA, timeout=60)
            if r.status_code != 200:
                missing.append(t)
                continue
            facts = r.json()
            out[t] = {k: signals_for(combined(facts, k)) for k in CONCEPTS}
        except Exception as e:  # network / parse
            print(f"[edgar] {t}: {e!r}")
            missing.append(t)
        time.sleep(pause)
    print(f"[edgar] fundamentals for {len(out)}/{len(tickers)} tickers; missing {len(missing)}")
    return out


def panels(data: dict, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    rev = {t: d["revenue"] for t, d in data.items()}
    ni = {t: d["net_income"] for t, d in data.items()}
    return {"rev_yoy": to_panel(rev, "yoy", dates), "ni_yoy": to_panel(ni, "yoy", dates),
            "ni_ttm": to_panel(ni, "ttm", dates)}
