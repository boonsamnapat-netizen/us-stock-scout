"""Universe = S&P 500 + Nasdaq-100 (from Wikipedia, falling back to a snapshot)."""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_CSV_URL = ("https://raw.githubusercontent.com/Gary-Strauss/NASDAQ100_Constituents/"
               "master/data/nasdaq100_constituents.csv")
UA = {"User-Agent": "us-stock-scout/1.0 (personal research tool)"}

# yfinance sector names -> GICS sector names
YF_TO_GICS = {
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Basic Materials": "Materials",
}

SECTOR_TH = {
    "Information Technology": "เทคโนโลยีสารสนเทศ",
    "Communication Services": "สื่อสาร/อินเทอร์เน็ต",
    "Consumer Discretionary": "สินค้าฟุ่มเฟือย",
    "Consumer Staples": "สินค้าจำเป็น",
    "Health Care": "สุขภาพ/การแพทย์",
    "Financials": "การเงิน",
    "Industrials": "อุตสาหกรรม",
    "Energy": "พลังงาน",
    "Materials": "วัสดุ/เคมี",
    "Real Estate": "อสังหาฯ",
    "Utilities": "สาธารณูปโภค",
    "Unknown": "ไม่ระบุ",
}


def normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def _read_tables(url: str) -> list[pd.DataFrame]:
    html = requests.get(url, headers=UA, timeout=30).text
    return pd.read_html(StringIO(html))


def _find(tables, required):
    for t in tables:
        if all(c in t.columns for c in required):
            return t
    raise ValueError(f"no table with columns {required}")


def fetch_sp500() -> pd.DataFrame:
    sp = _find(_read_tables(SP500_URL), ["Symbol", "Security", "GICS Sector"])
    sp = sp.rename(columns={"Symbol": "ticker", "Security": "name",
                            "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    return sp[["ticker", "name", "sector", "industry"]].assign(in_sp500=True)


def fetch_ndx() -> pd.DataFrame:
    # Wikipedia no longer has a Nasdaq-100 components table; use a GitHub-maintained CSV.
    # Its sector column is not GICS, so sectors are left empty and filled from yfinance.
    nd = pd.read_csv(StringIO(requests.get(NDX_CSV_URL, headers=UA, timeout=30).text))
    nd = nd.rename(columns={"Ticker": "ticker", "Company": "name"})
    if "ticker" not in nd.columns or not 90 <= len(nd) <= 110:
        raise ValueError(f"unexpected Nasdaq-100 CSV ({len(nd)} rows, {list(nd.columns)})")
    return nd[["ticker", "name"]].assign(in_ndx=True)


def _merge(sp: pd.DataFrame, nd: pd.DataFrame) -> pd.DataFrame:
    sp = sp.assign(ticker=sp["ticker"].map(normalize_ticker))
    nd = nd.assign(ticker=nd["ticker"].map(normalize_ticker))
    u = sp.merge(nd, on="ticker", how="outer", suffixes=("", "_n"))
    for c in ("name", "sector", "industry"):
        if c + "_n" in u.columns:
            u[c] = u[c].fillna(u[c + "_n"])
            u = u.drop(columns=c + "_n")
    u["in_sp500"] = u["in_sp500"].fillna(False).astype(bool)
    u["in_ndx"] = u["in_ndx"].fillna(False).astype(bool)
    return u.sort_values("ticker").reset_index(drop=True)


def load_universe(cfg: dict, base_dir: str = ".") -> tuple[pd.DataFrame, str]:
    """Return (universe, source_label). Each index falls back to the snapshot on its own."""
    ucfg = cfg.get("universe", {})
    snap = pd.read_csv(f"{base_dir}/{ucfg.get('snapshot', 'data/universe.csv')}")
    snap["ticker"] = snap["ticker"].map(normalize_ticker)
    if not ucfg.get("refresh_from_wikipedia", True):
        return snap, "snapshot"
    parts, labels = [], []
    for label, fetch, flag, cols in (
            ("S&P500:wikipedia", fetch_sp500, "in_sp500", ["ticker", "name", "sector", "industry"]),
            ("NDX:github-csv", fetch_ndx, "in_ndx", ["ticker", "name"])):
        try:
            parts.append(fetch())
            labels.append(label)
        except Exception as e:  # network / layout change -> snapshot for this index
            print(f"[universe] {label} failed: {e!r}; using snapshot")
            parts.append(snap.loc[snap[flag], cols].assign(**{flag: True}))
            labels.append(label.split(":")[0] + ":snapshot")
    u = _merge(*parts)
    # keep known GICS sectors from the snapshot for Nasdaq-only names
    known = snap.set_index("ticker")[["sector", "industry"]]
    for c in ("sector", "industry"):
        u[c] = u[c].fillna(u["ticker"].map(known[c]))
    return u, " + ".join(labels)


def fill_sectors(universe: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Fill missing GICS sectors from yfinance's sector field."""
    u = universe.set_index("ticker").copy()
    if "sector" in fundamentals.columns:
        yf_sec = fundamentals["sector"].map(lambda s: YF_TO_GICS.get(s, s))
        u["sector"] = u["sector"].fillna(yf_sec.reindex(u.index))
    if "industry" in fundamentals.columns:
        u["industry"] = u["industry"].fillna(fundamentals["industry"].reindex(u.index))
    u["sector"] = u["sector"].fillna("Unknown")
    return u.reset_index()
