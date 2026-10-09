"""Universe = S&P 500 + Nasdaq-100 (from Wikipedia, falling back to a snapshot)."""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
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


def fetch_from_wikipedia() -> pd.DataFrame:
    sp = _find(_read_tables(SP500_URL), ["Symbol", "Security", "GICS Sector"])
    sp = sp.rename(columns={"Symbol": "ticker", "Security": "name",
                            "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    sp = sp[["ticker", "name", "sector", "industry"]].assign(in_sp500=True)

    nd = None
    for t in _read_tables(NDX_URL):
        tick = next((c for c in ("Ticker", "Symbol") if c in t.columns), None)
        comp = next((c for c in ("Company", "Security") if c in t.columns), None)
        if tick and comp and 90 <= len(t) <= 110:
            nd = t.rename(columns={tick: "ticker", comp: "name"})
            break
    if nd is None:
        raise ValueError("Nasdaq-100 table not found")
    nd = nd.rename(columns={"GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    cols = [c for c in ["ticker", "name", "sector", "industry"] if c in nd.columns]
    nd = nd[cols].assign(in_ndx=True)
    return _merge(sp, nd)


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
    """Return (universe, source_label)."""
    ucfg = cfg.get("universe", {})
    if ucfg.get("refresh_from_wikipedia", True):
        try:
            u = fetch_from_wikipedia()
            if len(u) >= 450:
                return u, "wikipedia"
        except Exception as e:  # network / layout change -> snapshot
            print(f"[universe] wikipedia failed: {e!r}; using snapshot")
    u = pd.read_csv(f"{base_dir}/{ucfg.get('snapshot', 'data/universe.csv')}")
    u["ticker"] = u["ticker"].map(normalize_ticker)
    return u, "snapshot"


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
