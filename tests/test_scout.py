import numpy as np
import pandas as pd

from scout import backtest, data, report, scoring, telegram


def test_short_score_has_no_lookahead():
    _, close, _, _, _ = data.demo_data(n=30)
    i = len(close) - 50
    before = scoring.short_score(scoring.snapshot(scoring.price_features(close), i))
    future = close.copy()
    future.iloc[i + 1:] *= np.random.default_rng(0).uniform(0.5, 2, future.iloc[i + 1:].shape)
    after = scoring.short_score(scoring.snapshot(scoring.price_features(future), i))
    pd.testing.assert_series_equal(before, after)


def test_short_score_requires_uptrend_and_prefers_momentum():
    days = pd.bdate_range("2020-01-01", periods=400)
    t = np.arange(400)
    close = pd.DataFrame({
        "UP_FAST": 100 * 1.003 ** t,
        "UP_SLOW": 100 * 1.001 ** t,
        "DOWN": 100 * 0.998 ** t,
    }, index=days)
    s = scoring.short_score(scoring.snapshot(scoring.price_features(close)))
    assert np.isnan(s["DOWN"])            # below 200-day SMA -> excluded
    assert s["UP_FAST"] > s["UP_SLOW"]


def test_long_score_is_sector_relative():
    _, close, _, _, fund = data.demo_data(n=40)
    u, _, _, _, _ = data.demo_data(n=40)
    sc = scoring.score_all(close, fund, u)
    assert sc["long"].between(0, 1).all()
    assert sc["mid"].notna().sum() > 30


def test_backtest_and_report_run():
    u, close, _, bench, fund = data.demo_data(n=40)
    bt = backtest.backtest_short(close, bench["SPY"], fee_per_side_pct=0.1)
    assert bt["stats"]["n"] > 30
    sc = scoring.score_all(close, fund, u)
    sections = report.build_report(sc, fund, u, close, bench, bt, {"report": {"top_n": 5}}, "demo")
    msgs = telegram.pack(sections)
    assert all(len(m) <= telegram.LIMIT for m in msgs)
    assert "ระยะสั้น" in msgs[0]


def test_fee_is_deducted():
    days = pd.bdate_range("2020-01-01", periods=400)
    close = pd.DataFrame({f"T{i}": 100 * 1.0005 ** np.arange(400) for i in range(12)}, index=days)
    bench = close["T0"].rename("SPY")
    bt = backtest.backtest_short(close, bench, top_n=10, fee_per_side_pct=0.5)
    assert np.isclose(bt["stats"]["avg_excess"], -0.01)


def test_split_text_limits():
    text = "\n".join("x" * 100 for _ in range(200))
    chunks = telegram.split_text(text, 1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == "x" * 20000


def _html_balanced(text):
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []

        def handle_starttag(self, tag, attrs):
            assert tag in ("b", "i", "code"), tag
            self.stack.append(tag)

        def handle_endtag(self, tag):
            assert self.stack and self.stack.pop() == tag, tag

    p = P()
    p.feed(text)
    return not p.stack


def test_brief_messages(tmp_path):
    from scout import brief, heatmap
    u, close, _, bench, fund = data.demo_data(n=40)
    bt = backtest.backtest_short(close, bench["SPY"], top_n=5)
    sc = scoring.score_all(close, fund, u)
    hm = heatmap.sector_heatmap(sc, fund, str(tmp_path / "hm.png"), close.index[-1])
    msgs = brief.build(sc, fund, u, close, bench, bt, {"report": {"top_n": 5}}, hm, weekly=True)
    texts = [m["text"] for m in msgs if "text" in m]
    assert any("photo" in m for m in msgs)
    assert len(texts) == 5                      # summary + 3 cards + weekly backtest
    for t in texts:
        assert len(t) <= telegram.LIMIT
        assert _html_balanced(t)
    assert texts[0].count("<code>T") == 15      # 5 per horizon
    tickers = [ln.split("</code>")[0][6:].strip() for ln in texts[0].splitlines() if ln.startswith("<code>")]
    assert len(set(tickers)) == 15              # no stock repeated across horizons


def test_grades():
    from scout.brief import grade
    assert [grade(x) for x in (0.95, 0.7, 0.5, 0.3, 0.1, float("nan"))] == ["A", "B", "C", "D", "F", "–"]


def test_portfolio_sim_respects_caps():
    from scout.portfolio_bt import Rules, simulate
    _, close, _, bench, _ = data.demo_data(n=80, years=4)
    res = simulate(close, bench["SPY"], Rules(gate="nobuy", max_trades_month=5))
    tr = res["trades"]
    assert len(tr) > 0
    per_month = tr.groupby(tr["date"].dt.to_period("M")).size()
    assert per_month.max() <= 5
    held = 0
    for side in tr["side"]:
        held += 1 if side == "buy" else -1
        assert 0 <= held <= 5
    assert 0.99 < res["equity"].iloc[0] <= 1.0   # day-1 buys pay fees


def _uptrend_demo():
    import numpy as np
    u, close, _, bench, fund = data.demo_data(n=60, years=3, seed=3)
    spy = pd.Series(300 * 1.0006 ** np.arange(len(close)), index=close.index)   # always above SMA200
    return u, close, spy, fund


def test_live_model_portfolio_matches_backtest(tmp_path):
    from scout import model_portfolio as mp
    from scout.portfolio_bt import simulate
    u, close, spy, fund = _uptrend_demo()
    cfg = {"model_portfolio": {"max_vol": 0.9}, "fees": {"per_side_pct": 0.1}}
    rules = mp.rules_from_cfg(cfg)
    n, k = len(close), 80
    path = str(tmp_path / "pf.json")
    for i in range(n - k, n):
        mp.update(close.iloc[:i + 1], spy.iloc[:i + 1], cfg, path)
    live = mp.load(path)
    sim = simulate(close, spy, rules, start=n - k)
    assert set(live["state"]["units"]) == set(sim["state"]["units"])
    assert abs(live["state"]["cash"] - sim["state"]["cash"]) < 1e-9
    assert len(live["trades"]) == len(sim["trades"]) > 0
    assert len(live["history"]) == k


def test_model_portfolio_idempotent_and_message(tmp_path):
    from scout import model_portfolio as mp
    u, close, spy, fund = _uptrend_demo()
    cfg = {"model_portfolio": {"max_vol": 0.9}}
    path = str(tmp_path / "pf.json")
    v1 = mp.update(close, spy, cfg, path)
    n_trades = len(mp.load(path)["trades"])
    v2 = mp.update(close, spy, cfg, path)             # same closing date -> no new trades
    assert len(mp.load(path)["trades"]) == n_trades
    assert [t["ticker"] for t in v1["today"]] == [t["ticker"] for t in v2["today"]]
    sc = scoring.score_all(close, fund, u)
    msg = mp.message(v2, sc, spy)
    assert "ซื้อ" in msg and _html_balanced(msg)
    assert 1 <= len(v1["doc"]["state"]["units"]) <= 5


def test_sector_cap():
    from scout.portfolio_bt import Rules, decide_day, new_state
    rank = pd.Series({"A1": 0.01, "A2": 0.02, "A3": 0.03, "B1": 0.04, "C1": 0.05})
    price = pd.Series(100.0, index=rank.index)
    vol = pd.Series(0.2, index=rank.index)
    sectors = pd.Series({"A1": "Energy", "A2": "Energy", "A3": "Energy", "B1": "Tech", "C1": "Health"})
    st = new_state()
    decide_day(st, pd.Timestamp("2026-10-08"), rank, price, vol, True,
               Rules(max_per_sector=2, entry_pct=1.0, max_vol=0.6), sectors)
    assert set(st["units"]) == {"A1", "A2", "B1", "C1"}      # A3 skipped: 3rd Energy name


def _facts(rows):
    return {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": rows}}}}}


def _r(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def test_edgar_quarterly_q4_and_restatements():
    from scout import fundamentals as fu
    rows = [
        _r("2024-01-01", "2024-03-31", 10, "2024-05-01"),
        _r("2024-01-01", "2024-03-31", 99, "2024-08-01"),          # restated later -> ignored
        _r("2024-04-01", "2024-06-30", 20, "2024-08-01"),
        _r("2024-07-01", "2024-09-30", 30, "2024-11-01"),
        _r("2024-01-01", "2024-12-31", 100, "2025-02-20", "10-K"),  # Q4 = 100-60 = 40
    ]
    q = fu.quarterly(_facts(rows), "NetIncomeLoss")
    assert list(q["val"]) == [10, 20, 30, 40]
    assert q["avail"].iloc[0] == pd.Timestamp("2024-05-01")
    assert q["avail"].iloc[3] == pd.Timestamp("2025-02-20")


def test_edgar_panel_has_no_lookahead():
    from scout import fundamentals as fu
    rows = [_r(f"{y}-{m:02d}-01", e, v, f) for y, m, e, v, f in [
        (2023, 1, "2023-03-31", 10, "2023-05-01"), (2023, 4, "2023-06-30", 10, "2023-08-01"),
        (2023, 7, "2023-09-30", 10, "2023-11-01"), (2023, 10, "2023-12-31", 10, "2024-02-15"),
        (2024, 1, "2024-03-31", 15, "2024-05-01")]]
    s = fu.signals_for(fu.quarterly(_facts(rows), "NetIncomeLoss"))
    dates = pd.bdate_range("2024-04-25", "2024-05-10")
    panel = fu.to_panel({"X": s}, "yoy", dates)["X"]
    assert panel.loc["2024-05-01"] != 0.5           # filed that day -> usable only the next day
    assert abs(panel.loc["2024-05-02"] - 0.5) < 1e-9   # Q1'24 vs Q1'23: 15 vs 10 = +50%
    ttm = fu.to_panel({"X": s}, "ttm", dates)["X"]
    assert ttm.loc["2024-05-02"] == 45                 # 10+10+10+15


def test_portfolio_f_only_buys_eligible(tmp_path):
    from scout import model_portfolio as mp
    u, close, spy, fund = _uptrend_demo()
    fund = fund.copy()
    fund["profitMargins"] = 0.1
    fund["earningsQuarterlyGrowth"] = 0.1
    fund["revenueGrowth"] = 0.1
    fund["trailingEps"], fund["forwardEps"] = 1.0, 2.0
    bad = list(close.columns[::2])
    fund.loc[bad, "revenueGrowth"] = -0.1                    # half the universe fails the filter
    elig = mp.fundamental_filter(fund, close.columns)
    assert not elig[bad].any() and elig.drop(bad).all()
    v = mp.update(close, spy, {"model_portfolio": {"max_vol": 0.9}}, str(tmp_path / "f.json"), eligible=elig)
    held = set(v["doc"]["state"]["units"])
    assert held and not held & set(bad)
    v_e = mp.update(close, spy, {"model_portfolio": {"max_vol": 0.9}}, str(tmp_path / "e.json"))
    line = mp.compare_line(v_e, v, spy)
    assert "พอร์ตทดลอง F" in line and _html_balanced(line)


def _series_close(paths: dict) -> pd.DataFrame:
    n = len(next(iter(paths.values())))
    idx = pd.bdate_range(end="2026-10-08", periods=n)
    return pd.DataFrame(paths, index=idx)


def test_stage_detection_breakout_and_pullback():
    import numpy as np
    from scout import stages
    n = 520
    t = np.arange(n)
    # A: slow uptrend, then a flat 6-month base, then a breakout in the last few days
    a = np.concatenate([100 * 1.002 ** t[:350], np.full(165, 100 * 1.002 ** 349), 100 * 1.002 ** 349 * np.array([1.03, 1.05, 1.06, 1.06, 1.07])])
    # B: strong uptrend then a ~15% pullback
    b = np.concatenate([100 * 1.003 ** t[:500], 100 * 1.003 ** 499 * np.linspace(1, 0.85, 20)])
    # C: downtrend
    c = 100 * 0.998 ** t
    close = _series_close({"A": a, "B": b, "C": c})
    st = stages.stage_today(stages.stage_panels(close))
    assert st["A"] == "breakout"
    assert st["B"] == "pullback"
    assert st["C"] == "none"


def test_scout_classify_and_alerts():
    from scout import estimates, scout_report as sr
    import numpy as np
    tick = ["GOOD", "EMRG", "BAD"]
    fund = pd.DataFrame({
        "profitMargins": [0.2, -0.1, 0.1], "revenueGrowth": [0.15, 0.40, -0.05],
        "operatingMargins": [0.25, -0.05, 0.05], "returnOnEquity": [0.3, -0.2, 0.05],
        "sector": ["Tech"] * 3, "industry": ["Software"] * 3, "longName": tick,
        "forwardPE": [25, -50, 12], "targetMeanPrice": [110, 50, 40],
        "earningsTimestamp": [np.nan] * 3, "earningsTimestampStart": [np.nan] * 3,
    }, index=tick)
    est = pd.DataFrame({
        "eps_g_cy": [0.2, 0.5, -0.1], "eps_g_ny": [0.18, 0.9, -0.05], "rev_g_cy": [0.12, 0.4, -0.02],
        "rev_g_ny": [0.10, 0.35, -0.01], "eps_rev_30": [0.01, 0.02, -0.03], "eps_rev_90": [0.05, 0.1, -0.08],
        "up30": [5, 3, 0], "down30": [1, 0, 4], "n_analysts": [20, 10, 8], "eps_ny": [5, 0.3, 1],
        "ni_q0": [1e9, -1e8, 1e8], "ni_q4": [8e8, -3e8, 2e8], "ni_ttm": [4e9, -5e8, 5e8],
    }, index=tick)
    cls = sr.classify(fund, est, tick)
    assert cls.loc["GOOD", "good"] and not cls.loc["GOOD", "emerging"]
    assert cls.loc["EMRG", "emerging"] and not cls.loc["EMRG", "good"]
    assert not cls.loc["BAD", "good"] and not cls.loc["BAD", "emerging"]
    close = _series_close({t: np.linspace(50, 100, 300) for t in tick})
    stage = pd.Series({"GOOD": "breakout", "EMRG": "pullback", "BAD": "breakout"})
    dd = pd.Series({"GOOD": 0.0, "EMRG": -0.15, "BAD": 0.0})
    msg = sr.daily_alerts(close.index[-1], cls, fund, est, stage, {}, dd, close)
    assert "GOOD" in msg["text"] and "EMRG" in msg["text"] and "BAD" not in msg["text"]
    assert _html_balanced(msg["text"])
    # same stages as yesterday -> no alert
    assert sr.daily_alerts(close.index[-1], cls, fund, est, stage, stage.to_dict(), dd, close) is None
    c = sr.card("GOOD", cls, fund, est, stage, dd, close, pd.Series({"Tech": 20.0}))
    assert _html_balanced(c["text"])


def test_scout_main_demo(tmp_path):
    import scout_main
    assert scout_main.main(["--demo", "--mode", "weekly", "--out", str(tmp_path)]) == 0
    assert scout_main.main(["--demo", "--mode", "daily", "--out", str(tmp_path)]) == 0
