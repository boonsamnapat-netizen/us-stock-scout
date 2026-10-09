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
