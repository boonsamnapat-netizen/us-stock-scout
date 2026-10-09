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
