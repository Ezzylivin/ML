"""
test_risk_exits.py — guardrail tests for the money-critical math.

Locks in the properties that MUST hold or real money is at stake:
  1. Position sizing caps a losing trade to ~risk% of the book (the stop works).
  2. trend_ride has NO take-profit cap — it captures far more of a trend than a
     fixed-TP exit (this is the validated edge).
  3. trend_ride exits when the signal flips.
  4. The simulator returns sane, finite metrics.

Run:  pytest test_risk_exits.py -v     (from /root/Project/ML, venv active)
"""
import numpy as np
import pandas as pd
import pytest

from exit_lab import _simulate_exit, EXIT_STYLES


def _frame(closes, atr=1.0, high=None, low=None):
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "close": c,
        "high": c if high is None else np.asarray(high, dtype=float),
        "low":  c if low is None else np.asarray(low, dtype=float),
        "atr":  np.full(n, float(atr)),
    })


def _long_signal(n, enter_at=1, hold=True):
    """votes/weighted that go long at `enter_at` and (optionally) stay long."""
    votes = np.zeros(n); weighted = np.zeros(n)
    for i in range(enter_at, n if hold else enter_at + 1):
        votes[i] = 1.0; weighted[i] = 0.5
    return votes, weighted


def test_stop_caps_loss_to_risk_pct():
    # Enter long at 100, then price gaps straight down through the 2xATR stop (98).
    closes = [100, 100, 97, 97, 97]
    low    = [100, 100, 96, 96, 96]
    df = _frame(closes, atr=1.0, low=low)
    votes, weighted = _long_signal(len(closes), enter_at=1, hold=False)
    m = _simulate_exit(df, votes, weighted, direction="LONG", rule="OR",
                       min_weighted=0.3, risk_pct=1.0, style=EXIT_STYLES["trend_ride"],
                       initial_balance=1000.0)
    loss = 1000.0 - m["final_balance"]
    # Should lose ~1% (=$10) plus a little fee/slippage — NOT 5%+, NOT ~0.
    assert m["total_trades"] == 1
    assert 8.0 <= loss <= 14.0, f"stop did not cap loss near 1%: lost ${loss:.2f}"


def test_trend_ride_has_no_take_profit_cap():
    # A long, strong uptrend where the signal never flips. trend_ride should ride
    # it; a fixed 2:1 take-profit should cap the gain far lower.
    closes = [100] + list(np.linspace(100, 130, 11))  # +30 ATR run, sig stays long
    df = _frame(closes, atr=1.0)
    votes, weighted = _long_signal(len(closes), enter_at=1, hold=True)
    common = dict(direction="LONG", rule="OR", min_weighted=0.3, risk_pct=1.0, initial_balance=1000.0)
    ride = _simulate_exit(df, votes, weighted, style=EXIT_STYLES["trend_ride"], **common)
    capped = _simulate_exit(df, votes, weighted, style=EXIT_STYLES["fixed_2to1"], **common)
    assert ride["roi"] > capped["roi"] + 2.0, (
        f"trend_ride ({ride['roi']}%) should beat capped TP ({capped['roi']}%) in a trend")
    assert ride["roi"] > 0


def test_trend_ride_exits_on_signal_flip():
    # Long entry, then the signal flips short while price is flat -> must close.
    n = 8
    df = _frame([100] * n, atr=1.0)
    votes = np.zeros(n); weighted = np.zeros(n)
    votes[1] = 1.0; weighted[1] = 0.5        # enter long
    votes[4] = -1.0; weighted[4] = -0.5      # flip short -> trend_ride should exit
    m = _simulate_exit(df, votes, weighted, direction="LONG", rule="OR",
                       min_weighted=0.3, risk_pct=1.0, style=EXIT_STYLES["trend_ride"],
                       initial_balance=1000.0)
    assert m["total_trades"] >= 1  # the long was opened and closed on the flip


def test_metrics_are_finite_and_sane():
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 1, 200))
    closes = np.clip(closes, 1, None)
    df = _frame(closes, atr=1.0)
    votes = rng.choice([-1.0, 0.0, 1.0], size=200)
    weighted = votes * 0.5
    m = _simulate_exit(df, votes, weighted, direction="LONG", rule="OR",
                       min_weighted=0.3, risk_pct=1.0, style=EXIT_STYLES["trend_ride"],
                       initial_balance=1000.0)
    for k in ("roi", "final_balance", "win_rate", "profit_factor", "expectancy_r", "max_drawdown"):
        assert np.isfinite(m[k]), f"{k} not finite"
    assert m["final_balance"] > 0
    assert 0.0 <= m["win_rate"] <= 100.0


def test_higher_risk_pct_scales_exposure():
    # Same losing setup at 1% vs 2% risk -> 2% should lose roughly twice as much.
    closes = [100, 100, 97, 97, 97]; low = [100, 100, 96, 96, 96]
    df = _frame(closes, atr=1.0, low=low)
    votes, weighted = _long_signal(len(closes), enter_at=1, hold=False)
    base = dict(direction="LONG", rule="OR", min_weighted=0.3, style=EXIT_STYLES["trend_ride"], initial_balance=1000.0)
    loss1 = 1000.0 - _simulate_exit(df, votes, weighted, risk_pct=1.0, **base)["final_balance"]
    loss2 = 1000.0 - _simulate_exit(df, votes, weighted, risk_pct=2.0, **base)["final_balance"]
    assert loss2 > loss1 * 1.6, f"2% risk should lose much more than 1% ({loss2:.2f} vs {loss1:.2f})"
