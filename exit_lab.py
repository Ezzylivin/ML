"""
exit_lab.py — exit / risk / position-sizing laboratory.

WHY THIS EXISTS
---------------
Every edge_discovery config shared ONE hardcoded exit (3xATR TP, 1.5xATR SL,
trailing = SL distance, always on). We optimized *entries* across hundreds of
combos and never moved a single exit knob. Pros will tell you entries are ~20%
of the result and exits + sizing are ~80%. This module tests that 80%.

It reuses the PROVEN pieces:
  - entry signal      -> Backtester._strategy_votes (identical to the live bot)
  - costs             -> taker fee + slippage + funding (identical to _simulate)
  - features          -> apply_mega_features (identical to trainer + live)
  - data + folds      -> edge_discovery._load_tf + walk-forward slicing

...and opens up the exit dimension with named EXIT STYLES:
  fixed TP:R          classic fixed take-profit / stop
  trail-only          no fixed target; ride a trailing ATR stop
  breakeven-runner    move stop to breakeven after +1R, then trail (let winners run)
  partial-runner      scale out half at +2R, move rest to BE, trail the remainder
  time-momentum       trail + a hard time stop (momentum decays -> cut it)
  trend-ride          hold until the entry signal flips (pure trend capture)

PLUS richer risk metrics that a ROI number hides: expectancy in R, profit
factor, max drawdown, average R, and — critically — CROSS-COIN GENERALIZATION.
A style only "wins" if it clears the bar on MULTIPLE coins, not one. That single
rule is what separates a real effect from the per-coin overfit that has killed
every candidate so far.

Endpoints (wired in main4.py):
  POST /api/exitlab/run        -> optimize_exits(...)  (writes exit_lab_results.json)
  GET  /api/exitlab/results    -> last results
  POST /api/exitlab/portfolio  -> portfolio_test(...)  (many-coin combined equity)

Run standalone:  python exit_lab.py
"""
import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config2 import DATA_DIR, DEFAULT_TAKER_FEE
from app.backtest2 import Backtester
from app.verify.engineer_and_train import apply_mega_features
from edge_discovery import _load_tf, _attach_btc

log = logging.getLogger("exit_lab")
OUT_PATH = os.path.join(DATA_DIR, "exit_lab_results.json")


def _fmt_ts(ix):
    """Best-effort ISO string for a bar index value (Timestamp or anything)."""
    try:
        return ix.isoformat()
    except Exception:
        return str(ix)


# ---- validation eligibility registry -------------------------------------
# The bridge between the Strategy Lab's hard validation and the LIVE fleet: a
# config may only be used live on the coins that SURVIVED both the out-of-sample
# holdout AND the cost-stress test. validate_exit() writes this; fleet_start reads
# it to decide which coins may pyramid. A coin that didn't clear falls back to the
# safe single-leg behavior.
ELIG_PATH = os.path.join(DATA_DIR, "exit_validation.json")


def _elig_key(timeframe, entry, direction, style, max_legs):
    return f"{timeframe}:{entry}:{direction}:{style}:legs{int(max_legs)}"


def all_eligibility():
    """The whole validation registry (every validated config -> cleared coins)."""
    try:
        if not os.path.exists(ELIG_PATH):
            return {}
        with open(ELIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def load_eligibility(timeframe="4h", entry="regime", direction="LONG", style="trend_ride", max_legs=1):
    """Coins CLEARED (survived holdout + cost stress) for this exact config, from
    the persisted registry. [] if the config was never validated."""
    rec = all_eligibility().get(_elig_key(timeframe, entry, direction, style, max_legs))
    return list(rec.get("cleared_coins", [])) if rec else []

# ---- universe -------------------------------------------------------------
SYMBOLS    = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]
TIMEFRAMES = ["4h", "1d"]          # 1h is fee-dominated; focus where survival is plausible
DIRECTIONS = ["LONG", "BOTH"]

# ---- ENTRY signals (deliberately simple + robust; we are testing EXITS) ---
# Each entry is a small strategy set fed to the SAME vectorized StrategyBrain the
# live bot uses, so entries stay honest while we vary the exit.
ENTRIES = {
    "trend":    [{"code": "supertrend"}, {"code": "ema_cloud"}],   # trend-follow
    "momentum": [{"code": "macd_crossover"}],                       # momentum
    "regime":   [{"code": "ema_cloud"}, {"code": "btc_regime"}],    # trend + only when BTC risk-on
}

# ---- EXIT STYLES — the whole point of this lab ----------------------------
# Keys:
#   tp_atr       fixed take-profit distance in ATRs (None = no fixed target)
#   sl_atr       initial stop distance in ATRs
#   trail_atr    trailing-stop distance in ATRs (None = no trailing)
#   be_atr       move stop to breakeven once price is +be_atr ATR in favor (None=off)
#   ptp_atr      partial take-profit distance in ATRs (None = no scale-out)
#   ptp_frac     fraction of the position to close at ptp_atr (0..1)
#   time_bars    force-close after this many bars held (None = off)
#   trend_exit   True = also exit when the entry signal flips against the position
EXIT_STYLES = {
    "fixed_2to1":     dict(tp_atr=3.0, sl_atr=1.5, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=False),
    "fixed_3to1":     dict(tp_atr=4.5, sl_atr=1.5, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=False),
    "trail_tight":    dict(tp_atr=None, sl_atr=2.0, trail_atr=2.0, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=False),
    "trail_wide":     dict(tp_atr=None, sl_atr=3.0, trail_atr=3.0, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=False),
    "be_runner":      dict(tp_atr=None, sl_atr=1.5, trail_atr=2.5, be_atr=1.0, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=False),
    "partial_runner": dict(tp_atr=None, sl_atr=1.5, trail_atr=3.0, be_atr=None, ptp_atr=2.0, ptp_frac=0.5, time_bars=None, trend_exit=False),
    "time_momentum":  dict(tp_atr=None, sl_atr=2.0, trail_atr=2.5, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=10, trend_exit=False),
    "trend_ride":     dict(tp_atr=None, sl_atr=2.0, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    # ---- finer sweep around the trend_ride winner (stop distance + loose trail) ----
    "tr_sl15":        dict(tp_atr=None, sl_atr=1.5, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    "tr_sl25":        dict(tp_atr=None, sl_atr=2.5, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    "tr_sl30":        dict(tp_atr=None, sl_atr=3.0, trail_atr=None, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    "tr_trail3":      dict(tp_atr=None, sl_atr=2.0, trail_atr=3.0, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    "tr_trail4":      dict(tp_atr=None, sl_atr=2.0, trail_atr=4.0, be_atr=None, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
    "tr_be1":         dict(tp_atr=None, sl_atr=2.0, trail_atr=None, be_atr=1.5, ptp_atr=None, ptp_frac=0.0, time_bars=None, trend_exit=True),
}

# ---- walk-forward / edge bar ---------------------------------------------
FOLDS                = 6
MIN_ROWS             = 250
MIN_TRADES_PER_FOLD  = 3
MIN_PROFITABLE_FOLDS = 4
# generalization: a (entry,tf,dir,style) config must clear the per-coin bar on at
# least this many coins to count as a real, non-overfit effect.
MIN_COINS_GENERALIZE = 3

# ── Hardened "survives" bar (per coin, per holdout run) ──────────────────
# A coin counts as surviving a holdout only if it cleared a MEANINGFUL bar, not
# merely ROI > 0 on a couple of lucky trades (the old test). These are module
# globals so a recalibration run can tighten them AT WILL without code changes
# (see main4 /api/fleet/recalibrate):
#   MIN_HOLDOUT_TRADES     - enough closed trades for the result to mean anything
#   MIN_SURV_EXPECTANCY_R  - positive edge WITH margin above breakeven (in R)
#   MIN_SURV_PROFIT_FACTOR - gross wins / gross losses must clear > 1 with margin
MIN_HOLDOUT_TRADES     = 6
MIN_SURV_EXPECTANCY_R  = 0.02
MIN_SURV_PROFIT_FACTOR = 1.05


# ==========================================================================
# core simulator — same costs as _simulate, fully parameterized exits
# ==========================================================================
def _simulate_exit(df, votes, weighted, *, direction, rule, min_weighted,
                   risk_pct, style, cooldown_bars=0, initial_balance=1000.0,
                   fee_override=None, slip_bps=None, return_series=False):
    """Faithful single-slice simulation with a fully parameterized exit engine.

    Entry = the live StrategyBrain signal (votes/weighted already computed for the
    whole frame). Exit = the EXIT_STYLES bundle in `style`. Costs = taker fee +
    slippage + funding, identical to Backtester._simulate. Returns a metrics dict
    including expectancy in R, profit factor, max drawdown and average R.

    fee_override / slip_bps let a stress test rerun with harsher costs; both
    default to the standard taker fee and 5 bps slippage."""
    tp_atr    = style.get("tp_atr")
    sl_atr    = float(style.get("sl_atr", 1.5))
    trail_atr = style.get("trail_atr")
    be_atr    = style.get("be_atr")
    ptp_atr   = style.get("ptp_atr")
    ptp_frac  = float(style.get("ptp_frac", 0.0) or 0.0)
    time_bars = style.get("time_bars")
    trend_exit = bool(style.get("trend_exit", False))

    fee_rate  = float(DEFAULT_TAKER_FEE if fee_override is None else fee_override)
    slip      = (5.0 if slip_bps is None else float(slip_bps)) / 10000.0  # bps adverse fill
    funding_pb = 0.00005                    # per-bar carry on shorts/margin
    is_margin = (direction in ("BOTH", "SHORT"))

    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    idx    = df.index
    atr_a  = df["atr"].values.astype(float) if "atr" in df.columns else (closes * 0.01)

    balance = float(initial_balance)
    position = None
    entries = 0
    trade_pnls = []      # net $ per closed trade (sum of partial + final)
    trade_rs   = []      # R multiple per closed trade
    equity = [balance]
    equity_ts = [{"ts": _fmt_ts(idx[0]), "equity": round(float(balance), 2)}] if (return_series and len(idx)) else []
    trade_log = []       # per-trade detail, only populated when return_series=True
    signal_has_reset = True
    last_exit_bar = -10**9

    def _sig(i):
        v = votes[i]; wv = weighted[i]
        n_strats = max(1, int(round(np.nanmax(np.abs(votes)) if len(votes) else 1)))
        if rule == "AND":
            if v >= n_strats and wv > 0:    return 1
            if v <= -n_strats and wv < 0:   return -1
            return 0
        if v > 0 and wv >= min_weighted:    return 1
        if v < 0 and wv <= -min_weighted:   return -1
        return 0

    def _close(pos, size, fill_price, bar_i, tag, bank):
        """Realize `size` units of `pos` at fill_price; return net $ and add to bank."""
        if pos["type"] == "long":
            gross = (fill_price - pos["entry"]) * size
        else:
            gross = (pos["entry"] - fill_price) * size
        fee = abs(size * fill_price) * fee_rate
        held = max(0, bar_i - pos["entry_bar"])
        fund = (abs(size * pos["entry"]) * funding_pb * held) if (pos["type"] == "short" or is_margin) else 0.0
        net = gross - fee - fund
        bank["net"] += net
        return net

    for i in range(len(df)):
        price = closes[i]

        if position is not None:
            atr0 = position["entry_atr"]
            # --- trailing stop update ---
            if trail_atr is not None and atr0 > 0:
                if position["type"] == "long":
                    position["sl"] = max(position["sl"], price - trail_atr * atr0)
                else:
                    position["sl"] = min(position["sl"], price + trail_atr * atr0)
            # --- breakeven bump ---
            if be_atr is not None and not position["be_done"] and atr0 > 0:
                if position["type"] == "long" and highs[i] >= position["entry"] + be_atr * atr0:
                    position["sl"] = max(position["sl"], position["entry"]); position["be_done"] = True
                elif position["type"] == "short" and lows[i] <= position["entry"] - be_atr * atr0:
                    position["sl"] = min(position["sl"], position["entry"]); position["be_done"] = True
            # --- partial take-profit (scale out, move remainder to BE) ---
            if ptp_atr is not None and not position["ptp_done"] and ptp_frac > 0 and atr0 > 0:
                tgt = (position["entry"] + ptp_atr * atr0) if position["type"] == "long" else (position["entry"] - ptp_atr * atr0)
                hit = (highs[i] >= tgt) if position["type"] == "long" else (lows[i] <= tgt)
                if hit:
                    fill = tgt * (1 - slip) if position["type"] == "long" else tgt * (1 + slip)
                    part = position["size"] * ptp_frac
                    _close(position, part, fill, i, "PTP", position["bank"])
                    position["size"] -= part
                    position["sl"] = (max(position["sl"], position["entry"]) if position["type"] == "long"
                                      else min(position["sl"], position["entry"]))
                    position["ptp_done"] = True

            # --- exit checks for the remainder ---
            exit_price = None; reason = None
            hit_sl = (lows[i] <= position["sl"]) if position["type"] == "long" else (highs[i] >= position["sl"])
            hit_tp = False
            if tp_atr is not None:
                hit_tp = (highs[i] >= position["tp"]) if position["type"] == "long" else (lows[i] <= position["tp"])
            # conservative intrabar ordering: stop assumed first
            if hit_sl:
                exit_price, reason = position["sl"], "SL"
            elif hit_tp:
                exit_price, reason = position["tp"], "TP"
            elif time_bars is not None and (i - position["entry_bar"]) >= int(time_bars):
                exit_price, reason = price, "TIME"
            elif trend_exit:
                s = _sig(i)
                if (position["type"] == "long" and s < 0) or (position["type"] == "short" and s > 0):
                    exit_price, reason = price, "FLIP"

            if exit_price is not None:
                fill = exit_price * (1 - slip) if position["type"] == "long" else exit_price * (1 + slip)
                _close(position, position["size"], fill, i, reason, position["bank"])
                net = position["bank"]["net"]
                balance += net
                trade_pnls.append(net)
                risk0 = position["risk0"]
                trade_rs.append(net / risk0 if risk0 > 0 else 0.0)
                equity.append(balance)
                if return_series:
                    trade_log.append({
                        "entry_ts": _fmt_ts(idx[position["entry_bar"]]),
                        "exit_ts": _fmt_ts(idx[i]),
                        "direction": position["type"],
                        "entry_price": round(float(position["entry"]), 6),
                        "exit_price": round(float(fill), 6),
                        "pnl": round(float(net), 2),
                        "r": round(float(net / risk0), 3) if risk0 > 0 else 0.0,
                        "reason": reason,
                    })
                    equity_ts.append({"ts": _fmt_ts(idx[i]), "equity": round(float(balance), 2)})
                position = None
                last_exit_bar = i

        if position is None:
            s = _sig(i)
            if s == 1 and direction == "SHORT": s = 0
            if s == -1 and direction == "LONG": s = 0
            if s == 0:
                signal_has_reset = True
            can_enter = signal_has_reset and (i - last_exit_bar) >= cooldown_bars
            atr = atr_a[i]
            if s != 0 and can_enter and atr > 0 and np.isfinite(atr):
                stop_dist = atr * sl_atr
                size = (balance * (risk_pct / 100.0)) / stop_dist if stop_dist > 0 else 0.0
                notional = size * price
                if notional > balance:            # spot: no leverage
                    size = balance / price; notional = size * price
                if size > 0:
                    entry_fill = price * (1 + slip) if s == 1 else price * (1 - slip)
                    balance -= notional * fee_rate
                    tp = None
                    if tp_atr is not None:
                        tp = entry_fill + atr * tp_atr if s == 1 else entry_fill - atr * tp_atr
                    sl = entry_fill - atr * sl_atr if s == 1 else entry_fill + atr * sl_atr
                    position = {
                        "type": "long" if s == 1 else "short", "entry": entry_fill,
                        "size": size, "tp": tp, "sl": sl, "entry_bar": i, "entry_atr": atr,
                        "be_done": False, "ptp_done": False,
                        "risk0": stop_dist * size,          # initial $ risk -> defines 1R
                        "bank": {"net": 0.0},
                    }
                    entries += 1
                    signal_has_reset = False

    # close any open position at the last bar
    if position is not None:
        fill = closes[-1] * (1 - slip) if position["type"] == "long" else closes[-1] * (1 + slip)
        _close(position, position["size"], fill, len(df) - 1, "EOD", position["bank"])
        net = position["bank"]["net"]
        balance += net
        trade_pnls.append(net)
        risk0 = position["risk0"]
        trade_rs.append(net / risk0 if risk0 > 0 else 0.0)
        equity.append(balance)
        if return_series:
            trade_log.append({
                "entry_ts": _fmt_ts(idx[position["entry_bar"]]),
                "exit_ts": _fmt_ts(idx[len(df) - 1]),
                "direction": position["type"],
                "entry_price": round(float(position["entry"]), 6),
                "exit_price": round(float(fill), 6),
                "pnl": round(float(net), 2),
                "r": round(float(net / risk0), 3) if risk0 > 0 else 0.0,
                "reason": "EOD",
            })
            equity_ts.append({"ts": _fmt_ts(idx[len(df) - 1]), "equity": round(float(balance), 2)})

    roi = ((balance - initial_balance) / initial_balance) * 100.0
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = round(100.0 * len(wins) / len(trade_pnls), 1) if trade_pnls else 0.0
    gross_win = sum(wins); gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy_r = round(float(np.mean(trade_rs)), 3) if trade_rs else 0.0
    avg_r = expectancy_r
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    max_dd = round(float(np.max((peak - eq) / peak) * 100.0), 1) if len(eq) else 0.0
    out = {
        "roi": round(roi, 2), "final_balance": round(balance, 2),
        "total_trades": entries, "win_rate": win_rate,
        "profit_factor": profit_factor if np.isfinite(profit_factor) else 99.0,
        "expectancy_r": expectancy_r, "avg_r": avg_r, "max_drawdown": max_dd,
    }
    if return_series:
        out["equity"] = equity_ts
        out["trades"] = trade_log
    return out


# ==========================================================================
# PYRAMIDING variant — holds up to max_legs concurrent legs, adding ONLY in a
# confirmed trend (never in chop). Separate from _simulate_exit so the validated
# single-leg math stays byte-for-byte identical; this runs only when max_legs > 1.
# ==========================================================================
def _simulate_exit_pyr(df, votes, weighted, *, direction, rule, min_weighted,
                       risk_pct, style, max_legs=2, add_atr=1.0, add_spacing=2,
                       cooldown_bars=0, initial_balance=1000.0,
                       fee_override=None, slip_bps=None, return_series=False):
    """Like _simulate_exit but pyramids: it ADDS a leg (up to max_legs) only when
    the trend keeps confirming — the entry signal is still active in the same
    direction AND price has advanced >= add_atr ATR beyond the most recent leg.
    That price-progress gate is the anti-chop rule: a ranging/choppy market never
    makes sustained ATR-scaled progress, so no extra legs are stacked; a real
    trend does, so winners get added to. Each leg carries its own stop/trail/BE; a
    signal flip (or per-leg stop/time) closes legs. Same cost model as the single
    simulator. Partial take-profit is intentionally omitted here (adds replace it)."""
    tp_atr    = style.get("tp_atr")
    sl_atr    = float(style.get("sl_atr", 1.5))
    trail_atr = style.get("trail_atr")
    be_atr    = style.get("be_atr")
    time_bars = style.get("time_bars")
    trend_exit = bool(style.get("trend_exit", False))

    fee_rate  = float(DEFAULT_TAKER_FEE if fee_override is None else fee_override)
    slip      = (5.0 if slip_bps is None else float(slip_bps)) / 10000.0
    funding_pb = 0.00005
    is_margin = (direction in ("BOTH", "SHORT"))

    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    idx    = df.index
    atr_a  = df["atr"].values.astype(float) if "atr" in df.columns else (closes * 0.01)

    balance = float(initial_balance)
    legs = []
    entries = 0
    trade_pnls = []
    trade_rs   = []
    equity = [balance]
    equity_ts = [{"ts": _fmt_ts(idx[0]), "equity": round(float(balance), 2)}] if (return_series and len(idx)) else []
    trade_log = []
    last_exit_bar = -10**9
    last_add_bar = -10**9
    signal_has_reset = True
    max_legs = max(1, int(max_legs))
    add_dist = float(add_atr) if add_atr is not None else 1.0
    add_spacing = max(1, int(add_spacing))

    def _sig(i):
        v = votes[i]; wv = weighted[i]
        n_strats = max(1, int(round(np.nanmax(np.abs(votes)) if len(votes) else 1)))
        if rule == "AND":
            if v >= n_strats and wv > 0:    return 1
            if v <= -n_strats and wv < 0:   return -1
            return 0
        if v > 0 and wv >= min_weighted:    return 1
        if v < 0 and wv <= -min_weighted:   return -1
        return 0

    def _open_leg(sdir, price, atr, i):
        stop_dist = atr * sl_atr
        if stop_dist <= 0:
            return None
        size = (balance * (risk_pct / 100.0)) / stop_dist
        notional = size * price
        if notional > balance:            # spot: no leverage
            size = balance / price; notional = size * price
        if size <= 0:
            return None
        entry_fill = price * (1 + slip) if sdir > 0 else price * (1 - slip)
        tp = None
        if tp_atr is not None:
            tp = entry_fill + atr * tp_atr if sdir > 0 else entry_fill - atr * tp_atr
        sl = entry_fill - atr * sl_atr if sdir > 0 else entry_fill + atr * sl_atr
        return {"type": "long" if sdir > 0 else "short", "entry": entry_fill,
                "size": size, "tp": tp, "sl": sl, "entry_bar": i, "entry_atr": atr,
                "be_done": False, "risk0": stop_dist * size, "_fee": notional * fee_rate}

    def _record_close(leg, fill, bar_i, reason):
        nonlocal balance
        size = leg["size"]
        if leg["type"] == "long":
            gross = (fill - leg["entry"]) * size
        else:
            gross = (leg["entry"] - fill) * size
        fee = abs(size * fill) * fee_rate
        held = max(0, bar_i - leg["entry_bar"])
        fund = (abs(size * leg["entry"]) * funding_pb * held) if (leg["type"] == "short" or is_margin) else 0.0
        net = gross - fee - fund
        balance += net
        trade_pnls.append(net)
        risk0 = leg["risk0"]
        r_mult = net / risk0 if risk0 > 0 else 0.0
        trade_rs.append(r_mult)
        equity.append(balance)
        if return_series:
            trade_log.append({
                "entry_ts": _fmt_ts(idx[leg["entry_bar"]]), "exit_ts": _fmt_ts(idx[bar_i]),
                "direction": leg["type"], "entry_price": round(float(leg["entry"]), 6),
                "exit_price": round(float(fill), 6), "pnl": round(float(net), 2),
                "r": round(float(r_mult), 3), "reason": reason})
            equity_ts.append({"ts": _fmt_ts(idx[bar_i]), "equity": round(float(balance), 2)})

    for i in range(len(df)):
        price = closes[i]
        dir_sign = (1 if legs[0]["type"] == "long" else -1) if legs else 0

        if legs:
            # trailing / breakeven per leg
            for leg in legs:
                atr0 = leg["entry_atr"]
                if trail_atr is not None and atr0 > 0:
                    if leg["type"] == "long":
                        leg["sl"] = max(leg["sl"], price - trail_atr * atr0)
                    else:
                        leg["sl"] = min(leg["sl"], price + trail_atr * atr0)
                if be_atr is not None and not leg["be_done"] and atr0 > 0:
                    if leg["type"] == "long" and highs[i] >= leg["entry"] + be_atr * atr0:
                        leg["sl"] = max(leg["sl"], leg["entry"]); leg["be_done"] = True
                    elif leg["type"] == "short" and lows[i] <= leg["entry"] - be_atr * atr0:
                        leg["sl"] = min(leg["sl"], leg["entry"]); leg["be_done"] = True

            # signal flip closes EVERY leg at once
            flip = False
            if trend_exit:
                s0 = _sig(i)
                flip = (dir_sign > 0 and s0 < 0) or (dir_sign < 0 and s0 > 0)
            if flip:
                for leg in legs:
                    fill = price * (1 - slip) if leg["type"] == "long" else price * (1 + slip)
                    _record_close(leg, fill, i, "FLIP")
                legs = []; last_exit_bar = i
            else:
                survivors = []
                for leg in legs:
                    exit_price = None; reason = None
                    hit_sl = (lows[i] <= leg["sl"]) if leg["type"] == "long" else (highs[i] >= leg["sl"])
                    hit_tp = False
                    if tp_atr is not None and leg.get("tp") is not None:
                        hit_tp = (highs[i] >= leg["tp"]) if leg["type"] == "long" else (lows[i] <= leg["tp"])
                    if hit_sl:
                        exit_price, reason = leg["sl"], "SL"
                    elif hit_tp:
                        exit_price, reason = leg["tp"], "TP"
                    elif time_bars is not None and (i - leg["entry_bar"]) >= int(time_bars):
                        exit_price, reason = price, "TIME"
                    if exit_price is not None:
                        fill = exit_price * (1 - slip) if leg["type"] == "long" else exit_price * (1 + slip)
                        _record_close(leg, fill, i, reason)
                    else:
                        survivors.append(leg)
                if legs and not survivors:
                    last_exit_bar = i
                legs = survivors

        s = _sig(i)
        if s == 1 and direction == "SHORT": s = 0
        if s == -1 and direction == "LONG": s = 0
        if s == 0:
            signal_has_reset = True

        if not legs:
            can_enter = signal_has_reset and (i - last_exit_bar) >= cooldown_bars
            atr = atr_a[i]
            if s != 0 and can_enter and atr > 0 and np.isfinite(atr):
                leg = _open_leg(s, price, atr, i)
                if leg:
                    balance -= leg.pop("_fee"); legs.append(leg)
                    entries += 1; last_add_bar = i; signal_has_reset = False
        elif len(legs) < max_legs and (i - last_add_bar) >= add_spacing:
            # ADD a leg only in a confirmed trend: signal still same-direction AND
            # price advanced >= add_dist ATR beyond the last leg (the anti-chop gate).
            last_leg = legs[-1]; atr0 = last_leg["entry_atr"]; atr = atr_a[i]
            advanced = (price >= last_leg["entry"] + add_dist * atr0) if dir_sign > 0 else (price <= last_leg["entry"] - add_dist * atr0)
            if s == dir_sign and atr0 > 0 and advanced and atr > 0 and np.isfinite(atr):
                leg = _open_leg(dir_sign, price, atr, i)
                if leg:
                    balance -= leg.pop("_fee"); legs.append(leg)
                    entries += 1; last_add_bar = i

    for leg in legs:
        fill = closes[-1] * (1 - slip) if leg["type"] == "long" else closes[-1] * (1 + slip)
        _record_close(leg, fill, len(df) - 1, "EOD")
    legs = []

    roi = ((balance - initial_balance) / initial_balance) * 100.0
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = round(100.0 * len(wins) / len(trade_pnls), 1) if trade_pnls else 0.0
    gross_win = sum(wins); gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy_r = round(float(np.mean(trade_rs)), 3) if trade_rs else 0.0
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    max_dd = round(float(np.max((peak - eq) / peak) * 100.0), 1) if len(eq) else 0.0
    out = {
        "roi": round(roi, 2), "final_balance": round(balance, 2),
        "total_trades": entries, "win_rate": win_rate,
        "profit_factor": profit_factor if np.isfinite(profit_factor) else 99.0,
        "expectancy_r": expectancy_r, "avg_r": expectancy_r, "max_drawdown": max_dd,
    }
    if return_series:
        out["equity"] = equity_ts
        out["trades"] = trade_log
    return out


# ==========================================================================
# per-config walk-forward (one coin)
# ==========================================================================
_FEAT_CACHE = {}
_VOTE_CACHE = {}

def _prep(symbol, tf):
    """Prepared feature frame + BTC reference column, cached per (symbol,tf)."""
    key = (symbol, tf)
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    raw = _load_tf(symbol, tf)
    if raw is None or len(raw) < MIN_ROWS:
        _FEAT_CACHE[key] = None
        return None
    try:
        feat, _ = apply_mega_features(raw.copy())
    except Exception as e:
        log.warning(f"{symbol} {tf} feature build failed: {e}")
        _FEAT_CACHE[key] = None
        return None
    if feat is None or len(feat) < MIN_ROWS:
        _FEAT_CACHE[key] = None
        return None
    feat = _attach_btc(feat, symbol, tf)
    _FEAT_CACHE[key] = feat
    return feat


def _votes_for(symbol, tf, entry_name, feat):
    """Whole-frame (votes, weighted) for an entry set, cached per (symbol,tf,entry)."""
    key = (symbol, tf, entry_name)
    if key in _VOTE_CACHE:
        return _VOTE_CACHE[key]
    strats = ENTRIES[entry_name]
    bt = Backtester({"symbol": symbol, "timeframe": tf})
    v, w = bt._strategy_votes(feat, strats)
    _VOTE_CACHE[key] = (v, w)
    return v, w


def _score_one(symbol, tf, entry_name, direction, style_name):
    """Walk-forward a single (coin, tf, entry, direction, exit-style) over FOLDS."""
    feat = _prep(symbol, tf)
    if feat is None:
        return None
    v, w = _votes_for(symbol, tf, entry_name, feat)
    style = EXIT_STYLES[style_name]
    n = len(feat)
    edges = [int(n * k / FOLDS) for k in range(FOLDS + 1)]
    rois, rs, pfs, dds, trades = [], [], [], [], 0
    for k in range(FOLDS):
        a, b = edges[k], edges[k + 1]
        if b - a < 30:
            continue
        sl_df = feat.iloc[a:b]
        try:
            m = _simulate_exit(sl_df, v[a:b], w[a:b], direction=direction, rule="OR",
                               min_weighted=0.3, risk_pct=1.0, style=style)
        except Exception as e:
            log.warning(f"{symbol} {tf} {entry_name}/{style_name} fold {k} failed: {e}")
            continue
        rois.append(m["roi"]); rs.append(m["expectancy_r"]); pfs.append(m["profit_factor"])
        dds.append(m["max_drawdown"]); trades += m["total_trades"]
    if not rois:
        return None
    prof = sum(1 for r in rois if r > 0)
    mean_roi = float(np.mean(rois))
    mean_r = float(np.mean(rs))
    has_edge = (prof >= MIN_PROFITABLE_FOLDS and mean_roi > 0 and mean_r > 0
                and trades >= MIN_TRADES_PER_FOLD * len(rois))
    return {
        "symbol": symbol, "timeframe": tf, "entry": entry_name, "direction": direction,
        "style": style_name, "folds": len(rois), "profitable_folds": prof,
        "mean_roi": round(mean_roi, 2), "worst_roi": round(min(rois), 2),
        "best_roi": round(max(rois), 2), "mean_expectancy_r": round(mean_r, 3),
        "mean_profit_factor": round(float(np.mean([p for p in pfs if np.isfinite(p)])), 2) if pfs else 0.0,
        "mean_max_dd": round(float(np.mean(dds)), 1), "total_trades": trades,
        "coin_edge": bool(has_edge),
    }


# ==========================================================================
# single on-demand run — powers the Strategy Lab UI (read-only backtest)
# ==========================================================================
def lab_options():
    """The choices the Strategy Lab UI offers, incl. the live fleet's defaults."""
    return {
        "symbols": SYMBOLS, "timeframes": TIMEFRAMES,
        "directions": ["LONG", "SHORT", "BOTH"],
        "entries": list(ENTRIES.keys()), "styles": list(EXIT_STYLES.keys()),
        "fleet_default": {"entry": "regime", "style": "trend_ride",
                          "long_tf": "4h", "short_tf": "1d",
                          "note": "The live fleet runs entry=regime, exit=trend_ride: "
                                  "LONG on 4h, SHORT on 1d. Pick those to backtest what "
                                  "the fleet actually trades."},
    }


def run_single(symbol="BTC-USD", timeframe="4h", entry="regime", direction="LONG",
               style="trend_ride", risk_pct=1.0, start=None, end=None,
               initial_balance=1000.0, max_legs=1, add_atr=1.0):
    """One config, full detail — the SAME validated simulator the fleet's exit
    research uses (_simulate_exit), so the Strategy Lab shows exactly what the
    fleet would trade. Returns metrics + equity curve + per-trade log + a
    buy&hold benchmark over the same window. Read-only; never touches live bots."""
    if entry not in ENTRIES:
        return {"error": f"unknown entry '{entry}'", "known_entries": list(ENTRIES.keys())}
    style_d = EXIT_STYLES.get(style)
    if style_d is None:
        return {"error": f"unknown style '{style}'", "known_styles": list(EXIT_STYLES.keys())}
    direction = (direction or "LONG").upper()
    if direction not in ("LONG", "SHORT", "BOTH"):
        direction = "LONG"
    feat = _prep(symbol, timeframe)
    if feat is None:
        return {"error": f"insufficient data for {symbol} {timeframe}"}
    v, w = _votes_for(symbol, timeframe, entry, feat)
    n = len(feat)
    a, b = 0, n
    # Date-range slice. The feature index is tz-aware (UTC); the incoming date
    # strings are tz-naive, so align them to the index's tz before comparing
    # (a tz-aware vs tz-naive comparison raises and would drop the whole filter).
    _tz = getattr(feat.index, "tz", None)
    def _align(x):
        t = pd.Timestamp(x)
        if _tz is not None and t.tzinfo is None:
            t = t.tz_localize(_tz)
        elif _tz is None and t.tzinfo is not None:
            t = t.tz_localize(None)
        return t
    try:
        if start:
            a = int(feat.index.searchsorted(_align(start)))
        if end:
            b = int(feat.index.searchsorted(_align(end), side="right"))
    except Exception as e:
        log.warning(f"run_single date-slice failed ({start}..{end}): {e}")
        a, b = 0, n
    a = max(0, min(a, n)); b = max(a, min(b, n))
    if b - a < 30:
        return {"error": "selected window too small (need >= 30 bars of history)"}
    sl_df = feat.iloc[a:b]
    try:
        _legs = max(1, int(max_legs or 1))
        if _legs > 1:
            m = _simulate_exit_pyr(sl_df, v[a:b], w[a:b], direction=direction, rule="OR",
                                   min_weighted=0.3, risk_pct=float(risk_pct or 1.0), style=style_d,
                                   max_legs=_legs, add_atr=float(add_atr or 1.0),
                                   initial_balance=float(initial_balance), return_series=True)
        else:
            m = _simulate_exit(sl_df, v[a:b], w[a:b], direction=direction, rule="OR",
                               min_weighted=0.3, risk_pct=float(risk_pct or 1.0), style=style_d,
                               initial_balance=float(initial_balance), return_series=True)
    except Exception as e:
        log.warning(f"run_single {symbol} {timeframe} {entry}/{style} failed: {e}")
        return {"error": f"simulation failed: {e}"}
    closes = sl_df["close"].astype(float).values
    bh = ((float(closes[-1]) / float(closes[0])) - 1.0) * 100.0 if len(closes) >= 2 and closes[0] > 0 else 0.0
    return {
        "symbol": symbol, "timeframe": timeframe, "entry": entry,
        "direction": direction, "style": style, "risk_pct": float(risk_pct or 1.0),
        "max_legs": max(1, int(max_legs or 1)), "add_atr": float(add_atr or 1.0),
        "initial_balance": float(initial_balance),
        "window": {"start": _fmt_ts(sl_df.index[0]), "end": _fmt_ts(sl_df.index[-1]), "bars": int(b - a)},
        "metrics": {k: m.get(k) for k in ("roi", "final_balance", "total_trades",
                                          "win_rate", "profit_factor", "expectancy_r",
                                          "avg_r", "max_drawdown")},
        "equity": m.get("equity", []),
        "trades": m.get("trades", []),
        "buy_hold_pct": round(float(bh), 2),
    }


# ==========================================================================
# full optimizer — ranked by CROSS-COIN GENERALIZATION
# ==========================================================================
def optimize_exits(symbols=None, timeframes=None, entries=None, styles=None,
                   directions=None, top=25, write=True):
    """Sweep (entry x tf x direction x exit-style) and, for each, evaluate across
    ALL coins. A config is ranked by how many coins it clears the edge bar on
    (generalization) — the guardrail against the per-coin overfit that has killed
    every prior candidate. Writes exit_lab_results.json."""
    symbols   = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    entries   = entries or list(ENTRIES.keys())
    styles    = styles or list(EXIT_STYLES.keys())
    directions = directions or DIRECTIONS

    per_coin = []      # every (config, coin) row
    for tf in timeframes:
        for entry_name in entries:
            for direction in directions:
                for style_name in styles:
                    for symbol in symbols:
                        r = _score_one(symbol, tf, entry_name, direction, style_name)
                        if r:
                            per_coin.append(r)

    # aggregate to configs (across coins)
    agg = {}
    for r in per_coin:
        key = (r["timeframe"], r["entry"], r["direction"], r["style"])
        g = agg.setdefault(key, {"rows": [], "coins_edge": 0})
        g["rows"].append(r)
        g["coins_edge"] += 1 if r["coin_edge"] else 0

    configs = []
    for (tf, entry_name, direction, style_name), g in agg.items():
        rows = g["rows"]
        coins_edge = g["coins_edge"]
        mean_r = float(np.mean([x["mean_expectancy_r"] for x in rows]))
        mean_roi = float(np.mean([x["mean_roi"] for x in rows]))
        worst_coin_roi = min(x["mean_roi"] for x in rows)
        mean_pf = float(np.mean([x["mean_profit_factor"] for x in rows]))
        mean_dd = float(np.mean([x["mean_max_dd"] for x in rows]))
        total_trades = sum(x["total_trades"] for x in rows)
        generalizes = coins_edge >= MIN_COINS_GENERALIZE and mean_r > 0 and mean_roi > 0
        configs.append({
            "timeframe": tf, "entry": entry_name, "direction": direction, "style": style_name,
            "coins_tested": len(rows), "coins_with_edge": coins_edge,
            "mean_expectancy_r": round(mean_r, 3), "mean_roi_per_fold": round(mean_roi, 2),
            "worst_coin_mean_roi": round(worst_coin_roi, 2), "mean_profit_factor": round(mean_pf, 2),
            "mean_max_dd": round(mean_dd, 1), "total_trades": total_trades,
            "generalizes": bool(generalizes),
            "per_coin": [{"symbol": x["symbol"], "edge": x["coin_edge"],
                          "profit_folds": x["profitable_folds"], "mean_roi": x["mean_roi"],
                          "exp_r": x["mean_expectancy_r"], "trades": x["total_trades"]} for x in rows],
        })

    configs.sort(key=lambda c: (c["generalizes"], c["coins_with_edge"],
                                c["mean_expectancy_r"], c["mean_roi_per_fold"]), reverse=True)
    winners = [c for c in configs if c["generalizes"]]
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "configs_tested": len(configs), "generalizing_configs": len(winners),
        "criteria": {"folds": FOLDS, "min_profitable_folds": MIN_PROFITABLE_FOLDS,
                     "min_coins_generalize": MIN_COINS_GENERALIZE,
                     "note": "A config counts only if it clears the per-coin bar on >= "
                             f"{MIN_COINS_GENERALIZE} coins AND has positive mean expectancy (R)."},
        "winners": winners, "top": configs[:top],
    }
    if write:
        try:
            with open(OUT_PATH, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"exit_lab: {len(winners)} generalizing / {len(configs)} configs -> {OUT_PATH}")
        except Exception as e:
            log.warning(f"exit_lab write failed: {e}")
    return report


# ==========================================================================
# portfolio test — the ensemble avenue
# ==========================================================================
def portfolio_test(symbols=None, timeframe="1d", entry="trend", direction="LONG",
                   style="be_runner", risk_pct=1.0):
    """Run ONE config across many coins as a combined portfolio and measure the
    blended equity curve. Several marginal-but-uncorrelated coin streams can add
    up to a smoother, positive whole even when each alone is break-even — the
    diversification avenue. Splits risk equally across coins (risk_pct / N each)."""
    symbols = symbols or SYMBOLS
    style_d = EXIT_STYLES.get(style, EXIT_STYLES["be_runner"])
    per = risk_pct / max(1, len(symbols))
    coin_curves = {}
    combined = None
    coin_stats = []
    for symbol in symbols:
        feat = _prep(symbol, timeframe)
        if feat is None:
            continue
        v, w = _votes_for(symbol, timeframe, entry, feat)
        # full-period (not folded) run for the equity curve
        m = _simulate_exit(feat, v, w, direction=direction, rule="OR", min_weighted=0.3,
                           risk_pct=per, style=style_d, initial_balance=1000.0)
        coin_stats.append({"symbol": symbol, **m})
    if not coin_stats:
        return {"error": "no coins produced results"}
    total_roi = float(np.mean([c["roi"] for c in coin_stats]))   # equal-weight blend
    mean_dd = float(np.mean([c["max_drawdown"] for c in coin_stats]))
    mean_r = float(np.mean([c["expectancy_r"] for c in coin_stats]))
    pos = sum(1 for c in coin_stats if c["roi"] > 0)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe, "entry": entry, "direction": direction, "style": style,
        "coins": len(coin_stats), "coins_positive": pos,
        "blended_roi": round(total_roi, 2), "mean_expectancy_r": round(mean_r, 3),
        "mean_max_dd": round(mean_dd, 1),
        "diversified": pos >= MIN_COINS_GENERALIZE and total_roi > 0,
        "per_coin": coin_stats,
    }


# ==========================================================================
# HARD VALIDATION of a single winning config
#   1) TIME HOLDOUT  — score ONLY on the untouched final `holdout_frac` of bars,
#      one config (not 96), so multiple-comparison inflation is removed and the
#      most recent regime is tested on its own.
#   2) FEE/SLIP STRESS — rerun the holdout with fee_mult x fee + slip_mult x slip;
#      a real edge survives harsher costs, an artifact of thin margins doesn't.
#   3) PER-COIN detail — see the coins that FAIL, not just the average.
# ==========================================================================
def validate_exit(symbol=None, timeframe="4h", entry="trend", direction="LONG",
                  style="trend_ride", holdout_frac=0.25, fee_mult=2.0, slip_mult=3.0,
                  max_legs=1, add_atr=1.0, write=True):
    """Confirm ONE exit config out-of-sample + under cost stress, per coin.
    Returns a verdict: ROBUST only if it stays positive (ROI and expectancy R) on
    >= MIN_COINS_GENERALIZE coins in BOTH the clean holdout AND the stressed one."""
    style_d = EXIT_STYLES.get(style)
    if style_d is None:
        return {"error": f"unknown style '{style}'", "known": list(EXIT_STYLES.keys())}
    if entry not in ENTRIES:
        return {"error": f"unknown entry '{entry}'", "known": list(ENTRIES.keys())}
    syms = [symbol] if symbol else SYMBOLS
    base_fee = float(DEFAULT_TAKER_FEE)
    rows = []
    for sym in syms:
        feat = _prep(sym, timeframe)
        if feat is None:
            rows.append({"symbol": sym, "skip": "insufficient data"})
            continue
        v, w = _votes_for(sym, timeframe, entry, feat)
        n = len(feat)
        cut = int(n * (1.0 - holdout_frac))
        ho = slice(cut, n)
        if n - cut < 40:
            rows.append({"symbol": sym, "skip": f"holdout too small ({n - cut} bars)"})
            continue

        def _run(fee, slb):
            if int(max_legs) > 1:
                return _simulate_exit_pyr(feat.iloc[ho], v[ho], w[ho], direction=direction,
                                          rule="OR", min_weighted=0.3, risk_pct=1.0, style=style_d,
                                          max_legs=int(max_legs), add_atr=float(add_atr),
                                          fee_override=fee, slip_bps=slb)
            return _simulate_exit(feat.iloc[ho], v[ho], w[ho], direction=direction,
                                  rule="OR", min_weighted=0.3, risk_pct=1.0, style=style_d,
                                  fee_override=fee, slip_bps=slb)
        clean  = _run(base_fee, 5.0)
        stress = _run(base_fee * fee_mult, 5.0 * slip_mult)
        rows.append({
            "symbol": sym, "holdout_bars": n - cut,
            "oos_roi": clean["roi"], "oos_expR": clean["expectancy_r"],
            "oos_pf": clean["profit_factor"], "oos_win": clean["win_rate"],
            "oos_trades": clean["total_trades"], "oos_dd": clean["max_drawdown"],
            "stress_roi": stress["roi"], "stress_expR": stress["expectancy_r"],
            "stress_trades": stress["total_trades"],
            "survives": bool(clean["roi"] > 0 and clean["expectancy_r"] >= MIN_SURV_EXPECTANCY_R and clean["total_trades"] >= MIN_HOLDOUT_TRADES and clean["profit_factor"] >= MIN_SURV_PROFIT_FACTOR),
            "survives_stress": bool(stress["roi"] > 0 and stress["expectancy_r"] >= MIN_SURV_EXPECTANCY_R and stress["total_trades"] >= MIN_HOLDOUT_TRADES and stress["profit_factor"] >= MIN_SURV_PROFIT_FACTOR),
        })
    valid = [r for r in rows if "oos_roi" in r]
    n_pos = sum(1 for r in valid if r["survives"])
    n_str = sum(1 for r in valid if r["survives_stress"])
    robust = n_pos >= MIN_COINS_GENERALIZE and n_str >= MIN_COINS_GENERALIZE
    # A coin is CLEARED for live use only if it survives BOTH the clean holdout
    # AND the cost-stress rerun. This exact list gates live pyramiding per coin.
    cleared = [r["symbol"] for r in valid if r.get("survives") and r.get("survives_stress")]
    cfg = {"timeframe": timeframe, "entry": entry, "direction": direction,
           "style": style, "holdout_frac": holdout_frac,
           "fee_mult": fee_mult, "slip_mult": slip_mult,
           "max_legs": int(max_legs), "add_atr": float(add_atr)}
    if write:
        try:
            reg = all_eligibility()
            reg[_elig_key(timeframe, entry, direction, style, max_legs)] = {
                "generated": datetime.now(timezone.utc).isoformat(),
                "verdict": "ROBUST" if robust else "NOT ROBUST",
                "cleared_coins": cleared, "config": cfg,
                "per_coin": [{"symbol": r.get("symbol"), "survives": bool(r.get("survives")),
                              "survives_stress": bool(r.get("survives_stress"))} for r in valid],
            }
            with open(ELIG_PATH, "w") as f:
                json.dump(reg, f, indent=2)
            log.info(f"eligibility: {_elig_key(timeframe, entry, direction, style, max_legs)} cleared={cleared}")
        except Exception as e:
            log.warning(f"eligibility write failed: {e}")
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "coins_tested": len(valid), "survives_oos": n_pos, "survives_stress": n_str,
        "cleared_coins": cleared,
        "verdict": "ROBUST" if robust else "NOT ROBUST",
        "note": ("Holdout is the final slice of history rerun for THIS one config only "
                 "(no 96-way selection), then stressed with harsher costs. A ROBUST "
                 "verdict warrants forward PAPER trading next — never direct capital."),
        "per_coin": rows,
    }


def validate_top(timeframe="4h", entry="trend", direction="LONG", style="trend_ride"):
    """Convenience: print a readable per-coin validation of one config."""
    rep = validate_exit(timeframe=timeframe, entry=entry, direction=direction, style=style)
    print(f"\nVALIDATE {entry}/{style} {timeframe} {direction}  ->  {rep.get('verdict')}")
    print(f"  survives OOS: {rep.get('survives_oos')}/{rep.get('coins_tested')}   "
          f"survives STRESS: {rep.get('survives_stress')}/{rep.get('coins_tested')}")
    for r in rep.get("per_coin", []):
        if "oos_roi" not in r:
            print(f"  {r['symbol']:9} SKIP ({r.get('skip')})"); continue
        print(f"  {r['symbol']:9} oosROI {r['oos_roi']:+7.2f}  expR {r['oos_expR']:+.3f}  "
              f"PF {r['oos_pf']:>5}  win {r['oos_win']:>5}  {r['oos_trades']:>4}t  dd {r['oos_dd']:>4}  "
              f"| stress ROI {r['stress_roi']:+7.2f} expR {r['stress_expR']:+.3f}  "
              f"{'OK' if r['survives'] else 'x'}/{'OK' if r['survives_stress'] else 'x'}")
    return rep


# ==========================================================================
# CONFIG MATRIX — validate a whole slate of candidates at once, one table
# ==========================================================================
DEPLOY_CANDIDATES = [
    ("4h", "trend",    "LONG", "trend_ride"),
    ("4h", "regime",   "LONG", "trend_ride"),
    ("4h", "momentum", "LONG", "trend_ride"),
    ("4h", "trend",    "BOTH", "trend_ride"),
    ("1d", "trend",    "LONG", "trend_ride"),
    ("1d", "regime",   "LONG", "trend_ride"),
    ("1d", "momentum", "LONG", "trend_ride"),
    ("4h", "trend",    "LONG", "be_runner"),
    ("1d", "trend",    "LONG", "fixed_3to1"),
]

# Short / long+short slate. `regime` SHORT is the most promising short: its
# btc_regime leg only votes down when BTC itself is below its trend EMA, so it
# shorts in confirmed risk-off regimes instead of fading every up-drift.
SHORT_CANDIDATES = [
    ("4h", "regime",   "SHORT", "trend_ride"),
    ("1d", "regime",   "SHORT", "trend_ride"),
    ("4h", "trend",    "SHORT", "trend_ride"),
    ("1d", "trend",    "SHORT", "trend_ride"),
    ("4h", "momentum", "SHORT", "trend_ride"),
    ("1d", "momentum", "SHORT", "trend_ride"),
    ("4h", "regime",   "BOTH",  "trend_ride"),
    ("1d", "regime",   "BOTH",  "trend_ride"),
    ("4h", "trend",    "BOTH",  "trend_ride"),
    ("1d", "trend",    "BOTH",  "trend_ride"),
    ("4h", "momentum", "BOTH",  "trend_ride"),
    ("1d", "momentum", "BOTH",  "trend_ride"),
]

# Finer sweep AROUND the winners: vary the stop distance / trailing on the top
# directional configs to see if the trend_ride defaults can be tightened.
_SWEEP_STYLES = ["tr_sl15", "trend_ride", "tr_sl25", "tr_sl30", "tr_trail3", "tr_trail4", "tr_be1"]
SWEEP_CANDIDATES = (
    [("4h", "trend", "LONG", s) for s in _SWEEP_STYLES] +
    [("1d", "trend", "LONG", s) for s in _SWEEP_STYLES] +
    [("1d", "regime", "SHORT", s) for s in _SWEEP_STYLES]
)

SLATES = {"long": DEPLOY_CANDIDATES, "short": SHORT_CANDIDATES,
          "sweep": SWEEP_CANDIDATES,
          "all": DEPLOY_CANDIDATES + SHORT_CANDIDATES}


def validate_matrix(configs=None):
    """Run validate_exit across a slate of candidates so you can see, in one table,
    exactly which (tf, entry, direction, style) combos are deploy-worthy (survive
    the recent-holdout + cost-stress on >=3 coins) and which aren't."""
    configs = configs or DEPLOY_CANDIDATES
    out = []
    for tf, e, d, s in configs:
        r = validate_exit(timeframe=tf, entry=e, direction=d, style=s)
        out.append({"timeframe": tf, "entry": e, "direction": d, "style": s,
                    "verdict": r.get("verdict"), "survives_oos": r.get("survives_oos"),
                    "survives_stress": r.get("survives_stress"),
                    "coins_tested": r.get("coins_tested")})
    out.sort(key=lambda x: (x["verdict"] == "ROBUST", x["survives_stress"] or 0,
                            x["survives_oos"] or 0), reverse=True)
    return out


# ==========================================================================
# REGIME SPLIT — the risk the cost-stress test CANNOT see. Split the holdout
# into contiguous sub-windows; a durable edge is profitable across MOST of them,
# not carried by one lucky trend. buyhold column shows the market's own drift in
# each window so you can tell strategy-edge from just-long-in-a-bull.
# ==========================================================================
def validate_regime(timeframe="4h", entry="trend", direction="LONG", style="trend_ride",
                    holdout_frac=0.25, sub=4):
    style_d = EXIT_STYLES.get(style)
    if style_d is None:
        return {"error": f"unknown style '{style}'"}
    if entry not in ENTRIES:
        return {"error": f"unknown entry '{entry}'"}
    rows = []
    for sym in SYMBOLS:
        feat = _prep(sym, timeframe)
        if feat is None:
            rows.append({"symbol": sym, "skip": "insufficient data"}); continue
        v, w = _votes_for(sym, timeframe, entry, feat)
        n = len(feat); cut = int(n * (1.0 - holdout_frac))
        ho = feat.iloc[cut:n]; hov = v[cut:n]; how = w[cut:n]
        m = len(ho); edges = [int(m * k / sub) for k in range(sub + 1)]
        wins = 0; windows = []
        for k in range(sub):
            a, b = edges[k], edges[k + 1]
            if b - a < 20:
                continue
            seg = ho.iloc[a:b]
            res = _simulate_exit(seg, hov[a:b], how[a:b], direction=direction, rule="OR",
                                 min_weighted=0.3, risk_pct=1.0, style=style_d)
            bh = (float(seg["close"].iloc[-1]) / float(seg["close"].iloc[0]) - 1.0) * 100.0 if len(seg) else 0.0
            windows.append({"roi": res["roi"], "buyhold": round(bh, 1), "trades": res["total_trades"]})
            wins += 1 if res["roi"] > 0 else 0
        rows.append({"symbol": sym, "windows": len(windows), "profitable_windows": wins,
                     "detail": windows,
                     "robust_across_regimes": bool(wins >= max(1, int(0.6 * len(windows))))})
    valid = [r for r in rows if "windows" in r]
    strong = sum(1 for r in valid if r["robust_across_regimes"])
    return {"config": {"timeframe": timeframe, "entry": entry, "direction": direction,
                       "style": style, "sub": sub},
            "coins_tested": len(valid), "coins_robust_across_regimes": strong,
            "verdict": "REGIME-ROBUST" if strong >= MIN_COINS_GENERALIZE else "REGIME-FRAGILE",
            "note": "Profitable across most sub-windows = not carried by a single trend. "
                    "If strategy ROI >> buyhold in chop windows, the edge is real timing, "
                    "not just market beta.",
            "per_coin": rows}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "matrix":
        slate = sys.argv[2] if len(sys.argv) > 2 else "long"
        rep = validate_matrix(SLATES.get(slate, DEPLOY_CANDIDATES))
        print(f"\n[slate: {slate}]")
        print(f"{'verdict':11}{'tf':4}{'entry':9}{'dir':5}{'style':13}{'OOS':>5}{'STRESS':>8}")
        for r in rep:
            print(f"{r['verdict']:11}{r['timeframe']:4}{r['entry']:9}{r['direction']:5}{r['style']:13}"
                  f"{str(r['survives_oos'])+'/'+str(r['coins_tested']):>5}"
                  f"{str(r['survives_stress'])+'/'+str(r['coins_tested']):>8}")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "regime":
        a = sys.argv[2:]
        rep = validate_regime(timeframe=a[0] if len(a) > 0 else "4h",
                              entry=a[1] if len(a) > 1 else "trend",
                              direction=a[2] if len(a) > 2 else "LONG",
                              style=a[3] if len(a) > 3 else "trend_ride")
        print(f"\nREGIME {rep['config']} -> {rep.get('verdict')} "
              f"({rep.get('coins_robust_across_regimes')}/{rep.get('coins_tested')} coins robust)")
        for r in rep.get("per_coin", []):
            if "windows" not in r:
                print(f"  {r['symbol']:9} SKIP ({r.get('skip')})"); continue
            segs = " ".join(f"[{d['roi']:+5.1f} vs bh{d['buyhold']:+5.1f}]" for d in r["detail"])
            print(f"  {r['symbol']:9} {r['profitable_windows']}/{r['windows']} win  {segs}")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # python exit_lab.py validate [tf] [entry] [direction] [style]
        a = sys.argv[2:]
        validate_top(timeframe=a[0] if len(a) > 0 else "4h",
                     entry=a[1] if len(a) > 1 else "trend",
                     direction=a[2] if len(a) > 2 else "LONG",
                     style=a[3] if len(a) > 3 else "trend_ride")
        sys.exit(0)
    rep = optimize_exits()
    print(f"\nTested {rep['configs_tested']} exit configs; "
          f"{rep['generalizing_configs']} generalized across coins.\n")
    print(f"{'GEN':4}{'tf':4}{'entry':9}{'dir':5}{'style':16}"
          f"{'coinsEdge':10}{'expR':>7}{'roi/f':>8}{'worstCoin':>10}{'PF':>6}{'DD%':>6}{'trades':>8}")
    for c in rep["top"]:
        tag = "YES " if c["generalizes"] else "    "
        print(f"{tag}{c['timeframe']:4}{c['entry']:9}{c['direction']:5}{c['style']:16}"
              f"{str(c['coins_with_edge'])+'/'+str(c['coins_tested']):10}"
              f"{c['mean_expectancy_r']:>7.3f}{c['mean_roi_per_fold']:>8.2f}"
              f"{c['worst_coin_mean_roi']:>10.2f}{c['mean_profit_factor']:>6.2f}"
              f"{c['mean_max_dd']:>6.1f}{c['total_trades']:>8}")
