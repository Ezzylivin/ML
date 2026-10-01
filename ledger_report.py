#!/usr/bin/env python3
"""
ledger_report.py — honest performance report from the trade ledger.

This is the REAL edge signal: it reads the trades the bot actually took
(data/trade_ledger.db) and computes the metrics that tell you whether there's
an edge, net of the fees already baked into each recorded PnL.

Metrics: trade count, win rate, gross/net PnL, profit factor, expectancy per
trade, average win / average loss, largest win/loss, max drawdown on the
realized-PnL curve, and a breakdown by direction and exit reason.

Usage:
  python ledger_report.py                 # all trades
  python ledger_report.py --symbol BTC-USD
  python ledger_report.py --mode paper    # paper | live
"""
import os
import sys
import argparse
import sqlite3

sys.path.append(os.getcwd())
try:
    from app.config2 import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LEDGER = os.path.join(DATA_DIR, "trade_ledger.db")


def _rows(where, params):
    conn = sqlite3.connect(LEDGER)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT pnl, direction, reason FROM trades {where} ORDER BY id ASC", params)
    rows = cur.fetchall()
    conn.close()
    return rows


def _metrics(pnls):
    n = len(pnls)
    if n == 0:
        return {"trades": 0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    net = sum(pnls)
    # max drawdown on the cumulative realized-PnL curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "trades": n,
        "win_rate": round(len(wins) / n * 100, 1),
        "net_pnl": round(net, 2),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "expectancy": round(net / n, 4),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "largest_win": round(max(pnls), 2),
        "largest_loss": round(min(pnls), 2),
        "max_drawdown": round(max_dd, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--mode")
    args = ap.parse_args()

    if not os.path.exists(LEDGER):
        print(f"No ledger yet at {LEDGER}. Run some paper trades first.")
        return

    clauses, params = [], []
    if args.symbol:
        clauses.append("symbol = ?"); params.append(args.symbol)
    if args.mode:
        clauses.append("mode = ?"); params.append(args.mode)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = _rows(where, params)
    pnls = [float(r["pnl"] or 0) for r in rows]
    m = _metrics(pnls)

    scope = " ".join(filter(None, [args.symbol, args.mode])) or "ALL"
    print("=" * 60)
    print(f"  TRADE LEDGER REPORT  [{scope}]")
    print("=" * 60)
    if not m["trades"]:
        print("  No trades match.")
        return
    print(f"  Trades .............. {m['trades']}")
    print(f"  Win rate ............ {m['win_rate']}%")
    print(f"  Net PnL ............. {m['net_pnl']:+}")
    print(f"  Profit factor ....... {m['profit_factor']}   (>1 = winning; >1.3 healthy)")
    print(f"  Expectancy/trade .... {m['expectancy']:+}")
    print(f"  Avg win / avg loss .. {m['avg_win']:+} / {m['avg_loss']:+}")
    print(f"  Largest win / loss .. {m['largest_win']:+} / {m['largest_loss']:+}")
    print(f"  Max drawdown ........ {m['max_drawdown']:+}")
    print("-" * 60)
    for dim, label in (("direction", "BY DIRECTION"), ("reason", "BY EXIT REASON")):
        print(f"  {label}")
        groups = {}
        for r in rows:
            groups.setdefault(r[dim] or "?", []).append(float(r["pnl"] or 0))
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            mm = _metrics(v)
            net_s = f"{mm['net_pnl']:+.2f}".ljust(10)
            print(f"    {k:<16} n={mm['trades']:<4} win={str(mm['win_rate']):<5}%  "
                  f"net={net_s} PF={mm['profit_factor']}")
        print("-" * 60)

    verdict = ("LIKELY EDGE" if m["profit_factor"] >= 1.3 and m["trades"] >= 30
               else "INCONCLUSIVE — need more trades" if m["trades"] < 30
               else "NO CLEAR EDGE")
    print(f"  VERDICT: {verdict}")
    print(f"  (Rule of thumb: trust nothing under ~30 trades; profit factor")
    print(f"   below 1.0 loses money; costs are already in each PnL.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
