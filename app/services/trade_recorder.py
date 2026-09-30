"""
trade_recorder.py — durable capture of every closed trade.

Purpose
-------
The live engine keeps trade history in memory (ACTIVE_BOTS) and RESETS it on
every bot start, so nothing accumulates across sessions. This module writes a
permanent, append-only record of each closed trade — including the feature
snapshot taken at entry and the realized outcome — to a SQLite ledger that
survives restarts.

That ledger is the foundation for "learning from every trade": it is the
labeled dataset of the bot's own decisions (features at entry -> win/loss),
usable later for a trade-quality meta-model and for auditing what actually
works. It is intentionally decoupled from model retraining (which runs on
market data) so a write here can never stall or corrupt the trading loop.

Location: app/services/trade_recorder.py
Ledger:   data/trade_ledger.db  (SQLite)
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger("TradeRecorder")

try:
    from app.config2 import DATA_DIR
except Exception:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

LEDGER_PATH = os.path.join(DATA_DIR, "trade_ledger.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,       -- ISO close time (UTC)
    user_id       TEXT,
    symbol        TEXT,
    timeframe     TEXT,
    direction     TEXT,       -- long | short
    mode          TEXT,       -- paper | live
    entry_price   REAL,
    exit_price    REAL,
    size          REAL,
    pnl           REAL,
    label         INTEGER,    -- 1 = profitable, 0 = not
    reason        TEXT,       -- Take Profit | Trailing Stop | Manual | ...
    entry_conf    REAL,       -- raw P(bull) at entry
    composite     REAL,       -- TradeQualityScorer composite at entry
    adx_at_entry  REAL,
    atr_at_entry  REAL,
    vol_ratio     REAL,
    features_json TEXT,       -- JSON dict: FEATURE_COLUMNS snapshot at entry
    hold_secs     REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades(ts);
"""


def _connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(LEDGER_PATH, timeout=10)
    return conn


def init_ledger():
    """Create the ledger table if it does not exist. Safe to call repeatedly."""
    try:
        conn = _connect()
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"init_ledger failed: {e}")


def record_exit(*, user_id, symbol, timeframe, direction, mode,
                entry_price, exit_price, size, pnl, reason,
                entry_conf=None, composite=None,
                adx_at_entry=None, atr_at_entry=None, vol_ratio=None,
                features=None, hold_secs=None):
    """
    Append one closed-trade record. Best-effort and fully guarded: a failure
    here logs a warning but never raises into the trading loop.
    """
    try:
        label = 1 if (pnl is not None and float(pnl) > 0) else 0
        feats_json = json.dumps(features) if isinstance(features, dict) else None
        conn = _connect()
        conn.execute(
            """INSERT INTO trades
               (ts, user_id, symbol, timeframe, direction, mode,
                entry_price, exit_price, size, pnl, label, reason,
                entry_conf, composite, adx_at_entry, atr_at_entry, vol_ratio,
                features_json, hold_secs)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(), user_id, symbol, timeframe,
             direction, mode, entry_price, exit_price, size, pnl, label, reason,
             entry_conf, composite, adx_at_entry, atr_at_entry, vol_ratio,
             feats_json, hold_secs),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"record_exit failed (non-fatal): {e}")


def stats(symbol=None):
    """Quick aggregate for dashboards/inspection. Returns a dict."""
    try:
        conn = _connect()
        cur = conn.cursor()
        if symbol:
            cur.execute("SELECT COUNT(*), AVG(label), SUM(pnl) FROM trades WHERE symbol=?", (symbol,))
        else:
            cur.execute("SELECT COUNT(*), AVG(label), SUM(pnl) FROM trades")
        n, winrate, total = cur.fetchone()
        conn.close()
        return {"trades": n or 0,
                "win_rate": round((winrate or 0) * 100, 1),
                "total_pnl": round(total or 0, 2)}
    except Exception as e:
        logger.warning(f"stats failed: {e}")
        return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0}


def summary(recent_limit=25, user_id=None):
    """Aggregated view of the ledger for the dashboard / API. Never raises.

    Pass user_id to scope the whole report to one bot/user (multi-user
    isolation); omit for the global view.
    """
    out = {"totals": {}, "by_symbol": [], "by_direction": [],
           "by_reason": [], "recent": [], "cum_pnl": []}
    # Optional per-user filter, applied to every query.
    w = "WHERE user_id = ?" if user_id else ""
    p = (user_id,) if user_id else ()
    try:
        conn = _connect(); conn.row_factory = sqlite3.Row; c = conn.cursor()

        c.execute(f"SELECT COUNT(*) n, AVG(label) wr, SUM(pnl) tp, AVG(pnl) ap FROM trades {w}", p)
        r = c.fetchone()
        out["totals"] = {"trades": r["n"] or 0,
                         "win_rate": round((r["wr"] or 0) * 100, 1),
                         "total_pnl": round(r["tp"] or 0, 2),
                         "avg_pnl": round(r["ap"] or 0, 2)}

        c.execute(f"SELECT symbol, COUNT(*) n, AVG(label) wr, SUM(pnl) tp "
                  f"FROM trades {w} GROUP BY symbol ORDER BY n DESC", p)
        out["by_symbol"] = [{"symbol": x["symbol"], "trades": x["n"],
                             "win_rate": round((x["wr"] or 0) * 100, 1),
                             "total_pnl": round(x["tp"] or 0, 2)} for x in c.fetchall()]

        c.execute(f"SELECT direction, COUNT(*) n, AVG(label) wr, SUM(pnl) tp "
                  f"FROM trades {w} GROUP BY direction", p)
        out["by_direction"] = [{"direction": x["direction"], "trades": x["n"],
                                "win_rate": round((x["wr"] or 0) * 100, 1),
                                "total_pnl": round(x["tp"] or 0, 2)} for x in c.fetchall()]

        c.execute(f"SELECT reason, COUNT(*) n, AVG(label) wr, SUM(pnl) tp "
                  f"FROM trades {w} GROUP BY reason ORDER BY n DESC", p)
        out["by_reason"] = [{"reason": x["reason"], "trades": x["n"],
                             "win_rate": round((x["wr"] or 0) * 100, 1),
                             "total_pnl": round(x["tp"] or 0, 2)} for x in c.fetchall()]

        c.execute(f"SELECT ts, symbol, direction, mode, entry_price, exit_price, "
                  f"pnl, label, reason, entry_conf, composite "
                  f"FROM trades {w} ORDER BY id DESC LIMIT ?", (*p, int(recent_limit)))
        out["recent"] = [dict(x) for x in c.fetchall()]

        c.execute(f"SELECT ts, pnl FROM trades {w} ORDER BY id ASC", p)
        cum = 0.0; series = []
        for x in c.fetchall():
            cum += float(x["pnl"] or 0)
            series.append({"ts": x["ts"], "cum_pnl": round(cum, 2)})
        out["cum_pnl"] = series

        conn.close()
    except Exception as e:
        logger.warning(f"summary failed: {e}")
    return out


if __name__ == "__main__":
    init_ledger()
    print("Ledger initialized at", LEDGER_PATH)
    print(stats())
