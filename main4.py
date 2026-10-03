import sys
import os

# ============================================================
# ⚙️ RUNTIME ENVIRONMENT PATH INJECTION
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "Project", "ML"))
if "/root/Project/ML" not in sys.path:
    sys.path.append("/root/Project/ML")

import asyncio
import logging
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
import time
import pandas_ta as ta
import ccxt.async_support as ccxt
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
import aiohttp
from app.verify.engineer_and_train import apply_mega_features

# 🟢 SOCKET HELPERS (Must be async/await)
from app.services.socket_emitter import emit_log, emit_status, emit_trade_alert

# ============================================================
# 🔧 IMPORT CONFIGURATION
# ============================================================
from app.config2 import (
    MODEL_DIR,
    RESULTS_DIR,
    DEFAULT_TAKER_FEE,
    KRAKEN_TAKER_FEE,
    FEATURE_COLUMNS,
)
from app.predictors.stacking_predictor import StackingPredictor
from app.predictors.model_factory import clear_model_cache
from app.services import trade_recorder

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

GLOBAL_SESSION: Optional[aiohttp.ClientSession] = None
ACTIVE_BOTS = {}
TASK_REGISTRY = {}
# 🛑 Global emergency kill switch. When on, ALL bots stop opening NEW entries
# (open positions are still managed by their stops/exits). Toggle via
# POST /api/fleet/killswitch. Resets to off on restart.
KILL_SWITCH = {"on": False}

# ============================================================
# 🧠 PREDICTOR MODEL IN-MEMORY CACHE
# ============================================================
_PREDICTOR_CACHE: Dict[str, StackingPredictor] = {}

def get_cached_predictor(symbol: str, timeframe: str = "1h") -> StackingPredictor:
    cache_key = f"{symbol}_{timeframe}"
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = StackingPredictor(symbol=symbol, timeframe=timeframe)
        logger.info(f"🧠 Cached predictor for {cache_key}")
    return _PREDICTOR_CACHE[cache_key]


# ============================================================
# 🧠 LEDGER MODEL (trained on the bot's OWN closed trades) — optional entry gate
# ============================================================
_LEDGER_MODEL_CACHE: Dict[str, Any] = {}

def _ledger_pwin(symbol: str, timeframe: str, feat_row) -> Optional[float]:
    """P(win) for the current setup from the ledger-trained model (see
    ledger_trainer.py), or None if no model exists. Cached + fully guarded so it
    never disturbs the trading loop."""
    try:
        key = f"{symbol}_{timeframe}"
        if key not in _LEDGER_MODEL_CACHE:
            _p = os.path.join(MODEL_DIR, f"{symbol.split('-')[0].lower()}_{timeframe}_ledger_model.joblib")
            _LEDGER_MODEL_CACHE[key] = joblib.load(_p) if os.path.exists(_p) else None
        payload = _LEDGER_MODEL_CACHE[key]
        if not payload:
            return None
        model = payload["model"]; feats = payload.get("feature_names", FEATURE_COLUMNS)
        X = np.array([[float(feat_row.get(c, 0.0)) for c in feats]])
        proba = model.predict_proba(X)[0]
        return float(proba[1] if len(proba) > 1 else proba[0])
    except Exception:
        return None


# ============================================================
# 🚨 SYSTEM RISK BREAKER (multi-day live drawdown -> halt new live entries)
# ============================================================
_SYS_HALT_CACHE: Dict[str, Any] = {"ts": 0.0, "halted": {}}

def _system_live_halt(user_id: str, ref_capital: float, window_days: int, max_dd_pct: float) -> bool:
    """True when this user's realized LIVE PnL over the last `window_days` is worse
    than -max_dd_pct%% of ref_capital. Reads the trade ledger; cached ~5 min; fully
    guarded so it never disturbs the trading loop."""
    try:
        now = time.time()
        if now - _SYS_HALT_CACHE["ts"] > 300:
            _SYS_HALT_CACHE["ts"] = now
            _SYS_HALT_CACHE["halted"] = {}
        if user_id in _SYS_HALT_CACHE["halted"]:
            return _SYS_HALT_CACHE["halted"][user_id]
        from app.config2 import DATA_DIR as _DD
        lp = os.path.join(_DD, "trade_ledger.db")
        if not os.path.exists(lp) or not ref_capital:
            _SYS_HALT_CACHE["halted"][user_id] = False
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_days))).isoformat()
        conn = sqlite3.connect(lp, timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades "
                    "WHERE user_id = ? AND mode = 'live' AND ts >= ?", (user_id, cutoff))
        total = float(cur.fetchone()[0] or 0.0)
        conn.close()
        halted = total <= -abs(float(max_dd_pct)) / 100.0 * float(ref_capital)
        _SYS_HALT_CACHE["halted"][user_id] = halted
        return halted
    except Exception:
        return False


# ============================================================
# 🚢 FLEET-LEVEL RISK CAP (portfolio drawdown guard across a whole fleet)
# ============================================================
FLEET_STATE: Dict[str, Any] = {}   # {fleet_id: {"peak": float, "halted": bool, "ts": float}}

def _fleet_risk_halt(fleet_id: str, max_dd_pct: float) -> bool:
    """Portfolio guard: halt NEW entries across an ENTIRE fleet when its combined
    (realized) balance draws down more than max_dd_pct%% from its peak. Individual
    bots have their own breakers; this catches the case where the whole roster
    bleeds together. Cheap + in-memory (sums sibling balances), cached ~20s. Open
    positions are always still managed."""
    try:
        if not fleet_id or max_dd_pct <= 0:
            return False
        st = FLEET_STATE.setdefault(fleet_id, {"peak": 0.0, "halted": False, "ts": 0.0})
        now = time.time()
        if now - st["ts"] < 20:
            return st["halted"]
        st["ts"] = now
        prefix = f"{fleet_id}::"
        total = 0.0; seen = 0
        for k, b in ACTIVE_BOTS.items():
            if k.startswith(prefix):
                total += float(b.get("balance", 0) or 0); seen += 1
        if seen == 0:
            st["halted"] = False
            return False
        st["peak"] = max(st["peak"], total) if st["peak"] > 0 else total
        dd = (st["peak"] - total) / st["peak"] if st["peak"] > 0 else 0.0
        st["halted"] = dd >= (abs(float(max_dd_pct)) / 100.0)
        return st["halted"]
    except Exception:
        return False


# ============================================================
# 🔐 ORDER IDEMPOTENCY (opt-in tag for the live order path)
# ============================================================
def _client_order_id(user_id: str, symbol: str, side: str) -> str:
    """Deterministic idempotency tag for a live order (opt-in via useClientOrderId).
    Buckets to ~2s so a rapid retry of the SAME logical order reuses the id (the
    exchange dedupes it) while distinct orders differ — and every fill can be
    reconciled back to intent. Short + alphanumeric for broad exchange support."""
    import hashlib
    raw = f"{user_id}|{symbol}|{side}|{int(time.time() // 2)}"
    return "neo" + hashlib.sha1(raw.encode()).hexdigest()[:18]


# ============================================================
# 🔁 MODEL HOT-RELOAD (paired with retrain_and_promote.py)
# ============================================================
RELOAD_SENTINEL = os.path.join(MODEL_DIR, ".reload")

def _maybe_hot_reload(state: dict):
    """If the retrainer promoted new models, drop cached models so the council
    reloads fresh weights on the next prediction. Cheap file-stat per call."""
    try:
        if not os.path.exists(RELOAD_SENTINEL):
            return
        mtime = os.path.getmtime(RELOAD_SENTINEL)
        if state.get('_last_reload_mtime') != mtime:
            state['_last_reload_mtime'] = mtime
            clear_model_cache()
            _PREDICTOR_CACHE.clear()
            _LEDGER_MODEL_CACHE.clear()
            logger.info("🔁 Hot-reload: promoted models detected — model caches cleared.")
    except Exception as e:
        logger.warning(f"hot-reload check failed: {e}")


# ==========================================
# 🗄️ DATABASE HANDLER
# ==========================================
# ------------------------------------------------------------------
# 🔐 SECRET REDACTION
# ------------------------------------------------------------------
# Exchange credentials arrive inside the bot config and must stay in
# memory only (ACTIVE_BOTS) for order placement. They must never be
# written to disk, echoed in responses, or logged.
_SECRET_KEYS = {
    "api_keys", "apiKey", "secret", "passphrase",
    "krakenKey", "krakenSecret", "coinbaseKey", "coinbaseSecret",
}


def _redact_secrets(config):
    """Return a shallow copy of a config dict with secret fields removed."""
    if not isinstance(config, dict):
        return config
    return {k: v for k, v in config.items() if k not in _SECRET_KEYS}


class DatabaseHandler:
    DB_FILE = "bot_state.db"

    @classmethod
    def init_db(cls):
        try:
            conn = sqlite3.connect(cls.DB_FILE)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS bot_sessions
                         (user_id TEXT PRIMARY KEY, config TEXT, balance REAL,
                          positions TEXT, trade_history TEXT, equity_curve TEXT, logs TEXT,
                          status TEXT, last_update TIMESTAMP)''')
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully.")
        except Exception as e:
            print(f"❌ Database Init Error: {e}")

    @classmethod
    def save_state(cls, user_id, bot_data):
        conn = sqlite3.connect(cls.DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO bot_sessions
                     (user_id, config, balance, positions, trade_history, equity_curve, logs, status, last_update)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id,
                   json.dumps(_redact_secrets(bot_data['config'])),
                   bot_data['balance'],
                   json.dumps(bot_data['positions']),
                   json.dumps(bot_data['trade_history']),
                   json.dumps(bot_data.get('equityCurve', [])),
                   json.dumps(bot_data.get('logs', [])),
                   bot_data['status'],
                   datetime.now().isoformat()))
        conn.commit()
        conn.close()

    @classmethod
    def load_state(cls, user_id):
        conn = sqlite3.connect(cls.DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        try:
            c.execute("SELECT * FROM bot_sessions WHERE user_id = ?", (user_id,))
        except sqlite3.OperationalError:
            conn.close()
            cls.init_db()
            return None
        row = c.fetchone()
        conn.close()
        if row:
            try:
                return {
                    "config":        json.loads(row['config']),
                    "balance":       row['balance'],
                    "positions":     json.loads(row['positions']),
                    "trade_history": json.loads(row['trade_history']),
                    "equityCurve":   json.loads(row['equity_curve']),
                    "logs":          json.loads(row['logs']),
                    "status":        "stopped"
                }
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
                return None
        return None

    @classmethod
    def list_running_ids(cls):
        """Return the user_ids of every session persisted as status='running' — the
        bots that were live when the engine last went down. Used by the startup
        auto-resume so a deploy/reboot no longer silently stops them."""
        try:
            conn = sqlite3.connect(cls.DB_FILE)
            c = conn.cursor()
            try:
                c.execute("SELECT user_id FROM bot_sessions WHERE status = 'running'")
                ids = [r[0] for r in c.fetchall()]
            except sqlite3.OperationalError:
                ids = []
            conn.close()
            return ids
        except Exception as e:
            logger.error(f"list_running_ids failed: {e}")
            return []

DatabaseHandler.init_db()
trade_recorder.init_ledger()

# --- REQUEST MODELS ---
class BotStartRequest(BaseModel):
    userId: str
    config: Dict[str, Any]

class BotStopRequest(BaseModel):
    userId: str

class BotClosePositionRequest(BaseModel):
    userId: str
    symbol: str

class StrategyConfig(BaseModel):
    code: str
    params: Dict[str, Any]

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    initialBalance: float = 1000.0
    riskPercentage: float = 1.0
    code: str
    mlModel: str = "stacking"
    ml_confidence_threshold: Optional[float] = 0.8
    mlThresholdLong: Optional[float] = 0.8
    mlThresholdShort: Optional[float] = 0.8
    trend_strategy: Optional[str] = "atr_breakout"
    range_strategy: Optional[str] = "bollinger_reversal"
    trade_direction: Optional[str] = "BOTH"
    params: Optional[Dict[str, Any]] = {}

BacktestRequest.model_rebuild()

class ComboRequest(BaseModel):
    symbol: str
    timeframe: str
    startDate: str
    endDate: str
    initialBalance: float
    strategies: List[StrategyConfig]
    combinationRule: str = "OR"
    riskPercentage: float = 1.0
    take_profit: Optional[float] = 0.06
    stop_loss: Optional[float] = 0.03
    trailing_stop: Optional[float] = 0.02
    mlModel: Optional[str] = "stacking"
    mlThresholdLong: float = 0.8
    mlThresholdShort: float = 0.8
    mlMode: Optional[str] = None
    advanced_filters: Optional[Dict] = {}
    params: Optional[Dict[str, Any]] = {}

ComboRequest.model_rebuild()

# ============================================================
# 🧠 SELF-LEARNING LOOP (autonomous retrain → validate → promote → hot-reload)
# ============================================================
# The engine periodically refreshes market data, retrains the XGBoost + Random
# Forest experts, and PROMOTES a fresh model only if it beats the live one on a
# recent holdout (champion/challenger). On promotion it drops app/models/.reload
# so the running council hot-reloads on its next tick — no restart, no human.
# Toggle/tune with env: SELF_LEARN_ENABLED, SELF_LEARN_INTERVAL_HOURS,
# SELF_LEARN_SYMBOLS, SELF_LEARN_TIMEFRAME.
SELF_LEARN_ENABLED = os.getenv("SELF_LEARN_ENABLED", "true").lower() == "true"
SELF_LEARN_HOURS   = float(os.getenv("SELF_LEARN_INTERVAL_HOURS", "6"))
SELF_LEARN_SYMBOLS = [s.strip() for s in os.getenv("SELF_LEARN_SYMBOLS", "BTC-USD,ETH-USD,SOL-USD").split(",") if s.strip()]
SELF_LEARN_TF      = os.getenv("SELF_LEARN_TIMEFRAME", "1h")
# Every Nth cycle also retrain the heavier models (LSTM transformer + stacking
# judge). Default 4 => with a 6h cycle, a deep refresh once a day.
SELF_LEARN_DEEP_EVERY = int(os.getenv("SELF_LEARN_DEEP_EVERY", "4"))

# The live FLEET trades these coins on these timeframes. The self-learning ledger
# model must be trained for exactly these (coin, tf) pairs — otherwise the file
# the fleet bots look up ({coin}_{tf}_ledger_model.joblib) never exists and the
# gate stays a no-op. (train_ledger_model keys the FILE by tf but reads a coin's
# whole history, so training 4h + 1d just produces the names the fleet queries.)
LEDGER_FLEET_SYMBOLS = [s.strip() for s in os.getenv(
    "LEDGER_FLEET_SYMBOLS", "BTC-USD,ETH-USD,SOL-USD,DOGE-USD,XRP-USD").split(",") if s.strip()]
LEDGER_FLEET_TFS = [s.strip() for s in os.getenv("LEDGER_FLEET_TFS", "4h,1d").split(",") if s.strip()]


def _run_self_learn_once(deep=False):
    """Blocking (runs in a worker thread). Each cycle: retrain XGB+RF experts
    (champion/challenger) and the ledger model (bot's own trades). On a DEEP
    cycle also retrain the LSTM transformer and the stacking judge. Touches the
    reload sentinel if anything changed so the live council hot-reloads."""
    from retrain_and_promote import retrain_symbol, RELOAD_SENTINEL as _SENT
    changed = False

    # 1. Experts (XGBoost + RandomForest) — fast, promote only if better.
    for sym in SELF_LEARN_SYMBOLS:
        try:
            res = retrain_symbol(sym, SELF_LEARN_TF, dry_run=False)
            changed = changed or any(r.get("promoted") for r in res.values())
            logger.info(f"🧠 experts {sym}: {res}")
        except Exception as e:
            logger.error(f"🧠 experts {sym} failed: {e}")

    # 2. Ledger model — learns from the bot's OWN closed trades (features->win/loss).
    #    Train the (coin, tf) pairs the LIVE FLEET actually queries (5 coins x
    #    4h/1d) PLUS the self-learn TF, so the self-learning gate can activate for
    #    the fleet — not just the 1h self-learn models. Each (coin, tf) is a no-op
    #    until that coin has enough closed trades (ledger_trainer.MIN_SAMPLES).
    try:
        from ledger_trainer import train_ledger_model
        _led_syms = sorted(set(SELF_LEARN_SYMBOLS) | set(LEDGER_FLEET_SYMBOLS))
        _led_tfs = sorted(set([SELF_LEARN_TF]) | set(LEDGER_FLEET_TFS))
        for sym in _led_syms:
            for tf in _led_tfs:
                try:
                    lr = train_ledger_model(sym, tf)
                    changed = changed or bool(lr.get("trained"))
                    if lr.get("trained"):
                        logger.info(f"🧠 ledger {sym} {tf}: {lr}")
                except Exception as e:
                    logger.error(f"🧠 ledger {sym} {tf} failed: {e}")
    except Exception as e:
        logger.error(f"🧠 ledger training failed: {e}")

    # 3. DEEP: LSTM transformer + stacking judge (heavier; less often).
    if deep:
        try:
            from trainTransformers import train_transformer
            for sym in SELF_LEARN_SYMBOLS:
                try:
                    if train_transformer(sym):
                        changed = True
                except Exception as e:
                    logger.error(f"🧠 transformer {sym} failed: {e}")
        except Exception as e:
            logger.error(f"🧠 transformer import failed: {e}")
        try:
            from train_judge import retrain_turbo_judge
            retrain_turbo_judge()
            changed = True
            logger.info("🧠 judge retrained (stacking meta-model).")
        except Exception as e:
            logger.error(f"🧠 judge retrain failed: {e}")

        # Autonomous edge search: walk-forward the config grid and persist the
        # winners to data/discovered_edges.json (served at GET /api/discover/edges).
        try:
            from edge_discovery import discover
            rep = discover(SELF_LEARN_SYMBOLS, None, 20, True)
            logger.info(f"🧠 edge discovery: {rep.get('edges_found', 0)} edge(s) / {rep.get('tested', 0)} configs tested.")
        except Exception as e:
            logger.error(f"🧠 edge discovery failed: {e}")

    if changed:
        try:
            with open(_SENT, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
            logger.info("🧠 self-learn: models updated — reload sentinel touched.")
        except Exception as e:
            logger.warning(f"🧠 self-learn sentinel write failed: {e}")
    return changed


async def _self_learning_loop():
    # Delay the first cycle so training CPU doesn't fight cold-start.
    await asyncio.sleep(120)
    cycle = 0
    while True:
        deep = (cycle % max(1, SELF_LEARN_DEEP_EVERY) == 0)  # first cycle is deep
        try:
            logger.info(f"🧠 self-learn cycle {cycle} (deep={deep}) on {SELF_LEARN_SYMBOLS} @ {SELF_LEARN_TF}…")
            await asyncio.to_thread(_run_self_learn_once, deep)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"🧠 self-learn loop error: {e}")
        cycle += 1
        await asyncio.sleep(max(0.25, SELF_LEARN_HOURS) * 3600)


# ============================================================
# 🧭 MACRO REGIME ROUTER (BTC-driven risk-on / risk-off tilt)
# ============================================================
# A light supervisor that classifies the WHOLE market from BTC's daily trend and
# publishes it to MACRO_REGIME. Fleet bots with macroRegimeGate=True consult it to
# hold back LONGS in risk-off and SHORTS in risk-on — so the fleet leans WITH the
# market, not just each coin with itself. Only gates NEW entries; open positions
# are always still managed. Any bot that doesn't opt in is unaffected.
MACRO_REGIME = {"state": "neutral", "ts": 0.0, "detail": {}}
# Per-timeframe BTC trend feed for the live `btc_regime` strategy:
#   {timeframe: {"vote": +1/-1, "price": .., "ema50": .., "ts": ..}}
BTC_TREND = {}
MACRO_REFRESH_SEC = int(os.getenv("MACRO_REGIME_REFRESH_SEC", "600"))

async def _macro_regime_loop():
    await asyncio.sleep(20)  # let the app settle before the first fetch
    while True:
        try:
            daily_closes = None
            # Per-timeframe BTC trend (drives the live btc_regime strategy).
            for _tf in ("1h", "4h", "1d"):
                rows = await fetch_live_candles_ccxt("BTC-USD", _tf, 260)
                if not rows or len(rows) < 55:
                    continue
                c = pd.Series([float(r["close"]) for r in rows])
                price = float(c.iloc[-1]); ema50 = float(ta.ema(c, length=50).iloc[-1])
                BTC_TREND[_tf] = {"vote": (1 if price > ema50 else -1),
                                  "price": round(price, 2), "ema50": round(ema50, 2),
                                  "ts": time.time()}
                if _tf == "1d":
                    daily_closes = c
            # Macro risk-on/off from the BTC daily stack.
            if daily_closes is not None and len(daily_closes) >= 60:
                price  = float(daily_closes.iloc[-1])
                ema50  = float(ta.ema(daily_closes, length=50).iloc[-1])
                ema200 = float(ta.ema(daily_closes, length=200).iloc[-1]) if len(daily_closes) >= 200 else ema50
                if price > ema50 and ema50 >= ema200:
                    state = "risk_on"
                elif price < ema50 and ema50 <= ema200:
                    state = "risk_off"
                else:
                    state = "neutral"
                prev = MACRO_REGIME.get("state")
                MACRO_REGIME.update({"state": state, "ts": time.time(),
                    "detail": {"btc_price": round(price, 2), "ema50": round(ema50, 2),
                               "ema200": round(ema200, 2)}})
                if state != prev:
                    logger.info(f"🧭 MACRO REGIME → {state.upper()} "
                                f"(BTC {price:.0f} | ema50 {ema50:.0f} | ema200 {ema200:.0f})")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"🧭 macro regime loop error: {e}")
        await asyncio.sleep(max(60, MACRO_REFRESH_SEC))


# ============================================================
# 🗓️ MONTHLY EDGE RE-VALIDATION (structural audit of the exit edge)
# ============================================================
# Re-runs exit_lab's cross-coin walk-forward on a slow cadence and refreshes
# exit_lab_results.json. This is the STRUCTURAL check ("is the edge still real
# across years of history?") — NOT the fast market-change responder. Real-time
# adaptation is already handled by the macro regime router + per-bar trend gates
# + the drift monitor; a walk-forward over years barely moves week-to-week, so a
# ~monthly cadence is correct (weekly would be noise). Fully guarded + off the
# event loop so it never disturbs trading.
AUDIT_ENABLED       = os.getenv("EXIT_AUDIT_ENABLED", "true").lower() == "true"
AUDIT_INTERVAL_DAYS = float(os.getenv("EXIT_AUDIT_INTERVAL_DAYS", "30"))

async def _monthly_audit_loop():
    if not AUDIT_ENABLED:
        logger.info("🗓️ Monthly exit audit disabled (EXIT_AUDIT_ENABLED != true).")
        return
    from app.config2 import DATA_DIR as _DD
    results_path = os.path.join(_DD, "exit_lab_results.json")
    await asyncio.sleep(45)  # let the app settle before any heavy work
    while True:
        try:
            due = True
            if os.path.exists(results_path):
                age_days = (time.time() - os.path.getmtime(results_path)) / 86400.0
                due = age_days >= AUDIT_INTERVAL_DAYS
            if due:
                logger.info("🗓️ Exit-edge audit starting (cross-coin walk-forward)…")
                from exit_lab import optimize_exits
                rep = await asyncio.to_thread(optimize_exits, None, None, None, None, None, 25, True)
                logger.info(f"🗓️ Exit-edge audit done: {rep.get('generalizing_configs')}"
                            f"/{rep.get('configs_tested')} configs generalize "
                            f"→ exit_lab_results.json refreshed.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"🗓️ monthly audit error: {e}")
        # Re-check daily; the file-age gate above enforces the ~monthly cadence
        # and also catches up if the engine was down when an audit came due.
        await asyncio.sleep(86400)


# ============================================================
# ♻️  VALIDATION RECALIBRATION ("harden the system" — on demand + on a cadence)
# ============================================================
# Re-runs the HARD out-of-sample + cost-stress validation that decides which
# coins may pyramid live (exit_lab.validate_exit rewrites the eligibility
# registry the fleet reads). POST /api/fleet/recalibrate runs it now; a loop
# runs it on a cadence. "Levels" tighten the per-coin survive bar (min trades,
# expectancy-R margin, profit factor), how many coins must generalize, the
# out-of-sample size, and the cost stress. All off the event loop — never blocks
# trading, and thresholds are always restored so no other caller inherits them.
RECALIB_ENABLED  = os.getenv("RECALIB_ENABLED", "true").lower() == "true"
RECALIB_HOURS    = float(os.getenv("RECALIB_INTERVAL_HOURS", "24"))
RECALIB_LEVEL    = os.getenv("RECALIB_LEVEL", "strict")
RECALIB_MAX_LEGS = int(os.getenv("RECALIB_MAX_LEGS", "2"))  # pyramids capped at 1-2 (fewer legs = less fee drag + less overfit)
RECALIB_LEVELS = {
    "normal":   {"min_trades": 6,  "exp_r": 0.02, "pf": 1.05, "coins": 3, "holdout": 0.25, "fee_mult": 2.0, "slip_mult": 3.0},
    "strict":   {"min_trades": 10, "exp_r": 0.05, "pf": 1.15, "coins": 4, "holdout": 0.30, "fee_mult": 3.0, "slip_mult": 4.0},
    "paranoid": {"min_trades": 15, "exp_r": 0.10, "pf": 1.25, "coins": 4, "holdout": 0.35, "fee_mult": 4.0, "slip_mult": 6.0},
}
RECALIB_STATE = {"running": False, "last": None}


async def _run_recalibration(level="strict", max_legs=None):
    """Apply exit_lab's hardened thresholds for this run, re-validate the fleet's
    config (4h / regime / LONG / trend_ride) at each pyramiding depth, and let
    validate_exit rewrite the eligibility registry. Thresholds are ALWAYS restored
    afterwards, and RECALIB_STATE['running'] is always cleared."""
    RECALIB_STATE["running"] = True
    summary = None
    try:
        import exit_lab
        lv = RECALIB_LEVELS.get(level, RECALIB_LEVELS["strict"])
        legs_max = max(1, int(max_legs or RECALIB_MAX_LEGS))
        snap = (exit_lab.MIN_HOLDOUT_TRADES, exit_lab.MIN_SURV_EXPECTANCY_R,
                exit_lab.MIN_SURV_PROFIT_FACTOR, exit_lab.MIN_COINS_GENERALIZE)
        by_legs = []
        try:
            exit_lab.MIN_HOLDOUT_TRADES     = lv["min_trades"]
            exit_lab.MIN_SURV_EXPECTANCY_R  = lv["exp_r"]
            exit_lab.MIN_SURV_PROFIT_FACTOR = lv["pf"]
            exit_lab.MIN_COINS_GENERALIZE   = lv["coins"]
            for legs in range(1, legs_max + 1):
                rep = await asyncio.to_thread(
                    exit_lab.validate_exit, None, "4h", "regime", "LONG", "trend_ride",
                    float(lv["holdout"]), float(lv["fee_mult"]), float(lv["slip_mult"]),
                    int(legs), 1.0)
                by_legs.append({
                    "legs": legs, "verdict": rep.get("verdict"),
                    "cleared_coins": rep.get("cleared_coins", []),
                    "survives_oos": rep.get("survives_oos"),
                    "survives_stress": rep.get("survives_stress"),
                    "coins_tested": rep.get("coins_tested"),
                })
        finally:
            (exit_lab.MIN_HOLDOUT_TRADES, exit_lab.MIN_SURV_EXPECTANCY_R,
             exit_lab.MIN_SURV_PROFIT_FACTOR, exit_lab.MIN_COINS_GENERALIZE) = snap
        summary = {
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "level": level, "thresholds": lv, "by_legs": by_legs,
        }
        RECALIB_STATE["last"] = summary
        try:
            from app.config2 import DATA_DIR as _DD
            with open(os.path.join(_DD, "recalibration_status.json"), "w") as _f:
                json.dump(summary, _f, indent=2)
        except Exception as _e:
            logger.warning(f"♻️ recalibration status save failed: {_e}")
    except Exception as e:
        logger.error(f"♻️ recalibration failed: {e}")
    finally:
        RECALIB_STATE["running"] = False
    return summary


async def _recalibration_loop():
    if not RECALIB_ENABLED:
        logger.info("♻️ Auto-recalibration disabled (RECALIB_ENABLED != true).")
        return
    from app.config2 import DATA_DIR as _DD
    status_path = os.path.join(_DD, "recalibration_status.json")
    await asyncio.sleep(90)  # let the app settle before any heavy work
    while True:
        try:
            due = True
            if os.path.exists(status_path):
                age_h = (time.time() - os.path.getmtime(status_path)) / 3600.0
                due = age_h >= RECALIB_HOURS
            if due and not RECALIB_STATE.get("running"):
                logger.info(f"♻️ Auto-recalibration starting (level={RECALIB_LEVEL})…")
                rep = await _run_recalibration(RECALIB_LEVEL, RECALIB_MAX_LEGS)
                if rep:
                    logger.info("♻️ Auto-recalibration done: "
                                + ", ".join(f"x{r['legs']}:{r['verdict']}" for r in rep.get("by_legs", [])))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"♻️ recalibration loop error: {e}")
        await asyncio.sleep(max(1.0, min(RECALIB_HOURS, 6.0)) * 3600)


# ============================================================
# 🔬 AUTOMATED RESEARCH (always hunt for the best-performing configs)
# ============================================================
# Sweeps exit/sizing styles across coins (walk-forward, ranked by CROSS-COIN
# generalization) via exit_lab.optimize_exits — the "find the best performers"
# search. POST /api/fleet/research runs it now; a loop runs it on a cadence.
# Heavy, so it runs off the event loop with a file-age gate. Results land in
# data/exit_lab_results.json (same file the Evidence panel + /results read).
RESEARCH_ENABLED = os.getenv("RESEARCH_ENABLED", "true").lower() == "true"
RESEARCH_HOURS   = float(os.getenv("RESEARCH_INTERVAL_HOURS", "168"))  # weekly
RESEARCH_FOLDS   = int(os.getenv("RESEARCH_FOLDS", "25"))
RESEARCH_STATE = {"running": False, "last_ran": None, "summary": None}


async def _run_research():
    """Run the cross-coin exit/sizing sweep (optimize_exits) and refresh
    data/exit_lab_results.json with the ranked best performers. Off the event
    loop; a running flag lets the UI show progress."""
    RESEARCH_STATE["running"] = True
    try:
        from exit_lab import optimize_exits
        rep = await asyncio.to_thread(optimize_exits, None, None, None, None, None,
                                      RESEARCH_FOLDS, True)
        RESEARCH_STATE["last_ran"] = datetime.now(timezone.utc).isoformat()
        RESEARCH_STATE["summary"] = {
            "ran_at": RESEARCH_STATE["last_ran"],
            "configs_tested": rep.get("configs_tested"),
            "generalizing_configs": rep.get("generalizing_configs"),
        }
    except Exception as e:
        logger.error(f"🔬 research failed: {e}")
    finally:
        RESEARCH_STATE["running"] = False
    return RESEARCH_STATE.get("summary")


async def _research_loop():
    if not RESEARCH_ENABLED:
        logger.info("🔬 Auto-research disabled (RESEARCH_ENABLED != true).")
        return
    from app.config2 import DATA_DIR as _DD
    results_path = os.path.join(_DD, "exit_lab_results.json")
    await asyncio.sleep(120)  # settle after boot (and after the first recalibration)
    while True:
        try:
            due = True
            if os.path.exists(results_path):
                age_h = (time.time() - os.path.getmtime(results_path)) / 3600.0
                due = age_h >= RESEARCH_HOURS
            if due and not RESEARCH_STATE.get("running"):
                logger.info("🔬 Auto-research starting (cross-coin exit sweep)…")
                rep = await _run_research()
                if rep:
                    logger.info(f"🔬 Auto-research done: {rep.get('generalizing_configs')}"
                                f"/{rep.get('configs_tested')} configs generalize.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"🔬 research loop error: {e}")
        await asyncio.sleep(max(1.0, min(RESEARCH_HOURS, 24.0)) * 3600)


# ============================================================
# 🔄 CROSS-RESTART PERSISTENCE (auto-resume running bots on startup)
# ============================================================
# On boot, relaunch every bot that was 'running' when the engine last went down
# (saved in bot_state.db). Each bot's full state — config, balance, open
# positions, history — is restored and its heartbeat relaunched, so a deploy or
# reboot no longer silently stops the Live bot + the Fleet. SAFETY: saved configs
# have redacted secrets, so a resumed LIVE bot auto-falls-back to PAPER (it can't
# move real funds without keys being re-injected) — exactly what we want.
AUTO_RESUME_BOTS = os.getenv("AUTO_RESUME_BOTS", "true").lower() == "true"

async def _resume_saved_bots():
    if not AUTO_RESUME_BOTS:
        logger.info("🔄 Auto-resume disabled (AUTO_RESUME_BOTS != true).")
        return
    await asyncio.sleep(8)  # let the app + global session settle first
    try:
        ids = await asyncio.to_thread(DatabaseHandler.list_running_ids)
    except Exception as e:
        logger.error(f"🔄 auto-resume: list failed: {e}")
        return
    if not ids:
        logger.info("🔄 Auto-resume: no running bots to restore.")
        return
    logger.info(f"🔄 Auto-resume: restoring {len(ids)} bot(s)…")
    for uid in ids:
        if uid in ACTIVE_BOTS:
            continue  # already live (shouldn't happen at boot)
        try:
            state = await asyncio.to_thread(DatabaseHandler.load_state, uid)
            if not state:
                continue
            state["status"] = "running"           # load_state forces 'stopped'; we ARE resuming
            state.setdefault("logs", [])
            state.setdefault("equityCurve", [])
            state.setdefault("positions", [])
            state.setdefault("trade_history", [])
            state["startedAt"] = state.get("startedAt") or datetime.now(timezone.utc).isoformat()
            ACTIVE_BOTS[uid] = state
            TASK_REGISTRY[uid] = asyncio.create_task(live_neural_heartbeat(uid))
            logger.info(f"🔄 Resumed {uid} (balance ${state.get('balance')}, "
                        f"{len(state.get('positions', []))} open position(s))")
            await asyncio.sleep(1.5)              # stagger candle fetches across bots
        except Exception as e:
            logger.error(f"🔄 auto-resume {uid} failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_SESSION
    GLOBAL_SESSION = aiohttp.ClientSession()
    _self_learn_task = None
    if SELF_LEARN_ENABLED:
        _self_learn_task = asyncio.create_task(_self_learning_loop())
        logger.info(f"🧠 Self-learning ENABLED: every {SELF_LEARN_HOURS}h on {SELF_LEARN_SYMBOLS} ({SELF_LEARN_TF}).")
    else:
        logger.info("🧠 Self-learning disabled (set SELF_LEARN_ENABLED=true to enable).")
    _macro_task = asyncio.create_task(_macro_regime_loop())
    logger.info("🧭 Macro regime router started (BTC risk-on/off tilt for opt-in fleet bots).")
    _resume_task = asyncio.create_task(_resume_saved_bots())
    logger.info(f"🔄 Auto-resume {'ENABLED' if AUTO_RESUME_BOTS else 'disabled'} "
                f"(restores running bots after a restart).")
    _audit_task = asyncio.create_task(_monthly_audit_loop())
    logger.info(f"🗓️ Monthly exit-edge audit {'ENABLED' if AUDIT_ENABLED else 'disabled'} "
                f"(every {AUDIT_INTERVAL_DAYS:.0f}d; structural re-validation, off the event loop).")
    _recalib_task = asyncio.create_task(_recalibration_loop())
    logger.info(f"♻️ Auto-recalibration {'ENABLED' if RECALIB_ENABLED else 'disabled'} "
                f"(every {RECALIB_HOURS:.0f}h, level={RECALIB_LEVEL}; re-validates the live-pyramiding gate).")
    _research_task = asyncio.create_task(_research_loop())
    logger.info(f"🔬 Auto-research {'ENABLED' if RESEARCH_ENABLED else 'disabled'} "
                f"(every {RESEARCH_HOURS:.0f}h; hunts for the best-performing configs).")
    yield
    if _self_learn_task is not None:
        _self_learn_task.cancel()
    _macro_task.cancel()
    _resume_task.cancel()
    try:
        _audit_task.cancel()
    except Exception:
        pass
    try:
        _recalib_task.cancel()
    except Exception:
        pass
    try:
        _research_task.cancel()
    except Exception:
        pass
    await GLOBAL_SESSION.close()
    # Persist running bots for CROSS-RESTART RESUME. With auto-resume ON we keep
    # their status as-is ('running') so _resume_saved_bots relaunches them on the
    # next boot; with it OFF we mark them 'stopped' (legacy behavior). A bot the
    # USER stopped is already removed from ACTIVE_BOTS + saved 'stopped', so only
    # genuinely-running bots are persisted here.
    for user_id, bot in list(ACTIVE_BOTS.items()):
        if not AUTO_RESUME_BOTS:
            bot["status"] = "stopped"
        DatabaseHandler.save_state(user_id, bot)

app = FastAPI(title="NEO-V25.14 Sovereign Engine", lifespan=lifespan)

# CORS: restrict to an explicit allowlist. Set ALLOWED_ORIGINS in the
# environment (comma-separated) for production; defaults to local dev.
# A wildcard "*" with allow_credentials=True is invalid and unsafe.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
    ).split(",") if o.strip()
]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ============================================================
# 🔐 ENGINE API KEY (shared secret with the Node backend)
# ============================================================
# When ENGINE_API_KEY is set, every /api/* call (except /api/health) must carry a
# matching X-Internal-Key header. Unset = disabled (default), so this is safe to
# deploy now; set it on BOTH this engine and the Node backend to lock down :8000.
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")

@app.middleware("http")
async def _require_internal_key(request, call_next):
    if ENGINE_API_KEY:
        p = request.url.path
        if p.startswith("/api/") and p != "/api/health":
            if request.headers.get("x-internal-key") != ENGINE_API_KEY:
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Do NOT echo or print exc.body — request bodies can contain exchange
    # API keys/secrets. Log only the validation errors (field + type).
    logger.warning(f"❌ Request validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ==========================================
# HISTORICAL DATA MANAGEMENT LAYER
# ==========================================
def load_data_robust(symbol, timeframe):
    possible_paths = [
        f"data/{symbol.replace('/', '-')}-{timeframe}.csv",
        f"Project/ML/data/{symbol.replace('/', '-')}-{timeframe}.csv",
        f"{symbol.replace('/', '-')}-{timeframe}.csv"
    ]
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
    if not file_path:
        logger.error(f"❌ File not found. Searched: {possible_paths}")
        return None
    try:
        df = pd.read_csv(file_path)
        time_col = next((c for c in df.columns if c.lower() in ['timestamp', 'time', 'date']), None)
        if time_col:
            first_val = df[time_col].iloc[0]
            if isinstance(first_val, (int, float, np.number)):
                unit = 'ms' if first_val > 1e11 else 's'
                df[time_col] = pd.to_datetime(df[time_col], unit=unit, utc=True)
            else:
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
            df.rename(columns={time_col: 'timestamp'}, inplace=True)
            df.set_index('timestamp', inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df
    except Exception as e:
        logger.error(f"Data Load Error: {e}")
    return None


async def ensure_full_data(symbol, timeframe, start_str, end_str, *args, **kwargs):
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts   = int(pd.to_datetime(end_str).timestamp()   * 1000)
    df = load_data_robust(symbol, timeframe)
    if df is None:
        df = pd.DataFrame()
    needs_fetch = False
    if df.empty:
        needs_fetch = True
        current_since = start_ts
    else:
        local_start = int(df.index.min().timestamp() * 1000)
        if local_start > start_ts + 3600000:
            needs_fetch = True
            current_since = start_ts
        else:
            current_since = int(df.index.max().timestamp() * 1000)
    if needs_fetch or current_since < end_ts:
        print(f"📡 DATA GAP: Fetching {symbol} from Coinbase...")
        exchange = ccxt.coinbase({'enableRateLimit': True})
        all_new_candles = []
        fetch_symbol = symbol.replace("-", "/")
        try:
            while current_since < end_ts:
                new_batch = await exchange.fetch_ohlcv(fetch_symbol, timeframe,
                                                       since=current_since, limit=300)
                if not new_batch:
                    break
                all_new_candles.extend(new_batch)
                current_since = new_batch[-1][0] + 1
            await exchange.close()
        except Exception as e:
            print(f"⚠️ Coinbase Fetch Error: {e}")
            await exchange.close()
        if all_new_candles:
            new_df = pd.DataFrame(all_new_candles,
                                  columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['ts'], unit='ms', utc=True)
            new_df.set_index('timestamp', inplace=True)
            new_df.drop(columns=['ts'], inplace=True)
            df = pd.concat([df, new_df]).sort_index()
            df = df[~df.index.duplicated(keep='first')]
            try:
                os.makedirs("data", exist_ok=True)
                save_path = f"data/{symbol.replace('/', '-')}-{timeframe}.csv"
                df.to_csv(save_path)
                print(f"💾 Cached {len(df)} candles → {save_path}")
            except Exception as e:
                print(f"⚠️ Could not save candle cache: {e}")
    return df


# ==========================================
# DIAGNOSTIC LAYOUT METRIC COMPILER
# ==========================================
class DiagnosticLayer:
    @staticmethod
    def render_progress(current, target, reverse=False):
        try:
            # progress = how far `current` has advanced toward `target`.
            # (reverse kept for signature compatibility; all callers want this.)
            pct = (current / target) if target else 0.0
            pct = min(1.0, max(0.0, pct))
            filled = int(pct * 10)
            bar = "┃" + "█" * filled + "░" * (10 - filled) + "┃"
            return f"{bar} {int(pct * 100)}%"
        except:
            return "[----------] 0%"

    @staticmethod
    def get_pending_conditions(df, config, conf, ui_limit):
        strategies = config.get('strategies', [])
        try:
            current_price = float(df['close'].iloc[-1])
            # Only surface the AI veto when the neural gate is actually active.
            # In Bypass (mlMode == 'off') the gate is open (see StrategyBrain:
            # gate_passed = True), so showing "AI VETO" is misleading — fall
            # through to the pending-conditions view instead.
            if config.get('mlMode') != 'off' and conf < ui_limit:
                return f"🛑 AI VETO: Needs {int(ui_limit*100)}% (At {int(conf*100)}%)"
            pending = []
            for strat in strategies:
                code = strat.get('code')
                if code == "rsi_threshold":
                    val = ta.rsi(df['close']).iloc[-1]
                    target = 30 if current_price < ta.ema(df['close'], 200).iloc[-1] else 70
                    pending.append(f"RSI: {DiagnosticLayer.render_progress(val, target)}")
                elif code == "sma_crossover":
                    fast = ta.sma(df['close'], 50).iloc[-1]
                    slow = ta.sma(df['close'], 200).iloc[-1]
                    pending.append(f"SMA Cross: {DiagnosticLayer.render_progress(fast, slow)}")
                elif code == "supertrend":
                    st = ta.supertrend(df['high'], df['low'], df['close']).iloc[-1, 0]
                    pending.append(f"ST Dist: {DiagnosticLayer.render_progress(current_price, st)}")
                elif code == "macd_crossover":
                    macd = ta.macd(df['close'])
                    pending.append(f"MACD Gap: {DiagnosticLayer.render_progress(macd.iloc[-1, 0], macd.iloc[-1, 2])}")
                elif code == "atr_breakout":
                    atr   = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
                    ema20 = ta.ema(df['close'], 20).iloc[-1]
                    pending.append(f"ATR Break: {DiagnosticLayer.render_progress(current_price, ema20 + atr)}")
                elif code == "bb_fade":
                    bb = ta.bbands(df['close']).iloc[-1]
                    target = bb[0] if current_price < bb[1] else bb[2]
                    pending.append(f"BB Wall: {DiagnosticLayer.render_progress(current_price, target)}")
                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    pending.append(f"Stoch: {DiagnosticLayer.render_progress(k, 20)}")
                elif code == "ema_cloud":
                    fast = ta.ema(df['close'], 9).iloc[-1]
                    slow = ta.ema(df['close'], 21).iloc[-1]
                    pending.append(f"Cloud: {DiagnosticLayer.render_progress(fast, slow)}")
                elif code == "pa_breakout":
                    high_20 = df['high'].rolling(20).max().iloc[-1]
                    pending.append(f"PA High: {DiagnosticLayer.render_progress(current_price, high_20)}")
                elif code == "vol_profile":
                    vol_ma = ta.sma(df['volume'], 20).iloc[-1]
                    pending.append(f"Vol Surge: {DiagnosticLayer.render_progress(df['volume'].iloc[-1], vol_ma * 1.5)}")
            # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
            rule = _get_combo_rule(config)
            if not pending:
                return f"🔍 TARGETS ({rule}): Scanning Setup..."
            return f"🔍 TARGETS ({rule}): " + " | ".join(pending[:2])
        except Exception:
            return "🔍 Scanning Market Conditions..."


# ============================================================
# 🔮 PREDICTIVE REGIME OPTIMIZER
# ============================================================
class PredictiveRegimeOptimizer:
    REGIME_DEBOUNCE_TICKS = 5

    @staticmethod
    def get_regime_key(ai_score: float, current_adx: float) -> str:
        if ai_score >= 0.68 or (ai_score > 0.55 and current_adx > 30):
            return "trend"
        elif ai_score <= 0.32 or (ai_score < 0.45 and current_adx > 30):
            return "bear"
        else:
            return "range"

    @staticmethod
    def get_strategies_for_regime(regime: str) -> List[Dict[str, Any]]:
        if regime == "trend":
            return [
                {"code": "supertrend",    "params": {"st_atr": 10, "st_factor": 3.0}},
                {"code": "pa_breakout",   "params": {"lookback": 20, "buffer": 0.01}},
                {"code": "ema_cloud",     "params": {"fast_ema": 9, "slow_ema": 21}},
                {"code": "sma_crossover", "params": {"fast_sma": 20, "slow_sma": 100}}
            ]
        elif regime == "bear":
            return [
                {"code": "supertrend",     "params": {"st_atr": 10, "st_factor": 2.5}},
                {"code": "atr_breakout",   "params": {"atr_length": 14, "multiplier": 1.5}},
                {"code": "macd_crossover", "params": {"fast": 12, "slow": 26, "signal": 9}},
                {"code": "vol_profile",    "params": {"vol_ma": 20, "threshold": 1.2}}
            ]
        else:
            return [
                {"code": "bb_fade",       "params": {"bb_period": 20, "bb_std": 2.0}},
                {"code": "rsi_threshold", "params": {"rsi_length": 14, "oversold": 30, "overbought": 70}},
                {"code": "stoch",         "params": {"k_period": 14, "d_period": 3}}
            ]

    @staticmethod
    def dynamically_tune_strategies(bot_config: Dict[str, Any], ai_score: float,
                                     current_adx: float,
                                     locked_regime: Optional[str] = None) -> List[Dict[str, Any]]:
        # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
        if _get_combo_rule(bot_config) == "AND":
            return bot_config.get("strategies", [])
        regime = locked_regime or PredictiveRegimeOptimizer.get_regime_key(ai_score, current_adx)
        return PredictiveRegimeOptimizer.get_strategies_for_regime(regime)


# ==========================================
# 🧠 NEURAL MACHINE LEARNING PREDICTOR
# ==========================================
class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame, symbol: str = "BTC-USD") -> float:
        """
        Output contract: returns P(next candle move is net-positive / long).
        Expected range [0.0, 1.0] — 1.0 = strong bull, 0.0 = strong bear, 0.5 = neutral.
        """
        try:
            council    = get_cached_predictor(symbol=symbol, timeframe="1h")
            if hasattr(council, 'get_prediction_score'):
                prediction = council.get_prediction_score(df)
            else:
                prediction = council.predict_direction(df)
            val = float(prediction)
            # If the model always returns exactly 0.5 the predictor is likely
            # failing silently (missing model file, NaN features, wrong input
            # shape).  Log a warning so it shows up distinctly in Render logs.
            if val == 0.5:
                logger.warning(
                    "🧠 Predictor returned exactly 0.5 — possible silent failure "
                    "(check model path, feature columns, and NaN inputs)"
                )
            return val
        except Exception as e:
            # exc_info=True attaches the full traceback so the root cause
            # (missing file, shape mismatch, etc.) is visible in logs.
            logger.error(f"🧠 Council Predictor Error: {e}", exc_info=True)
            return 0.5


# ============================================================
# 🔧 COMBO RULE HELPER
# ============================================================
def _get_combo_rule(config: Dict[str, Any]) -> str:
    """
    Read the combination rule with a two-key fallback chain so a stale or
    missing comboConfig never silently forces AND mode on a user who chose OR.

    Priority:
      1. config["comboConfig"]["combinationRule"]  — set explicitly by handleConfirmStart
      2. config["hybridMode"]                      — top-level spread from formConfig
      3. "OR"                                      — safe default
    """
    from_combo = config.get("comboConfig", {}).get("combinationRule")
    if from_combo:
        return from_combo
    from_hybrid = config.get("hybridMode")
    if from_hybrid:
        return from_hybrid
    return "OR"


# ==========================================
# PACKET FORMATTING LAYER
# ==========================================
async def process_data_packet(df: pd.DataFrame, strategies: list) -> list:
    # FIX #12: apply_mega_features is a heavy synchronous feature build; run it
    # in a worker thread so the ~10s UI broadcast (and start_bot) never block
    # the event loop for every other bot.
    df, feats = await asyncio.to_thread(apply_mega_features, df)
    keys = ['bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 'sma_fast', 'sma_slow',
            'supertrend', 'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            'atr_upper', 'atr_lower', 'pa_high', 'pa_low', 'vol_ma']
    candles_to_send = []
    for _, row in df.tail(100).iterrows():
        ts = int(row['time']) if 'time' in row else int(row.name.timestamp())
        c_obj = {"time": ts, "open": row['open'], "high": row['high'],
                 "low": row['low'], "close": row['close']}
        for k in keys:
            if k in row and not pd.isna(row[k]):
                c_obj[k] = round(float(row[k]), 2)
        candles_to_send.append(c_obj)
    return candles_to_send


# ============================================================
# 🔧 STRATEGY BRAIN
# ============================================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df_raw: pd.DataFrame, df_ai: pd.DataFrame,
                          config: Dict[str, Any], l_thresh: float, s_thresh: float,
                          symbol: str = "BTC-USD",
                          precomputed_conf: Optional[float] = None,
                          live_price: Optional[float] = None):
        active_thoughts, votes = [], 0
        signals_map    = {}
        directions_map = {}
        strategies     = config.get('strategies', [])

        current_price = live_price if live_price is not None else float(df_raw['close'].iloc[-1])

        ema20  = ta.ema(df_raw['close'], length=20).iloc[-1]
        ema200 = ta.ema(df_raw['close'], length=200).iloc[-1]
        bb     = ta.bbands(df_raw['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)
        weighted_votes = 0.0

        for strat in strategies:
            code = strat.get('code')
            p    = strat.get('params', {})
            raw_vote   = 0
            confidence = 0.5
            try:
                if code == "rsi_threshold":
                    digits = int(p.get('rsi_length', 14))
                    rsi    = ta.rsi(df_raw['close'], length=digits).iloc[-1]
                    dist   = min(abs(rsi - 30), abs(rsi - 70))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if rsi < p.get('oversold', 30):    raw_vote = 1;  active_thoughts.append("RSI Low")
                    elif rsi > p.get('overbought', 70): raw_vote = -1; active_thoughts.append("RSI High")
                elif code == "sma_crossover":
                    f   = ta.sma(df_raw['close'], length=int(p.get('fast_sma', 50))).iloc[-1]
                    s   = ta.sma(df_raw['close'], length=int(p.get('slow_sma', 200))).iloc[-1]
                    gap = abs(f - s) / s
                    # FIX: symmetric — larger crossover gap = stronger conviction
                    # regardless of direction.  Prior code gave long=1.0 always
                    # while large bear gaps produced near-zero short confidence.
                    confidence = max(0.1, min(1.0, 0.5 + gap * 25))
                    raw_vote   = 1 if f > s else -1
                elif code == "macd_crossover":
                    macd      = ta.macd(df_raw['close'], fast=int(p.get('fast', 12))).iloc[-1]
                    norm_hist = abs(macd[1]) / (current_price * 0.0005)
                    confidence = max(0.1, min(1.0, norm_hist))
                    raw_vote   = 1 if macd[0] > macd[2] else -1
                elif code == "supertrend":
                    st_data = ta.supertrend(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    dist    = abs(current_price - st_data[0]) / current_price
                    # FIX: symmetric — further from the SuperTrend line = stronger signal.
                    # Prior code gave long=1.0 but shrank bear confidence as the
                    # distance grew, so deep downtrends showed minimum conviction.
                    confidence = max(0.1, min(1.0, 0.5 + dist * 10))
                    raw_vote   = 1 if st_data[1] == 1 else -1
                elif code == "bb_fade":
                    # FIX: prior code used pr/100 which goes negative when price
                    # is below the lower band — the strongest long signal got
                    # confidence clamped to 0.1 (minimum).  Now confidence scales
                    # with how far price has broken outside the relevant band edge.
                    band_width = max(upper - lower, current_price * 0.001)
                    if current_price < lower:
                        raw_vote   = 1
                        excess     = (lower - current_price) / band_width
                        confidence = max(0.1, min(1.0, 0.5 + excess * 2))
                    elif current_price > upper:
                        raw_vote   = -1
                        excess     = (current_price - upper) / band_width
                        confidence = max(0.1, min(1.0, 0.5 + excess * 2))
                    else:
                        raw_vote   = 0
                        confidence = max(0.1, min(0.4, abs(pr - 50) / 125))
                elif code == "atr_breakout":
                    atr    = ta.atr(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1]
                    target = ema20 + (atr * float(p.get('multiplier', 1.5)))
                    # FIX: prior code used price/target which gave confidence > 1
                    # only for longs and < 1 for shorts — making bears near target
                    # appear confident while bears far below it looked weak.
                    dist_pct   = abs(current_price - target) / target
                    confidence = max(0.1, min(1.0, 0.5 + dist_pct * 5))
                    raw_vote   = 1 if current_price > target else -1
                elif code == "pa_breakout":
                    lb      = int(p.get('lookback', 20))
                    high_lb = df_raw['high'].tail(lb).max()
                    # FIX: prior code used price/high_lb — approaching the high
                    # from below gave high confidence for the SHORT vote while
                    # being far below it (stronger bear) gave low confidence.
                    dist_pct   = abs(current_price - high_lb) / high_lb
                    confidence = max(0.1, min(1.0, 0.5 + dist_pct * 5))
                    raw_vote   = 1 if current_price >= high_lb else -1
                elif code == "vol_profile":
                    v_ma  = ta.sma(df_raw['volume'], length=int(p.get('vol_ma', 20))).iloc[-1]
                    ratio = df_raw['volume'].iloc[-1] / (v_ma * float(p.get('threshold', 1.5)))
                    confidence = max(0.1, min(1.0, ratio))
                    raw_vote   = (1 if current_price > mid else -1) if ratio >= 1.0 else 0
                elif code == "stoch":
                    k    = ta.stoch(df_raw['high'], df_raw['low'], df_raw['close']).iloc[-1][0]
                    dist = min(abs(k - 20), abs(k - 80))
                    confidence = max(0.1, min(1.0, 1.0 - (dist / 40)))
                    if k < 20:   raw_vote = 1
                    elif k > 80: raw_vote = -1
                elif code == "ema_cloud":
                    f_ema = ta.ema(df_raw['close'], length=int(p.get('fast_ema', 9))).iloc[-1]
                    s_ema = ta.ema(df_raw['close'], length=int(p.get('slow_ema', 21))).iloc[-1]
                    gap   = abs(f_ema - s_ema) / s_ema
                    # FIX: symmetric — larger cloud gap = stronger conviction.
                    # Prior code: long=1.0 always; bear gaps reduced short confidence.
                    confidence = max(0.1, min(1.0, 0.5 + gap * 50))
                    raw_vote   = 1 if f_ema > s_ema else -1
                elif code == "btc_regime":
                    # Cross-asset regime (ported from the backtest so the validated
                    # `regime` configs run live, incl. 1d regime SHORT = the 5/5
                    # champion). Votes with BTC's own trend on this timeframe:
                    # long-biased when BTC is above its 50-EMA, short-biased below.
                    # Reads the BTC_TREND feed (macro loop); neutral until it fills.
                    _btf = str(config.get('timeframe', '1d'))
                    _bt  = BTC_TREND.get(_btf) or BTC_TREND.get('1d') or {}
                    raw_vote   = int(_bt.get('vote', 0))
                    confidence = 0.6 if raw_vote != 0 else 0.5
                else:
                    confidence = 0.5
                signals_map[code]    = confidence
                directions_map[code] = raw_vote
                votes             += raw_vote
                weighted_votes    += raw_vote * confidence
            except Exception:
                signals_map[code]    = 0.0
                directions_map[code] = 0

        # FIX #4: gate the ML confidence on the side we're ABOUT TO TRADE, not
        # on the price-vs-EMA200 regime. The intended direction is the sign of
        # the strategy vote tally (same basis final_sig is derived from below).
        # Previously a counter-trend short in an up-regime (price > EMA200) was
        # gated as a long — conf>=limit instead of (1-conf)>=limit — so the AI
        # veto passed/failed backwards on every counter-trend entry.
        intended_dir   = 1 if votes > 0 else (-1 if votes < 0 else 0)
        is_short_trade = intended_dir < 0
        ui_limit = (float(config.get('mlThresholdShort', 0.55)) if is_short_trade
                    else float(config.get('mlThresholdLong', 0.55)))

        conf = precomputed_conf if precomputed_conf is not None else NeuralPredictor.get_prediction(
            config.get('mlModel', 'stacking'), df_ai, symbol=symbol)

        if config.get('mlMode') == 'off':
            gate_passed = True
        else:
            gate_passed = ((1.0 - conf) >= ui_limit) if is_short_trade else (conf >= ui_limit)

        # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
        rule      = _get_combo_rule(config)
        final_sig = 0
        if gate_passed:
            if rule == "AND":
                if votes >= len(strategies) and weighted_votes > 0:    final_sig = 1
                elif votes <= -len(strategies) and weighted_votes < 0: final_sig = -1
            else:
                min_weighted = float(config.get('minWeightedSignal', 0.3))
                if votes > 0  and weighted_votes >= min_weighted:  final_sig = 1
                elif votes < 0 and weighted_votes <= -min_weighted: final_sig = -1

        return final_sig, active_thoughts, directions_map, conf, signals_map


# ============================================================================
# 🎯 PRE-TRADE QUALITY SCORER
# ============================================================================
class TradeQualityScorer:
    """
    Synthesises market context, bot intelligence, and strategy alignment into
    a single 0–100 entry quality score.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Market Context  (35 pts) — trend stack, ATR regime, ADX, volume     │
    │  Bot Intelligence (35 pts) — ML margin, confidence trend, track rec  │
    │  Strategy Alignment (30 pts) — vote %, mean confidence, diversity    │
    │                                                                      │
    │  Entry approved only when composite ≥ MIN_ENTRY_SCORE (default 65)  │
    └──────────────────────────────────────────────────────────────────────┘
    """

    MIN_ENTRY_SCORE = 65  # raise for more selectivity, lower for more activity

    # Maps each strategy code to a broad analytical category.
    # Category diversity in the agreeing votes is worth bonus points because
    # three momentum strategies all agreeing is weaker evidence than a trend
    # follower, a momentum signal, and a mean-reversion signal all aligning.
    STRATEGY_CATEGORIES: Dict[str, str] = {
        "supertrend":     "trend",
        "sma_crossover":  "trend",
        "ema_cloud":      "trend",
        "pa_breakout":    "trend",
        "macd_crossover": "momentum",
        "atr_breakout":   "momentum",
        "vol_profile":    "momentum",
        "rsi_threshold":  "mean_rev",
        "bb_fade":        "mean_rev",
        "stoch":          "mean_rev",
    }

    # ── COMPONENT 1: MARKET CONTEXT ──────────────────────────────────────────
    @staticmethod
    def _score_market_context(df: pd.DataFrame, sig: int, current_price: float,
                               current_atr: float, current_adx: float,
                               vol_ratio: float) -> dict:
        """
        Scores the macro market backdrop — 35 points max.

        EMA stack alignment : 15 pts   (4 EMAs stacked + ordered in direction)
        ATR regime quality  :  8 pts   (volatility expanding healthily?)
        ADX trend strength  :  7 pts   (market actually trending?)
        Volume quality      :  5 pts   (volume backing the move?)
        """
        scores: Dict[str, int] = {}

        # 1. EMA stack — proxies multi-timeframe alignment on the 1h feed
        try:
            ema20  = float(ta.ema(df['close'], length=20).iloc[-1])
            ema50  = float(ta.ema(df['close'], length=50).iloc[-1])
            ema100 = float(ta.ema(df['close'], length=100).iloc[-1])
            ema200 = float(ta.ema(df['close'], length=200).iloc[-1])

            if sig == 1:   # long — price should sit above all EMAs in order
                levels  = [current_price > ema20, current_price > ema50,
                           current_price > ema100, current_price > ema200]
                ordered = ema20 > ema50 > ema100 > ema200
            else:          # short — price should sit below all EMAs in order
                levels  = [current_price < ema20, current_price < ema50,
                           current_price < ema100, current_price < ema200]
                ordered = ema20 < ema50 < ema100 < ema200

            stack_count = sum(levels)
            base_pts    = {4: 11, 3: 7, 2: 3, 1: 0, 0: 0}.get(stack_count, 0)
            scores['ema_stack'] = min(15, base_pts + (4 if ordered else 0))
        except Exception:
            scores['ema_stack'] = 5  # neutral if indicators fail

        # 2. ATR regime — ideal is mildly expanding (1.0–1.5× average)
        try:
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)
            atr_avg    = float(ta.sma(atr_series, length=20).iloc[-1])
            atr_ratio  = current_atr / atr_avg if atr_avg > 0 else 1.0
            if   1.0 <= atr_ratio <= 1.5: scores['atr_regime'] = 8
            elif 0.7 <= atr_ratio <  1.0: scores['atr_regime'] = 5
            elif 1.5 <  atr_ratio <= 2.0: scores['atr_regime'] = 4
            elif atr_ratio > 2.0:         scores['atr_regime'] = 2
            else:                         scores['atr_regime'] = 1
        except Exception:
            scores['atr_regime'] = 4

        # 3. ADX strength
        if   current_adx >= 40: scores['adx'] = 7
        elif current_adx >= 30: scores['adx'] = 5
        elif current_adx >= 22: scores['adx'] = 3
        else:                   scores['adx'] = 1

        # 4. Volume
        if   vol_ratio >= 1.5: scores['volume'] = 5
        elif vol_ratio >= 1.0: scores['volume'] = 3
        elif vol_ratio >= 0.7: scores['volume'] = 2
        else:                  scores['volume'] = 0

        return {"total": sum(scores.values()), "max": 35, "breakdown": scores}

    # ── COMPONENT 2: BOT INTELLIGENCE ────────────────────────────────────────
    @staticmethod
    def _score_bot_intelligence(bot: dict, ml_score: float, ui_limit: float,
                                ml_active: bool = True) -> dict:
        """
        Scores the bot's internal conviction — 35 points max.

        ML confidence margin  : 15 pts  (headroom above gate threshold)
        Confidence trajectory : 10 pts  (is confidence genuinely rising?)
        Recent trade record   : 10 pts  (win/loss context over last 6 trades)
        """
        scores: Dict[str, int] = {}

        # 1. Confidence margin above the ML gate.
        # When ML is off the AI must not judge, so award a neutral pass rather
        # than penalizing for a signal that isn't in use.
        if not ml_active:
            scores['ml_margin'] = 11
        else:
            margin = ml_score - ui_limit
            if   margin >= 0.25: scores['ml_margin'] = 15
            elif margin >= 0.15: scores['ml_margin'] = 11
            elif margin >= 0.08: scores['ml_margin'] = 7
            elif margin >= 0.02: scores['ml_margin'] = 3
            else:                scores['ml_margin'] = 0

        # 2. Confidence trajectory across recent equity-curve ticks
        try:
            curve = bot.get('equityCurve', [])[-8:]
            confs = [p.get('confidence', 50) / 100.0 for p in curve]
            if len(confs) >= 4:
                half      = len(confs) // 2
                early_avg = sum(confs[:half])  / half
                late_avg  = sum(confs[half:])  / (len(confs) - half)
                delta     = late_avg - early_avg
                if   delta >= 0.12: scores['conf_trajectory'] = 10
                elif delta >= 0.06: scores['conf_trajectory'] = 7
                elif delta >= 0.01: scores['conf_trajectory'] = 4
                elif delta >= -0.03: scores['conf_trajectory'] = 2
                else:               scores['conf_trajectory'] = 0
            else:
                scores['conf_trajectory'] = 3  # insufficient history → neutral
        except Exception:
            scores['conf_trajectory'] = 3

        # 3. Recent trade record (last 6 closed exits)
        try:
            exits = [t for t in bot.get('trade_history', [])
                     if t.get('type') == 'exit'][-6:]
            if exits:
                wins     = sum(1 for t in exits if float(t.get('pnl', 0)) > 0)
                rate     = wins / len(exits)
                # Count current win streak
                streak = 0
                for t in reversed(exits):
                    if float(t.get('pnl', 0)) > 0: streak += 1
                    else: break
                if   rate >= 0.67 or streak >= 3: scores['recent_perf'] = 10
                elif rate >= 0.50:                scores['recent_perf'] = 7
                elif rate >= 0.33:                scores['recent_perf'] = 4
                else:                             scores['recent_perf'] = 1
            else:
                scores['recent_perf'] = 5  # first trade in session → neutral
        except Exception:
            scores['recent_perf'] = 5

        return {"total": sum(scores.values()), "max": 35, "breakdown": scores}

    # ── COMPONENT 3: STRATEGY ALIGNMENT ──────────────────────────────────────
    @staticmethod
    def _score_strategy_alignment(signals_map: dict, active_strategies: list,
                                   sig: int, directions_map: Optional[dict] = None) -> dict:
        """
        Scores the quality of strategy agreement — 30 points max.

        Vote unanimity     : 12 pts  (% of strategies whose DIRECTION matches)
        Mean confidence    : 10 pts  (avg confidence of the agreeing strategies)
        Category diversity :  8 pts  (trend + momentum + mean-rev = strongest)

        NOTE: signals_map holds confidence MAGNITUDE (0.1-1.0); the buy/sell
        direction lives in directions_map (+1/-1/0). Agreement must be measured
        with directions_map, not by thresholding magnitude at 0.5.
        """
        directions_map = directions_map or {}
        scores: Dict[str, int] = {}

        if not active_strategies:
            return {"total": 10, "max": 30,
                    "breakdown": {"vote_unanimity": 4, "mean_confidence": 4, "diversity": 2}}

        # 1. Vote unanimity - strategies whose signed vote matches the trade
        agreeing = [s for s in active_strategies
                    if directions_map.get(s.get('code', ''), 0) == sig and sig != 0]
        pct = len(agreeing) / len(active_strategies) if active_strategies else 0
        if   pct >= 0.90: scores['vote_unanimity'] = 12
        elif pct >= 0.75: scores['vote_unanimity'] = 9
        elif pct >= 0.60: scores['vote_unanimity'] = 6
        elif pct >= 0.51: scores['vote_unanimity'] = 3
        else:             scores['vote_unanimity'] = 0

        # 2. Mean confidence across the AGREEING strategies (fallback: all)
        basis      = agreeing if agreeing else active_strategies
        conf_vals  = [signals_map.get(s.get('code', ''), 0.5) for s in basis]
        mean_conf  = sum(conf_vals) / len(conf_vals) if conf_vals else 0.5
        if   mean_conf >= 0.80: scores['mean_confidence'] = 10
        elif mean_conf >= 0.65: scores['mean_confidence'] = 7
        elif mean_conf >= 0.50: scores['mean_confidence'] = 4
        elif mean_conf >= 0.35: scores['mean_confidence'] = 2
        else:                   scores['mean_confidence'] = 0

        # 3. Category diversity - among the AGREEING strategies
        categories = {
            TradeQualityScorer.STRATEGY_CATEGORIES.get(s.get('code', ''), 'other')
            for s in (agreeing if agreeing else active_strategies)
        }
        if   len(categories) >= 3: scores['diversity'] = 8
        elif len(categories) == 2: scores['diversity'] = 5
        else:                      scores['diversity'] = 2  # single-family = fragile

        return {"total": sum(scores.values()), "max": 30, "breakdown": scores}

    # ── MASTER EVALUATION ─────────────────────────────────────────────────────
    @classmethod
    def evaluate(cls,
                 df_raw:            pd.DataFrame,
                 bot:               dict,
                 ml_score:          float,
                 ui_limit:          float,
                 signals_map:       dict,
                 active_strategies: list,
                 sig:               int,
                 current_price:     float,
                 current_atr:       float,
                 current_adx:       float,
                 vol_ratio:         float,
                 directions_map:    Optional[dict] = None,
                 min_score:         Optional[float] = None,
                 ml_active:         bool = True) -> dict:
        """
        Master evaluation. Returns a verdict dict with full breakdown.

        Keys:
          approved        bool     — whether the trade clears the threshold
          composite_score float    — 0–100 final score
          market          dict     — market context sub-scores
          bot             dict     — bot intelligence sub-scores
          strategies      dict     — strategy alignment sub-scores
          verdict         str      — one-line plain-English summary
          log_line        str      — ready-to-emit_log formatted string
        """
        mkt   = cls._score_market_context(df_raw, sig, current_price,
                                           current_atr, current_adx, vol_ratio)
        intel = cls._score_bot_intelligence(bot, ml_score, ui_limit, ml_active)
        strat = cls._score_strategy_alignment(signals_map, active_strategies, sig, directions_map)

        composite = mkt['total'] + intel['total'] + strat['total']
        _min = cls.MIN_ENTRY_SCORE if min_score is None else float(min_score)
        approved  = composite >= _min

        if   composite >= 85: verdict = "ELITE SETUP — maximum multi-factor conviction"
        elif composite >= 75: verdict = "HIGH QUALITY — strong cross-domain alignment"
        elif composite >= 65: verdict = "ACCEPTABLE — threshold cleared, proceed"
        elif composite >= 50: verdict = "MARGINAL — insufficient edge"
        else:                 verdict = "POOR SETUP — multiple factors weak"

        # Identify the weakest pillar for actionable rejection feedback
        pillars = {
            "market context":     mkt['total']   / mkt['max'],
            "AI conviction":      intel['total'] / intel['max'],
            "strategy alignment": strat['total'] / strat['max'],
        }
        weakest = min(pillars, key=pillars.get)

        direction = "LONG" if sig == 1 else "SHORT"
        status    = "✅ APPROVED" if approved else "❌ REJECTED"
        log_line  = (
            f"🎯 TRADE SCORER [{direction}] {status}: {composite:.0f}/100 "
            f"(Mkt {mkt['total']}/{mkt['max']} · "
            f"AI {intel['total']}/{intel['max']} · "
            f"Strat {strat['total']}/{strat['max']}) "
            f"— {verdict}"
            + (f"  ↳ Weakest pillar: {weakest} "
               f"({int(pillars[weakest]*100)}%)" if not approved else "")
        )

        return {
            "approved":        approved,
            "composite_score": composite,
            "market":          mkt,
            "bot":             intel,
            "strategies":      strat,
            "verdict":         verdict,
            "log_line":        log_line,
        }


# ============================================================================
# 🚀 CORE ENGINE HEARTBEAT
# ============================================================================
async def live_neural_heartbeat(user_id: str):
    signal_has_reset = True
    last_log         = 0
    last_ui_update   = 0
    # Persistent public-data exchange + market-data cache (perf refactor)
    read_exchange     = None
    read_exchange_id  = None
    last_candle_fetch = 0.0
    last_diag         = 0.0
    feed_cache: Dict[str, Any] = {}

    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]:
            ACTIVE_BOTS[user_id]["equityCurve"] = [
                {"time": datetime.now().isoformat(),
                 "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]:
            ACTIVE_BOTS[user_id]["logs"] = []

    try:
        while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
            bot           = ACTIVE_BOTS[user_id]
            config        = bot.get('config', {})
            params        = config.get('params', {})
            strategies    = config.get('strategies', [])

            # FIX #5: pre-validate the config values that the setup block below
            # parses (symbol + numeric casts). These lines run INSIDE the while
            # loop but OUTSIDE the per-tick try, so a malformed config used to
            # raise here, escape the never-awaited coroutine, and leave
            # status='running' on a dead bot with an orphaned position. Catch it
            # up-front, surface it, mark the bot 'error', and stop cleanly.
            try:
                _ = config['symbol']
                float(config.get('capitalAllocation', config.get('initialBalance', 200.0)))
                float(config.get('leverage', 1.0))
                int(config.get('maxPyramiding', 5))
            except (KeyError, ValueError, TypeError) as _cfg_err:
                logger.error(f"Bot {user_id} halted: invalid config ({_cfg_err})", exc_info=True)
                try:
                    await emit_log(user_id, f"\U0001f6d1 BOT HALTED: invalid config — {_cfg_err}")
                except Exception:
                    pass
                bot['status'] = 'error'
                try:
                    DatabaseHandler.save_state(user_id, bot)
                except Exception:
                    pass
                break

            symbol        = config['symbol'].replace('-', '/')
            ticker_symbol = config['symbol']

            max_p         = min(5, int(config.get('maxPyramiding', 5)))
            start_capital = float(config.get('capitalAllocation', config.get('initialBalance', 200.0)))
            api_keys      = bot.get('api_keys') or config.get('api_keys', {})
            trading_mode  = config.get('trading_mode', 'paper').lower()
            use_margin    = config.get('enable_shorting', False)
            leverage_val  = float(config.get('leverage', 1.0))

            target_exchange = "kraken" if use_margin else "coinbase"
            has_valid_keys  = bool(api_keys.get('krakenKey') if use_margin else api_keys.get('apiKey'))
            is_live_trading = (trading_mode == 'live') and has_valid_keys

            if trading_mode == 'live' and not is_live_trading:
                logger.warning(f"User {user_id} selected LIVE but lacks API keys. Falling back to PAPER.")
                await emit_log(user_id, "⚠️ LIVE MODE FAILED: Missing API Keys. Forcing PAPER MODE.")
                config['trading_mode'] = 'paper'
                bot['config'] = config
                DatabaseHandler.save_state(user_id, bot)

            fee_rate = KRAKEN_TAKER_FEE if use_margin else DEFAULT_TAKER_FEE

            try:
                current_time = datetime.now(timezone.utc)
                _maybe_hot_reload(feed_cache)

                # ── TIER 1: MARKET FEED ──────────────────────────────────────
                # Perf: reuse ONE public-data exchange across ticks, refetch
                # OHLCV + closed-candle features only every candle_refresh_secs,
                # and pull the lightweight ticker every tick so exits stay live.
                now_mono            = time.monotonic()
                candle_refresh_secs = float(config.get('candleRefreshSecs', 20))

                # CANDLE/TICKER FEED: always read from Binance.US. It supports every
                # timeframe we trade (1h/4h/1d) AND is the exact source the strategy
                # was validated on. Coinbase has NO 4h candles, which silently killed
                # the feed of every 4h bot (the whole long side). Order ROUTING still
                # uses target_exchange (coinbase spot / kraken margin) separately.
                READ_FEED   = "binanceus"
                read_symbol = f"{ticker_symbol.split('-')[0]}/USDT"
                if read_exchange is None or read_exchange_id != READ_FEED:
                    if read_exchange is not None:
                        try: await read_exchange.close()
                        except Exception: pass
                    read_exchange     = getattr(ccxt, READ_FEED)({'enableRateLimit': True})
                    read_exchange_id  = READ_FEED
                    last_candle_fetch = 0.0

                if (now_mono - last_candle_fetch) >= candle_refresh_secs or 'ohlcv_raw' not in feed_cache:
                    try:
                        _ohlcv = await read_exchange.fetch_ohlcv(
                            read_symbol, config.get('timeframe', '1h'), limit=500)
                    except Exception:
                        await emit_log(user_id, f"⚠️ Feed Unstable ({read_symbol} {config.get('timeframe','1h')}) - Retrying...")
                        await asyncio.sleep(5); continue
                    if not _ohlcv:
                        await emit_log(user_id, f"⚠️ {target_exchange.upper()} Feed Empty - Retrying...")
                        await asyncio.sleep(5); continue
                    feed_cache['ohlcv_raw'] = [
                        {"time": int(c[0] / 1000), "open": c[1], "high": c[2], "low": c[3],
                         "close": c[4], "volume": c[5] if len(c) > 5 else 0}
                        for c in _ohlcv
                    ]
                    last_candle_fetch = now_mono

                # Live ticker every tick (cheap) for entry/exit responsiveness
                try:
                    ticker        = await read_exchange.fetch_ticker(read_symbol)
                    current_price = float(ticker['last'])
                except Exception:
                    await emit_log(user_id, f"⚠️ {target_exchange.upper()} Ticker Unstable - Retrying...")
                    # FIX #22: the `continue` below skips the whole tick including
                    # the UI-broadcast tier, so a flaky feed used to blank the
                    # panels. Emit a light keep-alive with last-known state first.
                    try:
                        await emit_status(user_id, {
                            "status":          "running",
                            "currentBalance":  bot.get("currentBalance", bot.get("balance", 0)),
                            "activePositions": bot.get("positions", []),
                            "equityCurve":     bot.get("equityCurve", []),
                            "startedAt":       bot.get("startedAt"),
                        })
                    except Exception:
                        pass
                    await asyncio.sleep(2); continue

                # Copy cached candles, patch the live price onto the forming bar
                ohlcv_raw = [dict(c) for c in feed_cache['ohlcv_raw']]
                ohlcv_raw[-1]['close'] = current_price
                df_raw = pd.DataFrame(ohlcv_raw)

                atr_tp_mult = float(params.get('atr_tp_mult', config.get('atrTpMultiplier', 3.0)))
                atr_sl_mult = float(params.get('atr_sl_mult', config.get('atrSlMultiplier', 1.5)))

                # Recompute closed-candle features/indicators only on candle refresh
                if feed_cache.get('feature_stamp') != last_candle_fetch:
                    _dch    = df_raw.iloc[:-1].copy()
                    _dai, _ = await asyncio.to_thread(apply_mega_features, _dch)
                    feed_cache['df_closed_history'] = _dch
                    feed_cache['df_ai']             = _dai
                    feed_cache['current_ema200']    = float(ta.ema(df_raw['close'], length=200).iloc[-1])
                    feed_cache['current_atr']       = float(ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1])
                    try:
                        feed_cache['current_adx'] = float(ta.adx(df_raw['high'], df_raw['low'], df_raw['close'], length=14).iloc[-1, 0])
                    except Exception:
                        feed_cache['current_adx'] = 25.0
                    try:
                        _volc = _dch['volume']
                        _vma  = float(ta.sma(_volc, length=20).iloc[-1])
                        feed_cache['vol_ratio'] = float(_volc.iloc[-1]) / _vma if _vma > 0 else 1.0
                    except Exception:
                        feed_cache['vol_ratio'] = 1.0
                    feed_cache['feature_stamp'] = last_candle_fetch

                df_closed_history = feed_cache['df_closed_history']
                df_ai             = feed_cache['df_ai']
                current_ema200    = feed_cache['current_ema200']
                current_atr       = feed_cache['current_atr']
                current_adx       = feed_cache['current_adx']
                vol_ratio         = feed_cache['vol_ratio']

                # ── TIER 2: AI COUNCIL ────────────────────────────────────────
                # ML off = AI fully out of the loop: skip the predictor entirely
                # (no bias, no compute) and use a neutral 0.5.
                # FIX #9: the council (xgboost + RandomForest + transformer forward
                # pass, plus a cold model load on first use) is heavy, blocking CPU
                # and its inputs (df_ai) only change on candle refresh. Run it in a
                # worker thread so it never freezes the shared event loop, and cache
                # the score so it recomputes once per closed candle, not every tick.
                if config.get('mlMode') == 'on':
                    if feed_cache.get('conf_stamp') != last_candle_fetch or 'conf_score' not in feed_cache:
                        feed_cache['conf_score'] = await asyncio.to_thread(
                            NeuralPredictor.get_prediction,
                            config.get('mlModel', 'stacking'), df_ai, ticker_symbol)
                        feed_cache['conf_stamp'] = last_candle_fetch
                    conf_score = feed_cache['conf_score']
                else:
                    conf_score = 0.5

                desired_regime = PredictiveRegimeOptimizer.get_regime_key(conf_score, current_adx)
                if 'active_regime' not in bot:
                    bot['active_regime']        = desired_regime
                    bot['regime_pending_ticks'] = 0
                elif desired_regime != bot['active_regime']:
                    bot['regime_pending_ticks'] = bot.get('regime_pending_ticks', 0) + 1
                    if bot['regime_pending_ticks'] >= PredictiveRegimeOptimizer.REGIME_DEBOUNCE_TICKS:
                        prev_regime             = bot['active_regime']
                        bot['active_regime']    = desired_regime
                        bot['regime_pending_ticks'] = 0
                        await emit_log(user_id, f"🔄 REGIME SHIFT: {prev_regime.upper()} → {desired_regime.upper()}")
                else:
                    bot['regime_pending_ticks'] = 0

                active_strategies = (
                    PredictiveRegimeOptimizer.dynamically_tune_strategies(
                        config, conf_score, current_adx,
                        locked_regime=bot['active_regime'])
                    if config.get('mlMode') == 'on'
                    else config.get('strategies', strategies)
                )
                temp_config = {**config, 'strategies': active_strategies}

                # FIX #11: calculate_signals runs ~20 pandas_ta indicator passes;
                # offload to a worker thread so it doesn't block the event loop
                # every tick. (precomputed_conf is passed, so it does NO ML work.)
                sig, thoughts, nums, score, signals_map = await asyncio.to_thread(
                    StrategyBrain.calculate_signals,
                    df_closed_history, df_ai, temp_config, 0.5, 0.5,
                    ticker_symbol, conf_score, current_price)

                # ── TIER 3: PROP-MAPPING ──────────────────────────────────────
                # NeuralPredictor returns P(bull). In a bear market high conviction
                # = low raw score, so invert once here.  Use effective_score for
                # ALL downstream display and gate comparisons.  Raw `score` is kept
                # only for StrategyBrain's own gate (already written with inversion)
                # and for climb_satisfied (compares to stored entry_conf).
                is_short_market = current_price < current_ema200
                effective_score  = (1.0 - score) if is_short_market else score

                sentiment = "STRONG BUY" if effective_score > 0.85 else "BUY" if effective_score > 0.70 else "NEUTRAL"
                if effective_score < 0.20:   sentiment = "STRONG SELL"
                elif effective_score < 0.35: sentiment = "SELL"

                last_pos = bot['positions'][-1] if bot['positions'] else None
                if last_pos:
                    pos_is_profitable = (
                        (last_pos['type'] == 'long'  and current_price > last_pos['entry']) or
                        (last_pos['type'] == 'short' and current_price < last_pos['entry'])
                    )
                    # Direction-aware conviction climb. entry_conf is the raw
                    # P(bull) at entry; for shorts, conviction rises as P(bull)
                    # falls, so compare in effective (1 - p) space.
                    _entry_conf = last_pos.get('entry_conf', 0)
                    if last_pos['type'] == 'short':
                        _conf_climb = (1.0 - score) >= (1.0 - _entry_conf) + 0.10
                    else:
                        _conf_climb = score >= _entry_conf + 0.10
                    climb_satisfied = _conf_climb and pos_is_profitable
                else:
                    climb_satisfied = True

                is_short_allowed = (sig != -1) or config.get('enable_shorting', False)

                upnl = sum([
                    (current_price - p['entry']) * p['size'] if p['type'] == 'long'
                    else (p['entry'] - current_price) * p['size']
                    for p in bot['positions']
                ])
                current_equity = bot['balance'] + upnl

                active_summary = ""
                if bot['positions']:
                    p        = bot['positions'][-1]
                    stop_pct = round((abs(current_price - p['tsl']) / current_price) * 100, 2)
                    buffer_bar = DiagnosticLayer.render_progress(stop_pct, 2.0, reverse=True)
                    active_summary = f"⚡ ACTIVE: {len(bot['positions'])} POS (${upnl:,.2f}) | TSL {buffer_bar} {stop_pct}% | "

                # ── TIER 4: RISK GATEWAY ──────────────────────────────────────
                ui_limit = (float(config.get('mlThresholdShort', 0.55))
                            if current_price < current_ema200
                            else float(config.get('mlThresholdLong', 0.55)))
                # FIX #11: get_pending_conditions also runs several pandas_ta calls
                # each tick — offload it to a worker thread too.
                waiting_msg = await asyncio.to_thread(
                    DiagnosticLayer.get_pending_conditions, df_raw, temp_config, effective_score, ui_limit)

                base_threshold = ui_limit
                if current_adx > 30:   adaptive_threshold = max(0.45, base_threshold - 0.05)
                elif current_adx < 20: adaptive_threshold = min(0.80, base_threshold + 0.05)
                else:                  adaptive_threshold = base_threshold

                recent_trades      = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit'][-5:]
                consecutive_losses = 0
                for t in reversed(recent_trades):
                    if float(t.get('pnl', 0)) < 0: consecutive_losses += 1
                    else: break

                if consecutive_losses >= 3:
                    adaptive_threshold = min(0.80, adaptive_threshold + (consecutive_losses - 2) * 0.03)
                    if consecutive_losses == 3 and (datetime.now().timestamp() - last_log) >= 15:
                        await emit_log(user_id, f"⚠️ COLD STREAK: {consecutive_losses} losses — raising bar to {int(adaptive_threshold*100)}%")

                # FIX: effective_score is direction-aware — bear conviction (low P(bull))
                # now correctly clears the adaptive threshold in short markets.
                adaptive_gate = effective_score >= adaptive_threshold if config.get('mlMode') == 'on' else True

                atr_pct    = (current_atr / current_price) * 100
                # Loosened volatility band: lower ATR floor (0.3 -> 0.1) and higher
                # ceiling (3.0 -> 5.0) so quieter/hotter markets aren't filtered out.
                # NOTE: looser = more trades in marginal conditions; validate in a
                # backtest before trusting live.
                min_atr_pct = float(config.get('minAtrPct', 0.1))
                max_atr_pct = float(config.get('maxAtrPct', 5.0))
                is_volatility_safe = min_atr_pct <= atr_pct <= max_atr_pct

                # Daily loss breaker with a real UTC-day reset; measured on
                # equity (includes open-position PnL), not just realized balance.
                _today = current_time.date().isoformat()
                if bot.get('circuit_day') != _today:
                    bot['circuit_day']      = _today
                    bot['day_start_equity'] = current_equity
                _day_anchor = float(bot.get('day_start_equity', start_capital))
                is_circuit_breaker_tripped = (
                    current_equity <= _day_anchor * (1.0 - float(config.get('maxDailyLoss', 5.0)) / 100.0))

                # Max-drawdown breaker: session equity peak-to-trough guard.
                bot['peak_equity'] = max(float(bot.get('peak_equity', current_equity)), current_equity)
                _dd_limit = float(config.get('maxDrawdown', 100.0))
                is_drawdown_tripped = current_equity <= bot['peak_equity'] * (1.0 - _dd_limit / 100.0)

                # Daily trade cap (maxTradesPerDay was previously ignored).
                if bot.get('trade_day') != _today:
                    bot['trade_day']    = _today
                    bot['trades_today'] = 0
                _max_trades = int(config.get('maxTradesPerDay', 10000))
                under_daily_trade_cap = int(bot.get('trades_today', 0)) < _max_trades

                # Minimum agreeing votes (minVotesRequired was previously ignored).
                _required_votes = int(config.get('minVotesRequired', 1))
                agreeing_votes = (sum(1 for s in active_strategies
                                      if nums.get(s.get('code'), 0) == sig) if sig != 0 else 0)
                enough_votes = (sig == 0) or (agreeing_votes >= _required_votes)

                if len(bot['positions']) == 0 and sig == 0: signal_has_reset = True
                elif len(bot['positions']) > 0:             signal_has_reset = False
                market_gate_passed = signal_has_reset if len(bot['positions']) == 0 else True

                try:    adx_trending = current_adx >= float(config.get('minAdx', 20.0))
                except: adx_trending = True

                volume_confirmed = vol_ratio >= float(config.get('minVolRatio', 0.2))
                if config.get('requireTrendAlignment', True):
                    is_trend_aligned = (
                        (sig == 1 and current_price > current_ema200) or
                        (sig == -1 and current_price < current_ema200) or
                        (sig == 0)
                    )
                else:
                    is_trend_aligned = True

                # ── DYNAMIC TIME GATE ────────────────────────────────────────
                # Standard cooldown is 10 minutes.  The gate can be bypassed
                # early once a 2-minute hard floor has elapsed AND the AI
                # confidence has both improved since the last entry AND is
                # trending upward over recent ticks — meaning the market is
                # genuinely improving, not just bouncing off a local noise peak.
                #
                # Hard floor  → prevents re-hammering on fast reversals
                # Normal gate → 10-minute default inter-trade spacing
                # Surge path  → 2-min floor + conf ≥8 pts above last entry
                #                + recent 3-tick trend rising ≥5 pts
                MIN_COOLDOWN_SECS  = 2 * 60
                STANDARD_GATE_SECS = float(config.get('minMinutesBetweenTrades', 10)) * 60

                time_since_last = 0.0
                if bot.get('last_trade_time'):
                    try:
                        time_since_last = (
                            current_time - datetime.fromisoformat(bot['last_trade_time'])
                        ).total_seconds()
                    except Exception:
                        time_since_last = STANDARD_GATE_SECS  # treat as expired

                past_floor  = time_since_last >= MIN_COOLDOWN_SECS
                past_normal = time_since_last >= STANDARD_GATE_SECS

                # How much has confidence improved since the last trade entry?
                conf_at_last_entry = float(bot.get('last_trade_conf', 0.0))
                conf_improvement   = score - conf_at_last_entry  # positive = risen

                # Is confidence trending up over the last 6 equity-curve points?
                recent_curve  = bot.get('equityCurve', [])[-6:]
                recent_confs  = [p.get('confidence', 50) / 100.0 for p in recent_curve]
                if len(recent_confs) >= 4:
                    half          = len(recent_confs) // 2
                    earlier_avg   = sum(recent_confs[:half])  / half
                    later_avg     = sum(recent_confs[half:])  / half
                    conf_trending = (later_avg - earlier_avg) >= 0.05
                else:
                    conf_trending = False  # not enough history yet

                # Gate passes when:
                #   A) Normal: standard gap elapsed, OR
                #   B) Surge:  2-min floor cleared + conf rose ≥8 pts + trend up
                time_gate_passed = (
                    not bot.get('last_trade_time') or
                    past_normal or
                    (past_floor and conf_improvement >= 0.08 and conf_trending)
                )

                # FIX #3: system-level multi-day LIVE drawdown breaker. Halts NEW
                # live entries (open positions are still managed live) when this
                # bot's realized LIVE PnL over the window breaches the limit.
                system_halt = False
                if is_live_trading:
                    system_halt = _system_live_halt(
                        user_id, start_capital,
                        int(config.get('systemDrawdownDays', 3)),
                        float(config.get('maxSystemDrawdownPct', 15.0)))
                    if system_halt and not bot.get('_sys_halt_alerted'):
                        logger.error(f"🚨 SYSTEM RISK HALT {user_id}: multi-day live drawdown limit breached.")
                        await emit_log(user_id, "🚨 SYSTEM RISK HALT: multi-day live drawdown limit hit — new live entries disabled (open positions still managed). Reset to clear.")
                        bot['_sys_halt_alerted'] = True

                # DIRECTION FILTER: honor trade_direction (LONG / SHORT / BOTH) so a
                # bot can run a pure long OR pure short stream. The validated edge is
                # DIRECTIONAL (strong longs on up-trends + daily shorts on risk-off
                # legs); naive BOTH is weaker. Shorts ALSO require enable_shorting
                # (is_short_allowed) on top of this, so a SHORT bot needs both
                # trade_direction='SHORT' and enable_shorting=true.
                _td = str(config.get('trade_direction', 'BOTH')).upper()
                direction_allowed = (
                    sig == 0 or _td == 'BOTH' or
                    (_td == 'LONG' and sig == 1) or (_td == 'SHORT' and sig == -1))

                # MACRO REGIME TILT (opt-in via macroRegimeGate): hold LONGS in a
                # BTC risk-off regime and SHORTS in risk-on, so the fleet leans WITH
                # the whole market. Neutral lets both sides trade (the per-coin trend
                # gate still applies). Only gates NEW entries; open positions keep
                # being managed. Reads MACRO_REGIME (updated by _macro_regime_loop).
                macro_ok = True
                if config.get('macroRegimeGate', False) and sig != 0:
                    _ms = MACRO_REGIME.get("state", "neutral")
                    if _ms == "risk_off" and sig == 1:
                        macro_ok = False
                    elif _ms == "risk_on" and sig == -1:
                        macro_ok = False

                # FLEET RISK CAP: halt new entries for every bot in this fleet if the
                # roster's combined balance draws down past the fleet limit.
                fleet_halt = False
                _fid = config.get('fleetId')
                if _fid:
                    fleet_halt = _fleet_risk_halt(_fid, float(config.get('fleetMaxDrawdownPct', 15.0)))
                    if fleet_halt and not bot.get('_fleet_halt_alerted'):
                        await emit_log(user_id, "🚨 FLEET RISK CAP: combined fleet drawdown limit hit — new entries paused across the fleet (open positions still managed).")
                        bot['_fleet_halt_alerted'] = True

                all_filters_pass = (
                    is_volatility_safe and not is_circuit_breaker_tripped and
                    not is_drawdown_tripped and under_daily_trade_cap and enough_votes and
                    is_trend_aligned and adx_trending and volume_confirmed and
                    adaptive_gate and market_gate_passed and is_short_allowed and time_gate_passed and
                    direction_allowed and macro_ok and not fleet_halt and not system_halt and
                    not KILL_SWITCH['on']
                )

                # STDOUT decision diagnostic (emit_log goes to the Node UI, not
                # stdout, so mirror the gate state here for server-side debugging).
                if (datetime.now().timestamp() - last_diag) >= 15:
                    logger.info(
                        f"🔎 DECIDE {user_id} | sig={sig} eff={effective_score:.2f} "
                        f"filters_pass={all_filters_pass} pos={len(bot['positions'])} || "
                        f"vol_safe={is_volatility_safe} cb={is_circuit_breaker_tripped} "
                        f"dd={is_drawdown_tripped} cap_ok={under_daily_trade_cap} "
                        f"votes_ok={enough_votes}({agreeing_votes}/{_required_votes}) "
                        f"trend_ok={is_trend_aligned} adx_ok={adx_trending} "
                        f"vol_ok={volume_confirmed} ai_gate={adaptive_gate} "
                        f"mkt_gate={market_gate_passed} short_ok={is_short_allowed} "
                        f"time_gate={time_gate_passed}"
                    )
                    last_diag = datetime.now().timestamp()

                # ── TIER 5: TELEMETRY + VETO TRACKER ─────────────────────────
                if "vetoed_signals" not in bot:
                    bot["vetoed_signals"] = []

                raw_sig = 0
                # Use signed per-strategy directions (nums), not confidence magnitude.
                votes_t = sum(nums.get(s['code'], 0) for s in active_strategies)
                # FIX: use helper so hybridMode is honoured when comboConfig is absent/stale
                rule    = _get_combo_rule(config)
                if rule == "AND":
                    if votes_t >= len(active_strategies):    raw_sig = 1
                    elif votes_t <= -len(active_strategies): raw_sig = -1
                else:
                    if votes_t > 0:  raw_sig = 1
                    elif votes_t < 0: raw_sig = -1

                if raw_sig != 0 and effective_score < ui_limit and config.get('mlMode') == 'on':
                    last_veto_ts = (datetime.fromisoformat(bot["vetoed_signals"][-1]["time"]).timestamp()
                                    if bot["vetoed_signals"] else 0)
                    if (current_time.timestamp() - last_veto_ts) > 300:
                        bot["vetoed_signals"].append({
                            "time": current_time.isoformat(), "signal": "Long" if raw_sig == 1 else "Short",
                            "conf_score": round(effective_score, 4), "limit": round(ui_limit, 4), "price": current_price
                        })
                        if len(bot["vetoed_signals"]) > 100: bot["vetoed_signals"].pop(0)

                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    hunting_summary = f"🏹 STALKING LEG {len(bot['positions'])+1}: {int(effective_score*100)}% ({sentiment}) | {waiting_msg}"
                else:
                    hunting_summary = "✅ PYRAMID FULL: Managing Exits"
                targets_str = " | ".join([
                    f"{s['code'].upper()}: {DiagnosticLayer.render_progress(int(signals_map.get(s['code'], 0)*100), 100, reverse=True)}"
                    for s in active_strategies
                ])
                combined_status = f"{active_summary}{hunting_summary} | 🔍 {targets_str}"
                now_ts = datetime.now().timestamp()
                if (now_ts - last_log) >= 15:
                    await emit_log(user_id, combined_status)
                    last_log = now_ts

                # ── TIER 5b: UI BROADCAST ─────────────────────────────────────
                if (now_ts - last_ui_update) >= 10:
                    exposure_pct = (round((sum([p['entry'] * p['size'] for p in bot['positions']])
                                           / current_equity) * 100, 1)
                                    if current_equity > 0 else 0)
                    latest_candles = await process_data_packet(df_raw, active_strategies)

                    if config.get('mlMode') == 'on':
                        active_regime = bot.get('active_regime', 'range')
                        if active_regime == "trend":
                            deployed_gear = "Trend Armor (SuperTrend, PA Breakout, EMA Cloud, SMA Cross)"
                        elif active_regime == "bear":
                            deployed_gear = "Capitulation Armor (SuperTrend, ATR Breakout, MACD Cross, Vol Profile)"
                        else:
                            deployed_gear = "Range Armor (Bollinger Bands, RSI Threshold, Stochastic)"
                    else:
                        deployed_gear = f"Custom Suite ({', '.join([s['code'].upper() for s in active_strategies])})"

                    if len(bot['positions']) >= max_p:
                        regime_title = "Portfolio Full"
                        regime_desc  = f"All slots filled ({len(bot['positions'])}/{max_p}). Managing exits."
                    elif sig != 0 and all_filters_pass:
                        regime_title = "Executing Entry"
                        regime_desc  = f"All checkpoints cleared. Dispatching route at ${current_price:,.2f}."
                    elif sig != 0 and not all_filters_pass:
                        regime_title = "Entry Guarded"
                        regime_desc  = "Defensive logic engaged. Holding until signals optimise."
                    else:
                        regime_title = "Quiet Market Structure"
                        regime_desc  = f"Consolidation at ${current_price:,.2f}. Standing guard."

                    closed_trades = [t for t in bot.get('trade_history', []) if t.get('type') == 'exit']
                    if closed_trades:
                        wins          = len([t for t in closed_trades if float(t.get('pnl', 0)) > 0])
                        win_rate      = round((wins / len(closed_trades)) * 100, 1)
                        gross_profit  = sum(float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) > 0)
                        gross_loss    = abs(sum(float(t.get('pnl', 0)) for t in closed_trades if float(t.get('pnl', 0)) < 0))
                        profit_factor = (round(gross_profit / gross_loss, 2) if gross_loss > 0
                                         else (round(gross_profit, 2) if gross_profit > 0 else 1.0))
                    else:
                        win_rate, profit_factor = 0.0, 1.0

                    # FIX: compute daily_profit for the MetricCard
                    daily_profit = round(current_equity - start_capital, 2)

                    # effective_score is already direction-aware (computed in Tier 3)
                    display_conf = int(effective_score * 100)

                    # 📊 READINESS METER — how close this bot is to firing an entry:
                    # the fraction of the 10 entry gates currently green (100 = all
                    # clear → dispatching). `_blocker` = the first gate still red, for
                    # the "closest to trading" ranking + per-bot meter in the UI.
                    _gate_flags = [
                        ("Volatility",   bool(is_volatility_safe)),
                        ("Votes",        bool(enough_votes)),
                        ("Trend align",  bool(is_trend_aligned)),
                        ("ADX trend",    bool(adx_trending)),
                        ("Volume",       bool(volume_confirmed)),
                        ("AI gate",      bool(adaptive_gate)),
                        ("Direction",    bool(direction_allowed)),
                        ("Macro tilt",   bool(macro_ok)),
                        ("Cooldown",     bool(market_gate_passed)),
                        ("Risk breaker", not (is_circuit_breaker_tripped or is_drawdown_tripped or fleet_halt or system_halt or KILL_SWITCH['on'])),
                    ]
                    _passed = sum(1 for _, _ok in _gate_flags if _ok)
                    _ntot = len(_gate_flags)
                    # Smooth the meter: when the ADX trend-strength gate is the holdup
                    # (the usual last blocker for longs), add fractional credit for how
                    # close ADX is to its threshold, so the bar GLIDES toward 100 and
                    # users can watch a bot climb toward firing instead of it jumping.
                    _frac = float(_passed)
                    if not adx_trending:
                        try:
                            _minadx = float(config.get('minAdx', 20.0)) or 20.0
                            _frac += max(0.0, min(1.0, float(current_adx) / _minadx))
                        except Exception:
                            pass
                    # A bot is only truly "about to fire" when the gates are clear AND a
                    # directional entry signal actually exists (sig != 0). All gates green
                    # with a FLAT signal = waiting for the setup to appear, NOT 100%.
                    _has_signal = (sig != 0)
                    if all_filters_pass and _has_signal:
                        _readiness = 100
                        _blocker = None
                    elif all_filters_pass:
                        _readiness = 95
                        _blocker = "Signal"
                    else:
                        _readiness = min(99, round(100 * _frac / _ntot))
                        _blocker = next((_k for _k, _ok in _gate_flags if not _ok), None)

                    bot.update({
                        "currentBalance":    round(current_equity, 2),
                        "unrealizedPnl":     round(upnl, 2),
                        "exposure":          exposure_pct,
                        "currentConfidence": display_conf,
                        "signalsMap":        signals_map,
                        "candles":           latest_candles,
                        "aiRegimeTitle":     regime_title,
                        "aiRegimeDesc":      regime_desc,
                        "aiDeployedGear":    deployed_gear,
                        "winRate":           win_rate,
                        "profitFactor":      profit_factor,
                        "dailyProfit":       daily_profit,
                        # 🧠 THINKING SNAPSHOT — what the bot sees RIGHT NOW: the
                        # signal + the live entry-gate checklist (which condition is
                        # blocking) + the regime read. Powers the Fleet 'neural flow'.
                        "thinking": {
                            "sig":     sig,
                            "signal":  ("LONG" if sig == 1 else "SHORT" if sig == -1 else "FLAT"),
                            "score":   display_conf,
                            "regime":  regime_title,
                            "note":    regime_desc,
                            "waiting": (str(waiting_msg)[:160] if waiting_msg else None),
                            "all_pass": bool(all_filters_pass),
                            "readiness": _readiness,
                            "blocker": _blocker,
                            "gates": [{"k": _k, "ok": _ok} for _k, _ok in _gate_flags],
                        },
                    })

                    # FIX: append equity curve point BEFORE emit so the payload
                    # contains the up-to-date curve (previously appended after emit).
                    bot["equityCurve"].append({
                        "time":       datetime.now().isoformat(),
                        "balance":    round(current_equity, 2),
                        "confidence": display_conf   # direction-aware (inverted for shorts)
                    })
                    if len(bot["equityCurve"]) > 300:
                        bot["equityCurve"].pop(0)

                    # FIX: diagnostic log so you can confirm emit_status is actually reached.
                    # Remove or set to DEBUG once charts are confirmed working.
                    logger.info(
                        f"📡 emit_status → {user_id} | "
                        f"balance=${bot['currentBalance']:.2f} | "
                        f"conf={bot['currentConfidence']}% | "
                        f"signals={list(signals_map.keys())} | "
                        f"equity_pts={len(bot['equityCurve'])}"
                    )

                    # FIX: emit_status payload now includes equityCurve, dailyProfit,
                    # startedAt, and tradeHistory — previously missing keys that caused
                    # the equity/confidence area charts to stay blank until the frontend
                    # accumulated 2+ events from its own local curve builder.
                    await emit_status(user_id, {
                        "status":            "running",
                        "currentBalance":    bot["currentBalance"],
                        "currentConfidence": bot["currentConfidence"],
                        "mlMode":            config.get('mlMode', 'off'),
                        "signalsMap":        bot["signalsMap"],
                        # FIX: equityCurve was never emitted via socket — charts built
                        # locally were blank until 2+ events arrived (min ~20 s).
                        "equityCurve":       bot["equityCurve"],
                        # FIX: dailyProfit was absent — MetricCard showed 0 on every reconnect.
                        "dailyProfit":       bot["dailyProfit"],
                        # FIX: startedAt was absent — uptime timer reset on reconnect.
                        "startedAt":         bot.get("startedAt"),
                        "exposure":          bot["exposure"],
                        "activePositions":   bot['positions'],
                        "unrealizedPnl":     bot["unrealizedPnl"],
                        # FIX: send under both keys so the frontend fallback chain always hits.
                        # FIX #23: cap to the last 200 to stop the payload growing
                        # without bound (it was sent in full, twice, every ~10s).
                        "tradeHistory":      bot['trade_history'][-200:],
                        "tradeMarkers":      bot['trade_history'][-200:],
                        "candles":           bot["candles"],
                        "initialCapital":    start_capital,
                        "aiRegimeTitle":     bot["aiRegimeTitle"],
                        "aiRegimeDesc":      bot["aiRegimeDesc"],
                        "aiDeployedGear":    bot["aiDeployedGear"],
                        "winRate":           bot["winRate"],
                        "profitFactor":      bot["profitFactor"],
                        "lastTradeScore":    bot.get("lastTradeScore"),
                    })

                    logger.info(f"✅ emit_status dispatched for {user_id}")
                    last_ui_update = now_ts

                # ── TIER 6: REJECTION LOGGER ──────────────────────────────────
                if sig != 0 and not all_filters_pass:
                    reasons = []
                    if not time_gate_passed:
                        elapsed_min    = time_since_last / 60
                        normal_min     = STANDARD_GATE_SECS / 60
                        floor_min      = MIN_COOLDOWN_SECS  / 60
                        surge_eligible = past_floor and conf_improvement >= 0.08
                        if not past_floor:
                            reasons.append(
                                f"hard floor ({elapsed_min:.0f}/{floor_min:.0f} min)"
                            )
                        elif not surge_eligible:
                            reasons.append(
                                f"time gate ({elapsed_min:.0f}/{normal_min:.0f} min, "
                                f"conf Δ{int(conf_improvement*100):+}% — need +8% surge to bypass)"
                            )
                        else:
                            # Floor cleared and conf rose, but trend not yet up
                            reasons.append(
                                f"time gate ({elapsed_min:.0f}/{normal_min:.0f} min, "
                                f"conf Δ{int(conf_improvement*100):+}% but trend still flat)"
                            )
                    if not is_short_allowed: reasons.append("Spot mode cannot short sell")
                    if not is_volatility_safe:
                        if atr_pct < min_atr_pct: reasons.append(f"ATR {atr_pct:.2f}% too flat (min {min_atr_pct}%)")
                        else:                     reasons.append(f"ATR {atr_pct:.1f}% too high (max {max_atr_pct}%)")
                    if is_circuit_breaker_tripped: reasons.append("Daily drawdown limit hit")
                    if is_drawdown_tripped:        reasons.append(f"Max drawdown {_dd_limit:.0f}% from peak")
                    if not under_daily_trade_cap:  reasons.append(f"Daily trade cap {bot.get('trades_today',0)}/{_max_trades}")
                    if not enough_votes:           reasons.append(f"Votes {agreeing_votes}/{_required_votes} agree")
                    if not is_trend_aligned:       reasons.append("Trend misaligned with 200 EMA")
                    if not adx_trending:           reasons.append(f"ADX {current_adx:.0f} below min {float(config.get('minAdx', 20.0)):.0f}")
                    if not volume_confirmed:       reasons.append(f"Volume {vol_ratio:.2f}x below min {float(config.get('minVolRatio', 0.2)):.1f}x")
                    if not adaptive_gate:          reasons.append(f"AI {int(effective_score*100)}% below threshold {int(adaptive_threshold*100)}%")
                    if not market_gate_passed:     reasons.append("Waiting for signal reset")
                    if not reasons: reasons.append("Pre-flight check failed")
                    await emit_log(user_id, f"⛔ ENTRY BLOCKED: {', '.join(reasons)}")

                # ── TIER 7: ENTRY DISPATCH ────────────────────────────────────
                if len(bot['positions']) < max_p and all_filters_pass:
                    if (sig == 1 and climb_satisfied) or (sig == -1 and climb_satisfied):

                        # ── PRE-TRADE QUALITY SCORE ───────────────────────────
                        # Runs AFTER all hard filters pass. Synthesises three
                        # independent perspectives into a 0–100 composite score:
                        #   • Market Context  (35 pts) — EMA stack, ATR regime,
                        #                                ADX, volume
                        #   • Bot Intelligence (35 pts) — ML margin above gate,
                        #                                 confidence trajectory,
                        #                                 recent win/loss record
                        #   • Strategy Alignment (30 pts) — vote unanimity,
                        #                                   mean confidence,
                        #                                   category diversity
                        # Trade only proceeds when composite ≥ MIN_ENTRY_SCORE.
                        trade_quality = TradeQualityScorer.evaluate(
                            df_raw            = df_raw,
                            bot               = bot,
                            ml_score          = score,
                            ui_limit          = ui_limit,
                            signals_map       = signals_map,
                            directions_map    = nums,
                            active_strategies = active_strategies,
                            sig               = sig,
                            current_price     = current_price,
                            current_atr       = current_atr,
                            current_adx       = current_adx,
                            vol_ratio         = vol_ratio,
                            min_score         = float(config.get('minEntryScore', TradeQualityScorer.MIN_ENTRY_SCORE)),
                            ml_active         = (config.get('mlMode') == 'on'),
                        )

                        # Always log the verdict so it appears in Neural Flow
                        await emit_log(user_id, trade_quality['log_line'])

                        # Store latest score on bot so frontend can display it
                        bot['lastTradeScore'] = {
                            "composite":  trade_quality['composite_score'],
                            "approved":   trade_quality['approved'],
                            "market":     trade_quality['market']['total'],
                            "bot":        trade_quality['bot']['total'],
                            "strategies": trade_quality['strategies']['total'],
                            "verdict":    trade_quality['verdict'],
                        }

                        # In Bypass (mlMode != 'on') the user has explicitly opted
                        # out of AI gating, so the quality score is ADVISORY only:
                        # it is logged above but does not block the entry. With ML
                        # on it stays a hard gate at minEntryScore.
                        _quality_bypass = config.get('mlMode') != 'on'

                        # 🧠 LEDGER GATE: veto setups the bot's own closed-trade model
                        # has learned tend to lose. No-op until a ledger model exists
                        # (needs enough trades); set useLedgerGate=false to disable.
                        _lpw = (_ledger_pwin(ticker_symbol, config.get('timeframe', '1h'), df_ai.iloc[-1])
                                if config.get('useLedgerGate', True) else None)
                        _ledger_veto = _lpw is not None and _lpw < float(config.get('ledgerMinPwin', 0.40))
                        if _ledger_veto:
                            await emit_log(user_id, f"🧠 LEDGER VETO: learned P(win) {int(_lpw*100)}% below floor {int(float(config.get('ledgerMinPwin', 0.40))*100)}% — skipping entry.")

                        if (not trade_quality['approved'] and not _quality_bypass) or _ledger_veto:
                            # Scorer rejected or ledger veto — skip entry; Tier 8 still runs.
                            pass

                        else:
                            # ── ORDER EXECUTION ───────────────────────────────
                            # Only reached when composite score ≥ MIN_ENTRY_SCORE
                            trade_type         = "long" if sig == 1 else "short"
                            raw_risk           = float(config.get('riskPercentage') or config.get('risk_percentage') or 1.0)
                            raw_risk           = min(raw_risk, float(config.get('maxRiskPerTradePct', 100.0)))
                            # CONVICTION SIZING (opt-in via sizeByConviction): scale
                            # the risked amount between sizeMinMult..sizeMaxMult by the
                            # trade-quality composite (0..100) — bet more on strong
                            # setups, less on weak ones. ATR is already in the sizing
                            # denominator, so size is volatility-aware; this adds a
                            # conviction tilt on top. Default off -> flat risk unchanged.
                            if config.get('sizeByConviction', False):
                                _slo = float(config.get('sizeMinMult', 0.5))
                                _shi = float(config.get('sizeMaxMult', 1.5))
                                _q   = max(0.0, min(1.0, float(trade_quality.get('composite_score', 65)) / 100.0))
                                raw_risk = raw_risk * (_slo + (_shi - _slo) * _q)
                            # FIX #3: size by MONEY-AT-RISK and stop distance, and let
                            # leverage scale the position. Previously size was a flat
                            # % of equity as notional, so riskPercentage was NOT the
                            # amount risked and leverage never affected the position.
                            #   risk_amount   = equity * risk%     (max loss if stop hit)
                            #   stop_distance = ATR * atr_sl_mult   (entry->stop, per unit)
                            #   size          = risk_amount / stop_distance
                            # Notional is capped to equity * leverage (margin <= equity),
                            # and to equity when spot (leverage = 1).
                            per_leg_risk_amt = current_equity * (raw_risk / 100.0) / max_p
                            stop_distance    = current_atr * atr_sl_mult
                            if stop_distance > 0:
                                size_in_crypto = per_leg_risk_amt / stop_distance
                            else:
                                size_in_crypto = per_leg_risk_amt / current_price
                            _lev         = max(1.0, leverage_val)
                            max_notional = current_equity * _lev
                            notional     = size_in_crypto * current_price
                            if notional > max_notional:
                                size_in_crypto = max_notional / current_price
                                notional       = max_notional
                            safe_size          = float(f"{size_in_crypto:.6f}")
                            actual_entry_price = current_price
                            if not bot.get('_risk_warned'):
                                _pe = (notional / current_equity * 100) if current_equity > 0 else 0
                                _sp = (stop_distance / current_price * 100) if current_price > 0 else 0
                                await emit_log(user_id, f"📐 SIZING: risk {raw_risk/max_p:.2f}%/leg (${per_leg_risk_amt:,.2f}) · stop {_sp:.2f}% → notional ${notional:,.2f} ({_pe:.0f}% equity · {_lev:.0f}x)")
                                bot['_risk_warned'] = True

                            if is_live_trading:
                                try:
                                    exchange_class = getattr(ccxt, target_exchange)
                                    async with exchange_class({
                                        'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                        'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                        'enableRateLimit': True
                                    }) as user_exchange:
                                        side         = "buy" if sig == 1 else "sell"
                                        order_params = {'leverage': leverage_val} if use_margin else {}
                                        if config.get('useClientOrderId', False):
                                            order_params = {**order_params, 'clientOrderId': _client_order_id(user_id, symbol, side)}
                                        await emit_log(user_id, f"🔗 ROUTING {side.upper()} TO {target_exchange.upper()}...")
                                        await user_exchange.load_markets()
                                        formatted_size = float(user_exchange.amount_to_precision(symbol, size_in_crypto))
                                        if formatted_size <= 0: raise Exception("Order size too small")
                                        order = await user_exchange.create_market_order(symbol, side, formatted_size, params=order_params)
                                        actual_entry_price = order.get('average') or order.get('price') or current_price
                                        safe_size          = order.get('filled') or formatted_size
                                except Exception as ex_err:
                                    await emit_log(user_id, f"❌ {target_exchange.upper()} ORDER FAILED: {str(ex_err)}")
                                    await asyncio.sleep(5); continue
                            else:
                                bot['balance'] -= (safe_size * actual_entry_price) * fee_rate

                            tp_price_calc = (actual_entry_price + current_atr * atr_tp_mult if sig == 1
                                             else actual_entry_price - current_atr * atr_tp_mult)
                            sl_price_calc = (actual_entry_price - current_atr * atr_sl_mult if sig == 1
                                             else actual_entry_price + current_atr * atr_sl_mult)
                            partial_tp    = (actual_entry_price + current_atr * 1.5 if sig == 1
                                             else actual_entry_price - current_atr * 1.5)

                            try:
                                _feat_row = df_ai.iloc[-1]
                                entry_features = {c: float(_feat_row[c]) for c in FEATURE_COLUMNS
                                                  if c in df_ai.columns and pd.notna(_feat_row[c])}
                            except Exception:
                                entry_features = {}
                            bot['positions'].append({
                                "symbol":             symbol,
                                "type":               trade_type,
                                "entry":              actual_entry_price,
                                "size":               safe_size,
                                "time":               current_time.isoformat(),
                                "entry_conf":         score,
                                "entry_score":        trade_quality['composite_score'],
                                "tp":                 tp_price_calc,
                                "sl":                 sl_price_calc,
                                "tsl":                sl_price_calc,
                                "partial_tp":         partial_tp,
                                "partial_taken":      False,
                                "atr_at_entry":       round(current_atr, 4),
                                "adx_at_entry":       round(current_adx, 1),
                                "vol_ratio_at_entry": round(vol_ratio, 2),
                                "entry_features":     entry_features,
                            })
                            # Record the ENTRY (type 'entry') in trade_history so the
                            # fleet activity feed + the UI can alert on it in real time.
                            # Exits are recorded separately below; the ledger logs only
                            # exits, and stats filter type=='exit', so this is UI-only.
                            bot.setdefault('trade_history', []).append({
                                "type": "entry", "side": trade_type,
                                "entry": round(actual_entry_price, 2),
                                "price": round(actual_entry_price, 2),
                                "reason": "Entry", "time": int(time.time() * 1000),
                            })
                            bot['last_trade_time'] = current_time.isoformat()
                            bot['last_trade_conf'] = score
                            bot['trades_today'] = int(bot.get('trades_today', 0)) + 1
                            rr = round(atr_tp_mult / atr_sl_mult, 1)
                            await emit_log(user_id,
                                           f"🚀 ENTERED {trade_type.upper()} @ ${actual_entry_price:,.2f} | "
                                           f"TP ${tp_price_calc:,.2f} | SL ${sl_price_calc:,.2f} | "
                                           f"R:R {rr} | Score {trade_quality['composite_score']:.0f}/100 | ADX {current_adx:.0f}")
                            await emit_trade_alert(user_id, {"action": "entry", "symbol": config.get('symbol'),
                                                            "side": trade_type, "price": round(actual_entry_price, 2)})
                            await asyncio.to_thread(DatabaseHandler.save_state, user_id, bot)  # FIX #13

                # ── TIER 8: EXIT MONITORING ───────────────────────────────────
                for pos in bot['positions'][:]:
                    closed      = False
                    exit_reason = ""
                    pos_atr     = pos.get('atr_at_entry', current_atr)
                    trail_dist  = pos_atr * atr_sl_mult

                    # BUGFIX #6: backfill exit levels for positions restored from an
                    # older schema so the exit monitor never KeyErrors and orphans them.
                    if 'tsl' not in pos or 'tp' not in pos:
                        _sd    = 1 if pos.get('type') == 'long' else -1
                        _entry = pos.get('entry', current_price)
                        pos.setdefault('tp',         _entry + _sd * current_atr * atr_tp_mult)
                        pos.setdefault('sl',         _entry - _sd * current_atr * atr_sl_mult)
                        pos.setdefault('tsl',        pos.get('sl', _entry - _sd * current_atr * atr_sl_mult))
                        pos.setdefault('partial_tp', _entry + _sd * current_atr * 1.5)

                    if config.get('enablePartialExit', False) and not pos.get('partial_taken', False):

                        if pos['type'] == 'long' and current_price >= pos.get('partial_tp', float('inf')):
                            half_size   = pos['size'] / 2.0
                            partial_pnl = (current_price - pos['entry']) * half_size
                            if is_live_trading:
                                try:
                                    exchange_class = getattr(ccxt, target_exchange)
                                    async with exchange_class({
                                        'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                        'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                        'enableRateLimit': True
                                    }) as user_exchange:
                                        await user_exchange.load_markets()
                                        fmt_size = float(user_exchange.amount_to_precision(symbol, half_size))
                                        order = await user_exchange.create_market_order(symbol, "sell", fmt_size)
                                        partial_pnl = ((order.get('average') or current_price) - pos['entry']) * half_size
                                except Exception as e:
                                    await emit_log(user_id, f"⚠️ Partial long exit failed: {e}")
                            else:
                                net_partial    = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl    = net_partial
                            pos['size']          = half_size
                            pos['partial_taken'] = True
                            pos['tsl']           = max(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL LONG EXIT: 50% at ${current_price:,.2f} | Locked ${partial_pnl:.2f} | Stop → Breakeven")
                            bot['trade_history'].append({"type": "partial_exit", "side": pos['type'], "price": current_price, "pnl": round(partial_pnl, 2), "time": int(time.time() * 1000)})
                            DatabaseHandler.save_state(user_id, bot)

                        elif pos['type'] == 'short' and current_price <= pos.get('partial_tp', float('-inf')):
                            half_size   = pos['size'] / 2.0
                            partial_pnl = (pos['entry'] - current_price) * half_size
                            if is_live_trading:
                                try:
                                    exchange_class = getattr(ccxt, target_exchange)
                                    async with exchange_class({
                                        'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                        'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                        'enableRateLimit': True
                                    }) as user_exchange:
                                        await user_exchange.load_markets()
                                        fmt_size = float(user_exchange.amount_to_precision(symbol, half_size))
                                        order = await user_exchange.create_market_order(symbol, "buy", fmt_size)
                                        partial_pnl = (pos['entry'] - (order.get('average') or current_price)) * half_size
                                except Exception as e:
                                    await emit_log(user_id, f"⚠️ Partial short exit failed: {e}")
                            else:
                                net_partial    = partial_pnl - (half_size * current_price * fee_rate)
                                bot['balance'] += net_partial
                                partial_pnl    = net_partial
                            pos['size']          = half_size
                            pos['partial_taken'] = True
                            pos['tsl']           = min(pos['tsl'], pos['entry'])
                            await emit_log(user_id, f"💎 PARTIAL SHORT EXIT: 50% at ${current_price:,.2f} | Locked ${partial_pnl:.2f} | Stop → Breakeven")
                            bot['trade_history'].append({"type": "partial_exit", "side": pos['type'], "price": current_price, "pnl": round(partial_pnl, 2), "time": int(time.time() * 1000)})
                            DatabaseHandler.save_state(user_id, bot)

                    # EXIT MODE SELECTOR
                    # 'trend_ride' = the VALIDATED edge (exit_lab.py): hold until the
                    # entry signal FLIPS against the position, protected only by the
                    # fixed initial ATR stop — NO trailing ratchet, NO take-profit cap
                    # — so winners run with the trend (this is where the money was).
                    # Anything else = the original trailing-stop + ATR-TP behavior
                    # (default; fully unchanged).
                    _exit_mode = str(config.get('exitMode', 'standard')).lower()
                    if _exit_mode == 'trend_ride':
                        if pos['type'] == 'long':
                            if current_price <= pos['tsl']:
                                closed = True; exit_reason = "Trend Stop"
                            elif sig == -1:
                                closed = True; exit_reason = "Signal Flip"
                        else:
                            if current_price >= pos['tsl']:
                                closed = True; exit_reason = "Trend Stop"
                            elif sig == 1:
                                closed = True; exit_reason = "Signal Flip"
                    # Trailing stop ratchet (standard / default)
                    elif pos['type'] == 'long':
                        new_tsl = current_price - trail_dist
                        if new_tsl > pos['tsl']: pos['tsl'] = new_tsl
                        if current_price >= pos['tp']:    closed = True; exit_reason = "Take Profit"
                        elif current_price <= pos['tsl']: closed = True; exit_reason = "Trailing Stop"
                    else:
                        new_tsl = current_price + trail_dist
                        if new_tsl < pos['tsl']: pos['tsl'] = new_tsl
                        if current_price <= pos['tp']:    closed = True; exit_reason = "Take Profit"
                        elif current_price >= pos['tsl']: closed = True; exit_reason = "Trailing Stop"

                    # FIX #1: a tripped daily-loss / max-drawdown breaker must
                    # LIQUIDATE open positions, not just block new entries.
                    # Previously the breakers only fed all_filters_pass (the entry
                    # gate), so an open losing position kept running to its ATR
                    # stop far past the configured loss cap.
                    if (is_circuit_breaker_tripped or is_drawdown_tripped) and not closed:
                        closed      = True
                        exit_reason = ("Circuit Breaker" if is_circuit_breaker_tripped
                                       else "Max Drawdown")

                    if closed:
                        actual_close_price = current_price
                        if is_live_trading:
                            try:
                                exchange_class = getattr(ccxt, target_exchange)
                                async with exchange_class({
                                    'apiKey': api_keys.get('krakenKey', api_keys.get('apiKey')),
                                    'secret': api_keys.get('krakenSecret', api_keys.get('secret')),
                                    'enableRateLimit': True
                                }) as user_exchange:
                                    close_side   = "sell" if pos['type'] == 'long' else "buy"
                                    order_params = {'leverage': leverage_val} if use_margin else {}
                                    if config.get('useClientOrderId', False):
                                        order_params = {**order_params, 'clientOrderId': _client_order_id(user_id, symbol, close_side)}
                                    await user_exchange.load_markets()
                                    fmt_size = float(user_exchange.amount_to_precision(symbol, pos['size']))
                                    order = await user_exchange.create_market_order(symbol, close_side, fmt_size, params=order_params)
                                    actual_close_price = order.get('average') or order.get('price') or current_price
                                    balance_data = await user_exchange.fetch_balance()
                                    real_fiat = balance_data.get('USD', {}).get('free') or balance_data.get('USDC', {}).get('free')
                                    if real_fiat: bot['balance'] = float(real_fiat)
                            except Exception as ex_err:
                                await emit_log(user_id, f"❌ EXIT FAILED: {str(ex_err)}")
                                continue

                        gross_pnl = ((actual_close_price - pos['entry']) * pos['size'] if pos['type'] == 'long'
                                     else (pos['entry'] - actual_close_price) * pos['size'])
                        net_pnl   = gross_pnl - ((pos['size'] * actual_close_price) * fee_rate)
                        if not is_live_trading: bot['balance'] += net_pnl

                        bot['positions'].remove(pos)
                        bot['trade_history'].append({
                            "type":   "exit",
                            "side":   pos['type'],
                            "entry":  pos['entry'],
                            "price":  actual_close_price,
                            "pnl":    round(net_pnl, 2),
                            "reason": exit_reason,
                            "time":   int(time.time() * 1000)
                        })
                        try:
                            _entry_dt = datetime.fromisoformat(pos['time'])
                            _hold = (current_time - _entry_dt).total_seconds()
                        except Exception:
                            _hold = None
                        trade_recorder.record_exit(
                            user_id=user_id, symbol=config['symbol'],
                            timeframe=config.get('timeframe', '1h'),
                            direction=pos['type'],
                            mode=('live' if is_live_trading else 'paper'),
                            entry_price=pos['entry'], exit_price=actual_close_price,
                            size=pos['size'], pnl=round(net_pnl, 2), reason=exit_reason,
                            entry_conf=pos.get('entry_conf'), composite=pos.get('entry_score'),
                            adx_at_entry=pos.get('adx_at_entry'), atr_at_entry=pos.get('atr_at_entry'),
                            vol_ratio=pos.get('vol_ratio_at_entry'),
                            features=pos.get('entry_features'), hold_secs=_hold)
                        await emit_log(user_id,
                                       f"💰 {exit_reason}: CLOSED {pos['type'].upper()} @ "
                                       f"${actual_close_price:,.2f} | Net PnL: ${round(net_pnl, 2):+}")
                        await emit_trade_alert(user_id, {"action": "exit", "symbol": config.get('symbol'),
                                                        "side": pos['type'], "price": round(actual_close_price, 2),
                                                        "pnl": round(net_pnl, 2), "reason": exit_reason})
                        await asyncio.to_thread(DatabaseHandler.save_state, user_id, bot)  # FIX #13

                # Periodic DB save
                # FIX #13: offload sqlite writes to a worker thread so the JSON-blob
                # write never blocks the shared event loop in the hot path.
                if (datetime.now().timestamp() - last_log) >= 60:
                    await asyncio.to_thread(DatabaseHandler.save_state, user_id, bot)

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ WS Stream Error: {e}", exc_info=True)
                await asyncio.sleep(5)
    finally:
        if read_exchange is not None:
            try:
                await read_exchange.close()
            except Exception:
                pass
        logger.info(f"🔌 Heartbeat loop terminated for {user_id}")


# =============================================================
# ENDPOINTS
# =============================================================
async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int,
                                   exchange_id: str = "coinbase"):
    exchange_class = getattr(ccxt, exchange_id)
    async with exchange_class({'enableRateLimit': True}) as ex:
        try:
            fetch_symbol = symbol.replace('-', '/')
            ohlcv = await ex.fetch_ohlcv(fetch_symbol, timeframe, limit=limit)
            return [{"time": int(c[0]/1000), "open": c[1], "high": c[2], "low": c[3],
                     "close": c[4], "volume": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except Exception as e:
            logger.error(f"❌ {exchange_id.upper()} Fetch Error: {e}")
            return []


@app.get("/api/ml/available-models")
@app.get("/ml/available-models")
@app.get("/api/models")
def list_models():
    if not os.path.exists(MODEL_DIR): return {"status": "success", "models": []}
    models = [{"id": f.rsplit('.', 1)[0], "name": f.rsplit('.', 1)[0]}
              for f in os.listdir(MODEL_DIR) if f.endswith(('.keras', '.joblib', '.pkl'))]
    return {"status": "success", "models": models}


# ============================================================
# 🧭 STRATEGY COMPATIBILITY (help users pick sets that actually trade)
# ============================================================
# Strategies fall into three behavioural families. Trend + Momentum REINFORCE
# each other (both go long in an uptrend). Mean-reversion FIGHTS them (it buys
# dips / sells rips), so mixing mean-reversion with trend/momentum makes their
# votes cancel to a net-zero signal — and the bot then never enters.
STRATEGY_FAMILY = {
    "supertrend": "trend", "sma_crossover": "trend", "ema_cloud": "trend", "pa_breakout": "trend",
    "macd_crossover": "momentum", "atr_breakout": "momentum", "vol_profile": "momentum",
    "rsi_threshold": "mean_reversion", "bb_fade": "mean_reversion", "stoch": "mean_reversion",
}

# Curated sets whose members vote coherently.
RECOMMENDED_STRATEGY_SETS = {
    "trend_rider":     ["supertrend", "ema_cloud", "sma_crossover", "pa_breakout"],
    "trend_momentum":  ["supertrend", "ema_cloud", "macd_crossover", "atr_breakout"],
    "range_reversion": ["rsi_threshold", "bb_fade", "stoch"],
}


def analyze_strategy_mix(strategies):
    """Classify a strategy list into families and flag the classic no-trade trap
    (mean-reversion mixed with trend/momentum -> votes cancel to sig=0)."""
    codes = [s.get('code') for s in (strategies or []) if isinstance(s, dict) and s.get('code')]
    fams: Dict[str, List[str]] = {}
    for c in codes:
        fams.setdefault(STRATEGY_FAMILY.get(c, 'other'), []).append(c)
    has_directional = bool(fams.get('trend') or fams.get('momentum'))
    has_meanrev     = bool(fams.get('mean_reversion'))
    conflict = has_directional and has_meanrev
    if not codes:
        suggestion = "No strategies selected."
    elif conflict:
        suggestion = (
            f"Mean-reversion ({', '.join(fams['mean_reversion'])}) fights your "
            f"trend/momentum strategies and cancels the signal (sig=0 → no entries). "
            f"Either run the mean-reversion set alone for ranging markets, or drop it "
            f"and keep a coherent trend set, e.g. {', '.join(RECOMMENDED_STRATEGY_SETS['trend_rider'])}."
        )
    else:
        suggestion = "Coherent set — these vote in the same direction, so signals won't cancel."
    return {"coherent": (not conflict) and bool(codes), "families": fams,
            "conflict": conflict, "suggestion": suggestion}


@app.get("/api/strategies/compatibility")
async def strategy_compatibility(codes: Optional[str] = None):
    """Strategy compatibility guide for the UI. Returns each strategy's family and
    curated coherent sets; if `codes` (comma-separated) is passed, also analyzes
    that specific selection so the UI can warn before the user starts a bot."""
    families: Dict[str, List[str]] = {}
    for code, fam in STRATEGY_FAMILY.items():
        families.setdefault(fam, []).append(code)
    result = {
        "families": families,
        "recommended_sets": RECOMMENDED_STRATEGY_SETS,
        "rule": "Trend + Momentum reinforce each other. Mean-reversion fights them — run it alone in ranging markets.",
    }
    if codes:
        result["analysis"] = analyze_strategy_mix([{"code": c.strip()} for c in codes.split(",") if c.strip()])
    return result


@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest):
    user_id    = data.userId.strip()
    raw_cap    = (data.config.get("capitalAllocation") or data.config.get("capital_allocation")
                  or data.config.get("initialBalance"))
    ui_capital = float(raw_cap) if raw_cap else 200.0

    if user_id in TASK_REGISTRY:
        try: TASK_REGISTRY[user_id].cancel(); logger.info(f"♻️ Registry: Cleaned old loop for {user_id}")
        except Exception as e: logger.error(f"⚠️ Registry Cleanup Error: {e}")

    # BUGFIX #7: guard required 'symbol' so a missing key returns a clean 422
    # instead of an unhandled KeyError -> HTTP 500.
    _symbol = data.config.get('symbol')
    if not _symbol:
        raise HTTPException(status_code=422, detail="config.symbol is required")

    initial_ohlcv = await fetch_live_candles_ccxt(
        _symbol, data.config.get('timeframe', '1h'), 350)
    processed_candles = []
    if initial_ohlcv:
        df_init = pd.DataFrame(initial_ohlcv)
        processed_candles = await process_data_packet(df_init, data.config.get('strategies', []))

    saved_state = DatabaseHandler.load_state(user_id)
    resume = bool(data.config.get('resumeOpenPositions', True))
    if saved_state and resume and saved_state.get('positions'):
        # RELIABILITY: recover open positions across a restart instead of
        # orphaning them. The engine resumes managing their TP/SL/TSL; balance
        # and history carry over. Use /api/bot/reset for a clean slate.
        ACTIVE_BOTS[user_id] = saved_state
        _recovered = saved_state.get('positions', [])
        _bal = float(saved_state.get('balance', ui_capital))
        ACTIVE_BOTS[user_id].update({
            "status": "running", "config": data.config, "balance": _bal,
            "startedAt": datetime.now(timezone.utc).isoformat(), "logs": [],
        })
        ACTIVE_BOTS[user_id].setdefault(
            "equityCurve",
            [{"time": datetime.now().isoformat(), "balance": _bal, "confidence": 50}])
        await emit_log(user_id, f"♻️ RECOVERED {len(_recovered)} open position(s); balance ${_bal:,.2f} — resuming management.")
    elif saved_state:
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id].update({
            "status": "running", "config": data.config, "balance": ui_capital,
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "trade_history": [], "logs": [], "positions": [],
            "equityCurve": [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}],
        })
        await emit_log(user_id, f"♻️ SESSION INITIALIZED: Fresh slate at ${ui_capital}")
    else:
        ACTIVE_BOTS[user_id] = {
            "status": "running", "config": data.config, "balance": ui_capital,
            "positions": [], "trade_history": [], "logs": [],
            "equityCurve": [{"time": datetime.now().isoformat(), "balance": ui_capital, "confidence": 50}],
            "startedAt": datetime.now(timezone.utc).isoformat(),
        }
        await emit_log(user_id, f"🚀 Engine Started. Portfolio: ${ui_capital}")

    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    # FIX #24: include positions / trades / signals so panels don't briefly blank
    # on (re)connect — especially after recovering open positions.
    await emit_status(user_id, {
        "status":     "running",
        "currentBalance": ACTIVE_BOTS[user_id]["balance"],
        "candles":    processed_candles,
        "startedAt":  ACTIVE_BOTS[user_id]["startedAt"],
        "equityCurve": ACTIVE_BOTS[user_id]["equityCurve"],
        "initialCapital": ui_capital,
        "activePositions": ACTIVE_BOTS[user_id].get("positions", []),
        "positions":       ACTIVE_BOTS[user_id].get("positions", []),
        "tradeHistory":    ACTIVE_BOTS[user_id].get("trade_history", [])[-200:],
        "tradeMarkers":    ACTIVE_BOTS[user_id].get("trade_history", [])[-200:],
        "signalsMap":      ACTIVE_BOTS[user_id].get("signalsMap", {}),
    })
    # 💡 STRATEGY COMPATIBILITY CHECK: warn immediately if the selected mix will
    # cancel its own votes (the classic 'bot never enters' trap) so the user can
    # fix it before waiting for trades that never come.
    _mix = analyze_strategy_mix(data.config.get('strategies', []))
    if _mix['conflict']:
        await emit_log(user_id, f"💡 STRATEGY TIP: {_mix['suggestion']}")

    loop = asyncio.get_event_loop()
    task = loop.create_task(live_neural_heartbeat(user_id))
    TASK_REGISTRY[user_id] = task
    return {"status": "running"}


@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    # FIX #2: strip the userId to match start/status/reset. A userId sent with
    # stray whitespace otherwise targeted a different key than the running bot,
    # so stop was a no-op and the live bot could not be stopped.
    user_id = data.userId.strip()
    if user_id in TASK_REGISTRY:
        TASK_REGISTRY[user_id].cancel()
        del TASK_REGISTRY[user_id]
        logger.info(f"💀 Registry: Task killed for {user_id}")
    if user_id in ACTIVE_BOTS:
        ACTIVE_BOTS[user_id].update({
            "status": "stopped", "positions": [], "trade_history": [], "equityCurve": [], "logs": []})
        await emit_status(user_id, {
            "status": "stopped", "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [], "tradeMarkers": [], "equityCurve": [], "startedAt": None})
        DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
        del ACTIVE_BOTS[user_id]
        await emit_log(user_id, "💀 SYSTEM PURGED: Engine stopped and session reset.")
        return {"status": "stopped", "message": "Bot killed and reset"}
    return {"status": "stopped"}


@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    # FIX #2: strip the userId to match start/status/reset (see stop_bot).
    _uid = data.userId.strip()
    if _uid in ACTIVE_BOTS and ACTIVE_BOTS[_uid]['positions']:
        pos = ACTIVE_BOTS[_uid]['positions'].pop(0)
        await emit_log(_uid, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
        DatabaseHandler.save_state(_uid, ACTIVE_BOTS[_uid])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")


@app.post('/api/backtest/run')
async def run_backtest(request: BacktestRequest):
    try:
        config = request.dict()
        final_thresh = config.get('mlThresholdLong') or config.get('ml_confidence_threshold') or 0.8
        config['mlThresholdLong']  = final_thresh
        config['mlThresholdShort'] = config.get('mlThresholdShort') or final_thresh
        config['risk_percentage']  = config.get('riskPercentage', 1.0)
        config['strategies']       = [{"code": request.code, "params": request.params}]
        from app.backtest2 import Backtester
        tester = Backtester(config)
        result = await tester.run()
        return json.loads(json.dumps(result, default=str))
    except Exception as e:
        logger.error(f"❌ Backtest Run Error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post('/api/backtest/combo')
async def run_combo_backtest(req: ComboRequest):
    async def event_generator():
        try:
            config = req.dict()
            config['mlThresholdLong']  = config.get('mlThresholdLong', 0.8)
            config['mlThresholdShort'] = config.get('mlThresholdShort', 0.8)
            config['risk_percentage']  = config.get('risk_percentage', 1.0)
            yield f"{json.dumps({'status': 'progress', 'percentage': 10, 'message': 'Assembling AI Council...'}, default=str)}\n"
            logger.info(f"⚖️ COMBO RUN START: {len(config.get('strategies', []))} Strategies")
            from app.backtest2 import Backtester
            tester = Backtester(config)
            yield f"{json.dumps({'status': 'progress', 'percentage': 30, 'message': 'Fetching Market History...'}, default=str)}\n"
            result = await tester.run()
            yield f"{json.dumps({'status': 'progress', 'percentage': 90, 'message': 'Finalizing Analytics...'}, default=str)}\n"
            yield f"{json.dumps({'status': 'success', 'result': result}, default=str)}\n"
        except Exception as e:
            logger.error(f"❌ Combo Stream Error: {e}")
            yield f"{json.dumps({'status': 'error', 'message': str(e)}, default=str)}\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"})


# ============================================================
# 🔬 WALK-FORWARD EVALUATION (out-of-sample edge check, FIX #5)
# ============================================================
class WalkForwardRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    startDate: str
    endDate: str
    initialBalance: float = 1000.0
    riskPercentage: float = 1.0
    strategies: List[StrategyConfig] = []
    code: Optional[str] = None          # single-strategy convenience
    combinationRule: str = "OR"
    mlModel: str = "off"
    mlThresholdLong: float = 0.6
    mlThresholdShort: float = 0.6
    trade_direction: str = "BOTH"
    folds: int = 4
    params: Optional[Dict[str, Any]] = {}

WalkForwardRequest.model_rebuild()


@app.post("/api/backtest/walkforward")
async def walkforward(req: WalkForwardRequest):
    """Split [startDate, endDate] into N consecutive OUT-OF-SAMPLE folds, run the
    backtest on each, and report consistency. A strategy with real edge is
    profitable across most folds — not just on one cherry-picked window."""
    try:
        from app.backtest2 import Backtester
        start = pd.to_datetime(req.startDate); end = pd.to_datetime(req.endDate)
        folds = max(1, min(12, int(req.folds)))
        edges = pd.date_range(start, end, periods=folds + 1)
        strategies = ([s.dict() for s in req.strategies] if req.strategies
                      else ([{"code": req.code, "params": req.params or {}}] if req.code else []))
        if not strategies:
            return JSONResponse(status_code=422, content={"status": "failed", "error": "provide strategies[] or code"})

        detail = []
        for k in range(folds):
            cfg = {
                "symbol": req.symbol, "timeframe": req.timeframe,
                "startDate": str(edges[k].date()), "endDate": str(edges[k + 1].date()),
                "initialBalance": req.initialBalance,
                "risk_percentage": req.riskPercentage, "riskPercentage": req.riskPercentage,
                "strategies": strategies, "combinationRule": req.combinationRule,
                "mlModel": req.mlModel, "mlThresholdLong": req.mlThresholdLong,
                "mlThresholdShort": req.mlThresholdShort, "trade_direction": req.trade_direction,
                "params": {**(req.params or {}), "hybridMode": req.combinationRule},
            }
            try:
                res = await Backtester(cfg).run()
                m = res.get("metrics", {}) if isinstance(res, dict) else {}
            except Exception as e:
                m = {"error": str(e)}
            detail.append({"fold": k + 1, "start": cfg["startDate"], "end": cfg["endDate"], **m})

        rois = [d["roi"] for d in detail if isinstance(d.get("roi"), (int, float))]
        agg = {
            "folds": len(detail),
            "mean_roi": round(sum(rois) / len(rois), 2) if rois else None,
            "median_roi": round(sorted(rois)[len(rois) // 2], 2) if rois else None,
            "worst_fold_roi": round(min(rois), 2) if rois else None,
            "best_fold_roi": round(max(rois), 2) if rois else None,
            "pct_profitable_folds": round(100 * sum(1 for x in rois if x > 0) / len(rois), 1) if rois else None,
            "total_trades": sum(int(d.get("total_trades", 0) or 0) for d in detail),
            "mean_win_rate": round(sum(float(d.get("win_rate", 0) or 0) for d in detail) / len(detail), 1) if detail else None,
        }
        return {"status": "success", "aggregate": agg, "folds_detail": detail}
    except Exception as e:
        logger.error(f"❌ Walk-forward error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


# ============================================================
# 🔎 AUTONOMOUS EDGE DISCOVERY
# ============================================================
@app.get("/api/discover/edges")
async def get_discovered_edges():
    """Latest autonomous edge-search results (walk-forward grid). Read-only."""
    try:
        from app.config2 import DATA_DIR as _DD
        p = os.path.join(_DD, "discovered_edges.json")
        if not os.path.exists(p):
            return {"status": "empty", "message": "No discovery run yet.", "edges": [], "top": []}
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/discover/run")
async def run_discovery(symbols: Optional[str] = None, timeframes: Optional[str] = None):
    """Trigger an edge-search sweep now (runs off the event loop). Optional CSV
    overrides, e.g. ?symbols=BTC-USD,ETH-USD&timeframes=4h,1d."""
    try:
        from edge_discovery import discover
        syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
        tfs  = [t.strip() for t in timeframes.split(",") if t.strip()] if timeframes else None
        return await asyncio.to_thread(discover, syms, tfs, 20, True)
    except Exception as e:
        logger.error(f"❌ Discovery error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/discover/ensure-data")
async def ensure_data(symbols: Optional[str] = None, years_1h: int = 3, years_1d: int = 6):
    """Fetch enough history per symbol: top up 1h + fetch long TRUE-daily. Runs off
    the event loop (first fetch can take a few minutes)."""
    try:
        from edge_discovery import ensure_history
        syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
        rep = await asyncio.to_thread(ensure_history, syms, int(years_1h), int(years_1d))
        return {"status": "success", "rows": rep}
    except Exception as e:
        logger.error(f"❌ ensure-data error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/discover/validate")
async def validate_candidate(symbol: str = "SOL-USD", timeframe: str = "1d", folds: int = 8):
    """Deep param-robustness walk-forward of a candidate (default SOL 1d MACD)."""
    try:
        from edge_discovery import validate_config
        return await asyncio.to_thread(validate_config, symbol, timeframe, int(folds))
    except Exception as e:
        logger.error(f"❌ validate error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


# ============================================================
# 🧪 EXIT / RISK / SIZING LABORATORY
# We optimized entries across hundreds of combos with ONE fixed exit. This tests
# the other ~80% (exits + sizing), ranked by CROSS-COIN generalization so a
# per-coin overfit can't masquerade as edge.
# ============================================================
@app.get("/api/exitlab/results")
async def get_exitlab_results():
    """Latest exit-lab sweep results. Read-only."""
    try:
        from app.config2 import DATA_DIR as _DD
        p = os.path.join(_DD, "exit_lab_results.json")
        if not os.path.exists(p):
            return {"status": "empty", "message": "No exit-lab run yet.", "winners": [], "top": []}
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/exitlab/run")
async def run_exitlab(symbols: Optional[str] = None, timeframes: Optional[str] = None,
                      entries: Optional[str] = None, styles: Optional[str] = None,
                      directions: Optional[str] = None):
    """Sweep exit/risk/sizing styles over a fixed entry, walk-forward per coin, and
    rank by how many coins each config generalizes to. Runs off the event loop.
    Optional CSV overrides, e.g. ?timeframes=1d&entries=trend&styles=be_runner."""
    try:
        from exit_lab import optimize_exits
        def _csv(v): return [s.strip() for s in v.split(",") if s.strip()] if v else None
        return await asyncio.to_thread(
            optimize_exits, _csv(symbols), _csv(timeframes), _csv(entries),
            _csv(styles), _csv(directions), 25, True)
    except Exception as e:
        logger.error(f"❌ exit-lab error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/exitlab/portfolio")
async def run_exitlab_portfolio(timeframe: str = "1d", entry: str = "trend",
                                direction: str = "LONG", style: str = "be_runner",
                                symbols: Optional[str] = None):
    """Run ONE exit config across many coins as a combined portfolio and report the
    blended result — the diversification avenue (many marginal streams can sum to a
    smoother positive whole). Runs off the event loop."""
    try:
        from exit_lab import portfolio_test
        syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
        return await asyncio.to_thread(portfolio_test, syms, timeframe, entry, direction, style)
    except Exception as e:
        logger.error(f"❌ exit-lab portfolio error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/exitlab/validate")
async def run_exitlab_validate(timeframe: str = "4h", entry: str = "trend",
                               direction: str = "LONG", style: str = "trend_ride",
                               holdout_frac: float = 0.25, fee_mult: float = 2.0,
                               slip_mult: float = 3.0, symbol: Optional[str] = None,
                               max_legs: int = 1, add_atr: float = 1.0):
    """Hard-validate ONE exit config: rerun it on ONLY the untouched final slice of
    history (one config, no 96-way selection) and again under fee/slippage stress,
    per coin. max_legs>1 validates the pyramiding variant. Verdict ROBUST only if it
    stays positive on >=3 coins in BOTH."""
    try:
        from exit_lab import validate_exit
        return await asyncio.to_thread(
            validate_exit, symbol, timeframe, entry, direction, style,
            float(holdout_frac), float(fee_mult), float(slip_mult),
            int(max_legs), float(add_atr))
    except Exception as e:
        logger.error(f"❌ exit-lab validate error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


# =============================================================
# STRATEGY LAB — on-demand single-config backtest for the UI.
# Reuses the VALIDATED exit_lab simulator, so what a user backtests here is
# exactly what the live fleet trades (regime entry + trend_ride fixed-stop exit).
# Read-only: never touches ACTIVE_BOTS or the ledger.
# =============================================================
@app.get("/api/strategylab/options")
async def strategylab_options():
    """Choices for the Strategy Lab UI (coins, timeframes, entries, exit styles,
    and the live fleet's default config)."""
    try:
        from exit_lab import lab_options
        return lab_options()
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/strategylab/run")
async def strategylab_run(symbol: str = "BTC-USD", timeframe: str = "4h",
                          entry: str = "regime", direction: str = "LONG",
                          style: str = "trend_ride", risk_pct: float = 1.0,
                          start: Optional[str] = None, end: Optional[str] = None,
                          initial_balance: float = 1000.0,
                          max_legs: int = 1, add_atr: float = 1.0):
    """One-config backtest using exit_lab (the validated money-math behind the
    fleet). max_legs>1 enables pyramiding (adds legs only in a confirmed trend,
    never in chop). Returns metrics + equity curve + per-trade log + a buy&hold
    benchmark. Read-only; runs off the event loop."""
    try:
        from exit_lab import run_single
        return await asyncio.to_thread(
            run_single, symbol, timeframe, entry, direction, style,
            float(risk_pct), start, end, float(initial_balance),
            int(max_legs), float(add_atr))
    except Exception as e:
        logger.error(f"❌ strategy-lab error: {e}")
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.get("/api/fleet/eligibility")
async def fleet_eligibility():
    """Which coins are CLEARED for live pyramiding, per validated config — the
    bridge from the Strategy Lab's hard validation to the live fleet. A coin
    appears here only if it survived BOTH the out-of-sample holdout and the
    cost-stress test. Read-only."""
    try:
        from exit_lab import all_eligibility
        return {"registry": all_eligibility()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})


@app.post("/api/fleet/recalibrate")
async def fleet_recalibrate(level: Optional[str] = None, max_legs: Optional[int] = None):
    """Harden the system NOW: re-run the hard out-of-sample + cost-stress validation
    that gates live pyramiding, at a chosen strictness (normal|strict|paranoid), and
    rewrite the eligibility registry. Runs in the background so this returns
    immediately; poll GET /api/fleet/recalibration for progress + the verdict."""
    if RECALIB_STATE.get("running"):
        return {"status": "already_running", "last": RECALIB_STATE.get("last")}
    lvl = level if level in RECALIB_LEVELS else RECALIB_LEVEL
    RECALIB_STATE["running"] = True
    asyncio.create_task(_run_recalibration(lvl, max_legs))
    return {"status": "started", "level": lvl}


@app.get("/api/fleet/recalibration")
async def fleet_recalibration_status():
    """Last recalibration result (verdict + cleared coins per pyramiding depth) and
    whether one is running now. Falls back to the persisted file after a restart."""
    last = RECALIB_STATE.get("last")
    if last is None:
        try:
            from app.config2 import DATA_DIR as _DD
            _p = os.path.join(_DD, "recalibration_status.json")
            if os.path.exists(_p):
                with open(_p) as _f:
                    last = json.load(_f)
        except Exception:
            last = None
    return {
        "running": bool(RECALIB_STATE.get("running")),
        "last": last,
        "levels": list(RECALIB_LEVELS.keys()),
        "enabled": RECALIB_ENABLED,
        "auto_hours": RECALIB_HOURS,
        "auto_level": RECALIB_LEVEL,
    }


@app.post("/api/fleet/research")
async def fleet_research():
    """Run the cross-coin research sweep NOW (find the best-performing exit/sizing
    configs). Background; poll GET /api/fleet/research for progress + the ranking."""
    if RESEARCH_STATE.get("running"):
        return {"status": "already_running"}
    RESEARCH_STATE["running"] = True
    asyncio.create_task(_run_research())
    return {"status": "started"}


@app.get("/api/fleet/research")
async def fleet_research_status():
    """Latest research ranking (top generalizing configs) + whether a sweep is
    running now + the automated cadence. Reads data/exit_lab_results.json."""
    results = None
    try:
        from app.config2 import DATA_DIR as _DD
        _p = os.path.join(_DD, "exit_lab_results.json")
        if os.path.exists(_p):
            with open(_p) as _f:
                _data = json.load(_f)
            results = {
                "generated": _data.get("generated"),
                "configs_tested": _data.get("configs_tested"),
                "generalizing_configs": _data.get("generalizing_configs"),
                "top": (_data.get("top") or [])[:10],
                "winners": (_data.get("winners") or [])[:10],
            }
    except Exception:
        results = None
    return {
        "running": bool(RESEARCH_STATE.get("running")),
        "enabled": RESEARCH_ENABLED,
        "auto_hours": RESEARCH_HOURS,
        "results": results,
    }


@app.get("/api/fleet/learning")
async def fleet_learning():
    """Self-learning status: the ledger models trained from the fleet's OWN closed
    trades (entry features -> win/loss), and each coin's progress toward the
    training threshold. The gate activates per coin/tf once it has enough trades.
    Read-only."""
    import glob
    try:
        from ledger_trainer import MIN_SAMPLES
    except Exception:
        MIN_SAMPLES = 60
    models = []
    try:
        for p in glob.glob(os.path.join(MODEL_DIR, "*_ledger_model.joblib")):
            try:
                payload = joblib.load(p)
                models.append({
                    "model": os.path.basename(p).replace("_ledger_model.joblib", ""),
                    "n_samples": int(payload.get("n_samples", 0)),
                    "win_rate": round(float(payload.get("win_rate", 0)) * 100, 1),
                })
            except Exception:
                pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})
    progress = []
    for sym in LEDGER_FLEET_SYMBOLS:
        try:
            st = trade_recorder.stats(sym)
            progress.append({"symbol": sym, "trades": int(st.get("trades", 0)), "needed": int(MIN_SAMPLES)})
        except Exception:
            pass
    return {"enabled": SELF_LEARN_ENABLED, "min_trades": int(MIN_SAMPLES),
            "refresh_hours": SELF_LEARN_HOURS, "models": models,
            "fleet_symbols": LEDGER_FLEET_SYMBOLS, "fleet_tfs": LEDGER_FLEET_TFS,
            "progress": progress}


@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot: bot = DatabaseHandler.load_state(userId.strip())
    if bot:
        return {
            "status":            bot.get("status", "stopped"),
            "currentBalance":    bot.get("currentBalance", bot.get("balance", 0)),
            "balance":           bot.get("balance", 0),
            "unrealizedPnl":     bot.get("unrealizedPnl", 0),
            "exposure":          bot.get("exposure", 0),
            "currentConfidence": bot.get("currentConfidence", 50),
            "signalsMap":        bot.get("signalsMap", {}),
            "equityCurve":       bot.get("equityCurve", []),
            "logs":              bot.get("logs", []),
            "activePositions":   bot.get("positions", []),
            "positions":         bot.get("positions", []),
            "startedAt":         bot.get("startedAt"),
            "config":            bot.get("config"),
            "candles":           bot.get("candles", []),
            "trade_history":     bot.get('trade_history', []),
            "tradeHistory":      bot.get('trade_history', []),
            "tradeMarkers":      bot.get('trade_history', []),
            "dailyProfit":       bot.get("dailyProfit", 0),
            "initialCapital":    (bot.get("config", {}).get("capitalAllocation") or
                                  bot.get("config", {}).get("initialBalance") or
                                  bot.get("balance", 0)),
            "aiRegimeTitle":     bot.get("aiRegimeTitle",  "Mean-Reverting Consolidation"),
            "aiRegimeDesc":      bot.get("aiRegimeDesc",   "Sideways Range"),
            "aiDeployedGear":    bot.get("aiDeployedGear", "Syncing Core Strategy Modules..."),
            "winRate":           bot.get("winRate",        0.0),
            "profitFactor":      bot.get("profitFactor",   1.0),
            "lastTradeScore":    bot.get("lastTradeScore"),
        }
    return {"status": "inactive", "balance": 0}


@app.post("/api/bot/reset")
async def reset_bot(data: BotStopRequest):
    _uid = data.userId.strip()
    # BUGFIX #8: cancel the heartbeat first so an in-flight tick can't re-save the
    # state we're about to clear (stop/tick race), and don't leak the task.
    if _uid in TASK_REGISTRY:
        try: TASK_REGISTRY[_uid].cancel()
        except Exception: pass
        del TASK_REGISTRY[_uid]
    if _uid in ACTIVE_BOTS:
        ACTIVE_BOTS[_uid].update({
            "status": "stopped", "positions": [], "trade_history": [], "equityCurve": [], "logs": []})
        DatabaseHandler.save_state(_uid, ACTIVE_BOTS[_uid])
        return {"status": "reset"}
    return {"status": "not_found"}


# =============================================================
# 🚢 FLEET ORCHESTRATION — run many validated bots in ONE session
# Each child runs as its own independent heartbeat (same validated code path),
# keyed by a composite id "<userId>::<symbol>::<suffix>". Purely additive: the
# single-bot endpoints are untouched. Automatic long/short SELECTION is emergent
# — the per-bot trend-alignment gate only lets a long enter in an up-trend and a
# short enter in a down-trend, so running both per coin means the market routes
# each stream to the side that currently has the edge.
# =============================================================
class FleetStartRequest(BaseModel):
    userId: str
    symbols: Optional[List[str]] = None
    capitalEach: Optional[float] = 1000.0
    longTimeframe: Optional[str] = "4h"
    shortTimeframe: Optional[str] = "1d"
    fleetMaxDrawdownPct: Optional[float] = 20.0   # portfolio-level drawdown halt
    sizeByConviction: Optional[bool] = False      # opt-in conviction position sizing
    longOnly: Optional[bool] = False              # True = SPOT mode (long legs only, no shorts/margin)
    riskPct: Optional[float] = 1.0                # % of each bot's capital risked per trade (to the ATR stop)
    maxLegs: Optional[int] = 1                    # pyramiding: max concurrent legs per LONG bot (only applied to validation-cleared coins)
    bots: Optional[List[Dict[str, Any]]] = None   # explicit override: [{suffix, config}]

FleetStartRequest.model_rebuild()


def _validated_fleet_specs(symbols, cap, long_tf="4h", short_tf="1d", include_shorts=True, risk_pct=1.0,
                           max_legs=1, cleared_coins=None):
    """The paper-deploy roster: per coin, the validated LONG stream (regime =
    ema_cloud + btc_regime, trend_ride) + the validated DAILY SHORT stream
    (regime, trend_ride). Longs harvest up-trends; daily shorts harvest risk-off
    legs. Each self-activates only in its favorable trend via the live
    trend-alignment gate.

    NOTE: the LONG entry is the top-ranked exit_lab config — 4h `regime`
    trend_ride scored mean expectancy 0.596 R / PF 1.92 across 4 of 5 coins,
    vs 0.139 R / PF 1.27 for the old `macd_crossover` momentum entry."""
    specs = []
    _cleared = set(cleared_coins or [])
    for sym in symbols:
        # Pyramiding is allowed ONLY when the user asked for it (max_legs>1) AND
        # this coin survived the hard validation (holdout + cost stress). Every
        # other coin — and every SHORT leg — stays at the validated single leg.
        # The live loop splits risk across legs (riskPercentage / maxPyramiding),
        # so total per-trade risk is unchanged, and only adds in a confirmed,
        # profitable climb (never in chop).
        _mp = int(max_legs) if (int(max_legs) > 1 and sym in _cleared) else 1
        specs.append({"suffix": "long", "config": {
            "symbol": sym, "timeframe": long_tf, "strategies": [{"code": "ema_cloud"}, {"code": "btc_regime"}],
            "comboConfig": {"combinationRule": "OR"}, "trade_direction": "LONG",
            "exitMode": "trend_ride", "atrSlMultiplier": 2.0, "mlMode": "off",
            "macroRegimeGate": True, "maxPyramiding": _mp,
            # Self-learning gate: once this coin has enough closed trades, the bot
            # trains a win/loss model on its OWN trades and skips setups it has
            # learned tend to lose (P(win) below the floor). No-op until trained.
            "useLedgerGate": True, "ledgerMinPwin": 0.30,
            # Safety stack, set explicitly (= the engine's active defaults) so the
            # fleet's guards are intentional and locked against a future default
            # change: ADX trend filter, volume floor, ATR volatility band,
            # price/200-EMA alignment, a 5%/day per-bot loss breaker, and a daily
            # trade-count runaway guard (the one previously effectively off).
            "minAdx": 20.0, "minVolRatio": 0.2, "minAtrPct": 0.1, "maxAtrPct": 5.0,
            "requireTrendAlignment": True, "maxDailyLoss": 5.0, "maxTradesPerDay": 8,
            "riskPercentage": float(risk_pct), "initialBalance": cap, "capitalAllocation": cap,
            "enablePartialExit": False}})
        # SPOT mode (longOnly) skips the margin SHORT legs entirely — long side is
        # the validated edge; shorts are an opt-in margin hedge. Shorts stay single
        # leg (the pyramiding validation was run on the LONG side).
        if include_shorts:
            specs.append({"suffix": "short", "config": {
                "symbol": sym, "timeframe": short_tf, "strategies": [{"code": "ema_cloud"}, {"code": "btc_regime"}],
                "comboConfig": {"combinationRule": "OR"}, "trade_direction": "SHORT",
                "enable_shorting": True, "leverage": 1, "exitMode": "trend_ride",
                "atrSlMultiplier": 2.0, "mlMode": "off", "macroRegimeGate": True, "maxPyramiding": 1,
                "useLedgerGate": True, "ledgerMinPwin": 0.30,
                "minAdx": 20.0, "minVolRatio": 0.2, "minAtrPct": 0.1, "maxAtrPct": 5.0,
                "requireTrendAlignment": True, "maxDailyLoss": 5.0, "maxTradesPerDay": 8,
                "riskPercentage": float(risk_pct), "initialBalance": cap, "capitalAllocation": cap,
                "enablePartialExit": False}})
    return specs


# Opt-in auth for fleet MUTATIONS (start/stop). Stays OPEN until ENGINE_API_KEY is
# set on the engine; the Node backend already forwards it as X-Internal-Key, so
# setting the SAME secret on both sides locks fleet control without breaking the
# UI mid-flight. Read-only status endpoints stay open.
FLEET_API_KEY = os.getenv("ENGINE_API_KEY", "")
def _require_fleet_key(request: Request):
    if FLEET_API_KEY and request.headers.get("x-internal-key") != FLEET_API_KEY:
        raise HTTPException(status_code=401, detail="Fleet control requires a valid internal key.")


@app.post("/api/fleet/start")
async def fleet_start(req: FleetStartRequest, request: Request):
    """Launch a roster of validated bots under one session. Default roster =
    LONG + DAILY-SHORT per coin across the given symbols. Reuses the exact
    single-bot start path per child, so nothing in the validated loop changes."""
    _require_fleet_key(request)
    base = req.userId.strip()
    # Reset this fleet's portfolio-drawdown peak so a fresh / re-ignited fleet
    # (especially a SMALLER one, e.g. a lower-budget tier) isn't instantly halted
    # by a stale peak left over from a larger previous session.
    FLEET_STATE.pop(base, None)
    symbols = req.symbols or ["BTC-USD", "ETH-USD", "SOL-USD"]
    cap = float(req.capitalEach or 1000.0)
    # Pyramiding gate: if the user enabled it (maxLegs>1), only coins CLEARED by
    # the Strategy Lab's hard validation may pyramid; everything else stays single.
    _legs = int(req.maxLegs or 1)
    _cleared = []
    if _legs > 1:
        try:
            from exit_lab import load_eligibility
            _cleared = load_eligibility(timeframe=(req.longTimeframe or "4h"), entry="regime",
                                        direction="LONG", style="trend_ride", max_legs=_legs)
            logger.info("⚙️ FLEET pyramiding x%d requested — validation-cleared coins: %s", _legs, _cleared)
        except Exception as e:
            logger.warning(f"fleet eligibility load failed (pyramiding disabled): {e}")
            _cleared = []
    specs = req.bots or _validated_fleet_specs(symbols, cap, req.longTimeframe, req.shortTimeframe,
                                               include_shorts=not bool(req.longOnly),
                                               risk_pct=float(req.riskPct or 1.0),
                                               max_legs=_legs, cleared_coins=_cleared)
    started, failed = [], []
    for spec in specs:
        cfg = spec.get("config", {})
        sym = cfg.get("symbol", "UNK")
        child = f"{base}::{sym}::{spec.get('suffix', 'bot')}"
        # Bind every child to the fleet so the portfolio-level risk cap applies,
        # and pass through the opt-in conviction sizing choice.
        cfg["fleetId"] = base
        cfg.setdefault("fleetMaxDrawdownPct", float(req.fleetMaxDrawdownPct or 20.0))
        if req.sizeByConviction:
            cfg.setdefault("sizeByConviction", True)
        try:
            await start_bot(BotStartRequest(userId=child, config=cfg))
            started.append(child)
        except Exception as e:
            logger.error(f"❌ fleet child {child} failed: {e}")
            failed.append({"id": child, "error": str(e)})
    return {"status": "started", "fleet": base, "count": len(started),
            "bots": started, "failed": failed}


@app.get("/api/fleet/status")
async def fleet_status(userId: str):
    """Aggregate every bot running under this session (all "<userId>::*" keys):
    combined balance, open positions, and a per-bot roster."""
    base = userId.strip()
    prefix = f"{base}::"
    bots = []
    total_eq = 0.0
    total_upnl = 0.0
    total_pos = 0
    for key, b in list(ACTIVE_BOTS.items()):
        if not key.startswith(prefix):
            continue
        bal = float(b.get("balance", 0) or 0)
        # LIVE equity (realized + unrealized), refreshed by the heartbeat every ~10s.
        eq = float(b.get("currentBalance", bal) or bal)
        upnl = float(b.get("unrealizedPnl", 0) or 0)
        total_eq += eq
        total_upnl += upnl
        cfg = b.get("config", {}) or {}
        npos = len(b.get("positions", []) or [])
        total_pos += npos
        bots.append({
            "id": key, "symbol": cfg.get("symbol"),
            "direction": cfg.get("trade_direction", "BOTH"),
            "timeframe": cfg.get("timeframe"), "exitMode": cfg.get("exitMode"),
            "status": b.get("status", "unknown"),
            "balance": round(eq, 2),              # live equity
            "realized": round(bal, 2),
            "unrealized": round(upnl, 2),
            "winRate": b.get("winRate", 0),
            "open_positions": npos,
            "trades": len([t for t in b.get("trade_history", []) if t.get("type") == "exit"]),
            # Readiness for the "closest to trading" meter + ranking. A bot already
            # in a position is trading (100); otherwise use its live gate readiness.
            "readiness": 100 if npos > 0 else int((b.get("thinking") or {}).get("readiness", 0) or 0),
            "blocker": "In trade" if npos > 0 else (b.get("thinking") or {}).get("blocker"),
            "signal": (b.get("thinking") or {}).get("signal"),
        })
    bots.sort(key=lambda x: (x.get("symbol") or "", x.get("direction") or ""))
    return {"fleet": base, "count": len(bots),
            "total_balance": round(total_eq, 2), "total_unrealized": round(total_upnl, 2),
            "open_positions": total_pos, "kill_switch": KILL_SWITCH["on"], "ts": time.time(), "bots": bots}


@app.get("/api/fleet/regime")
async def fleet_regime():
    """Current BTC-driven macro regime (risk_on / risk_off / neutral) that opt-in
    fleet bots tilt with. risk_on holds shorts; risk_off holds longs."""
    r = dict(MACRO_REGIME)
    age = (time.time() - float(r.get("ts", 0))) if r.get("ts") else None
    return {"state": r.get("state", "neutral"), "detail": r.get("detail", {}),
            "age_seconds": (round(age) if age is not None else None),
            "meaning": {"risk_on": "longs favored, shorts held",
                        "risk_off": "shorts favored, longs held",
                        "neutral": "both sides trade on their own per-coin trend"}}


@app.get("/api/fleet/drift")
async def fleet_drift(userId: str):
    """Compare the fleet's LIVE paper stats to the validated trend_ride profile
    (~26-40%% win, profit factor 1.2-1.7, positive expectancy). A large gap =
    the edge may be decaying in the current regime — catch it here, not in the
    balance. Read-only; computed from each bot's closed-trade history."""
    base = userId.strip()
    prefix = f"{base}::"
    baseline = {"win_rate_band": [20.0, 45.0], "min_profit_factor": 1.0,
                "reference": "validated trend_ride: ~26-40% win, PF 1.2-1.7, positive expectancy"}
    per = []
    agg_n = agg_w = 0
    gwin = gloss = net = 0.0
    for key, b in list(ACTIVE_BOTS.items()):
        if not key.startswith(prefix):
            continue
        exits = [t for t in b.get("trade_history", []) if t.get("type") == "exit"]
        pnls = [float(t.get("pnl", 0) or 0) for t in exits]
        n = len(pnls); w = sum(1 for p in pnls if p > 0)
        gw = sum(p for p in pnls if p > 0); gl = abs(sum(p for p in pnls if p <= 0))
        pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
        cfg = b.get("config", {}) or {}
        per.append({"id": key, "symbol": cfg.get("symbol"), "direction": cfg.get("trade_direction"),
                    "trades": n, "win_rate": round(100.0 * w / n, 1) if n else 0.0,
                    "profit_factor": (round(pf, 2) if pf < 99 else pf), "net_pnl": round(sum(pnls), 2)})
        agg_n += n; agg_w += w; gwin += gw; gloss += gl; net += sum(pnls)
    fleet_wr = round(100.0 * agg_w / agg_n, 1) if agg_n else 0.0
    fleet_pf = (gwin / gloss) if gloss > 0 else (99.0 if gwin > 0 else 0.0)
    if agg_n < 10:
        status = "INSUFFICIENT_DATA"
    elif fleet_pf < baseline["min_profit_factor"] or not (baseline["win_rate_band"][0] <= fleet_wr <= baseline["win_rate_band"][1]):
        status = "DRIFTING"
    else:
        status = "ON_TRACK"
    per.sort(key=lambda x: (x.get("symbol") or "", x.get("direction") or ""))
    return {"fleet": base, "status": status, "trades": agg_n, "win_rate": fleet_wr,
            "profit_factor": (round(fleet_pf, 2) if fleet_pf < 99 else fleet_pf),
            "net_pnl": round(net, 2), "baseline": baseline,
            "note": "DRIFTING = live paper stats diverge from the validated profile; "
                    "investigate before adding capital. Needs >=10 closed trades to judge.",
            "per_bot": per}


@app.get("/api/fleet/bot")
async def fleet_bot_detail(userId: str, symbol: str, side: str):
    """Per-coin fleet child detail for the unified Fleet page chart: positions,
    trade markers, live signals, balance, status. Candles are fetched separately
    by the client (market data). Scoped by the composite key, so only the caller's
    own fleet is reachable (the Node proxy injects the verified userId)."""
    suffix = str(side).strip().lower()
    key = f"{str(userId).strip()}::{str(symbol).strip()}::{suffix}"
    bot = ACTIVE_BOTS.get(key)
    if bot is None:
        bot = DatabaseHandler.load_state(key)
    if not bot:
        return {"id": key, "symbol": symbol, "side": suffix, "status": "stopped",
                "balance": 0, "unrealizedPnl": 0, "positions": [], "tradeMarkers": [],
                "signalsMap": {}, "trades": 0}
    cfg = bot.get("config", {}) or {}
    th = bot.get("trade_history", []) or []
    return {
        "id": key, "symbol": symbol, "side": suffix,
        "status": bot.get("status", "stopped"),
        "timeframe": cfg.get("timeframe"),
        "direction": cfg.get("trade_direction"),
        "balance": round(float(bot.get("currentBalance", bot.get("balance", 0)) or 0), 2),  # live equity
        "realized": round(float(bot.get("balance", 0) or 0), 2),
        "unrealizedPnl": round(float(bot.get("unrealizedPnl", 0) or 0), 2),
        "winRate": bot.get("winRate", 0),
        "profitFactor": bot.get("profitFactor", 1.0),
        "regimeTitle": bot.get("aiRegimeTitle"),
        "regimeDesc": bot.get("aiRegimeDesc"),
        "thinking": bot.get("thinking"),
        "positions": bot.get("positions", []),
        "activePositions": bot.get("positions", []),
        "tradeMarkers": th[-200:],
        "tradeHistory": th[-200:],
        "signalsMap": bot.get("signalsMap", {}),
        "currentConfidence": bot.get("currentConfidence", 50),
        "lastTradeScore": bot.get("lastTradeScore"),  # TradeQualityScorer 0-100 composite at last entry
        "trades": len([t for t in th if t.get("type") == "exit"]),
    }


@app.get("/api/fleet/activity")
async def fleet_activity(userId: str, limit: int = 20):
    """Recent CLOSED trades across the whole fleet, newest first — for the live
    trade feed. Aggregated from each child bot's in-memory history."""
    base = str(userId).strip()
    prefix = f"{base}::"
    events = []
    for key, b in list(ACTIVE_BOTS.items()):
        if not key.startswith(prefix):
            continue
        cfg = b.get("config", {}) or {}
        sym = cfg.get("symbol", "?")
        side = cfg.get("trade_direction", "?")
        for t in (b.get("trade_history", []) or []):
            tt = t.get("type")
            if tt in ("entry", "exit"):
                events.append({
                    "id": f"{key}:{t.get('time')}:{tt}:{t.get('price')}",
                    "symbol": sym, "side": side, "action": tt, "type": tt,
                    "entry": t.get("entry"), "price": t.get("price"),
                    "pnl": t.get("pnl"), "reason": t.get("reason"),
                    "time": t.get("time"),
                })
    events.sort(key=lambda e: e.get("time") or 0, reverse=True)
    return {"fleet": base, "count": len(events), "events": events[:int(limit)]}


@app.post("/api/fleet/stop")
async def fleet_stop(req: BotStopRequest, request: Request):
    """Stop every bot in this session (all "<userId>::*" keys)."""
    _require_fleet_key(request)
    base = req.userId.strip()
    prefix = f"{base}::"
    FLEET_STATE.pop(base, None)  # drop the portfolio-drawdown peak when the fleet stops
    keys = set(k for k in list(TASK_REGISTRY.keys()) if k.startswith(prefix))
    keys |= set(k for k in list(ACTIVE_BOTS.keys()) if k.startswith(prefix))
    stopped = []
    for key in keys:
        try:
            await stop_bot(BotStopRequest(userId=key))
            stopped.append(key)
        except Exception as e:
            logger.error(f"❌ fleet stop {key} failed: {e}")
    return {"status": "stopped", "fleet": base, "count": len(stopped), "stopped": sorted(stopped)}


@app.post("/api/fleet/killswitch")
async def fleet_killswitch(request: Request, on: bool = True):
    """Emergency kill switch: when on, every bot stops opening NEW entries (open
    positions keep being managed by their stops/exits). Requires the internal key."""
    _require_fleet_key(request)
    KILL_SWITCH["on"] = bool(on)
    logger.warning("🛑 KILL SWITCH %s", "ENGAGED — all new entries halted" if KILL_SWITCH["on"] else "released")
    return {"kill_switch": KILL_SWITCH["on"]}


@app.post("/api/fleet/risk")
async def fleet_set_risk(request: Request, userId: str, riskPct: float):
    """Update the risk %% per trade on a RUNNING fleet without re-igniting. The
    change is mutated into each child bot's live config in place, so it takes
    effect on the NEXT entry; positions already open keep the size and stop they
    were given at entry (we never resize a live trade). Requires the internal key."""
    _require_fleet_key(request)
    base = str(userId).strip()
    prefix = f"{base}::"
    rp = max(0.1, min(50.0, float(riskPct)))
    updated = []
    for key, b in list(ACTIVE_BOTS.items()):
        if not key.startswith(prefix):
            continue
        cfg = b.setdefault("config", {})
        cfg["riskPercentage"] = rp
        cfg["risk_percentage"] = rp
        try:
            DatabaseHandler.save_state(key, b)
        except Exception as e:
            logger.warning(f"fleet risk save_state {key} failed: {e}")
        updated.append(key)
    logger.info("⚙️ FLEET RISK %s -> %.2f%% on %d bots (new entries only)", base, rp, len(updated))
    return {"fleet": base, "risk_pct": rp, "updated": len(updated),
            "note": "Applies to NEW entries only; open positions keep their original size and stop."}


@app.post("/api/fleet/reset_history")
async def fleet_reset_history(request: Request, userId: str):
    """HARD RESET of all trade-history data behind the charts: wipes each bot's
    in-memory trade log + equity curve (the live feed, chart markers, drift and
    equity chart) AND the persistent trade ledger (the Track Record). Does NOT
    touch open positions or balances — it clears the HISTORY, not the money, so
    the fleet keeps running from a clean slate. Requires the internal key."""
    _require_fleet_key(request)
    base = str(userId).strip()
    prefix = f"{base}::"
    FLEET_STATE.pop(base, None)  # clear the portfolio-drawdown peak on a full reset
    reset = 0
    for key, b in list(ACTIVE_BOTS.items()):
        if not key.startswith(prefix):
            continue
        b["trade_history"] = []
        b["equityCurve"] = []
        b["winRate"] = 0
        b["profitFactor"] = 1.0
        try:
            DatabaseHandler.save_state(key, b)
        except Exception as e:
            logger.warning(f"reset_history save_state {key} failed: {e}")
        reset += 1
    removed = 0
    try:
        removed = trade_recorder.clear(base)  # clears base + all uid::* ledger rows
    except Exception as e:
        logger.warning(f"reset_history ledger clear failed: {e}")
    logger.warning("🧹 FLEET HISTORY RESET %s — %d bots cleared, %d ledger trades removed", base, reset, removed)
    return {"fleet": base, "bots_reset": reset, "ledger_removed": removed,
            "note": "Trade history + ledger cleared. Open positions and balances untouched."}


# =============================================================
# HEALTH CHECK
# =============================================================
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "active_bots": len(ACTIVE_BOTS),
        "running_tasks": len(TASK_REGISTRY),
        "ledger": trade_recorder.stats(),
    }


# =============================================================
# TRADE LEARNING LEDGER — API + DASHBOARD
# =============================================================
@app.get("/api/ledger/stats")
async def ledger_stats(recent: int = 25, user_id: Optional[str] = None):
    """Aggregated view of closed trades. Pass user_id to scope to one bot/user;
    omit for the global view. (The Node backend passes the authenticated user.)"""
    return trade_recorder.summary(recent_limit=recent, user_id=user_id)


@app.post("/api/ledger/clear")
async def ledger_clear(user_id: Optional[str] = None):
    """Clear the trade-learning ledger, scoped to a user when given. DESTRUCTIVE.
    The Node proxy always passes the AUTHENTICATED user, so a user can only ever
    clear their own trades."""
    removed = await asyncio.to_thread(trade_recorder.clear, user_id)
    return {"status": "cleared", "removed": removed, "user_id": user_id}


@app.get("/ledger", response_class=HTMLResponse)
async def ledger_dashboard():
    """Self-contained dashboard for the trade-learning ledger. Served from this
    origin so it can read /api/ledger/stats directly."""
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger_dashboard.html")
        with open(_p, "r", encoding="utf-8") as _f:
            return HTMLResponse(_f.read())
    except Exception as e:
        return HTMLResponse(f"<h1>Ledger dashboard unavailable</h1><pre>{e}</pre>", status_code=500)


@app.get("/fleet", response_class=HTMLResponse)
async def fleet_dashboard():
    """Self-contained fleet control panel: start/stop the validated long+short
    roster, live roster + combined balance, macro-regime badge, and the
    paper-vs-validation drift check. Served from this origin so it reads the
    /api/fleet/* endpoints directly (no Node proxy needed)."""
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_dashboard.html")
        with open(_p, "r", encoding="utf-8") as _f:
            return HTMLResponse(_f.read())
    except Exception as e:
        return HTMLResponse(f"<h1>Fleet dashboard unavailable</h1><pre>{e}</pre>", status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
