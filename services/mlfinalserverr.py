# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v82.1 - "The Complete Sovereign"
# 🛠 FIXES: Restored missing 'ensure_tables_exist', Full Audit Compliance.

import os, json, logging, traceback, math, threading, warnings, joblib, time, sqlite3
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas_ta as ta
from datetime import datetime, timezone

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("System")

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data/optimizer_results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(PROJECT_ROOT, "trading_state.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_TAKER_FEE = 0.0006
SLIPPAGE_BPS = 2.0 
MAX_TOTAL_RISK = 0.30 
GLOBAL_MODEL_CACHE = {}

app = FastAPI(title="Sovereign Executive v82.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}
bots_lock = threading.Lock()

# --- 2. PERSISTENCE LAYER ---
def ensure_tables_exist():
    """Restored: Ensures DB tables exist to prevent 'no such table' errors."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS bots (
            bot_id TEXT PRIMARY KEY, 
            user_id TEXT,
            cash REAL, 
            last_ts INTEGER
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS positions (
            bot_id TEXT, 
            user_id TEXT, 
            qty REAL, 
            entry REAL, 
            time TEXT, 
            peak REAL
        )''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"❌ DB Schema Init Failed: {e}")

# Run immediately on load
ensure_tables_exist()

def execute_db_with_retry(query, params=(), retries=5):
    conn = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(query, params)
            if query.strip().upper().startswith("SELECT"):
                res = cur.fetchall()
                conn.close()
                return res
            else:
                conn.commit()
                conn.close()
                return True
        except sqlite3.OperationalError as e:
            if conn: conn.close()
            # If table missing, try to heal
            if "no such table" in str(e) and attempt == 0:
                ensure_tables_exist()
                continue
            if "locked" in str(e):
                time.sleep(0.1 * (2 ** attempt))
            else:
                logger.error(f"DB Error: {e}")
                break
    return [] if query.strip().upper().startswith("SELECT") else False

# --- 3. DYNAMIC STRATEGY REGISTRY ---
def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        weight = float(strategy_conf.get('weight', 1.0))
        ml_confirm = strategy_conf.get('mlConfirm', True)

        def get_p(key, default): return float(p.get(key, default))

        ml_gate_sig = 0
        if ml_model is not None and ml_confirm:
            try:
                window = df_slice.tail(5).select_dtypes(include=[np.number])
                if hasattr(ml_model, "n_features_in_") and window.shape[1] != ml_model.n_features_in_:
                    ml_gate_sig = 0
                elif len(window) >= 5:
                    p_long = np.mean(ml_model.predict_proba(window)[:, 1])
                    if p_long > ml_thresh: ml_gate_sig = 1
                    elif p_long < (1 - ml_thresh): ml_gate_sig = -1
                    else: ml_gate_sig = 0 
            except: ml_gate_sig = 0

        sig = 0
        if len(df_slice) < 2: return 0
        
        t_row = df_slice.iloc[-1]
        t_prev = df_slice.iloc[-2]
        
        if code == 'sma_crossover':
            if t_row.get('fast_sma') > t_row.get('slow_sma') and t_prev.get('fast_sma') <= t_prev.get('slow_sma'): sig = 1
            elif t_row.get('fast_sma') < t_row.get('slow_sma') and t_prev.get('fast_sma') >= t_prev.get('slow_sma'): sig = -1
        elif code == 'macd_crossover':
            if t_row.get('MACD_12_26_9') > t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') <= t_prev.get('MACDs_12_26_9'): sig = 1
            elif t_row.get('MACD_12_26_9') < t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') >= t_prev.get('MACDs_12_26_9'): sig = -1
        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl_0.02_0.2')) and pd.isna(t_prev.get('PSARl_0.02_0.2')): sig = 1
            elif pd.notna(t_row.get('PSARs_0.02_0.2')) and pd.isna(t_prev.get('PSARs_0.02_0.2')): sig = -1
        elif code == 'rsi_divergence' or code == 'rsi_threshold_reversal':
            rsi = t_row.get('RSI_14', 50)
            if rsi < get_p('oversold_level', 30): sig = 1
            elif rsi > get_p('overbought_level', 70): sig = -1
        elif code == 'bollinger_bands':
            if t_row.get('close') < t_row.get('BBL_20_2.0'): sig = 1
            elif t_row.get('close') > t_row.get('BBU_20_2.0'): sig = -1
        elif code == 'atr_breakout':
            close = t_row.get('close')
            prev_close = t_prev.get('close')
            mult = get_p('atr_multiplier', 3.0)
            ema = t_row.get('EMA_20', close)
            atr = t_row.get('ATR_14', 0)
            if close > (ema + atr*mult) and prev_close <= (ema + atr*mult): sig = 1
            elif close < (ema - atr*mult) and prev_close >= (ema - atr*mult): sig = -1
        elif code == 'stochastic_crossover':
            k, d = t_row.get('STOCHk_14_3_3'), t_row.get('STOCHd_14_3_3')
            pk, pd_val = t_prev.get('STOCHk_14_3_3'), t_prev.get('STOCHd_14_3_3')
            if k > d and pk <= pd_val and k < 20: sig = 1
            elif k < d and pk >= pd_val and k > 80: sig = -1
        elif code == 'cci_oversold':
            cci = t_row.get('CCI_14_0.015')
            if cci < -100: sig = 1
            elif cci > 100: sig = -1
        elif code == 'ichimoku_system':
            tk, kj = t_row.get('ITS_9'), t_row.get('IKS_26')
            ptk, pkj = t_prev.get('ITS_9'), t_prev.get('IKS_26')
            if tk > kj and ptk <= pkj: sig = 1
            elif tk < kj and ptk >= pkj: sig = -1
        elif code == 'obv_signal':
            if t_row.get('OBV') > t_prev.get('OBV'): sig = 1
            elif t_row.get('OBV') < t_prev.get('OBV'): sig = -1

        if ml_gate_sig == 0: return float(sig * weight)
        if sig == ml_gate_sig: return float(sig * weight)
        return 0.0

    except Exception: return 0.0

# --- 4. RISK MANAGER ---
class PrecisionPyramidManager:
    def __init__(self, capital, fee=DEFAULT_TAKER_FEE, max_layers=3, base_risk=0.01, risk_mode="static"):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.fee = fee
        self.max_layers = max_layers
        self.base_risk = base_risk
        self.risk_mode = risk_mode 
        self.positions = []
        self.trades = []
        self.peak = float(capital)
        self.dd_series = [0]
        self.current_equity = float(capital)

    def step_equity(self, price):
        self.current_equity = self.cash + sum(p['qty'] * price for p in self.positions)
        self.peak = max(self.peak, self.current_equity)
        self.dd_series.append((self.peak - self.current_equity) / self.peak if self.peak > 0 else 0)
        return self.current_equity

    def handle(self, signal, price, low, high, time, atr, tsl_mult, slippage_bps=0):
        orders = []
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        
        slippage_rate = slippage_bps / 10000.0
        fill_price = price * (1 + slippage_rate) if signal == 1 else price * (1 - slippage_rate)

        # Risk Cap Check
        current_risk_exposure = len(self.positions) * self.base_risk
        if (current_risk_exposure + self.base_risk) > MAX_TOTAL_RISK:
            signal = 0 

        if signal == 1 and len(self.positions) < self.max_layers:
            dist = (atr * tsl_mult) if atr > 0 else (fill_price * 0.05)
            if dist <= 0: dist = fill_price * 0.05
            qty = (balance_metric * self.base_risk) / dist
            
            cost = qty * fill_price
            entry_fee = cost * self.fee
            
            if self.cash >= (cost + entry_fee):
                self.positions.append({'qty': qty, 'entry': fill_price, 'time': time, 'peak': fill_price})
                self.cash -= (cost + entry_fee)
                orders.append(('BUY', qty))
        
        elif signal == -1 and self.positions:
            q = sum(p['qty'] for p in self.positions)
            self.close_all(fill_price, time, "SIGNAL_REVERSAL")
            orders.append(('SELL', q))
            return orders

        dist = (atr * tsl_mult) if atr > 0 else (price * 0.05)
        remaining = []
        for p in self.positions:
            p['peak'] = max(p['peak'], high) 
            stop_level = p['peak'] - dist
            
            if low <= stop_level:
                stop_fill = stop_level * (1 - slippage_rate)
                self.close_position(p, stop_fill, time, 'TSL_STOP')
                orders.append(('SELL', p['qty']))
            else:
                remaining.append(p)
        self.positions = remaining
        return orders

    def close_all(self, price, time, reason):
        for p in self.positions: self.close_position(p, price, time, reason)
        self.positions = []

    def close_position(self, p, price, time, reason):
        proceeds = p['qty'] * price
        exit_fee = proceeds * self.fee
        entry_fee = p['entry'] * p['qty'] * self.fee
        pnl = proceeds - exit_fee - (p['entry'] * p['qty']) - entry_fee
        self.cash += (proceeds - exit_fee)
        self.trades.append({'entryTime': p['time'], 'exitTime': time, 'profit': pnl, 'type': reason})

# --- 5. LIVE TRADING BOT ENGINE ---
class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.user_id = config.get('userId', 'anon_user')
        
        clean_symbol = self.symbol.replace('/', '-')
        self.bot_id = f"{self.user_id}_{clean_symbol}_{self.tf}"
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.last_processed_ts = 0
        
        # Logger Setup
        self.logger = logging.getLogger(self.bot_id)
        if not self.logger.handlers:
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                log_path = os.path.join(LOG_DIR, f"{self.bot_id}.log")
                handler = logging.FileHandler(log_path)
                handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            except: pass

        self.params = config.get('params', {})
        self.tsl_mult = float(self.params.get('tslAtrMult', 3.5))
        self.min_adx = float(self.params.get('minAdxLevel', 0))
        
        # Calls the function defined above to ensure DB is ready
        ensure_tables_exist()

        exchange_id = config.get('exchange', 'coinbase').lower()
        if not hasattr(ccxt, exchange_id): raise ValueError(f"Exchange '{exchange_id}' not supported.")
        
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = None
        
        try:
            if not config.get('apiKey') and self.is_paper:
                self.exchange = exchange_class({'enableRateLimit': True})
            else:
                self.exchange = exchange_class({
                    'apiKey': config.get('apiKey', ''), 
                    'secret': config.get('secret', ''),
                    'enableRateLimit': True
                })
                self.exchange.check_required_credentials()
                if self.is_paper:
                    try:
                        bal = self.exchange.fetch_balance()
                        quote = self.symbol.split('/')[-1]
                        real_cash = bal['total'].get(quote, 0)
                        if real_cash > 0: config['initialBalance'] = real_cash
                    except: pass

            self.exchange.load_markets()
            if self.symbol not in self.exchange.markets: raise ValueError(f"Symbol {self.symbol} not found.")
        except Exception as e:
            self.logger.error(f"Init Failed: {e}")
            raise e

        self.ml_model = self._load_model(config.get('mlModel'))
        
        risk_pct = float(config.get('riskPercentage', 1)) / 100.0
        risk_mode = config.get('riskManagementMode', 'static')
        self.manager = PrecisionPyramidManager(
            capital=config.get('initialBalance', 1000), 
            base_risk=risk_pct, 
            risk_mode=risk_mode
        )
        self._load_state_and_recover()
        self.is_running = False

    def _load_model(self, name):
        if not name: return None
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        return joblib.load(path) if os.path.exists(path) else None

    def _load_state_and_recover(self):
        rows = execute_db_with_retry("SELECT cash, last_ts FROM bots WHERE bot_id=?", (self.bot_id,))
        if rows:
            self.manager.cash, self.last_processed_ts = rows[0][0], rows[0][1]
            pos_rows = execute_db_with_retry("SELECT qty, entry, time, peak FROM positions WHERE bot_id=?", (self.bot_id,))
            for r in pos_rows:
                self.manager.positions.append({'qty':r[0], 'entry':r[1], 'time':r[2], 'peak':r[3]})
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            self.manager.step_equity(ticker['last'])
        except: pass

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"STATUS | 🚀 Bot Active. User: {self.user_id} | ID: {self.bot_id}")
        
        while self.is_running:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=1000)
                latest_ts = ohlcv[-1][0]
                if latest_ts <= self.last_processed_ts:
                    time.sleep(10); continue

                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df.set_index(pd.to_datetime(df['ts'], unit='ms', utc=True), inplace=True)

                df.ta.sma(length=10, append=True, col_names="fast_sma")
                df.ta.sma(length=50, append=True, col_names="slow_sma")
                df.ta.macd(append=True); df.ta.psar(append=True)
                df.ta.atr(length=14, append=True); df.ta.adx(length=14, append=True)
                df.ta.rsi(length=14, append=True, col_names="RSI_14")
                df.ta.bbands(length=20, std=2, append=True)
                df.ta.stoch(append=True); df.ta.cci(length=14, append=True)
                df.ta.ichimoku(append=True); df.ta.obv(append=True)
                df.ta.ema(length=20, append=True, col_names="EMA_20")

                atr = df.filter(like='ATR').columns[-1]
                adx_col = df.filter(like='ADX').columns[0]
                
                price = df['close'].iloc[-1]
                low = df['low'].iloc[-1]
                high = df['high'].iloc[-1]
                current_adx = df[adx_col].iloc[-1]
                current_rsi = df['RSI_14'].iloc[-1]
                
                # --- VERBOSE DECISION LOGGING ---
                report = []
                report.append(f"--- 📊 MARKET SNAPSHOT [{df.index[-1]}] ---")
                
                trend_status = "Weak/Choppy" if current_adx < 25 else "Strong Trend"
                rsi_status = "Oversold" if current_rsi < 30 else ("Overbought" if current_rsi > 70 else "Neutral")
                report.append(f"• Price: ${price:.2f}")
                report.append(f"• Trend (ADX): {current_adx:.1f} ({trend_status})")
                report.append(f"• Momentum (RSI): {current_rsi:.1f} ({rsi_status})")

                current_exposure = len(self.manager.positions) * self.manager.base_risk * 100
                risk_status = "Safe" if current_exposure < (MAX_TOTAL_RISK * 100) else "MAX CAP HIT"
                report.append(f"• Risk Exposure: {current_exposure:.1f}% / {MAX_TOTAL_RISK*100}% Cap ({risk_status})")

                report.append("--- 🗳️ STRATEGY VOTES ---")
                
                if self.min_adx > 0 and current_adx < self.min_adx:
                    combined_sig = 0
                    report.append(f"⛔ TRADE BLOCKED: ADX {current_adx:.1f} is below minimum {self.min_adx}. Market too choppy.")
                else:
                    strats = self.config.get('strategies', [])
                    votes = []
                    for s in strats:
                        vote = compute_signal(df, s, self.ml_model, 0.5)
                        votes.append(vote)
                        vote_str = "🟢 BUY" if vote > 0 else ("🔴 SELL" if vote < 0 else "⚪ WAIT")
                        report.append(f"   {vote_str} | Strategy: {s.get('code')}")

                    combined_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
                    final_verdict = "🟢 BUY" if combined_sig == 1 else ("🔴 SELL" if combined_sig == -1 else "⚪ HOLD")
                    report.append(f"⚖️  FINAL VERDICT: {final_verdict} (Score: {sum(votes)})")

                orders = self.manager.handle(combined_sig, price, low, high, df.index[-1].isoformat(), df[atr].iloc[-1], self.tsl_mult)
                
                if orders:
                    report.append("--- ⚡ EXECUTION ---")
                    for action, qty in orders:
                        report.append(f"✅ EXECUTING: {action} {qty:.4f} units @ ${price:.2f}")
                        if not self.is_paper:
                            safe_qty = float(self.exchange.amount_to_precision(self.symbol, qty))
                            self.exchange.create_market_order(self.symbol, action.lower(), safe_qty)
                else:
                    report.append("💤 No trade actions taken this cycle.")

                self.manager.step_equity(price)
                self.last_processed_ts = latest_ts
                self._save_state(latest_ts)
                
                for line in report:
                    self.logger.info(f"DECISION | {line}")

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(60)

    def _save_state(self, last_ts):
        execute_db_with_retry("REPLACE INTO bots VALUES (?, ?, ?, ?)", (self.bot_id, self.user_id, self.manager.cash, last_ts))
        execute_db_with_retry("DELETE FROM positions WHERE bot_id=?", (self.bot_id,))
        for p in self.manager.positions:
            execute_db_with_retry("INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?)", (self.bot_id, self.user_id, p['qty'], p['entry'], p['time'], p['peak']))

    def stop(self): self.is_running = False

# --- 6. ENDPOINTS ---

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        user_id = body.get('userId')
        if not user_id: return JSONResponse({"error": "Missing userId"}, 400)
        
        clean_symbol = body['symbol'].replace('/', '-')
        bot_id = f"{user_id}_{clean_symbol}_{body['timeframe']}"
        
        with bots_lock:
            if bot_id in active_live_bots: 
                return JSONResponse({"error": "Bot active"}, 400)
            try:
                bot = LiveExecutiveBot(body)
                active_live_bots[bot_id] = bot
            except Exception as init_err:
                logger.error(f"❌ Bot Init Failed: {init_err}")
                return JSONResponse({"error": f"Failed to start: {str(init_err)}"}, 500)

        bg.add_task(bot.sync_and_trade)
        return {"status": "started", "bot_id": bot_id}
    except Exception as e:
        logger.error(f"🔥 Critical API Error: {e}")
        return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/stop')
async def stop_live_bot(request: Request):
    body = await request.json()
    user_id = body.get('userId')
    if not user_id: return JSONResponse({"error": "Missing userId"}, 400)
    
    clean_symbol = body['symbol'].replace('/', '-')
    bot_id = f"{user_id}_{clean_symbol}_{body['timeframe']}"
    
    with bots_lock:
        if bot_id in active_live_bots:
            active_live_bots[bot_id].stop()
            del active_live_bots[bot_id]
            return {"status": "stopped"}
    return JSONResponse({"error": "Bot not found"}, 404)

# --- 7. TELEMETRY ENDPOINTS ---

@app.get('/api/bot/status')
async def get_bot_status(userId: str = None):
    if not userId: return JSONResponse({"error": "userId required"}, 400)
    
    target_bot = None
    for bot_id, bot in active_live_bots.items():
        if bot_id.startswith(userId):
            target_bot = bot
            break
            
    if not target_bot:
        rows = execute_db_with_retry("SELECT cash, last_ts FROM bots WHERE user_id=?", (userId,))
        if rows:
            return {"status": "stopped", "currentBalance": rows[0][0], "active": False}
        return JSONResponse({"status": "not_found"}, 404)

    return {
        "status": "running",
        "symbol": target_bot.symbol,
        "timeframe": target_bot.tf,
        "currentBalance": target_bot.manager.current_equity,
        "positions": target_bot.manager.positions,
        "trades": target_bot.manager.trades,
        "performanceMetrics": {
            "totalTrades": len(target_bot.manager.trades),
            "netProfit": target_bot.manager.current_equity - target_bot.manager.initial_capital
        }
    }

@app.get('/api/bot/logs')
async def get_bot_logs(userId: str = None):
    if not userId: return JSONResponse({"error": "userId required"}, 400)

    log_files = [f for f in os.listdir(LOG_DIR) if f.startswith(userId)]
    if not log_files: return []

    latest_log = max([os.path.join(LOG_DIR, f) for f in log_files], key=os.path.getmtime)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()[-100:]
            
        parsed_logs = []
        for line in lines:
            parts = line.strip().split(' | ')
            if len(parts) >= 3:
                parsed_logs.append({
                    "timestamp": parts[0],
                    "type": parts[1].strip().lower(), 
                    "message": parts[2]
                })
        return parsed_logs[::-1]
    except Exception: return []

@app.post('/api/backtest/combo')
@app.post('/api/backtest/run')
@app.post('/api/ml/run-combo-backtest')
async def handle_backtest(request: Request):
    try:
        body = await request.json()
        params = body.get('params', {})
        
        symbol, tf = body.get('symbol', 'BTC-USD').replace('-', '/'), body.get('timeframe', '1h')
        exchange = ccxt.coinbase()
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df.set_index(pd.to_datetime(df['ts'], unit='ms', utc=True), inplace=True)

        df.ta.sma(length=10, append=True, col_names="fast_sma")
        df.ta.sma(length=50, append=True, col_names="slow_sma")
        df.ta.macd(append=True); df.ta.psar(append=True)
        df.ta.atr(length=14, append=True); df.ta.adx(length=14, append=True)
        df.ta.rsi(length=14, append=True, col_names="RSI_14")
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.stoch(append=True); df.ta.cci(length=14, append=True)
        df.ta.ichimoku(append=True); df.ta.obv(append=True)
        df.ta.ema(length=20, append=True, col_names="EMA_20")
        
        atr_col = df.filter(like='ATR').columns[-1]
        adx_col = df.filter(like='ADX').columns[0]
        
        df['ATR_SAFE'] = df[atr_col].shift(1)
        df['ADX_SAFE'] = df[adx_col].shift(1)

        ml_model = None
        if body.get('mlMode') == 'on':
            p = os.path.join(MODEL_DIR, f"{body['mlModel']}.joblib")
            if os.path.exists(p): ml_model = joblib.load(p)

        strats = body.get('strategies', []) or [{"code": body.get('code'), "params": body.get('params', {})}]
        
        tsl_mult = float(params.get('tslAtrMult', 3.5))
        min_adx = float(params.get('minAdxLevel', 0))
        risk_pct = float(body.get('riskPercentage', 1)) / 100.0
        risk_mode = body.get('riskManagementMode', 'static')
        max_layers = int(body.get('maxPyramiding', 3))
        slippage_bps = float(body.get('slippageBps', SLIPPAGE_BPS))

        mgr = PrecisionPyramidManager(
            capital=body.get('initialBalance', 1000),
            fee=DEFAULT_TAKER_FEE,
            max_layers=max_layers,
            base_risk=risk_pct,
            risk_mode=risk_mode
        )
        
        curve = []

        for i in range(50, len(df)):
            p = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            adx_val = df['ADX_SAFE'].iloc[i]
            atr_val = df['ATR_SAFE'].iloc[i]

            if min_adx > 0 and adx_val < min_adx:
                sig = 0 
            else:
                votes = [compute_signal(df.iloc[:i], s, ml_model, body.get('mlThreshold', 0.5)) for s in strats]
                sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
            mgr.handle(sig, p, low, high, df.index[i].isoformat(), atr_val, tsl_mult, slippage_bps=slippage_bps)
            curve.append({'timestamp': df.index[i].isoformat(), 'balance': mgr.step_equity(p)})

        final_bal = curve[-1]['balance']
        total_ret = ((final_bal - body.get('initialBalance', 1000)) / body.get('initialBalance', 1000)) * 100
        
        trades = len(mgr.trades)
        wins = len([t for t in mgr.trades if t['profit'] > 0])
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        periods_per_year = 8760
        if '15m' in tf: periods_per_year = 35040
        elif '5m' in tf: periods_per_year = 105120
        elif '1d' in tf: periods_per_year = 365
        
        returns = pd.Series([c['balance'] for c in curve]).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(periods_per_year)) if len(returns) > 0 and returns.std() != 0 else 0

        res = {
            "metrics": {
                "totalReturn": round(total_ret, 2),
                "maxDrawdown": round(max(mgr.dd_series) * 100, 2),
                "totalTrades": trades,
                "winRate": round(win_rate, 2),
                "sharpeRatio": round(sharpe, 2)
            },
            "equityCurve": curve,
            "tradeBreakdown": mgr.trades,
            "candleData": [{"time": i.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for i, r in df.iterrows()]
        }
        
        return JSONResponse(content={"combinedResult": res})

    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return []
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

@app.get("/api/bot/winners")
def get_winners():
    winners = []
    if not os.path.exists(RESULTS_DIR): return []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, f), 'r') as file:
                    data = json.load(file)
                    config_data = {"strategies": data} if isinstance(data, list) else data
                    winners.append({"id": f, "name": f.replace('.json',''), "config": config_data})
            except: pass
    winners.sort(key=lambda x: x['config'].get('metrics', {}).get('totalReturn', 0), reverse=True)
    return winners

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
