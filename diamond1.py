# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v77.0 - "The Titanium Standard"
# 🛠 FEATURES: Dynamic Exchange, Concurrency Locks, ML Validation, Deep History.

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
# Main System Logger
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
SLIPPAGE_BPS = 0.0002
GLOBAL_MODEL_CACHE = {}

app = FastAPI(title="Sovereign Executive v77.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}

# --- 2. PERSISTENCE LAYER (Concurrency Safe) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS bots (bot_id TEXT PRIMARY KEY, cash REAL, last_ts INTEGER)')
    cur.execute('CREATE TABLE IF NOT EXISTS positions (bot_id TEXT, qty REAL, entry REAL, time TEXT, peak REAL)')
    conn.commit()
    conn.close()

init_db()

def execute_db_with_retry(query, params=(), retries=5):
    """🛡️ TITANIUM UPGRADE: Robust Concurrency Handling"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for attempt in range(retries):
        try:
            cur.execute(query, params)
            conn.commit()
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                time.sleep(0.1 * (2 ** attempt)) # Exponential backoff
            else:
                logger.error(f"DB Error: {e}")
                break
    return False

# --- 3. DYNAMIC STRATEGY REGISTRY ---
def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        weight = float(strategy_conf.get('weight', 1.0))
        ml_confirm = strategy_conf.get('mlConfirm', True)

        def get_p(key, default): return float(p.get(key, default))

        # 🧠 ML GATE (TITANIUM UPGRADE: Input Validation)
        ml_gate_sig = 0
        if ml_model is not None and ml_confirm:
            try:
                window = df_slice.tail(5).select_dtypes(include=[np.number])
                # Check for feature mismatch if possible
                if hasattr(ml_model, "n_features_in_") and window.shape[1] != ml_model.n_features_in_:
                    # Soft fail: Default to neutral if features don't match
                    ml_gate_sig = 0 
                elif len(window) >= 5:
                    p_long = np.mean(ml_model.predict_proba(window)[:, 1])
                    if p_long > ml_thresh: ml_gate_sig = 1
                    elif p_long < (1 - ml_thresh): ml_gate_sig = -1
            except Exception: 
                ml_gate_sig = 0 # Default to Neutral on any ML error

        sig = 0
        t_row = df_slice.iloc[-1]
        t_prev = df_slice.iloc[-2] if len(df_slice) > 1 else t_row
        
        # 1. SMA Crossover
        if code == 'sma_crossover':
            if t_row.get('fast_sma') > t_row.get('slow_sma') and t_prev.get('fast_sma') <= t_prev.get('slow_sma'): sig = 1
            elif t_row.get('fast_sma') < t_row.get('slow_sma') and t_prev.get('fast_sma') >= t_prev.get('slow_sma'): sig = -1
        # 2. MACD Crossover
        elif code == 'macd_crossover':
            if t_row.get('MACD_12_26_9') > t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') <= t_prev.get('MACDs_12_26_9'): sig = 1
            elif t_row.get('MACD_12_26_9') < t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') >= t_prev.get('MACDs_12_26_9'): sig = -1
        # 3. PSAR Flip
        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl_0.02_0.2')) and pd.isna(t_prev.get('PSARl_0.02_0.2')): sig = 1
            elif pd.notna(t_row.get('PSARs_0.02_0.2')) and pd.isna(t_prev.get('PSARs_0.02_0.2')): sig = -1
        # 4. RSI Threshold
        elif code == 'rsi_divergence' or code == 'rsi_threshold_reversal':
            rsi = t_row.get('RSI_14', 50)
            if rsi < get_p('oversold_level', 30): sig = 1
            elif rsi > get_p('overbought_level', 70): sig = -1
        # 5. Bollinger Reversal
        elif code == 'bollinger_bands':
            if t_row.get('close') < t_row.get('BBL_20_2.0'): sig = 1
            elif t_row.get('close') > t_row.get('BBU_20_2.0'): sig = -1
        # 6. ATR Envelope
        elif code == 'atr_breakout':
            close = t_row.get('close')
            prev_close = t_prev.get('close')
            mult = get_p('atr_multiplier', 3.0)
            ema = t_row.get('EMA_20', close)
            atr = t_row.get('ATR_14', 0)
            if close > (ema + atr*mult) and prev_close <= (ema + atr*mult): sig = 1
            elif close < (ema - atr*mult) and prev_close >= (ema - atr*mult): sig = -1
        # 7. Stochastic
        elif code == 'stochastic_crossover':
            k, d = t_row.get('STOCHk_14_3_3'), t_row.get('STOCHd_14_3_3')
            pk, pd_val = t_prev.get('STOCHk_14_3_3'), t_prev.get('STOCHd_14_3_3')
            if k > d and pk <= pd_val and k < 20: sig = 1
            elif k < d and pk >= pd_val and k > 80: sig = -1
        # 8. CCI Threshold
        elif code == 'cci_oversold':
            cci = t_row.get('CCI_14_0.015')
            if cci < -100: sig = 1
            elif cci > 100: sig = -1
        # 9. Ichimoku Cross
        elif code == 'ichimoku_system':
            tk, kj = t_row.get('ITS_9'), t_row.get('IKS_26')
            ptk, pkj = t_prev.get('ITS_9'), t_prev.get('IKS_26')
            if tk > kj and ptk <= pkj: sig = 1
            elif tk < kj and ptk >= pkj: sig = -1
        # 10. OBV Slope
        elif code == 'obv_signal':
            if t_row.get('OBV') > t_prev.get('OBV'): sig = 1
            elif t_row.get('OBV') < t_prev.get('OBV'): sig = -1

        # Non-Blocking Gate
        if ml_gate_sig == 0: return float(sig * weight)
        if sig == ml_gate_sig: return float(sig * weight)
        return 0.0

    except Exception: return 0.0

# --- 4. RISK & EXECUTION ENGINE ---
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

    def handle(self, signal, price, time, atr, tsl_mult, slippage_rate=0):
        orders = []
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        
        fill_price = price * (1 + slippage_rate) if signal == 1 else price * (1 - slippage_rate)

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
            p['peak'] = max(p['peak'], price)
            if price <= (p['peak'] - dist):
                stop_fill = price * (1 - slippage_rate)
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
        self.bot_id = f"{self.symbol}_{self.tf}"
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.last_processed_ts = 0
        self.reconcile_counter = 0
        
        # 🛡️ TITANIUM UPGRADE: Per-Bot Logging
        self.logger = logging.getLogger(self.bot_id)
        handler = logging.FileHandler(os.path.join(LOG_DIR, f"{self.bot_id}.log"))
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.params = config.get('params', {})
        self.tsl_mult = float(self.params.get('tslAtrMult', 3.5))
        self.min_adx = float(self.params.get('minAdxLevel', 0))
        
        # 🛡️ TITANIUM UPGRADE: Dynamic Exchange Loading
        exchange_id = config.get('exchange', 'coinbase').lower()
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': config.get('apiKey', ''), 
            'secret': config.get('secret', ''),
            'enableRateLimit': True
        })
        
        # Validate Market
        try:
            self.exchange.load_markets()
            if self.symbol not in self.exchange.markets:
                self.logger.error(f"❌ Invalid Symbol: {self.symbol}")
                raise ValueError(f"Symbol {self.symbol} not found on {exchange_id}.")
        except Exception as e:
            self.logger.error(f"Exchange init failed: {e}")

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
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT cash, last_ts FROM bots WHERE bot_id=?", (self.bot_id,))
        row = cur.fetchone()
        if row:
            self.manager.cash, self.last_processed_ts = row[0], row[1]
            cur.execute("SELECT qty, entry, time, peak FROM positions WHERE bot_id=?", (self.bot_id,))
            for r in cur.fetchall():
                self.manager.positions.append({'qty':r[0], 'entry':r[1], 'time':r[2], 'peak':r[3]})
        conn.close()
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            self.manager.step_equity(ticker['last'])
            self.logger.info(f"✨ Bot Recovered. Equity: {self.manager.current_equity}")
        except: pass

    def reconcile_balance(self):
        if self.is_paper: return
        self.reconcile_counter += 1
        if self.reconcile_counter % 10 == 0:
            try:
                bal = self.exchange.fetch_balance()
                quote = self.symbol.split('/')[-1]
                self.manager.cash = bal['total'].get(quote, self.manager.cash)
            except: pass

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"🚀 Bot Started: {self.bot_id} | Mode: {'PAPER' if self.is_paper else 'REAL'}")
        
        while self.is_running:
            try:
                # 🛡️ TITANIUM UPGRADE: Deep History (1000 candles)
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=1000)
                latest_ts = ohlcv[-1][0]
                if latest_ts <= self.last_processed_ts:
                    time.sleep(20); continue

                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df.set_index(pd.to_datetime(df['ts'], unit='ms', utc=True), inplace=True)

                # Indicators
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
                
                atr = df[atr_col].iloc[-1]
                adx = df[adx_col].iloc[-1]
                price = df['close'].iloc[-1]
                
                if self.min_adx > 0 and adx < self.min_adx:
                    combined_sig = 0
                else:
                    votes = [compute_signal(df, s, self.ml_model, 0.5) for s in self.config.get('strategies', [])]
                    combined_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)

                orders = self.manager.handle(combined_sig, price, df.index[-1].isoformat(), atr, self.tsl_mult)
                
                if not self.is_paper:
                    for action, qty in orders:
                        safe_qty = float(self.exchange.amount_to_precision(self.symbol, qty))
                        self.exchange.create_market_order(self.symbol, action.lower(), safe_qty)
                        self.reconcile_balance()

                self.manager.step_equity(price)
                self.last_processed_ts = latest_ts
                self._save_state(latest_ts)
                self.logger.info(f"✅ Candle {df.index[-1]} | Sig: {combined_sig} | Eq: {self.manager.current_equity:.2f}")

            except Exception as e:
                self.logger.error(f"Bot Loop Error: {traceback.format_exc()}")
                time.sleep(60)

    def _save_state(self, last_ts):
        execute_db_with_retry("REPLACE INTO bots VALUES (?, ?, ?)", (self.bot_id, self.manager.cash, last_ts))
        execute_db_with_retry("DELETE FROM positions WHERE bot_id=?", (self.bot_id,))
        for p in self.manager.positions:
            execute_db_with_retry("INSERT INTO positions VALUES (?, ?, ?, ?, ?)", (self.bot_id, p['qty'], p['entry'], p['time'], p['peak']))

    def stop(self): self.is_running = False

# --- 6. ENDPOINTS ---

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    try:
        # 1. Log the incoming request payload
        body = await request.json()
        symbol = body.get('symbol', 'Unknown')
        mode = body.get('mode', 'Unknown')
        logger.info(f"📥 Received Start Request for {symbol} ({mode})")

        # 2. Check for duplicate bot instances
        bot_id = f"{body['symbol']}_{body['timeframe']}"
        if bot_id in active_live_bots:
            logger.warning(f"⚠️ Bot {bot_id} is already running.")
            return JSONResponse({"error": f"Bot {bot_id} is already active. Stop it first."}, 400)
        
        # 3. Attempt to Initialize the Bot
        # This will trigger exchange connection, database checks, and model loading.
        # If any of these fail, we catch specific errors below.
        try:
            bot = LiveExecutiveBot(body)
        except ccxt.AuthenticationError:
            logger.error("❌ Exchange Auth Failed: Invalid API Keys")
            return JSONResponse({"error": "Exchange Authentication Failed. Check API Keys."}, 401)
        except ccxt.NetworkError:
            logger.error("❌ Exchange Network Error: Connection refused")
            return JSONResponse({"error": "Exchange Connection Failed. Network Error."}, 503)
        except ValueError as ve:
            logger.error(f"❌ Configuration Error: {ve}")
            return JSONResponse({"error": f"Config Error: {str(ve)}"}, 400)
        except Exception as init_error:
            logger.error(f"❌ Critical Bot Init Failure: {traceback.format_exc()}")
            return JSONResponse({"error": f"Bot Failed to Start: {str(init_error)}"}, 500)

        # 4. Launch Background Loop
        bg.add_task(bot.sync_and_trade)
        active_live_bots[bot_id] = bot
        
        logger.info(f"✅ Bot {bot_id} Launched Successfully.")
        return {"status": "started", "bot_id": bot_id}

    except Exception as e:
        # Catch-all for request parsing or generic server errors
        logger.error(f"🚨 Unhandled Server Error: {traceback.format_exc()}")
        return JSONResponse({"error": f"Server Crash: {str(e)}"}, 500)

@app.post('/api/bot/stop')
async def stop_live_bot(request: Request):
    body = await request.json()
    bot_id = f"{body['symbol']}_{body['timeframe']}"
    if bot_id in active_live_bots:
        active_live_bots[bot_id].stop()
        del active_live_bots[bot_id]
        return {"status": "stopped"}
    return JSONResponse({"error": "Bot not found"}, 404)

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
            adx_val = df['ADX_SAFE'].iloc[i]
            
            if min_adx > 0 and adx_val < min_adx:
                sig = 0 
            else:
                votes = [compute_signal(df.iloc[:i+1], s, ml_model, body.get('mlThreshold', 0.5)) for s in strats]
                sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
            # 🚀 AUDIT 4.1: Apply Slippage (simulated)
            mgr.handle(sig, p, df.index[i].isoformat(), df['ATR_SAFE'].iloc[i], tsl_mult, slippage_rate=SLIPPAGE_BPS)
            curve.append({'timestamp': df.index[i].isoformat(), 'balance': mgr.step_equity(p)})

        final_bal = curve[-1]['balance']
        total_ret = ((final_bal - body.get('initialBalance', 1000)) / body.get('initialBalance', 1000)) * 100
        
        res = {
            "metrics": {
                "totalReturn": round(total_ret, 2),
                "maxDrawdown": round(max(mgr.dd_series) * 100, 2),
                "totalTrades": len(mgr.trades)
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
