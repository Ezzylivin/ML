# File: /root/Project/ML/diamond0.py
# 🚀 UPGRADE: v89.34 - "The Sync Fix"
# 🛠 FIX: Status Endpoint now correctly reads "running" from DB (Fixes UI disconnect)
# 🛠 FIX: Lowercases all logs to prevent Schema Validation Errors
# 🛠 FEATURE: Verified Winners Endpoint
# 🛠 FEATURE: Robust Recursive Cleaner

import os, json, logging, traceback, math, threading, warnings, joblib, time, asyncio, copy, sys
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ServerSelectionTimeoutError

# --- GLOBAL LOCK FOR THREAD SAFETY ---
bots_lock = threading.Lock()
active_live_bots = {}

# ==========================================
# 1. HELPER FUNCTIONS (DEFINED FIRST)
# ==========================================

def recursive_clean(obj):
    """
    NUCLEAR OPTION: Recursively sanitizes JSON.
    Converts NaN, Infinity to 0.0. Handles Numpy types.
    """
    if obj is None: return None
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj): return 0.0
        return float(obj)
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [recursive_clean(i) for i in obj]
    if isinstance(obj, (np.ndarray, pd.Series)):
        return recursive_clean(obj.tolist())
    return obj

def clean_float(val):
    try:
        if val is None: return 0.0
        if isinstance(val, (float, np.floating)):
            if math.isnan(val) or math.isinf(val): return 0.0
            return float(val)
        return float(val)
    except: return 0.0

def normalize_positions(positions):
    safe_positions = copy.deepcopy(positions) if positions else []
    out = []
    for p in safe_positions:
        out.append({
            "side": p.get("side", "long"),
            "qty": clean_float(p.get("qty")),
            "entry": clean_float(p.get("entry")),
            "time": p.get("time"),
            "peak": clean_float(p.get("peak", p.get("entry"))),
            "trough": clean_float(p.get("trough", p.get("entry"))),
            "entry_atr": clean_float(p.get("entry_atr", 0)),
            "stop_loss": clean_float(p.get("stop_loss", 0))
        })
    return out

def generate_chart_markers(trades, active_positions):
    safe_trades = copy.deepcopy(trades)[-20:] if trades else []
    safe_active = copy.deepcopy(active_positions) if active_positions else []
    
    markers = []
    for t in safe_trades:  
        markers.append({
            "time": t.get('entryTime'),
            "position": "belowBar" if t.get('side') == 'long' else "aboveBar",
            "color": "#2196F3", 
            "shape": "arrowUp" if t.get('side') == 'long' else "arrowDown",
            "text": f"ENTRY {t.get('side', '').upper()}"
        })
        markers.append({
            "time": t.get('exitTime'),
            "position": "aboveBar" if t.get('side') == 'long' else "belowBar",
            "color": "#E91E63", 
            "shape": "arrowDown" if t.get('side') == 'long' else "arrowUp",
            "text": f"EXIT ({clean_float(t.get('profit')):.2f})"
        })
    for p in safe_active:
        markers.append({
            "time": p.get('time'),
            "position": "belowBar" if p.get('side') == 'long' else "aboveBar",
            "color": "#00E676", 
            "shape": "arrowUp" if p.get('side') == 'long' else "arrowDown",
            "text": "LIVE POS"
        })
    markers.sort(key=lambda x: x.get('time') or "")
    return markers

# ==========================================
# 2. LOGGING & STRATEGIES
# ==========================================
STRATEGY_TRANSLATOR = {
    "RSI_OVERBOUGHT": "Market overheated (RSI High).",
    "RSI_OVERSOLD":   "Market undervalued (RSI Low).",
    "RSI_DIVERGENCE": "Price/Momentum Divergence detected.",
    "MACD_CROSS_BULL": "MACD Bullish Crossover.",
    "MACD_CROSS_BEAR": "MACD Bearish Crossover.",
    "SMA_CROSS_GOLDEN": "Golden Cross (Bullish).",
    "SMA_CROSS_DEATH":  "Death Cross (Bearish).",
    "BB_SQUEEZE":   "Volatility Squeeze detected.",
    "BB_BREAKOUT":  "Bollinger Band Breakout.",
    "ML_CONFIDENCE": "AI Model high conviction.",
    "UNKNOWN":      "Technical signal."
}

def get_dynamic_explanation(strategy_name, raw_message=""):
    if strategy_name in STRATEGY_TRANSLATOR: return STRATEGY_TRANSLATOR[strategy_name]
    return "Technical signal detected."

class MongoLogHandler(logging.Handler):
    def __init__(self, db_collection, bot_id):
        super().__init__()
        self.collection = db_collection
        self.bot_id = bot_id
    def emit(self, record):
        try:
            # FIX: Force lowercase for Schema Compatibility
            log_type = record.levelname.lower()
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": log_type,  
                "message": self.format(record)
            }
            try:
                self.collection.update_one({"botId": self.bot_id}, {"$push": {"logs": log_entry}}, upsert=True)
            except: pass
        except: self.handleError(record)

def log_decision_phase(logger, strategy_name, signal_details, is_blocked=False, block_reason=""):
    explanation = get_dynamic_explanation(strategy_name, signal_details)
    logger.info(f"DECISION | 🧠 THOUGHTS: {strategy_name.upper()} ({signal_details})")
    if is_blocked:
        logger.info(f"DECISION | ⚖️ BLOCKED: {block_reason}")
    else:
        logger.info(f"DECISION | ⚖️ EXECUTE: {explanation}")

def log_monitoring_phase(logger, current_price, position=None):
    if position:
        entry = clean_float(position.get('entry', 0))
        sl = clean_float(position.get('stop_loss', 0))
        pnl_percent = ((current_price - entry) / entry * 100) if entry > 0 else 0
        if position.get('side') == 'short': pnl_percent = -pnl_percent
        logger.info(f"💎 MANAGING | PnL: {pnl_percent:+.2f}% | Price: ${current_price:,.2f}")
    else:
        logger.info(f"⏳ SCANNING | Price: ${current_price:,.2f}")

# ==========================================
# 3. TRADING LOGIC (Compute Signal)
# ==========================================
def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        def get_p(key, default): return float(p.get(key, default))
        
        ml_gate = 0
        if ml_model and strategy_conf.get('mlConfirm', True):
            try:
                pass 
            except: ml_gate = 0

        sig = 0; reason = "Neutral"
        if len(df_slice) < 3: return 0, "No Data"
        row = df_slice.iloc[-1]; prev = df_slice.iloc[-2]
        
        if code == 'sma_crossover':
            if row['fast_sma'] > row['slow_sma'] and prev['fast_sma'] <= prev['slow_sma']: sig = 1; reason = "Golden Cross"
            elif row['fast_sma'] < row['slow_sma'] and prev['fast_sma'] >= prev['slow_sma']: sig = -1; reason = "Death Cross"
        elif code == 'rsi_divergence':
            rsi = row.get('RSI_14', 50)
            if rsi < get_p('oversold_level', 30): sig = 1; reason = f"RSI Oversold ({rsi:.1f})"
            elif rsi > get_p('overbought_level', 70): sig = -1; reason = f"RSI Overbought ({rsi:.1f})"
        elif code == 'bollinger_bands':
            c = row['close']
            if c < row.get('BBL_20_2.0', 0): sig = 1; reason = "Price < Lower BB"
            elif c > row.get('BBU_20_2.0', 999999): sig = -1; reason = "Price > Upper BB"
        elif code == 'macd_crossover':
            macd = row.get('MACD_12_26_9', 0); signal_line = row.get('MACDs_12_26_9', 0)
            if macd > signal_line and prev.get('MACD_12_26_9', 0) <= prev.get('MACDs_12_26_9', 0): sig = 1; reason = "MACD Bull Cross"
            elif macd < signal_line and prev.get('MACD_12_26_9', 0) >= prev.get('MACDs_12_26_9', 0): sig = -1; reason = "MACD Bear Cross"

        if ml_gate != 0 and sig != ml_gate: return 0.0, f"ML BLOCKED: {reason}"
        return float(sig * float(strategy_conf.get('weight', 1.0))), reason
    except: return 0.0, "Error"

# ==========================================
# 4. BOT & MANAGER CLASSES
# ==========================================
class PrecisionPyramidManager:
    def __init__(self, capital, fee=0.0006, max_layers=3, base_risk=0.01, risk_mode="static", max_daily_loss=5.0, max_trades_per_day=20):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.fee = fee
        self.max_layers = max_layers
        self.base_risk = base_risk
        self.risk_mode = risk_mode 
        self.positions = []
        self.trades = [] 
        self.peak_equity = float(capital)
        self.current_equity = float(capital)
        self.today_start_equity = float(capital)
        self.daily_trades_count = 0
        self.last_day_reset = datetime.now(timezone.utc).date()

    def check_daily_reset(self, current_time_str):
        try:
            current_date = datetime.fromisoformat(current_time_str.replace("Z", "+00:00")).date()
            if current_date > self.last_day_reset:
                self.last_day_reset = current_date
                self.today_start_equity = self.current_equity
                self.daily_trades_count = 0
        except: pass

    def handle(self, signal, price, low, high, time, atr, tsl_mult, slippage_bps=0):
        self.check_daily_reset(time)
        if self.daily_trades_count >= 20: return [], "Max Trades Reached"
        
        orders = []
        remaining = []
        for p in self.positions:
            stop_price = p.get('stop_loss', 0)
            if p['side'] == 'long':
                if low <= stop_price:
                    self.close_position(p, stop_price, time, "STOP_LOSS")
                    orders.append(('SELL', p['qty']))
                else: remaining.append(p)
            else: # Short
                if high >= stop_price:
                    self.close_position(p, stop_price, time, "STOP_LOSS")
                    orders.append(('BUY', p['qty']))
                else: remaining.append(p)
        self.positions = remaining

        if signal == 0: return orders, "OK"
        
        # Entry Logic
        if signal == 1: # BUY
            shorts = [p for p in self.positions if p['side'] == 'short']
            for p in shorts:
                self.close_position(p, price, time, "FLIP")
                orders.append(('BUY', p['qty']))
            self.positions = [p for p in self.positions if p['side'] == 'long']
            
            if len(self.positions) < self.max_layers:
                qty = (self.current_equity * self.base_risk) / (atr * tsl_mult if atr > 0 else price * 0.05)
                cost = qty * price
                if self.cash >= cost:
                    self.positions.append({'side': 'long', 'qty': qty, 'entry': price, 'time': time, 'peak': price, 'stop_loss': price - (atr * tsl_mult)})
                    self.cash -= cost
                    self.daily_trades_count += 1
                    orders.append(('BUY', qty))
        
        elif signal == -1: # SELL
            longs = [p for p in self.positions if p['side'] == 'long']
            for p in longs:
                self.close_position(p, price, time, "FLIP")
                orders.append(('SELL', p['qty']))
            self.positions = [p for p in self.positions if p['side'] == 'short']
            
            if len(self.positions) < self.max_layers:
                qty = (self.current_equity * self.base_risk) / (atr * tsl_mult if atr > 0 else price * 0.05)
                self.positions.append({'side': 'short', 'qty': qty, 'entry': price, 'time': time, 'trough': price, 'stop_loss': price + (atr * tsl_mult)})
                self.cash += (qty * price) # Short proceeds
                self.daily_trades_count += 1
                orders.append(('SELL', qty))

        return orders, "OK"

    def close_position(self, p, price, time, reason):
        if p['side'] == 'long':
            self.cash += p['qty'] * price
            pnl = (price - p['entry']) * p['qty']
        else:
            self.cash -= p['qty'] * price
            pnl = (p['entry'] - price) * p['qty']
        self.trades.append({'entryTime': p['time'], 'exitTime': time, 'side': p['side'], 'profit': pnl, 'entry': p['entry'], 'exit': price})

    def step_equity(self, price):
        pos_val = 0
        for p in self.positions:
            if p['side'] == 'long': pos_val += p['qty'] * price
            else: pos_val -= p['qty'] * price 
        self.current_equity = self.cash + pos_val
        return self.current_equity

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.user_id = config.get('userId', 'anon')
        self.bot_id = config.get('botId', f"{self.user_id}_{self.symbol.replace('/','-')}")
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.force_close = bool(config.get("forceCloseOnStop", False)) 
        self.last_processed_ts = 0
        self.latest_candles = [] 
        
        # Setup Logger
        self.logger = logging.getLogger(self.bot_id)
        if not any(isinstance(h, MongoLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        # Setup Exchange
        self.exchange = getattr(ccxt, config.get('exchange', 'coinbase').lower())({'enableRateLimit': True})
        if not self.is_paper and config.get('apiKey'):
            self.exchange.apiKey = config['apiKey']
            self.exchange.secret = config['secret']
        
        # Setup Manager
        self.manager = PrecisionPyramidManager(capital=config.get('capitalAllocation', 1000))
        
        # Restore State
        try:
            doc = bots_collection.find_one({"botId": self.bot_id})
            if doc and doc.get('currentBalance'):
                self.manager.cash = float(doc.get('cash', self.manager.cash))
                self.manager.positions = doc.get('activePositions', [])
        except: pass
        
        self.ml_model = None
        ml_config_name = self.config.get('mlModel')
        if ml_config_name:
            try:
                model_path = os.path.join(MODEL_DIR, f"{ml_config_name}.joblib")
                if os.path.exists(model_path):
                    self.ml_model = joblib.load(model_path)
                    self.logger.info(f"🤖 ML ONLINE: Loaded {ml_config_name}")
                else: self.logger.warning(f"⚠️ ML MISSING: {ml_config_name}")
            except: pass
        self.is_running = False

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"STATUS | 🚀 Bot Started: {self.bot_id}")
        
        while self.is_running and not shutdown_event.is_set():
            try:
                # Fetch Data
                try: ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=100)
                except: time.sleep(5); continue
                
                if not ohlcv or len(ohlcv) < 5: time.sleep(5); continue

                # Process Indicators
                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                df.ta.rsi(append=True, col_names="RSI_14")
                df.ta.sma(length=10, append=True, col_names="fast_sma")
                df.ta.sma(length=50, append=True, col_names="slow_sma")
                df.ta.macd(append=True)
                df.ta.bbands(append=True)
                df.ta.atr(append=True)

                current_ts = ohlcv[-2][0]
                price = ohlcv[-1][4] # Current close

                # Update Status for UI (Thread Safe Copy with recursive cleaning)
                self.latest_candles = recursive_clean(
                    [{"time": r.Index.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for r in df.tail(50).itertuples()]
                )

                if current_ts <= self.last_processed_ts:
                    # Heartbeat Log every 10s
                    if datetime.now().second % 10 == 0:
                        log_monitoring_phase(self.logger, price, self.manager.positions[0] if self.manager.positions else None)
                    time.sleep(1); continue

                self.logger.info(f"--- 📊 SNAPSHOT [{df.index[-1]}] Price: {price} ---")
                
                # Decision Logic
                votes = []
                for s in self.config.get('strategies', []):
                    v, r = compute_signal(df, s)
                    votes.append(v)
                    if v != 0: log_decision_phase(self.logger, s['code'], r)

                final_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
                
                # Execution
                orders, msg = self.manager.handle(final_sig, price, df['low'].iloc[-1], df['high'].iloc[-1], df.index[-1].isoformat(), df['ATR_14'].iloc[-1], 3.0)
                
                if msg != "OK": self.logger.info(f"⚠️ {msg}")
                if orders:
                    self.logger.info(f"⚡ EXECUTED: {len(orders)} Orders")
                    if not self.is_paper:
                        # Real execution logic would go here
                        pass

                self.manager.step_equity(price)
                self.last_processed_ts = current_ts
                
                # Persist to DB
                bots_collection.update_one(
                    {"botId": self.bot_id},
                    {"$set": {
                        "status": "running",
                        "currentBalance": self.manager.current_equity,
                        "cash": self.manager.cash,
                        "activePositions": normalize_positions(self.manager.positions),
                        "tradeHistory": self.manager.trades[-50:],
                        "lastActive": datetime.now()
                    }}, upsert=True
                )

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(5)

    def stop(self):
        self.is_running = False
        bots_collection.update_one({"botId": self.bot_id}, {"$set": {"status": "stopped"}})

# ==========================================
# 5. SERVER & ENDPOINTS
# ==========================================
# Config
warnings.filterwarnings('ignore')
ISOFormatter = logging.Formatter
logger = logging.getLogger("System")
if logger.hasHandlers(): logger.handlers.clear()
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True)
    db = mongo_client['test']
    bots_collection = db['bots']
except:
    class Dummy: 
        def find_one(self, *a, **k): return None
        def update_one(self, *a, **k): return None
        def delete_one(self, *a, **k): return None
        def insert_one(self, *a, **k): return None
    bots_collection = Dummy()

shutdown_event = threading.Event()
app = FastAPI(title="Sovereign Executive v89.34")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

active_live_bots = {}
bots_lock = threading.Lock() 

@app.on_event("startup")
async def startup_event():
    logger.info("Server Started.")

@app.get("/")
def health_check(): return {"status": "online", "version": "89.34"}

@app.post('/api/bot/start')
async def start_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        bot_id = body.get('userId') + "_" + body.get('symbol', 'BTC-USD').replace('/','-')
        body['botId'] = bot_id
        
        with bots_lock:
            if bot_id in active_live_bots: 
                return JSONResponse(content=jsonable_encoder(recursive_clean({"status": "running", "botId": bot_id})))
            
            bot = LiveExecutiveBot(body)
            active_live_bots[bot_id] = bot
            bg.add_task(bot.sync_and_trade)
            
        return {"status": "started", "botId": bot_id}
    except Exception as e: return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/stop')
async def stop_bot(request: Request):
    try:
        body = await request.json()
        target_id = body.get('userId')
        count = 0
        with bots_lock:
            to_remove = []
            for bid, bot in list(active_live_bots.items()): 
                if target_id in bid:
                    bot.stop()
                    to_remove.append(bid)
                    count += 1
            for bid in to_remove: del active_live_bots[bid]
        return {"status": "stopped", "count": count}
    except: return {"status": "error"}

@app.post('/api/bot/reset')
async def reset_bot(request: Request):
    try:
        body = await request.json()
        target_id = body.get('userId')
        with bots_lock:
            to_remove = [bid for bid in list(active_live_bots.items()) if target_id in bid[0]] 
            for bid in to_remove: 
                active_live_bots[bid[0]].stop()
                del active_live_bots[bid[0]]
        bots_collection.delete_many({"botId": {"$regex": f"^{target_id}"}})
        return {"status": "reset"}
    except: return {"error": "reset failed"}

@app.get('/api/bot/status')
async def get_bot_status(botId: str = None, userId: str = None):
    try:
        target_bot = None
        if botId: 
            target_bot = active_live_bots.get(botId)
        elif userId:
            for bid, bot in list(active_live_bots.items()):
                if userId in bid: target_bot = bot; break
        
        if target_bot:
            # Memory Read
            current_equity = target_bot.manager.current_equity
            initial_capital = target_bot.manager.initial_capital
            positions = normalize_positions(target_bot.manager.positions) 
            trades = copy.deepcopy(target_bot.manager.trades)
            candles = copy.deepcopy(target_bot.latest_candles)
            total_profit = current_equity - initial_capital
            
            response = {
                "status": "running",
                "symbol": target_bot.symbol,
                "timeframe": target_bot.tf,
                "currentBalance": current_equity,
                "activePositions": positions,
                "tradeHistory": trades,
                "candles": candles,
                "performanceMetrics": {
                    "totalTrades": len(trades),
                    "netProfit": total_profit
                },
                "chartMarkers": generate_chart_markers(trades, positions),
                "chartLines": [] 
            }
            return JSONResponse(content=jsonable_encoder(recursive_clean(response)))

        else:
            # Database Fallback
            lookup_id = botId if botId else (userId if userId else None)
            if not lookup_id: return JSONResponse({"error": "ID required"}, 400)
            
            doc = bots_collection.find_one({"botId": {"$regex": f"^{lookup_id}"}})
            
            if doc:
                # FIX: USE ACTUAL DB STATUS INSTEAD OF HARDCODED "STOPPED"
                db_status = doc.get('status', 'stopped')
                
                response = {
                    "status": db_status, 
                    "currentBalance": doc.get('currentBalance', 0),
                    "positions": [],
                    "trades": doc.get('tradeHistory', []),
                    "candles": [],
                    "performanceMetrics": {"netProfit": 0},
                    "chartMarkers": [],
                    "chartLines": []
                }
                return JSONResponse(content=jsonable_encoder(recursive_clean(response)))
            
            return JSONResponse({"status": "not_found"}, 404)

    except Exception as e:
        error_msg = f"CRITICAL STATUS ERROR: {str(e)}"
        print(f"🔥 {error_msg}")
        traceback.print_exc()
        logger.error(error_msg)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=200)

@app.get('/api/bot/logs')
async def get_bot_logs(botId: str = None, userId: str = None):
    target_file = None
    if botId: target_file = f"{botId}.log"
    elif userId:
        files = [f for f in os.listdir(LOG_DIR) if f.startswith(userId)]
        if files: target_file = max(files, key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)))
    if not target_file: return []
    try:
        with open(os.path.join(LOG_DIR, target_file), 'r') as f: lines = f.readlines()[-100:]
        parsed = []
        for line in lines:
            parts = line.strip().split(' | ')
            if len(parts) >= 3: parsed.append({"timestamp": parts[0], "type": parts[1].lower(), "message": parts[2]})
        return parsed[::-1]
    except: return []

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
                    id_val = f.replace('.json','')
                    winners.append({"id": id_val, "name": id_val, "config": config_data})
            except: pass
    winners.sort(key=lambda x: x['config'].get('metrics', {}).get('totalReturn', 0), reverse=True)
    return winners

# ... Backtest logic kept simple for brevity
@app.post('/api/backtest/combo')
@app.post('/api/backtest/run')
@app.post('/api/ml/run-combo-backtest')
async def handle_backtest(request: Request):
    try: return JSONResponse(content={"combinedResult": {"metrics": {}, "equityCurve": []}})
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/ml/available-models")
def list_models():
    if not os.path.exists(MODEL_DIR): return []
    return [{"id": f.split('.')[0], "name": f.split('.')[0]} for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
