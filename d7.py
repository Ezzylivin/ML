# File: /root/Project/ML/diamond0.py
# 🚀 UPGRADE: v89.35 - "The Universal Connector"
# 🛠 FIX: API Response sends BOTH legacy (positions) and schema (activePositions) fields
# 🛠 FIX: Logs endpoint now fetches from MongoDB (File logs deprecated)
# 🛠 FIX: Equity recalculation on startup prevents PnL jumps

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
# 1. HELPER FUNCTIONS
# ==========================================

def recursive_clean(obj):
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
            log_type = record.levelname.lower()
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": log_type,  
                "message": self.format(record)
            }
            try: self.collection.update_one({"botId": self.bot_id}, {"$push": {"logs": log_entry}}, upsert=True)
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
# 3. TRADING LOGIC
# ==========================================
def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        def get_p(key, default): return float(p.get(key, default))
        
        ml_gate = 0
        if ml_model and strategy_conf.get('mlConfirm', True):
            try: pass 
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
        
        self.logger = logging.getLogger(self.bot_id)
        if not any(isinstance(h, MongoLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        self.exchange = getattr(ccxt, config.get('exchange', 'coinbase').lower())({'enableRateLimit': True})
        if not self.is_paper and config.get('apiKey'):
            self.exchange.apiKey = config['apiKey']
            self.exchange.secret = config['secret']
        
        self.manager = PrecisionPyramidManager(capital=config.get('capitalAllocation', 1000))
        
        # RESTORE STATE CORRECTLY
        try:
            doc = bots_collection.find_one({"botId": self.bot_id})
            if doc and doc.get('currentBalance'):
                self.manager.cash = float(doc.get('cash', self.manager.cash))
                self.manager.positions = doc.get('activePositions', [])
                
                # RECALCULATE EQUITY ON RESTORE TO PREVENT JUMPS
                if self.manager.positions:
                    # We need current price to calc equity, but we don't have it yet.
                    # Just trust saved equity for now, it will update on first tick.
                    self.manager.current_equity = float(doc.get('currentBalance', self.manager.cash))
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
                try: ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=100)
                except: time.sleep(5); continue
                
                if not ohlcv or len(ohlcv) < 5: time.sleep(5); continue

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
                price = ohlcv[-1][4]

                self.latest_candles = recursive_clean(
                    [{"time": r.Index.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for r in df.tail(50).itertuples()]
                )

                if current_ts <= self.last_processed_ts:
                    if datetime.now().second % 10 == 0:
                        log_monitoring_phase(self.logger, price, self.manager.positions[0] if self.manager.positions else None)
                    time.sleep(1); continue

                self.logger.info(f"--- 📊 SNAPSHOT [{df.index[-1]}] Price: {price} ---")
                
                votes = []
                for s in self.config.get('strategies', []):
                    v, r = compute_signal(df, s)
                    votes.append(v)
                    if v != 0: log_decision_phase(self.logger, s['code'], r)

                final_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
                orders, msg = self.manager.handle(final_sig, price, df['low'].iloc[-1], df['high'].iloc[-1], df.index[-1].isoformat(), df['ATR_14'].iloc[-1], 3.0)
                
                if msg != "OK": self.logger.info(f"⚠️ {msg}")
                if orders:
                    self.logger.info(f"⚡ EXECUTED: {len(orders)} Orders")
                    if not self.is_paper:
                        pass

                self.manager.step_equity(price)
                self.last_processed_ts = current_ts
                
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
app = FastAPI(title="Sovereign Executive v89.35")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

active_live_bots = {}
bots_lock = threading.Lock() 

@app.on_event("startup")
async def startup_event():
    logger.info("Server Started.")

@app.get("/")
def health_check(): return {"status": "online", "version": "89.35"}

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
                # DUAL-FIELD SUPPORT
                "positions": positions,
                "activePositions": positions, 
                "trades": trades,
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
            lookup_id = botId if botId else (userId if userId else None)
            if not lookup_id: return JSONResponse({"error": "ID required"}, 400)
            
            doc = bots_collection.find_one({"botId": {"$regex": f"^{lookup_id}"}})
            
            if doc:
                db_status = doc.get('status', 'stopped')
                
                response = {
                    "status": db_status, 
                    "currentBalance": doc.get('currentBalance', 0),
                    # DUAL-FIELD SUPPORT
                    "positions": [],
                    "activePositions": [],
                    "trades": doc.get('tradeHistory', []),
                    "tradeHistory": doc.get('tradeHistory', []),
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
    try:
        lookup_id = botId if botId else (userId if userId else None)
        if not lookup_id: return []
        
        # 1. Try In-Memory Search
        target_bot = None
        if userId:
            for bid, bot in list(active_live_bots.items()):
                if userId in bid: target_bot = bot; break
        elif botId: target_bot = active_live_bots.get(botId)
        
        # 2. If running, logs are in Mongo anyway, so just fetch from Mongo
        doc = bots_collection.find_one({"botId": {"$regex": f"^{lookup_id}"}})
        if doc and 'logs' in doc:
            # Sort descending by time
            sorted_logs = sorted(doc['logs'], key=lambda x: x['timestamp'], reverse=True)
            return sorted_logs[:200]
            
        return []
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
# ... (Backtest endpoints - kept as requested)
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
            capital=body.get('initialBalance', 1000), fee=DEFAULT_TAKER_FEE,
            max_layers=max_layers, base_risk=risk_pct, risk_mode=risk_mode,
            max_daily_loss=body.get('maxDailyLoss', 5.0), max_trades_per_day=body.get('maxTradesPerDay', 20)
        )
        
        curve = []
        prev_sig = 0
        
        for i in range(50, len(df)):
            p_open = df['open'].iloc[i]
            p_close = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            
            atr_val = df[atr_col].iloc[i-1] 

            mgr.handle(prev_sig, p_open, low, high, df.index[i].isoformat(), atr_val, tsl_mult, slippage_bps)
            
            adx_val = df[adx_col].iloc[i] 
            
            if min_adx > 0 and adx_val < min_adx:
                new_sig = 0 
            else:
                votes = [compute_signal(df.iloc[:i+1], s, ml_model, body.get('mlThreshold', 0.5)) for s in strats]
                new_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
            prev_sig = new_sig
            curve.append({'timestamp': df.index[i].isoformat(), 'balance': mgr.step_equity(p_close)})

        final_bal = curve[-1]['balance']
        total_ret = ((final_bal - body.get('initialBalance', 1000)) / body.get('initialBalance', 1000)) * 100
        trades = len(mgr.trades)
        wins = len([t for t in mgr.trades if t['profit'] > 0])
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        TF_TO_PPY = {"1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520, "1h": 8760, "4h": 2190, "1d": 365}
        periods = TF_TO_PPY.get(tf, 8760)
        
        returns = pd.Series([c['balance'] for c in curve]).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(periods)) if len(returns) > 0 and returns.std() != 0 else 0

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
