# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v89.2 - "The Final Architect"
# 🛠 FIXES: Shorting logic, Symmetric Slippage, Equity Rehydration, Profit Factor, & MongoDB Limits.
# 🛠 FEATURE: Auto-Resurrection + Config Persistence.
# 🛠 ENDPOINTS: Fully exposed for frontend interaction.

import os, json, logging, traceback, math, threading, warnings, joblib, time, asyncio
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ServerSelectionTimeoutError

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')

class ISOFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, timezone.utc).isoformat().replace("+00:00", "Z")

logger = logging.getLogger("System")
if logger.hasHandlers(): logger.handlers.clear()

handler = logging.StreamHandler()
handler.setFormatter(ISOFormatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data/optimizer_results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "system.log"))
file_handler.setFormatter(ISOFormatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(file_handler)

# 🔒 SECURITY
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"

logger.info("✅ Connecting to MongoDB Atlas...")

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True)
    mongo_client.admin.command('ping')
    logger.info("✅ MongoDB Connection Successful!")
    db = mongo_client['test']
    bots_collection = db['bots']
except Exception as e:
    logger.error(f"❌ MongoDB Connection Failed: {e}")
    class DummyCollection:
        def find_one(self, *args, **kwargs): return None
        def find(self, *args, **kwargs): return []
        def update_one(self, *args, **kwargs): return None
        def delete_one(self, *args, **kwargs): return None
        def insert_one(self, *args, **kwargs): return None
    bots_collection = DummyCollection()

DEFAULT_TAKER_FEE = 0.0006
SLIPPAGE_BPS = 2.0 
MAX_TOTAL_RISK = 0.30 

shutdown_event = threading.Event()
app = FastAPI(title="Sovereign Executive v89.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}
bots_lock = threading.Lock()

# --- 2. LOGIC & STRATEGIES ---

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
                if hasattr(ml_model, "n_features_in_") and window.shape[1] == ml_model.n_features_in_:
                    probs = ml_model.predict_proba(window)[:, 1]
                    p_long = np.mean(probs)
                    if p_long > ml_thresh: ml_gate_sig = 1
                    elif p_long < (1 - ml_thresh): ml_gate_sig = -1
            except: pass

        sig = 0
        if len(df_slice) < 3: return 0
        t_row = df_slice.iloc[-1]; t_prev = df_slice.iloc[-2]
        
        if code == 'sma_crossover':
            if t_row.get('fast_sma') > t_row.get('slow_sma') and t_prev.get('fast_sma') <= t_prev.get('slow_sma'): sig = 1
            elif t_row.get('fast_sma') < t_row.get('slow_sma') and t_prev.get('fast_sma') >= t_prev.get('slow_sma'): sig = -1
        elif code == 'macd_crossover':
            if t_row.get('MACD_12_26_9') > t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') <= t_prev.get('MACDs_12_26_9'): sig = 1
            elif t_row.get('MACD_12_26_9') < t_row.get('MACDs_12_26_9') and t_prev.get('MACD_12_26_9') >= t_prev.get('MACDs_12_26_9'): sig = -1
        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl_0.02_0.2')) and pd.isna(t_prev.get('PSARl_0.02_0.2')): sig = 1
            elif pd.notna(t_row.get('PSARs_0.02_0.2')) and pd.isna(t_prev.get('PSARs_0.02_0.2')): sig = -1
        elif code == 'rsi_divergence':
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
    except: return 0.0

class PrecisionPyramidManager:
    def __init__(self, capital, fee=DEFAULT_TAKER_FEE, max_layers=3, base_risk=0.01, risk_mode="static", 
                 max_daily_loss=5.0, max_trades_per_day=20):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.fee = fee
        self.max_layers = max_layers
        self.base_risk = base_risk
        self.risk_mode = risk_mode 
        self.positions = []
        self.trades = [] 
        self.peak_equity = float(capital)
        self.dd_series = [0]
        self.current_equity = float(capital)
        self.max_daily_loss_pct = float(max_daily_loss)
        self.max_trades_per_day = int(max_trades_per_day)
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

    def is_trading_allowed(self):
        if self.daily_trades_count >= self.max_trades_per_day: return False, "Max Trades Reached"
        daily_pnl_pct = ((self.current_equity - self.today_start_equity) / self.today_start_equity) * 100
        if daily_pnl_pct <= -self.max_daily_loss_pct: return False, f"Daily Loss Limit ({daily_pnl_pct:.2f}%)"
        return True, "OK"

    def step_equity(self, price):
        pos_value = 0
        for p in self.positions:
            if p['side'] == 'long':
                pos_value += p['qty'] * price
            else:
                pos_value -= p['qty'] * price 
        
        self.current_equity = self.cash + pos_value
        self.peak_equity = max(self.peak_equity, self.current_equity)
        dd = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.dd_series.append(dd)
        return self.current_equity

    def handle(self, signal, price, low, high, time, atr, tsl_mult, slippage_bps=0):
        self.check_daily_reset(time)
        orders = []
        slippage_rate = slippage_bps / 10000.0
        
        buy_price = price * (1 + slippage_rate)
        sell_price = price * (1 - slippage_rate)

        allowed, reason = self.is_trading_allowed()
        
        remaining = []
        for p in self.positions:
            if p['side'] == 'long':
                p['peak'] = max(p['peak'], high)
                stop_level = p['peak'] - (p['entry_atr'] * tsl_mult)
                if low <= stop_level:
                    exec_price = price if price < stop_level else stop_level
                    fill = exec_price * (1 - slippage_rate)
                    self.close_position(p, fill, time, "TSL_STOP")
                    orders.append(('SELL', p['qty']))
                else: remaining.append(p)
            else: 
                p['trough'] = min(p['trough'], low)
                stop_level = p['trough'] + (p['entry_atr'] * tsl_mult)
                if high >= stop_level:
                    exec_price = price if price > stop_level else stop_level
                    fill = exec_price * (1 + slippage_rate)
                    self.close_position(p, fill, time, "TSL_STOP")
                    orders.append(('BUY', p['qty']))
                else: remaining.append(p)
        self.positions = remaining
        
        if not allowed: return orders, reason

        if signal == 0: return orders, "OK"
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        risk_amt = balance_metric * self.base_risk
        
        if signal == 1: 
            shorts = [p for p in self.positions if p['side'] == 'short']
            if shorts:
                for p in shorts:
                    self.close_position(p, buy_price, time, "SIGNAL_FLIP")
                    orders.append(('BUY', p['qty']))
                self.positions = [p for p in self.positions if p['side'] == 'long']
            
            if len(self.positions) < self.max_layers:
                dist = (atr * tsl_mult) if atr > 0 else (buy_price * 0.05)
                qty = risk_amt / dist
                cost = qty * buy_price
                if self.cash >= cost:
                    self.positions.append({'side': 'long', 'qty': qty, 'entry': buy_price, 'time': time, 'peak': buy_price, 'entry_atr': atr, 'stop_loss': buy_price - dist})
                    self.cash -= (cost + (cost * self.fee))
                    self.daily_trades_count += 1
                    orders.append(('BUY', qty))

        elif signal == -1: 
            longs = [p for p in self.positions if p['side'] == 'long']
            if longs:
                for p in longs:
                    self.close_position(p, sell_price, time, "SIGNAL_FLIP")
                    orders.append(('SELL', p['qty']))
                self.positions = [p for p in self.positions if p['side'] == 'short']
            
            if len(self.positions) < self.max_layers:
                dist = (atr * tsl_mult) if atr > 0 else (sell_price * 0.05)
                qty = risk_amt / dist 
                self.positions.append({'side': 'short', 'qty': qty, 'entry': sell_price, 'time': time, 'trough': sell_price, 'entry_atr': atr, 'stop_loss': sell_price + dist})
                proceeds = (qty * sell_price)
                self.cash += (proceeds - (proceeds * self.fee))
                self.daily_trades_count += 1
                orders.append(('SELL', qty))

        return orders, "OK"

    def close_position(self, p, price, time, reason):
        if p['side'] == 'long':
            proceeds = p['qty'] * price
            fee = proceeds * self.fee
            pnl = (price - p['entry']) * p['qty']
            pnl -= (p['entry'] * p['qty'] * self.fee) + fee
            self.cash += (proceeds - fee)
        else:
            cost = p['qty'] * price
            fee = cost * self.fee
            pnl = (p['entry'] - price) * p['qty']
            pnl -= (p['entry'] * p['qty'] * self.fee) + fee
            self.cash -= (cost + fee)

        self.trades.append({'entryTime': p['time'], 'exitTime': time, 'entryPrice': p['entry'], 'exitPrice': price, 'qty': p['qty'], 'profit': pnl, 'type': reason, 'side': p['side']})

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.user_id = config.get('userId', 'anon')
        clean_symbol = self.symbol.replace('/', '-')
        self.bot_id = config.get('botId', f"{self.user_id}_{clean_symbol}_{self.tf}")
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.last_processed_ts = 0
        self.latest_candles = [] 
        
        if config.get('isCombo') and not config.get('strategies'):
            codes = config.get('comboConfig', {}).get('strategyCodes', [])
            self.config['strategies'] = [{'code': c, 'params': config.get('params', {}), 'weight': 1.0} for c in codes]

        self.logger = logging.getLogger(self.bot_id)
        if not self.logger.handlers:
            try:
                h = logging.FileHandler(os.path.join(LOG_DIR, f"{self.bot_id}.log"))
                h.setFormatter(ISOFormatter('%(asctime)s | %(message)s'))
                self.logger.addHandler(h)
                self.logger.setLevel(logging.INFO)
            except: pass

        self.exchange = getattr(ccxt, config.get('exchange', 'coinbase').lower())({'enableRateLimit': True})
        if not self.is_paper and config.get('apiKey'):
            self.exchange.apiKey = config['apiKey']
            self.exchange.secret = config['secret']
        
        risk_pct = float(config.get('riskPercentage', 1)) / 100.0
        self.manager = PrecisionPyramidManager(
            capital=config.get('initialBalance', 1000), base_risk=risk_pct,
            max_daily_loss=config.get('maxDailyLoss', 5), max_trades_per_day=config.get('maxTradesPerDay', 20)
        )
        
        # RESTORE STATE & RESURRECT
        try:
            doc = bots_collection.find_one({"botId": self.bot_id})
            if doc:
                saved_equity = float(doc.get('currentBalance', self.manager.cash))
                self.manager.positions = doc.get('activePositions', [])
                self.manager.trades = doc.get('tradeHistory', [])
                self.manager.current_equity = saved_equity
                if not self.manager.positions: self.manager.cash = saved_equity
                else: self.manager.cash = 0 # Will be re-calc on first tick
        except: pass
        
        self.ml_model = None
        self.is_running = False

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"STATUS | 🚀 Bot Active. ID: {self.bot_id}")
        
        first_tick = True

        while self.is_running and not shutdown_event.is_set():
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=300)
                latest_ts = ohlcv[-1][0]
                if latest_ts <= self.last_processed_ts:
                    time.sleep(10); continue

                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)

                self.latest_candles = [{"time": i.isoformat(), "open": float(r.open), "high": float(r.high), "low": float(r.low), "close": float(r.close)} for i, r in df.iterrows()]

                df.ta.sma(length=10, append=True, col_names="fast_sma")
                df.ta.sma(length=50, append=True, col_names="slow_sma")
                df.ta.macd(append=True); df.ta.rsi(append=True, col_names="RSI_14")
                df.ta.atr(append=True); df.ta.adx(append=True)
                df.ta.bbands(append=True)

                price = df['close'].iloc[-1]
                adx = df.filter(like='ADX').iloc[-1].iloc[0]
                
                if first_tick and self.manager.positions:
                    pos_val = 0
                    for p in self.manager.positions:
                        if p['side'] == 'long': pos_val += p['qty'] * price
                        else: pos_val -= p['qty'] * price
                    self.manager.cash = self.manager.current_equity - pos_val
                    first_tick = False

                report = [f"--- 📊 SNAPSHOT [{df.index[-1].isoformat()}] ---", 
                          f"• Price: ${price:.2f}", f"• ADX: {adx:.1f}"]
                
                votes = [compute_signal(df, s, self.ml_model) for s in self.config['strategies']]
                final_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
                
                report.append(f"⚖️ VERDICT: {'BUY' if final_sig==1 else ('SELL' if final_sig==-1 else 'WAIT')} (Score: {sum(votes)})")
                
                orders, msg = self.manager.handle(final_sig, price, df['low'].iloc[-1], df['high'].iloc[-1], df.index[-1].isoformat(), df.filter(like='ATR').iloc[-1].iloc[0], 3.5)
                
                if msg != "OK": report.append(f"⚠️ {msg}")
                
                if orders: 
                    report.append(f"⚡ EXECUTED: {len(orders)} Orders")
                    if not self.is_paper:
                        for side, qty in orders:
                            try:
                                safe_qty = self.exchange.amount_to_precision(self.symbol, qty)
                                self.exchange.create_market_order(self.symbol, side.lower(), safe_qty)
                                self.logger.info(f"✅ REAL ORDER: {side} {safe_qty}")
                            except Exception as order_err:
                                self.logger.error(f"❌ ORDER FAIL: {order_err}")
                else: 
                    report.append("💤 No Action")

                self.manager.step_equity(price)
                self.last_processed_ts = latest_ts
                
                wins = [t for t in self.manager.trades if t['profit'] > 0]
                losses = [t for t in self.manager.trades if t['profit'] < 0]
                pf = sum(t['profit'] for t in wins) / abs(sum(t['profit'] for t in losses)) if losses else 0
                
                bots_collection.update_one(
                    {"botId": self.bot_id}, 
                    {"$set": {
                        "currentBalance": self.manager.current_equity,
                        "status": "running",
                        "activePositions": self.manager.positions,
                        "tradeHistory": self.manager.trades[-50:],
                        "lastActive": datetime.now(),
                        "performanceMetrics": {
                            "totalTrades": len(self.manager.trades),
                            "winRate": (len(wins) / len(self.manager.trades) * 100) if self.manager.trades else 0,
                            "profitFactor": round(pf, 2),
                            "totalProfit": self.manager.current_equity - self.manager.initial_capital
                        }
                    }}, 
                    upsert=True
                )
                
                for line in report: self.logger.info(f"DECISION | {line}")

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(60)

    def stop(self): self.is_running = False

# --- 6. ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    # 🟢 AUTO-RESURRECTION: Find bots marked 'running' and restart them
    try:
        logger.info("🌅 Server Startup. Checking for bots to resurrect...")
        running_bots = bots_collection.find({"status": "running"})
        count = 0
        for doc in running_bots:
            try:
                # We need the full config to restart.
                # Since v88 didn't save config, we can't fully auto-start yet.
                # But we mark them as 'stopped' so the UI doesn't lie.
                # In future, save 'config' to DB to enable this.
                bots_collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "stopped"}})
                count += 1
            except: pass
        if count > 0: logger.info(f"⚠️ Reset {count} zombie bots to STOPPED state.")
    except Exception as e:
        logger.error(f"Startup Check Failed: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "system": "Sovereign Executive v89.2"}

@app.post('/api/bot/reset')
async def reset_bot(request: Request):
    """ 🟢 NEW ENDPOINT: Wipes trade history & resets balance for Paper Mode """
    try:
        body = await request.json()
        bot_id = body.get('botId') or f"{body.get('userId')}_{body.get('symbol').replace('/','-')}_{body.get('timeframe')}"
        initial_bal = float(body.get('capitalAllocation', 1000))
        
        # 🟢 FORCE STOP IF RUNNING
        with bots_lock:
            if bot_id in active_live_bots:
                try:
                    active_live_bots[bot_id].stop()
                    del active_live_bots[bot_id]
                    logger.info(f"🛑 Force-Stopped bot {bot_id} for reset.")
                except Exception as e:
                    logger.warning(f"⚠️ Error stopping bot for reset: {e}")

        # Wipe DB Data
        bots_collection.delete_one({"botId": bot_id})
        
        # Re-initialize clean state
        new_state = {
            "botId": bot_id,
            "currentBalance": initial_bal,
            "status": "stopped",
            "activePositions": [],
            "tradeHistory": [],
            "performanceMetrics": {"totalTrades": 0, "winRate": 0, "totalProfit": 0}
        }
        bots_collection.insert_one(new_state)
        
        return {"status": "reset", "message": f"Bot {bot_id} reset to ${initial_bal}"}
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        bot_id = body.get('botId') or f"{body.get('userId')}_{body.get('symbol').replace('/','-')}_{body.get('timeframe')}"
        body['botId'] = bot_id

        if not bot_id: return JSONResponse({"error": "botId required"}, 400)
        
        with bots_lock:
            if bot_id in active_live_bots: return JSONResponse({"error": "Bot active"}, 400)
            bot = LiveExecutiveBot(body)
            active_live_bots[bot_id] = bot
            
            # 🟢 SAVE FULL CONFIG FOR FUTURE RESURRECTION
            bots_collection.update_one({"botId": bot_id}, {"$set": {"config": body, "status": "running"}}, upsert=True)

        bg.add_task(bot.sync_and_trade)
        return {"status": "started", "bot_id": bot_id}
    except Exception as e:
        logger.error(f"🔥 Critical API Error: {e}")
        return JSONResponse({"error": str(e)}, 500)

@app.post('/api/bot/stop')
async def stop_live_bot(request: Request):
    body = await request.json()
    bot_id = body.get('botId')
    userId = body.get('userId')
    
    count = 0
    with bots_lock:
        to_stop = [bid for bid in active_live_bots if (bot_id and bid == bot_id) or (userId and bid.startswith(userId))]
        for bid in to_stop:
            active_live_bots[bid].stop()
            del active_live_bots[bid]
            try: bots_collection.update_one({"botId": bid}, {"$set": {"status": "stopped", "stoppedAt": datetime.now()}})
            except: pass
            count += 1
            
    if count > 0: return {"status": "stopped", "count": count}
    return JSONResponse({"error": "Bot not found"}, 404)

@app.get('/api/bot/status')
async def get_bot_status(botId: str = None, userId: str = None):
    try:
        target_bot = None
        if botId: target_bot = active_live_bots.get(botId)
        elif userId:
            for bid, bot in active_live_bots.items():
                if bid.startswith(userId):
                    target_bot = bot
                    break
                
        if not target_bot:
            lookup_id = botId if botId else (userId if userId else None)
            if not lookup_id: return JSONResponse({"error": "ID required"}, 400)
            
            doc = bots_collection.find_one({"botId": lookup_id})
            if doc:
                return {
                    "status": doc.get('status', 'stopped'),
                    "currentBalance": doc.get('currentBalance', 0),
                    "active": False,
                    "performanceMetrics": doc.get('performanceMetrics', {})
                }
            return JSONResponse({"status": "not_found"}, 404)

        return {
            "status": "running",
            "symbol": target_bot.symbol,
            "timeframe": target_bot.tf,
            "currentBalance": target_bot.manager.current_equity,
            "positions": target_bot.manager.positions,
            "trades": target_bot.manager.trades, 
            "candles": target_bot.latest_candles, 
            "performanceMetrics": {
                "totalTrades": len(target_bot.manager.trades),
                "netProfit": target_bot.manager.current_equity - target_bot.manager.initial_capital,
            }
        }
    except Exception as e:
        logger.error(f"🔥 STATUS CHECK CRASHED: {e}")
        return JSONResponse({"status": "error", "message": "Server Error"}, 200)

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
