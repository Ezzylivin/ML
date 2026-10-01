# File: /root/Project/ML/diamond0.py
# 🚀 UPGRADE: v89.19 - "The Telepath"
# 🛠 FEATURE: "Thinking" Engine - Logs detailed reasons for every signal (Values, Crosses, Logic).
# 🛠 FEATURE: Verbose Risk Manager - Explains exactly why a trade was blocked.
# 🛠 INCLUDES: All previous safety, ML, and loop fixes.

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

# StreamHandler ensures logs appear in your Server Console
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

MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"
if not MONGO_URI:
    # Fail-safe allows import, runtime checks apply later
    pass 

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
app = FastAPI(title="Sovereign Executive v89.19")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}
bots_lock = threading.Lock()

# --- 2. HELPERS ---

def normalize_positions(positions):
    """Safely converts numpy types to native Python types for MongoDB"""
    out = []
    for p in positions:
        out.append({
            "side": p["side"],
            "qty": float(p["qty"]),
            "entry": float(p["entry"]),
            "time": p["time"],
            "peak": float(p.get("peak", p["entry"])),
            "trough": float(p.get("trough", p["entry"])),
            "entry_atr": float(p.get("entry_atr", 0)),
            "stop_loss": float(p.get("stop_loss", 0))
        })
    return out

# --- 3. LOGIC & STRATEGIES (The "Brain") ---

def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    """
    Returns: (signal, explanation_string)
    """
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        weight = float(strategy_conf.get('weight', 1.0))
        ml_confirm = strategy_conf.get('mlConfirm', True)
        def get_p(key, default): return float(p.get(key, default))

        ml_gate_sig = 0
        ml_reason = ""
        
        if ml_model is not None and ml_confirm:
            try:
                window = df_slice.tail(5).select_dtypes(include=[np.number])
                
                if hasattr(ml_model, "feature_names_in_"):
                    window = window.reindex(ml_model.feature_names_in_, axis=1)

                if hasattr(ml_model, "n_features_in_") and window.shape[1] == ml_model.n_features_in_:
                    probs = ml_model.predict_proba(window)[:, 1]
                    p_long = np.mean(probs)
                    if p_long > ml_thresh: 
                        ml_gate_sig = 1
                        ml_reason = f"ML Confirms BUY ({p_long:.2f} > {ml_thresh})"
                    elif p_long < (1 - ml_thresh): 
                        ml_gate_sig = -1
                        ml_reason = f"ML Confirms SELL ({p_long:.2f} < {1-ml_thresh:.2f})"
                    else:
                        ml_reason = f"ML Neutral ({p_long:.2f})"
                else:
                    ml_gate_sig = 0 
            except Exception as e:
                ml_gate_sig = 0
                ml_reason = "ML Error/Mismatch"

        sig = 0
        reason = "Neutral"
        
        if len(df_slice) < 3: return 0, "Insufficient Data"
        t_row = df_slice.iloc[-1]; t_prev = df_slice.iloc[-2]
        
        # --- STRATEGY LOGIC WITH REASONING ---
        if code == 'sma_crossover':
            fast, slow = t_row.get('fast_sma'), t_row.get('slow_sma')
            if fast > slow and t_prev.get('fast_sma') <= t_prev.get('slow_sma'): 
                sig = 1; reason = f"Golden Cross (Fast {fast:.2f} > Slow {slow:.2f})"
            elif fast < slow and t_prev.get('fast_sma') >= t_prev.get('slow_sma'): 
                sig = -1; reason = f"Death Cross (Fast {fast:.2f} < Slow {slow:.2f})"
        
        elif code == 'macd_crossover':
            macd, signal_line = t_row.get('MACD_12_26_9'), t_row.get('MACDs_12_26_9')
            if macd > signal_line and t_prev.get('MACD_12_26_9') <= t_prev.get('MACDs_12_26_9'): 
                sig = 1; reason = f"MACD Bullish Cross ({macd:.2f} > {signal_line:.2f})"
            elif macd < signal_line and t_prev.get('MACD_12_26_9') >= t_prev.get('MACDs_12_26_9'): 
                sig = -1; reason = f"MACD Bearish Cross ({macd:.2f} < {signal_line:.2f})"

        elif code == 'rsi_divergence':
            rsi = t_row.get('RSI_14', 50)
            low_level, high_level = get_p('oversold_level', 30), get_p('overbought_level', 70)
            if rsi < low_level: 
                sig = 1; reason = f"RSI Oversold ({rsi:.2f} < {low_level})"
            elif rsi > high_level: 
                sig = -1; reason = f"RSI Overbought ({rsi:.2f} > {high_level})"

        elif code == 'bollinger_bands':
            close = t_row.get('close')
            if close < t_row.get('BBL_20_2.0'): 
                sig = 1; reason = f"Price ({close:.2f}) Below Lower Band"
            elif close > t_row.get('BBU_20_2.0'): 
                sig = -1; reason = f"Price ({close:.2f}) Above Upper Band"

        elif code == 'atr_breakout':
            close = t_row.get('close'); prev = t_prev.get('close')
            mult = get_p('atr_multiplier', 3.0); ema = t_row.get('EMA_20', close); atr = t_row.get('ATR_14', 0)
            upper = ema + atr*mult; lower = ema - atr*mult
            if close > upper and prev <= upper: 
                sig = 1; reason = f"ATR Breakout UP (Price {close:.2f} > {upper:.2f})"
            elif close < lower and prev >= lower: 
                sig = -1; reason = f"ATR Breakout DOWN (Price {close:.2f} < {lower:.2f})"

        elif code == 'stochastic_crossover':
            k, d = t_row.get('STOCHk_14_3_3'), t_row.get('STOCHd_14_3_3')
            pk, pd_val = t_prev.get('STOCHk_14_3_3'), t_prev.get('STOCHd_14_3_3')
            if k > d and pk <= pd_val and k < 20: 
                sig = 1; reason = f"Stoch Cross UP in Oversold (K={k:.2f})"
            elif k < d and pk >= pd_val and k > 80: 
                sig = -1; reason = f"Stoch Cross DOWN in Overbought (K={k:.2f})"

        elif code == 'cci_oversold':
            cci = t_row.get('CCI_14_0.015')
            if cci < -100: sig = 1; reason = f"CCI Oversold ({cci:.2f} < -100)"
            elif cci > 100: sig = -1; reason = f"CCI Overbought ({cci:.2f} > 100)"

        elif code == 'ichimoku_system':
            tk, kj = t_row.get('ITS_9'), t_row.get('IKS_26')
            ptk, pkj = t_prev.get('ITS_9'), t_prev.get('IKS_26')
            if tk > kj and ptk <= pkj: sig = 1; reason = "Ichimoku TK Cross Bullish"
            elif tk < kj and ptk >= pkj: sig = -1; reason = "Ichimoku TK Cross Bearish"

        elif code == 'obv_signal':
            if t_row.get('OBV') > t_prev.get('OBV'): sig = 1; reason = "OBV Increasing"
            elif t_row.get('OBV') < t_prev.get('OBV'): sig = -1; reason = "OBV Decreasing"

        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl_0.02_0.2')) and pd.isna(t_prev.get('PSARl_0.02_0.2')): 
                sig = 1; reason = "PSAR Flip to Bullish"
            elif pd.notna(t_row.get('PSARs_0.02_0.2')) and pd.isna(t_prev.get('PSARs_0.02_0.2')): 
                sig = -1; reason = "PSAR Flip to Bearish"

        # --- ML GATE LOGIC ---
        if ml_gate_sig == 0: 
            return float(sig * weight), reason
        
        if sig == ml_gate_sig: 
            return float(sig * weight), f"{reason} + {ml_reason}"
        
        if sig != 0:
            return 0.0, f"🚫 ML BLOCKED: Strategy says {sig} ({reason}) but {ml_reason}"
            
        return 0.0, reason
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

# --- 4. RISK MANAGER (The "Muscle") ---

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
        
        # 1. Manage Existing Positions (Stops/TP)
        remaining = []
        for p in self.positions:
            if p['side'] == 'long':
                p['peak'] = max(p['peak'], high)
                stop_level = p['peak'] - (p['entry_atr'] * tsl_mult)
                if low <= stop_level:
                    exec_price = stop_level 
                    fill = exec_price * (1 - slippage_rate)
                    self.close_position(p, fill, time, "TSL_STOP")
                    orders.append(('SELL', p['qty']))
                else: remaining.append(p)
            else: 
                p['trough'] = min(p['trough'], low)
                stop_level = p['trough'] + (p['entry_atr'] * tsl_mult)
                if high >= stop_level:
                    exec_price = stop_level 
                    fill = exec_price * (1 + slippage_rate)
                    self.close_position(p, fill, time, "TSL_STOP")
                    orders.append(('BUY', p['qty']))
                else: remaining.append(p)
        self.positions = remaining
        
        if not allowed: return orders, f"🚫 Blocked: {reason}"

        if signal == 0: return orders, "OK"
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        risk_amt = balance_metric * self.base_risk
        
        # 2. Execute New Signals
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
            else:
                return orders, "🚫 Blocked: Max Pyramiding Reached"

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
            else:
                return orders, "🚫 Blocked: Max Pyramiding Reached"

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

# --- 5. LIVE BOT ---

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.user_id = config.get('userId', 'anon')
        clean_symbol = self.symbol.replace('/', '-')
        self.bot_id = config.get('botId', f"{self.user_id}_{clean_symbol}_{self.tf}")
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.force_close = bool(config.get("forceCloseOnStop", False)) 
        self.last_processed_ts = 0
        self.latest_candles = [] 
        self._db_tick = 0
        
        if config.get('isCombo') and not config.get('strategies'):
            codes = config.get('comboConfig', {}).get('strategyCodes', [])
            seen = set()
            unique_codes = [x for x in codes if not (x in seen or seen.add(x))]
            self.config['strategies'] = [{'code': c, 'params': config.get('params', {}), 'weight': 1.0} for c in unique_codes]

        self.logger = logging.getLogger(self.bot_id)
        if not self.logger.handlers:
            try:
                # 1. File Handler (Dashboard)
                fh = logging.FileHandler(os.path.join(LOG_DIR, f"{self.bot_id}.log"))
                fh.setFormatter(ISOFormatter('%(asctime)s | %(message)s'))
                self.logger.addHandler(fh)
                # 2. Stream Handler (Console)
                sh = logging.StreamHandler()
                sh.setFormatter(ISOFormatter('%(asctime)s | %(levelname)s | %(message)s'))
                self.logger.addHandler(sh)
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
        
        # RESTORE STATE
        try:
            doc = bots_collection.find_one({"botId": self.bot_id})
            if doc:
                saved_equity = float(doc.get('currentBalance', self.manager.cash))
                self.manager.positions = doc.get('activePositions', [])
                self.manager.trades = doc.get('tradeHistory', [])
                self.manager.current_equity = saved_equity
                self.manager.cash = float(doc.get("cash", saved_equity))
        except: pass
        
        # ML Loading
        self.ml_model = None
        ml_config_name = self.config.get('mlModel')
        if ml_config_name:
            try:
                model_path = os.path.join(MODEL_DIR, f"{ml_config_name}.joblib")
                if os.path.exists(model_path):
                    self.ml_model = joblib.load(model_path)
                    self.logger.info(f"🤖 ML ONLINE: Loaded {ml_config_name}")
                else:
                    self.logger.warning(f"⚠️ ML MISSING: Could not find {ml_config_name} at {model_path}")
            except Exception as e:
                self.logger.error(f"❌ ML LOAD ERROR: {e}")
        else:
            self.logger.info("ℹ️ ML OFF: No model configured")

        self.is_running = False

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"STATUS | 🚀 Bot Active. ID: {self.bot_id}")
        self.logger.info(f"STATUS | 🔄 Thread Started") 

        first_tick = True

        while self.is_running and not shutdown_event.is_set():
            try:
                # 🚀 REALITY CHECK
                market_price = None
                try:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    market_price = ticker['last']
                    server_ts = self.exchange.fetch_time()
                    local_ts = int(time.time() * 1000)
                    skew = abs(local_ts - server_ts)
                    if skew > 5000: 
                        self.logger.warning(f"⚠️ CLOCK SKEW: Your server is {skew}ms out of sync with Exchange.")
                except Exception as tick_err:
                    self.logger.warning(f"⚠️ Reality Check Failed: {tick_err}")
                
                try:
                    ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=300)
                except Exception as api_err:
                    self.logger.error(f"⚠️ API ERROR: {api_err}")
                    time.sleep(5) 
                    continue

                if not ohlcv or len(ohlcv) < 2:
                    self.logger.warning("⚠️ Empty Candle Data")
                    time.sleep(5)
                    continue

                processed_ts = ohlcv[-2][0]
                latest_ts_open = ohlcv[-1][0] 

                # 🚀 PRICE DRIFT CHECK
                if market_price:
                    candle_close = ohlcv[-1][4]
                    drift = abs(market_price - candle_close) / market_price * 100
                    if drift > 0.5: 
                        self.logger.warning(f"⚠️ DATA MISMATCH: Candle=${candle_close}, Ticker=${market_price} (Diff: {drift:.2f}%)")

                # 🛑 STALE DATA DETECTOR
                candle_time = datetime.fromtimestamp(latest_ts_open / 1000, timezone.utc)
                data_age = (datetime.now(timezone.utc) - candle_time).total_seconds()
                
                if data_age > 7200: 
                    self.logger.error(f"🛑 STALE DATA: Exchange Data is {int(data_age/60)} mins old. Halting.")
                    time.sleep(30)
                    continue

                if processed_ts <= self.last_processed_ts:
                    # 🛠 HEARTBEAT LOG (every ~60s)
                    if datetime.now().second < 10: 
                         self.logger.info(f"⏳ MONITORING | Price: ${ohlcv[-1][4]}")
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
                
                if self.ml_model:
                    report.append(f"• ML Status: ACTIVE ({type(self.ml_model).__name__})")
                else:
                    report.append("• ML Status: INACTIVE")

                # 🧠 THOUGHT ENGINE: Collect Reasons
                votes = []
                thoughts = []
                for s in self.config['strategies']:
                    sig, reason = compute_signal(df, s, self.ml_model)
                    votes.append(sig)
                    if sig != 0:
                        thoughts.append(f"{s['code'].upper()}: {reason}")
                
                if thoughts:
                    report.append(f"🧠 THOUGHTS: {'; '.join(thoughts)}")

                min_votes = int(self.config.get('comboConfig', {}).get('minVotesRequired', 1))
                buy_votes = sum(1 for v in votes if v > 0)
                sell_votes = sum(1 for v in votes if v < 0)
                
                final_sig = 0
                if buy_votes >= min_votes and sell_votes >= min_votes:
                    final_sig = 0 
                elif buy_votes >= min_votes:
                    final_sig = 1
                elif sell_votes >= min_votes:
                    final_sig = -1
                
                report.append(f"⚖️ VERDICT: {'BUY' if final_sig==1 else ('SELL' if final_sig==-1 else 'WAIT')} (Buys: {buy_votes}, Sells: {sell_votes}, Req: {min_votes})")
                
                orders, msg = self.manager.handle(final_sig, price, df['low'].iloc[-1], df['high'].iloc[-1], df.index[-1].isoformat(), df.filter(like='ATR').iloc[-1].iloc[0], 3.5)
                
                if msg != "OK": report.append(f"⚠️ {msg}")
                
                trade_happened = False
                if orders: 
                    report.append(f"⚡ EXECUTED: {len(orders)} Orders")
                    trade_happened = True
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
                self.last_processed_ts = processed_ts
                
                wins = [t for t in self.manager.trades if t['profit'] > 0]
                losses = [t for t in self.manager.trades if t['profit'] < 0]
                pf = sum(t['profit'] for t in wins) / abs(sum(t['profit'] for t in losses)) if losses else 0
                
                self._db_tick += 1
                
                # 🚀 BALANCE RESYNC
                if not self.is_paper and self._db_tick % 10 == 0:
                    try:
                        bal = self.exchange.fetch_balance()
                        quote_currency = self.symbol.split('/')[1] 
                        real_cash = bal.get(quote_currency, {}).get('free', 0)
                        if abs(real_cash - self.manager.cash) > (self.manager.cash * 0.01):
                            self.logger.info(f"💰 BALANCE SYNC: Internal=${self.manager.cash:.2f} -> Real=${real_cash:.2f}")
                            self.manager.cash = real_cash
                    except Exception as bal_err:
                        self.logger.warning(f"⚠️ Balance Sync Failed: {bal_err}")

                if trade_happened or self._db_tick % 3 == 0:
                    bots_collection.update_one(
                        {"botId": self.bot_id}, 
                        {"$set": {
                            "config": self.config, 
                            "currentBalance": self.manager.current_equity,
                            "status": "running",
                            "activePositions": normalize_positions(self.manager.positions),
                            "tradeHistory": self.manager.trades[-50:],
                            "lastActive": datetime.now(),
                            "performanceMetrics": {
                                "totalTrades": len(self.manager.trades),
                                "winRate": (len(wins) / len(self.manager.trades) * 100) if self.manager.trades else 0,
                                "profitFactor": round(pf, 2),
                                "totalProfit": self.manager.current_equity - self.manager.initial_capital
                            },
                            "stoppedAt": None,
                            "cash": self.manager.cash 
                        }}, 
                        upsert=True
                    )
                
                for line in report: self.logger.info(f"DECISION | {line}")

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(5) 
        
        self.logger.info(f"STATUS | 🛑 Thread Exiting. ID: {self.bot_id}") 

    def stop(self):
        self.is_running = False
        if self.force_close:
            self.manager.positions.clear()
            
        bots_collection.update_one(
            {"botId": self.bot_id},
            {"$set": {
                "status": "stopped",
                "stoppedAt": datetime.now(timezone.utc),
                "activePositions": normalize_positions(self.manager.positions),
                "cash": self.manager.cash,
                "currentBalance": self.manager.current_equity
            }}
        )

# --- 6. ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🌅 Server Startup. Resurrecting bots...")
        with bots_lock:
            running_bots = bots_collection.find({"status": "running"})
            count = 0
            for doc in running_bots:
                try:
                    if 'config' in doc:
                        bot_id = doc['botId']
                        if bot_id not in active_live_bots:
                            logger.info(f"⚡ Resurrecting Bot: {bot_id}")
                            
                            bot_logger = logging.getLogger(bot_id)
                            fh = logging.FileHandler(os.path.join(LOG_DIR, f"{bot_id}.log"))
                            fh.setFormatter(ISOFormatter('%(asctime)s | %(message)s'))
                            bot_logger.addHandler(fh)
                            
                            sh = logging.StreamHandler()
                            sh.setFormatter(ISOFormatter('%(asctime)s | %(message)s'))
                            bot_logger.addHandler(sh)
                            
                            bot_logger.info("INFO | ♻️ SYSTEM RECOVERY: Bot Resurrected after server restart.")
                            
                            bots_collection.update_one({"botId": bot_id}, {"$unset": {"stoppedAt": ""}})

                            bot = LiveExecutiveBot(doc['config'])
                            active_live_bots[bot_id] = bot
                            t = threading.Thread(target=bot.sync_and_trade)
                            t.daemon = True
                            t.start()
                            count += 1
                    else:
                        bots_collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "stopped"}})
                except Exception as e:
                    logger.error(f"Failed to resurrect {doc.get('botId')}: {e}")
                    try:
                        f_log = os.path.join(LOG_DIR, f"{doc.get('botId')}.log")
                        with open(f_log, "a") as f:
                            f.write(f"{datetime.now(timezone.utc).isoformat()} | ERROR | ☠️ CRITICAL: Resurrection Failed: {e}\n")
                    except: pass
        
        if count > 0: logger.info(f"✅ Resurrected {count} bots.")
    except Exception as e:
        logger.error(f"Startup Check Failed: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "system": "Sovereign Executive v89.19"}

@app.post('/api/bot/reset')
async def reset_bot(request: Request):
    try:
        body = await request.json()
        bot_id = body.get('botId') or f"{body.get('userId')}_{body.get('symbol').replace('/','-')}_{body.get('timeframe')}"
        initial_bal = float(body.get('capitalAllocation', 1000))
        
        with bots_lock:
            if bot_id in active_live_bots:
                try:
                    active_live_bots[bot_id].stop()
                    del active_live_bots[bot_id]
                    logger.info(f"🛑 Force-Stopped bot {bot_id} for reset.")
                except Exception as e:
                    logger.warning(f"⚠️ Error stopping bot for reset: {e}")

        bots_collection.delete_one({"botId": bot_id})
        
        new_state = {
            "botId": bot_id,
            "currentBalance": initial_bal,
            "status": "stopped",
            "activePositions": [],
            "tradeHistory": [],
            "performanceMetrics": {"totalTrades": 0, "winRate": 0, "totalProfit": 0},
            "cash": initial_bal 
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
            "positions": normalize_positions(target_bot.manager.positions),
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

# ... (Backtest endpoints omitted for brevity)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
