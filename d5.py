# File: /root/Project/ML/diamond0.py
# 🚀 UPGRADE: v89.22 - "The Narrator"
# 🛠 FEATURE: Integrated Strategy Translator (Human Readable Logs)
# 🛠 FEATURE: MongoLogHandler (Direct DB Logging)
# 🛠 FEATURE: "Glass Box" Monitoring

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

# ==========================================
# 1. THE STRATEGY TRANSLATOR (The "Brain")
# ==========================================
STRATEGY_TRANSLATOR = {
    # --- RSI Strategies ---
    "RSI_OVERBOUGHT": "The market is overheated (RSI High). Buyers are exhausted, which usually precedes a price drop.",
    "RSI_OVERSOLD":   "The market is undervalued (RSI Low). Sellers are exhausted, often leading to a bounce up.",
    "RSI_DIVERGENCE": "Price is moving one way, but momentum (RSI) is moving the other. A reversal is likely.",

    # --- MACD Strategies ---
    "MACD_CROSS_BULL": "Momentum Shift: The MACD line crossed above the signal line, indicating upward momentum.",
    "MACD_CROSS_BEAR": "Momentum Shift: The MACD line crossed below the signal line, indicating downward pressure.",

    # --- Moving Averages (SMA/EMA) ---
    "SMA_CROSS_GOLDEN": "Golden Cross: Short-term trend has crossed above the long-term average. Strongly Bullish.",
    "SMA_CROSS_DEATH":  "Death Cross: Short-term trend has dropped below the long-term average. Strongly Bearish.",
    "TREND_FOLLOWING":  "Price is consistently above the moving average. We are riding the trend.",

    # --- Bollinger Bands ---
    "BB_SQUEEZE":   "Volatility is dangerously low. A massive explosive move is imminent.",
    "BB_BREAKOUT":  "Price has broken outside the statistical bands. This signals a continuation of the breakout.",
    "BB_REVERSAL":  "Price touched the outer band and is snapping back to the mean.",

    # --- Machine Learning ---
    "ML_CONFIDENCE": "The AI model has detected a high-probability pattern based on historical training data.",
    
    # --- Candlestick Patterns ---
    "HAMMER":       "Reversal Pattern: Sellers pushed price down, but buyers rejected it aggressively.",
    "ENGULFING":    "Momentum Shift: The current candle completely overwhelmed the previous one.",
    
    # --- Fallback ---
    "UNKNOWN":      "Technical setup detected matching strategy parameters."
}

def get_dynamic_explanation(strategy_name, raw_message=""):
    """
    Scans the strategy name and message to find the best human explanation.
    """
    if strategy_name in STRATEGY_TRANSLATOR:
        return STRATEGY_TRANSLATOR[strategy_name]

    # Fuzzy Search
    for key, text in STRATEGY_TRANSLATOR.items():
        keywords = key.split('_')
        if all(k in strategy_name.upper() or k in raw_message.upper() for k in keywords):
            return text

    return "Technical signal detected. Market conditions meet strategy entry criteria."

# ==========================================
# 2. THE LOGGING HANDLER (The "Bridge")
# ==========================================
class MongoLogHandler(logging.Handler):
    def __init__(self, db_collection, bot_id):
        super().__init__()
        self.collection = db_collection
        self.bot_id = bot_id

    def emit(self, record):
        try:
            log_message = self.format(record)
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": record.levelname,  
                "message": log_message
            }
            # Efficient Push to MongoDB
            self.collection.update_one(
                {"botId": self.bot_id},
                {"$push": {"logs": log_entry}},
                upsert=True
            )
        except Exception:
            self.handleError(record)

# ==========================================
# 3. HELPER FUNCTIONS (The "Action")
# ==========================================
def log_decision_phase(logger, strategy_name, signal_details, is_blocked=False, block_reason=""):
    """ Logs the decision logic with human context. """
    explanation = get_dynamic_explanation(strategy_name, signal_details)
    logger.info(f"DECISION | 🧠 THOUGHTS: {strategy_name.upper()} Triggered ({signal_details})")
    logger.info(f"DECISION | 🗣️ CONTEXT: {explanation}")

    if is_blocked:
        logger.info(f"DECISION | ⚖️ VERDICT: Signal Generated but Blocked.")
        logger.info(f"DECISION | 🚫 REASON: {block_reason}")
    else:
        logger.info(f"DECISION | ⚖️ VERDICT: GO! Signal Executed.")

def log_monitoring_phase(logger, current_price, position=None):
    """ Logs the heartbeat with distances to targets. """
    if position:
        entry = float(position.get('entry', 0)) # Note: key is 'entry' in your norm func
        # If using stop_loss from manager, it might not have explicit TP in object unless added
        sl = float(position.get('stop_loss', 0))
        # Assuming a default TP distance for display if not in object, or calculate from Risk:Reward
        tp = float(position.get('take_profit', 0)) 
        side = position.get('side', 'long').lower()
        
        dist_sl = abs(current_price - sl)
        
        # Calculate PnL Percentage
        if entry > 0:
            raw_pnl = (current_price - entry) / entry * 100
            pnl_percent = raw_pnl if side == 'long' else -raw_pnl
        else:
            pnl_percent = 0.0

        emoji = "🟢" if pnl_percent >= 0 else "🔴"
        action = "Buying" if side == 'long' else "Shorting"

        logger.info(f"💎 STRATEGY  | {emoji} {action.upper()} from ${entry:,.2f} (PnL: {pnl_percent:+.2f}%)")
        logger.info(f"🎯 TARGETS   | Safety: ${sl:,.2f} (${dist_sl:.2f} away)")
    else:
        logger.info(f"⏳ MONITORING | Price: ${current_price:,.2f} | Scanning for setups...")

# --- CONFIGURATION ---
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

MONGO_URI = "mongodb+srv://ericjdickerson93:Dadeadend1!!@cluster0.wzztvhe.mongodb.net/?retryWrites=true&w=majority"
if not MONGO_URI:
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
app = FastAPI(title="Sovereign Executive v89.22")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}
bots_lock = threading.Lock()

# --- HELPERS ---
def generate_chart_markers(trades, active_positions):
    """Generates visual markers for the chart (Entry arrows, Exit arrows)"""
    markers = []
    
    # 1. Markers for History (Completed Trades)
    for t in trades[-20:]:  # Limit to last 20 to keep chart clean
        # Entry Marker
        markers.append({
            "time": t['entryTime'],
            "position": "belowBar" if t['side'] == 'long' else "aboveBar",
            "color": "#2196F3", # Blue for entry
            "shape": "arrowUp" if t['side'] == 'long' else "arrowDown",
            "text": f"ENTRY {t['side'].upper()}"
        })
        # Exit Marker
        markers.append({
            "time": t['exitTime'],
            "position": "aboveBar" if t['side'] == 'long' else "belowBar",
            "color": "#E91E63", # Pink/Red for exit
            "shape": "arrowDown" if t['side'] == 'long' else "arrowUp",
            "text": f"EXIT ({t['profit']:.2f})"
        })

    # 2. Markers for Active Positions (Current Open Trade)
    for p in active_positions:
        markers.append({
            "time": p['time'],
            "position": "belowBar" if p['side'] == 'long' else "aboveBar",
            "color": "#00E676", # Bright Green for active trade
            "shape": "arrowUp" if p['side'] == 'long' else "arrowDown",
            "text": "LIVE POS"
        })
        
    # Sort by time to prevent chart errors
    markers.sort(key=lambda x: x['time'])
    return markers


def clean_float(val):
    """Prevents 500 Errors by converting NaN/Inf to 0.0"""
    try:
        if val is None or math.isnan(val) or math.isinf(val):
            return 0.0
        return float(val)
    except:
        return 0.0



def normalize_positions(positions):
    out = []
    for p in positions:
        out.append({
            "side": p["side"],
            "qty": clean_float(p["qty"]),
            "entry": clean_float(p["entry"]),
            "time": p["time"],
            "peak": clean_float(p.get("peak", p["entry"])),
            "trough": clean_float(p.get("trough", p["entry"])),
            "entry_atr": clean_float(p.get("entry_atr", 0)),
            "stop_loss": clean_float(p.get("stop_loss", 0))
        })
    return out

# --- LOGIC & STRATEGIES ---

def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
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
                        ml_gate_sig = 1; ml_reason = f"ML BUY ({p_long:.2f})"
                    elif p_long < (1 - ml_thresh): 
                        ml_gate_sig = -1; ml_reason = f"ML SELL ({p_long:.2f})"
                    else: ml_reason = "ML Neutral"
                else: ml_gate_sig = 0 
            except: ml_gate_sig = 0

        sig = 0; reason = "Neutral"
        if len(df_slice) < 3: return 0, "No Data"
        t_row = df_slice.iloc[-1]; t_prev = df_slice.iloc[-2]
        
        # --- STRATEGIES ---
        if code == 'sma_crossover':
            fast, slow = t_row.get('fast_sma'), t_row.get('slow_sma')
            if fast > slow and t_prev.get('fast_sma') <= t_prev.get('slow_sma'): sig = 1; reason = "Golden Cross"
            elif fast < slow and t_prev.get('fast_sma') >= t_prev.get('slow_sma'): sig = -1; reason = "Death Cross"
        elif code == 'macd_crossover':
            macd, sl = t_row.get('MACD_12_26_9'), t_row.get('MACDs_12_26_9')
            if macd > sl and t_prev.get('MACD_12_26_9') <= t_prev.get('MACDs_12_26_9'): sig = 1; reason = "MACD Bull Cross"
            elif macd < sl and t_prev.get('MACD_12_26_9') >= t_prev.get('MACDs_12_26_9'): sig = -1; reason = "MACD Bear Cross"
        elif code == 'rsi_divergence':
            rsi = t_row.get('RSI_14', 50)
            if rsi < get_p('oversold_level', 30): sig = 1; reason = f"RSI Oversold ({rsi:.1f})"
            elif rsi > get_p('overbought_level', 70): sig = -1; reason = f"RSI Overbought ({rsi:.1f})"
        elif code == 'bollinger_bands':
            c = t_row.get('close')
            if c < t_row.get('BBL_20_2.0'): sig = 1; reason = "Price < Lower BB"
            elif c > t_row.get('BBU_20_2.0'): sig = -1; reason = "Price > Upper BB"
        elif code == 'atr_breakout':
            c = t_row.get('close'); prev = t_prev.get('close')
            mult = get_p('atr_multiplier', 3.0); ema = t_row.get('EMA_20', c); atr = t_row.get('ATR_14', 0)
            if c > (ema + atr*mult) and prev <= (ema + atr*mult): sig = 1; reason = "ATR Breakout Up"
            elif c < (ema - atr*mult) and prev >= (ema - atr*mult): sig = -1; reason = "ATR Breakout Down"
        elif code == 'stochastic_crossover':
            k, d = t_row.get('STOCHk_14_3_3'), t_row.get('STOCHd_14_3_3')
            pk, pd_val = t_prev.get('STOCHk_14_3_3'), t_prev.get('STOCHd_14_3_3')
            if k > d and pk <= pd_val and k < 20: sig = 1; reason = "Stoch Bull Cross"
            elif k < d and pk >= pd_val and k > 80: sig = -1; reason = "Stoch Bear Cross"
        elif code == 'cci_oversold':
            cci = t_row.get('CCI_14_0.015')
            if cci < -100: sig = 1; reason = "CCI Oversold"
            elif cci > 100: sig = -1; reason = "CCI Overbought"
        elif code == 'ichimoku_system':
            tk, kj = t_row.get('ITS_9'), t_row.get('IKS_26')
            ptk, pkj = t_prev.get('ITS_9'), t_prev.get('IKS_26')
            if tk > kj and ptk <= pkj: sig = 1; reason = "Tenkan > Kijun"
            elif tk < kj and ptk >= pkj: sig = -1; reason = "Tenkan < Kijun"
        elif code == 'obv_signal':
            if t_row.get('OBV') > t_prev.get('OBV'): sig = 1; reason = "OBV Rising"
            elif t_row.get('OBV') < t_prev.get('OBV'): sig = -1; reason = "OBV Falling"
        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl_0.02_0.2')) and pd.isna(t_prev.get('PSARl_0.02_0.2')): sig = 1; reason = "PSAR Bull Flip"
            elif pd.notna(t_row.get('PSARs_0.02_0.2')) and pd.isna(t_prev.get('PSARs_0.02_0.2')): sig = -1; reason = "PSAR Bear Flip"

        if ml_gate_sig == 0: return float(sig * weight), reason
        if sig == ml_gate_sig: return float(sig * weight), f"{reason} + {ml_reason}"
        if sig != 0: return 0.0, f"ML BLOCKED: {reason} but {ml_reason}"
            
        return 0.0, reason
    except: return 0.0, "Error"

# --- RISK MANAGER ---

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

# --- LIVE BOT ---

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
            self.config['strategies'] = [{'code': c, 'params': config.get('params', {}), 'weight': 1.0} for c in codes]

        self.logger = logging.getLogger(self.bot_id)
        # ADDED: Mongo Logger Attachment
        if not any(isinstance(h, MongoLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            try:
                fh = logging.FileHandler(os.path.join(LOG_DIR, f"{self.bot_id}.log"))
                fh.setFormatter(ISOFormatter('%(asctime)s | %(message)s'))
                self.logger.addHandler(fh)
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
                    if abs(local_ts - server_ts) > 5000:
                        self.logger.warning(f"⚠️ CLOCK SKEW: {abs(local_ts - server_ts)}ms")
                except: pass
                
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

                # 🛠 ORACLE FEATURE: Calculate Indicators LIVE
                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                
                df.ta.sma(length=10, append=True, col_names="fast_sma")
                df.ta.sma(length=50, append=True, col_names="slow_sma")
                df.ta.macd(append=True); df.ta.rsi(append=True, col_names="RSI_14")
                df.ta.atr(append=True); df.ta.adx(append=True)
                df.ta.bbands(append=True)
                
                processed_ts = ohlcv[-2][0]
                latest_ts_open = ohlcv[-1][0] 
                
                # Use current ticker if available, else latest close
                current_price = market_price if market_price else df['close'].iloc[-1]

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
                    # 🛠 ORACLE PULSE: Detailed Status Log via New Helper
                    if datetime.now().second < 10: 
                         active_pos = self.manager.positions[0] if self.manager.positions else None
                         log_monitoring_phase(self.logger, current_price, active_pos)
                    time.sleep(10); continue
                
                price = df['close'].iloc[-1]
                adx = df.filter(like='ADX').iloc[-1].iloc[0]
                
                if first_tick and self.manager.positions:
                    pos_val = 0
                    for p in self.manager.positions:
                        if p['side'] == 'long': pos_val += p['qty'] * price
                        else: pos_val -= p['qty'] * price
                    self.manager.cash = self.manager.current_equity - pos_val
                    first_tick = False

                self.logger.info(f"--- 📊 SNAPSHOT [{df.index[-1].isoformat()}] ---")
                self.latest_candles = [{"time": r.Index.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for r in df.tail(50).itertuples()]
                # 🧠 THOUGHT ENGINE
                votes = []
                for s in self.config['strategies']:
                    sig, reason = compute_signal(df, s, self.ml_model)
                    votes.append(sig)
                    
                    # 🛠 UPGRADE: Log Decision Phase
                    if sig != 0:
                        log_decision_phase(self.logger, s['code'], reason, is_blocked=False)

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
                
                orders, msg = self.manager.handle(final_sig, price, df['low'].iloc[-1], df['high'].iloc[-1], df.index[-1].isoformat(), df.filter(like='ATR').iloc[-1].iloc[0], 3.5)
                
                # 🛠 UPGRADE: Blocked Signal Logging
                if "Blocked" in msg:
                    log_decision_phase(self.logger, "RISK_MANAGER", "Trade Entry", is_blocked=True, block_reason=msg)
                elif msg != "OK": 
                    self.logger.info(f"⚠️ {msg}")
                
                trade_happened = False
                if orders: 
                    self.logger.info(f"⚡ EXECUTED: {len(orders)} Orders")
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
                    if final_sig != 0 and msg == "OK":
                        self.logger.info("💤 No Orders generated (Position likely already active).")

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

# --- ENDPOINTS ---

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
    return {"status": "online", "system": "Sovereign Executive v89.22"}

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
        # 1. Find the Target Bot
        target_bot = None
        if botId: target_bot = active_live_bots.get(botId)
        elif userId:
            for bid, bot in active_live_bots.items():
                if bid.startswith(userId):
                    target_bot = bot
                    break
        
        # 2. Handle "Bot Not Found" or "Stopped"
        if not target_bot:
            lookup_id = botId if botId else (userId if userId else None)
            if not lookup_id: return JSONResponse({"error": "ID required"}, 400)
            
            # Check MongoDB for stopped bot data
            doc = bots_collection.find_one({"botId": lookup_id})
            if doc:
                return {
                    "status": doc.get('status', 'stopped'),
                    "currentBalance": clean_float(doc.get('currentBalance', 0)),
                    "active": False,
                    "performanceMetrics": doc.get('performanceMetrics', {}),
                    # Return empty chart data for stopped bots so UI doesn't crash
                    "chartMarkers": [],
                    "chartLines": []
                }
            return JSONResponse({"status": "not_found"}, 404)

        # 3. Calculate Live Visualization Data
        total_profit = target_bot.manager.current_equity - target_bot.manager.initial_capital
        
        # Prepare Active Lines (Horizontal Levels for the Chart)
        active_lines = []
        if target_bot.manager.positions:
            p = target_bot.manager.positions[0]
            # Entry Line
            active_lines.append({
                "price": clean_float(p['entry']), 
                "color": "#2196F3", 
                "label": "Entry", 
                "style": "solid"
            })
            # Stop Loss Line
            if 'stop_loss' in p:
                active_lines.append({
                    "price": clean_float(p['stop_loss']), 
                    "color": "#FF5252", 
                    "label": "Stop Loss", 
                    "style": "dashed"
                })
        
        # 4. Return the Mega-Response
        return {
            "status": "running",
            "symbol": target_bot.symbol,
            "timeframe": target_bot.tf,
            "currentBalance": clean_float(target_bot.manager.current_equity),
            
            # Data for Lists/Tables
            # Note: normalize_positions must be updated to use clean_float as well
            "positions": normalize_positions(target_bot.manager.positions),
            "trades": target_bot.manager.trades, 
            
            # Data for Charting
            "candles": target_bot.latest_candles,
            "chartMarkers": generate_chart_markers(target_bot.manager.trades, target_bot.manager.positions),
            "chartLines": active_lines,
            
            "performanceMetrics": {
                "totalTrades": len(target_bot.manager.trades),
                "netProfit": clean_float(total_profit),
            }
        }
    except Exception as e:
        logger.error(f"🔥 STATUS CHECK CRASHED: {e}")
        # Return a safe error response instead of crashing with 500
        return JSONResponse({"status": "error", "message": str(e)}, 200)

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
# ... (Backtest endpoints kept intact)
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
