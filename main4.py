import asyncio
import logging
import os
import joblib
import json
import sqlite3
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt 
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# 🟢 SOCKET HELPERS
from app.services.socket_emitter import emit_log, emit_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-Engine")

ACTIVE_BOTS = {} 

# ==========================================
# 🗄️ 0. DATABASE HANDLER (Persistence Layer)
# ==========================================
# In main3.py / main4.py

class DatabaseHandler:
    DB_FILE = "bot_state.db"

    @classmethod
    def init_db(cls):
        """Initialize the SQLite database for persistence."""
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
        """Save the current bot state to DB."""
        # 🟢 SELF-HEALING: Ensure table exists before saving
        cls.init_db() 
        
        conn = sqlite3.connect(cls.DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO bot_sessions 
                     (user_id, config, balance, positions, trade_history, equity_curve, logs, status, last_update)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, 
                   json.dumps(bot_data['config']), 
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
        """Load bot state from DB if exists."""
        # 🟢 SELF-HEALING: Ensure table exists before loading
        if not os.path.exists(cls.DB_FILE):
            cls.init_db()
            return None
            
        conn = sqlite3.connect(cls.DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute("SELECT * FROM bot_sessions WHERE user_id = ?", (user_id,))
        except sqlite3.OperationalError:
            # 🟢 Table missing? Create it and return None (fresh start)
            conn.close()
            cls.init_db()
            return None

        row = c.fetchone()
        conn.close()
        
        if row:
            try:
                return {
                    "config": json.loads(row['config']),
                    "balance": row['balance'],
                    "positions": json.loads(row['positions']),
                    "trade_history": json.loads(row['trade_history']),
                    "equityCurve": json.loads(row['equity_curve']),
                    "logs": json.loads(row['logs']),
                    "status": "stopped" # Always load as stopped initially
                }
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
                return None
        return None

# Ensure this is called at the module level
DatabaseHandler.init_db()
# --- REQUEST MODELS ---
class BotStartRequest(BaseModel):
    userId: str
    config: Dict[str, Any]

class BotStopRequest(BaseModel):
    userId: str

class BotClosePositionRequest(BaseModel):
    userId: str
    symbol: str

class BacktestRequest(BaseModel):
    userId: str
    symbol: str
    timeframe: str
    start_date: str 
    end_date: str   
    strategies: List[Dict[str, Any]]
    initial_capital: float = 10000.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for user_id, bot in ACTIVE_BOTS.items():
        bot["status"] = "stopped"
        DatabaseHandler.save_state(user_id, bot)

app = FastAPI(title="NEO-V25.14 Sovereign Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 🧠 1. NEURAL PREDICTOR
# ==========================================
class DiagnosticLayer:
    @staticmethod
    def render_progress(current, target, reverse=False):
        """Generates a [|||||.....] visual bar."""
        try:
            # Calculate completion percentage
            pct = (target / current) if not reverse else (current / target)
            pct = min(1.0, max(0.0, pct))
            filled = int(pct * 10)
            bar = "┃" + "█" * filled + "░" * (10 - filled) + "┃"
            return f"{bar} {int(pct * 100)}%"
        except: return "[----------] 0%"

    @staticmethod
    def get_pending_conditions(df, config, conf, ui_limit):
        try:
            current_price = float(df['close'].iloc[-1])
            ema200_val = ta.ema(df['close'], length=200).iloc[-1]
            is_uptrend = current_price > ema200_val
            
            if conf < ui_limit:
                return f"🛑 AI VETO: Confidence {DiagnosticLayer.render_progress(conf, ui_limit, True)}"

            pending = []
            strategies = config.get('strategies', [])
            for strat in strategies:
                code = strat.get('code')
                p = strat.get('params', {})
                if code == "bb_fade":
                    bb = ta.bbands(df['close'], length=int(p.get('bb_period', 20)), std=float(p.get('bb_std', 2.0)))
                    target = bb.iloc[-1, 0] if is_uptrend else bb.iloc[-1, 2]
                    bar = DiagnosticLayer.render_progress(current_price, target, is_uptrend)
                    pending.append(f"Price to BB: {bar}")
                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close']).iloc[-1, 0]
                    target = 20 if is_uptrend else 80
                    bar = DiagnosticLayer.render_progress(k, target, not is_uptrend)
                    pending.append(f"Stoch to Trigg: {bar}")
            
            rule = config.get('combinationRule', 'OR')
            return f"🔍 TARGETS ({rule}): " + " | ".join(pending)
        except Exception: return "🔍 Scanning Market Conditions..."


class NeuralPredictor:
    @staticmethod
    def get_prediction(model_id: str, df: pd.DataFrame) -> float:
        try:
            recent = df.tail(10)
            momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            model_path = f"./models/{model_id}_model.pkl"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                return float(model.predict_proba([[momentum]])[0][1])
            base = 1.0 / (1.0 + np.exp(-momentum * 100))
            return float(min(1.0, max(0.0, base)))
        except Exception: return 0.5

# ==========================================
# 🧠 2. SHARED STRATEGY BRAIN
# ==========================================
class StrategyBrain:
    @staticmethod
    def calculate_signals(df: pd.DataFrame, config: Dict[str, Any], l_thresh: float, s_thresh: float):
        active_thoughts, votes = [], 0
        strategies = config.get('strategies', [])
        current_price = df['close'].iloc[-1]

        # 🟢 Indicators
        ema20 = ta.ema(df['close'], 20).iloc[-1]
        ema50 = ta.ema(df['close'], 50).iloc[-1]
        ema200 = ta.ema(df['close'], 200).iloc[-1]
        bb = ta.bbands(df['close'], length=20, std=2.0)
        lower, mid, upper = bb.iloc[-1, 0], bb.iloc[-1, 1], bb.iloc[-1, 2]
        pr = int((current_price - lower) / (upper - lower) * 100)

        # 🟢 1. FULL DYNAMIC STRATEGY EVALUATION
        for strat in strategies:
            code = strat.get('code')
            params = strat.get('params', {})
            try:
                if code == "rsi_threshold":
                    rsi = ta.rsi(df['close'], length=int(params.get('rsi_length', 14))).iloc[-1]
                    if rsi < params.get('oversold', 30): votes += 1; active_thoughts.append(f"RSI Low ({int(rsi)})")
                    elif rsi > params.get('overbought', 70): votes -= 1; active_thoughts.append(f"RSI High ({int(rsi)})")

                elif code == "stoch":
                    k = ta.stoch(df['high'], df['low'], df['close'], k=int(params.get('k_period', 14))).iloc[-1, 0]
                    if k < 20: votes += 1; active_thoughts.append(f"Stoch Low ({int(k)})")
                    elif k > 80: votes -= 1; active_thoughts.append(f"Stoch High ({int(k)})")

                elif code == "bb_fade":
                    # Note: Visualization handled in Heartbeat loop
                    p_bb = ta.bbands(df['close'], length=int(params.get('bb_period', 20)), std=float(params.get('bb_std', 2.0)))
                    if current_price < p_bb.iloc[-1, 0]: votes += 1; active_thoughts.append("Price < BB Floor")
                    elif current_price > p_bb.iloc[-1, 2]: votes -= 1; active_thoughts.append("Price > BB Ceiling")

                elif code == "sma_crossover":
                    fast = ta.sma(df['close'], length=int(params.get('fast_sma', 50))).iloc[-1]
                    slow = ta.sma(df['close'], length=int(params.get('slow_sma', 200))).iloc[-1]
                    if fast > slow: votes += 1; active_thoughts.append("SMA Golden Cross")

                elif code == "macd_crossover":
                    macd = ta.macd(df['close'], fast=int(params.get('fast', 12)), slow=int(params.get('slow', 26)), signal=int(params.get('signal', 9))).iloc[-1]
                    if macd.iloc[0] > macd.iloc[2]: votes += 1; active_thoughts.append("MACD Bullish")

                elif code == "supertrend":
                    st = ta.supertrend(df['high'], df['low'], df['close'], length=int(params.get('st_atr', 10)), multiplier=float(params.get('st_factor', 3.0))).iloc[-1]
                    if st.iloc[1] == 1: votes += 1; active_thoughts.append("SuperTrend Long")

                elif code == "ema_cloud":
                    if current_price > ema50: votes += 1; active_thoughts.append("Above EMA Cloud")

                elif code == "atr_breakout":
                    atr = ta.atr(df['high'], df['low'], df['close'], length=int(params.get('atr_length', 14))).iloc[-1]
                    if current_price > (ema20 + atr * float(params.get('multiplier', 1.5))): votes += 1; active_thoughts.append("ATR Breakout")

                elif code == "pa_breakout":
                    if current_price >= df['high'].tail(int(params.get('lookback', 20))).max(): votes += 1; active_thoughts.append("PA High Break")

                elif code == "vol_profile":
                    vol_ma = ta.sma(df['volume'], length=int(params.get('vol_ma', 20))).iloc[-1]
                    if df['volume'].iloc[-1] > vol_ma * float(params.get('threshold', 1.5)): 
                        votes += (1 if current_price > ema20 else -1)
                        active_thoughts.append("Volume Surge")

            except Exception as e: continue

        # 🚀 2. DYNAMIC DUAL-GATE LOGIC (Block 1 Logic)
        is_short = current_price < ema200
        ui_limit = float(config.get('mlThresholdShort', 0.90)) if is_short else float(config.get('mlThresholdLong', 0.80))
        
        conf = NeuralPredictor.get_prediction(config.get('mlModel', 'stacking'), df)
        gate_passed = conf >= ui_limit
        logic_desc = f"📊 LOGIC: {'SHORT' if is_short else 'LONG'} GATE {'PASSED' if gate_passed else 'VETOED'} ({int(conf*100)}% vs {int(ui_limit*100)}% UI Limit) {'🟢' if gate_passed else '🔴'}"

        # 🎯 3. INTENT (Block 1 Logic)
        signal_names = " + ".join(active_thoughts) if active_thoughts else "Scanning Setup"
        gap = int(abs(current_price - ema50))
        intent_desc = f"🎯 INTENT: STALKING {'SHORT' if is_short else 'LONG'} ({signal_names} | Gap: ${gap}) {'🔴' if is_short else '🟢'}"

        # 📡 4. MARKET CONTEXT (Rich Descriptions from Block 2)
        
        # 1. Trend Logic (Distance from 200EMA)
        trend_dist = current_price - ema200
        trend_pct = (trend_dist / ema200) * 100
        if trend_dist > 0:
            trend_str = "STRONG UPTREND" if trend_pct > 1.0 else "WEAK UPTREND"
        else:
            trend_str = "STRONG DOWNTREND" if trend_pct < -1.0 else "WEAK DOWNTREND"
            
        trend_text = f"📡 TREND: {trend_str} (Price is ${int(abs(trend_dist))} {'above' if trend_dist > 0 else 'below'} 200EMA)"

        # 2. Bias Logic (EMA Spread Strength)
        spread = ema20 - ema50
       
        bias_str = "BULLISH EXPANSION" if spread > 0 else "BEARISH CONTRACTION"
        bias_text = f"⚖️ BIAS: {bias_str} (Fast EMA is ${int(abs(spread))} {'above' if spread > 0 else 'below'} Slow EMA)"

        # 3. Mindset Logic (Bollinger Percentile)
        if pr >= 80: mindset_str = "⚠️ OVEREXTENDED (Expensive)"
        elif pr <= 20: mindset_str = "🎯 ACCUMULATION ZONE (Cheap)"
        else: mindset_str = "⚖️ EQUILIBRIUM (Balanced)"
        
        mindset_text = f"🤖 MINDSET: {mindset_str} - Price is at {pr}% of Bollinger Range"

        numeric_details = {
            "market": {
                "logic": logic_desc, 
                "intent": intent_desc,
                "trend": trend_text,
                "bias": bias_text,
                "mindset": mindset_text
            }
        }

        # Final Signal Calculation
        final_sig = 1 if votes > 0 and not is_short and gate_passed else (-1 if votes < 0 and is_short and gate_passed else 0)
        
        return final_sig, active_thoughts, numeric_details, conf



# ==========================================
# 🚀 3. THE HEARTBEAT (Dynamic Calculation Loop)
# ==========================================
async def live_neural_heartbeat(user_id: str):
    last_log = 0
    if user_id in ACTIVE_BOTS:
        if "equityCurve" not in ACTIVE_BOTS[user_id]: 
            ACTIVE_BOTS[user_id]["equityCurve"] = [{"time": datetime.now().isoformat(), "balance": ACTIVE_BOTS[user_id]["balance"], "confidence": 50}]
        if "logs" not in ACTIVE_BOTS[user_id]: ACTIVE_BOTS[user_id]["logs"] = []

    while user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        bot = ACTIVE_BOTS[user_id]
        config = bot.get('config', {})
        strategies = config.get('strategies', [])
        
        try:
            ohlcv_raw = await fetch_live_candles_ccxt(config['symbol'], config.get('timeframe', '1h'), 250)
            
            # 🟢 ADDED: Debug log if data is missing
            if not ohlcv_raw:
                logger.warning(f"⚠️ No candle data returned for {config['symbol']}")
                emit_log(user_id, "⚠️ Market Data Feed Unstable - Retrying...")
            
            if ohlcv_raw:
                df = pd.DataFrame(ohlcv_raw)
                
                current_price = float(df['close'].iloc[-1])
                ema200_val = ta.ema(df['close'], 200).iloc[-1]

                # 🟢 Also fix upnl calculation early to avoid errors
                upnl = sum([(current_price - p['entry']) * p['size'] if p['type'] == 'long' else (p['entry'] - current_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl
                # 🟢 DYNAMIC INDICATOR CALCULATION FOR CHART (ALL 10 STRATEGIES)
                for strat in strategies:
                    code = strat.get('code')
                    p = strat.get('params', {})
                    try:
                        if code == 'bb_fade':
                            bb = ta.bbands(df['close'], length=int(p.get('bb_period', 20)), std=float(p.get('bb_std', 2.0)))
                            df['bb_lower'] = bb.iloc[:, 0]
                            df['bb_upper'] = bb.iloc[:, 2]
                        elif code == 'ema_cloud':
                            df['ema_fast'] = ta.ema(df['close'], length=int(p.get('fast_ema', 9)))
                            df['ema_slow'] = ta.ema(df['close'], length=int(p.get('slow_ema', 21)))
                        elif code == 'sma_crossover':
                            df['sma_fast'] = ta.sma(df['close'], length=int(p.get('fast_sma', 50)))
                            df['sma_slow'] = ta.sma(df['close'], length=int(p.get('slow_sma', 200)))
                        elif code == 'supertrend':
                            st = ta.supertrend(df['high'], df['low'], df['close'], length=int(p.get('st_atr', 10)), multiplier=float(p.get('st_factor', 3.0)))
                            df['supertrend'] = st.iloc[:, 0]
                        elif code == 'pa_breakout':
                            lb = int(p.get('lookback', 20))
                            df['pa_high'] = df['high'].rolling(lb).max()
                            df['pa_low'] = df['low'].rolling(lb).min()
                        elif code == 'atr_breakout':
                            atr = ta.atr(df['high'], df['low'], df['close'], length=int(p.get('atr_length', 14)))
                            mult = float(p.get('multiplier', 1.5))
                            ema20 = ta.ema(df['close'], length=20)
                            df['atr_upper'] = ema20 + (atr * mult)
                            df['atr_lower'] = ema20 - (atr * mult)
                        
                        # 🟢 NEW: Oscillator Calculations (Data only, for tooltips/future use)
                        elif code == 'rsi_threshold':
                            df['rsi'] = ta.rsi(df['close'], length=int(p.get('rsi_length', 14)))
                        elif code == 'stoch':
                            stoch = ta.stoch(df['high'], df['low'], df['close'], k=int(p.get('k_period', 14)))
                            df['stoch_k'] = stoch.iloc[:, 0]
                            df['stoch_d'] = stoch.iloc[:, 1]
                        elif code == 'macd_crossover':
                            macd = ta.macd(df['close'], fast=int(p.get('fast', 12)), slow=int(p.get('slow', 26)), signal=int(p.get('signal', 9)))
                            df['macd'] = macd.iloc[:, 0]
                            df['macd_signal'] = macd.iloc[:, 2]
                        elif code == 'vol_profile':
                            df['vol_ma'] = ta.sma(df['volume'], length=int(p.get('vol_ma', 20)))

                    except Exception: continue

                # Logic & Signals
                sig, thoughts, nums, score = StrategyBrain.calculate_signals(df, config, 0.5, 0.5)

                ui_limit = float(config.get('mlThresholdLong', 0.5)) if current_price > ema200_val else float(config.get('mlThresholdShort', 0.5))
                waiting_msg = DiagnosticLayer.get_pending_conditions(df, config, score, ui_limit)

# 3. Inject into Logs
                if datetime.now().timestamp() - last_log >= 60:
                    emit_log(user_id, nums['market']['trend'])
                    emit_log(user_id, waiting_msg) # 🟢 The New Diagnostic Layer Output
                    emit_log(user_id, nums['market']['logic']) 
   
                    new_logs = [
                        {"time": datetime.now().isoformat(), "message": nums['market']['trend']},
                        {"time": datetime.now().isoformat(), "message": waiting_msg},
                        {"time": datetime.now().isoformat(), "message": nums['market']['logic']}
                    ]
                    bot["logs"] = (new_logs + bot["logs"])[:300]
                    
                    DatabaseHandler.save_state(user_id, bot)
                    last_log = datetime.now().timestamp()
             
                # 🟢 4. EXECUTION LOGIC (✅ ADDED THIS BLOCK)
                # This actually opens the trade when sig != 0
                if len(bot['positions']) < int(config.get('maxPyramiding', 1)):
                    
                    # LONG
                    if sig == 1:
                        risk_pct = float(config.get('riskPercentage', 1)) / 100
                        size = (bot['balance'] * risk_pct) / current_price 
                        pos = {
                            "type": "long", "entry": current_price, "size": size, "time": datetime.now().isoformat(),
                            "sl": current_price * (1 - float(config['params'].get('stop_loss', 0.05))),
                            "tp": current_price * (1 + float(config['params'].get('take_profit', 0.10)))
                        }
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "buy", "price": current_price, "time": datetime.now().isoformat()})
                        emit_log(user_id, f"🚀 LONG EXECUTED @ ${current_price} (Signal: {thoughts[0] if thoughts else 'Manual'})")
                        DatabaseHandler.save_state(user_id, bot)

                    # SHORT
                    elif sig == -1 and config.get('enable_shorting', True):
                        risk_pct = float(config.get('riskPercentage', 1)) / 100
                        size = (bot['balance'] * risk_pct) / current_price 
                        pos = {
                            "type": "short", "entry": current_price, "size": size, "time": datetime.now().isoformat(),
                            "sl": current_price * (1 + float(config['params'].get('stop_loss', 0.05))),
                            "tp": current_price * (1 - float(config['params'].get('take_profit', 0.10)))
                        }
                        bot['positions'].append(pos)
                        bot['trade_history'].append({"type": "short", "price": current_price, "time": datetime.now().isoformat()})
                        emit_log(user_id, f"🔻 SHORT EXECUTED @ ${current_price} (Signal: {thoughts[0] if thoughts else 'Manual'})")
                        DatabaseHandler.save_state(user_id, bot)

                # 🟢 5. EXIT LOGIC (✅ ADDED THIS BLOCK)
                # This checks SL/TP and closes positions
                active_pos = bot['positions'][:]
                for pos in active_pos:
                    pnl, closed = 0, False
                    if pos['type'] == 'long':
                        if current_price >= pos['tp']: pnl = (current_price - pos['entry']) * pos['size']; closed = True; emit_log(user_id, f"💰 TP HIT (Long): +${round(pnl, 2)}")
                        elif current_price <= pos['sl']: pnl = (current_price - pos['entry']) * pos['size']; closed = True; emit_log(user_id, f"🛑 SL HIT (Long): -${round(abs(pnl), 2)}")
                    elif pos['type'] == 'short':
                        if current_price <= pos['tp']: pnl = (pos['entry'] - current_price) * pos['size']; closed = True; emit_log(user_id, f"💰 TP HIT (Short): +${round(pnl, 2)}")
                        elif current_price >= pos['sl']: pnl = (pos['entry'] - current_price) * pos['size']; closed = True; emit_log(user_id, f"🛑 SL HIT (Short): -${round(abs(pnl), 2)}")

                    if closed:
                        bot['balance'] += pnl
                        bot['positions'].remove(pos)
                        bot['trade_history'].append({"type": "exit", "price": current_price, "pnl": pnl, "time": datetime.now().isoformat()})
                        DatabaseHandler.save_state(user_id, bot)
                
                # Markers
                markers = [{"time": int(c['time']), "position": "belowBar", "color": "#10b981", "shape": "circle", "text": thoughts[0] if thoughts else ""} for c in ohlcv_raw[-1:] if thoughts]

                # Update PnL & History
                latest_price = df['close'].iloc[-1]
                upnl = sum([(latest_price - p['entry']) * p['size'] if p['type'] == 'buy' else (p['entry'] - latest_price) * p['size'] for p in bot['positions']])
                current_equity = bot['balance'] + upnl

                if datetime.now().timestamp() - last_log >= 60:
                    bot["equityCurve"].append({"time": datetime.now().isoformat(), "balance": round(current_equity, 2), "confidence": int(score * 100)})
                    if len(bot["equityCurve"]) > 100: bot["equityCurve"].pop(0)
                    
                    new_logs = []
                    for key in ['trend', 'bias', 'mindset', 'logic', 'intent']:
                        msg = nums['market'][key]
                        emit_log(user_id, msg)
                        new_logs.append({"time": datetime.now().isoformat(), "message": msg})
                    bot["logs"] = (new_logs + bot["logs"])[:300]
                    
                    DatabaseHandler.save_state(user_id, bot)
                    last_log = datetime.now().timestamp()

                # Package Candle Data (Handling Dynamic Columns)
                candles_to_send = []
                # List of potential keys to send to frontend if they exist
                keys_to_check = [
                    'bb_lower', 'bb_upper', 'ema_fast', 'ema_slow', 
                    'sma_fast', 'sma_slow', 'supertrend', 'pa_high', 'pa_low', 
                    'atr_upper', 'atr_lower', 
                    'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'vol_ma' # 🟢 Added Oscillators
                ]
                
                for index, row in df.tail(100).iterrows():
                    c_obj = {"time": int(row['time']), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close']}
                    for key in keys_to_check:
                        if key in row and not pd.isna(row[key]): c_obj[key] = float(row[key])
                    candles_to_send.append(c_obj)
                    
                    bot["candles"] = candles_to_send
               

                if ACTIVE_BOTS[user_id]["status"] != "running":
                   logger.info("Stop detected. Aborting final emit.")
                   return

                emit_status(user_id, {
                    "status": "running", 
                    "currentBalance": round(current_equity, 2),
                    "unrealizedPnl": round(upnl, 2),
                    "activePositions": bot['positions'], 
                    "tradeMarkers": bot['trade_history'] + markers, 
                    "equityCurve": bot["equityCurve"], 
                    # 🟢 NEW: Send start time so timer works
                    "startedAt": bot.get("startedAt"), 
                    "candles": candles_to_send
                })



            await asyncio.sleep(15)
        except Exception as e: 
            logger.error(f"Sync Error: {e}"); await asyncio.sleep(10)

async def fetch_live_candles_ccxt(symbol: str, timeframe: str, limit: int):
    async with ccxt.coinbase() as ex:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol.replace('-', '/'), timeframe, limit=limit)
            return [{"time": c[0]/1000, "open": c[1], "high": c[2], "low": c[3], "close": c[4], "vol": c[5] if len(c) > 5 else 0} for c in ohlcv]
        except: return []

@app.post("/api/bot/start")
async def start_bot(data: BotStartRequest, background_tasks: BackgroundTasks):
    user_id = data.userId.strip()
    
    # 1. Get the capital from the UI (e.g., $125)
    raw_cap = data.config.get("capitalAllocation") or data.config.get("capital_allocation")
    ui_capital = float(raw_cap) if raw_cap else 200.0 

    # 2. Check if already active
    if user_id in ACTIVE_BOTS and ACTIVE_BOTS[user_id]["status"] == "running":
        return {"status": "running", "message": "Bot already active"}
    
    # 3. Load from DB (This might have the old $200)
    saved_state = DatabaseHandler.load_state(user_id)
    
    if saved_state:
        # 🟢 RESUME SESSION
        ACTIVE_BOTS[user_id] = saved_state
        ACTIVE_BOTS[user_id]["status"] = "running"
        ACTIVE_BOTS[user_id]["config"] = data.config
        
        # 🚀 FORCE OVERWRITE: This is the specific fix!
        # Even though we loaded $200 from the DB, we replace it with $125 immediately.
        ACTIVE_BOTS[user_id]["balance"] = ui_capital 
        
        # 🚀 RESET TIMER
        ACTIVE_BOTS[user_id]["startedAt"] = datetime.now(timezone.utc).isoformat()
        
        emit_log(user_id, f"♻️ SESSION RESET: Balance updated to ${ui_capital}")

    else:
        # 🟢 FRESH START
        ACTIVE_BOTS[user_id] = {
            "status": "running", 
            "config": data.config, 
            "balance": ui_capital,
            "positions": [], 
            "trade_history": [], 
            "equityCurve": [], 
            "logs": [],
            "startedAt": datetime.now(timezone.utc).isoformat()
        }
        emit_log(user_id, f"🚀 Engine Started. Portfolio: ${ui_capital}")

    # 4. PRE-FILL CHART (Prevents "Charts not showing")
    if not ACTIVE_BOTS[user_id].get("equityCurve"):
        ACTIVE_BOTS[user_id]["equityCurve"] = [{
            "time": datetime.now().isoformat(), 
            "balance": ui_capital, 
            "confidence": 50
        }]

    # 5. SAVE IMMEDIATELY
    # This overwrites the "bad" $200 in the file with the new $125 automatically.
    DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
    
    background_tasks.add_task(live_neural_heartbeat, user_id)
    return {"status": "running"}

@app.post("/api/bot/stop")
async def stop_bot(data: BotStopRequest):
    user_id = data.userId
    if user_id in ACTIVE_BOTS:
        # 1. Kill the loop flag
        ACTIVE_BOTS[user_id]["status"] = "stopped"
        
        # 2. Reset the bot data (The "Reset" part)
        ACTIVE_BOTS[user_id]["positions"] = []
        ACTIVE_BOTS[user_id]["trade_history"] = []
        ACTIVE_BOTS[user_id]["equityCurve"] = []
        ACTIVE_BOTS[user_id]["logs"] = []
        
        # 3. Tell Frontend to clear everything NOW
        emit_status(user_id, {
            "status": "stopped",
            "currentBalance": ACTIVE_BOTS[user_id]['balance'],
            "activePositions": [],
            "tradeMarkers": [],
            "equityCurve": [],
            "startedAt": None 
        })
        
        # 4. Wipe the Database state for this user
        DatabaseHandler.save_state(user_id, ACTIVE_BOTS[user_id])
        
        # 5. Remove from active memory to ensure total death
        del ACTIVE_BOTS[user_id]
        
        emit_log(user_id, "💀 SYSTEM PURGED: Engine stopped and session reset.")
        return {"status": "stopped", "message": "Bot killed and reset"}
    
    return {"status": "stopped"}

@app.post("/api/bot/close_position")
async def close_position(data: BotClosePositionRequest):
    if data.userId in ACTIVE_BOTS and ACTIVE_BOTS[data.userId]['positions']:
        pos = ACTIVE_BOTS[data.userId]['positions'].pop(0)
        emit_log(data.userId, f"⚠️ Manual Exit: {pos['type'].upper()} closed.")
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "closed"}
    raise HTTPException(status_code=400, detail="No active positions")

@app.post("/api/bot/backtest")
async def run_backtest(data: BacktestRequest):
    try:
        async with ccxt.coinbase() as exchange:
            since = exchange.parse8601(data.start_date)
            ohlcv = await exchange.fetch_ohlcv(data.symbol.replace('-', '/'), data.timeframe, since=since, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        balance, position, trades, curve = data.initial_capital, None, [], []
        for i in range(20, len(df)):
            window = df.iloc[:i+1]
            sig, _, _, _ = StrategyBrain.calculate_signals(window, {"strategies": data.strategies}, 0.5, 0.5)
            price, ts = df.iloc[i]['close'], datetime.fromtimestamp(df.iloc[i]['time']/1000, tz=timezone.utc).isoformat()
            if sig == 1 and position is None:
                position = {"type": "long", "entry": price, "size": balance / price}
                trades.append({"type": "buy", "price": price, "time": ts})
            elif sig == -1 and position is None:
                position = {"type": "short", "entry": price, "size": balance / price}
                trades.append({"type": "short", "price": price, "time": ts})
            elif (sig == -1 and position and position['type'] == 'long') or (sig == 1 and position and position['type'] == 'short'):
                pnl = (price - position['entry']) * position['size'] if position['type'] == 'long' else (position['entry'] - price) * position['size']
                balance += pnl; position = None
                trades.append({"type": "exit", "price": price, "time": ts, "pnl": pnl})
            curve.append({"time": ts, "equity": balance})
        return {"status": "success", "metrics": {"final_balance": round(balance, 2), "trade_count": len(trades)}, "trades": trades, "equity_curve": curve}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/api/bot/status")
async def get_status(userId: str):
    bot = ACTIVE_BOTS.get(userId.strip())
    if not bot:
        bot = DatabaseHandler.load_state(userId.strip())
    
    if bot:
        return {
            "status": bot["status"], 
            "balance": bot["balance"],
            "equityCurve": bot.get("equityCurve", []),
            "logs": bot.get("logs", []),
            "positions": bot.get("positions", []),
            # 🟢 NEW: Return start time
            "startedAt": bot.get("startedAt"),
            "config": bot.get("config"),
            "candles": bot.get("candles", [])
        }
    return {"status": "inactive", "balance": 0}

@app.post("/api/bot/reset")
async def reset_bot(data: BotStopRequest): # Uses same model as stop
    if data.userId in ACTIVE_BOTS:
        ACTIVE_BOTS[data.userId]["status"] = "stopped"
        ACTIVE_BOTS[data.userId]["positions"] = []
        ACTIVE_BOTS[data.userId]["trade_history"] = []
        ACTIVE_BOTS[data.userId]["equityCurve"] = []
        ACTIVE_BOTS[data.userId]["logs"] = []
        DatabaseHandler.save_state(data.userId, ACTIVE_BOTS[data.userId])
        return {"status": "reset"}
    return {"status": "not_found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

(venv) root@intelligent-mendel:~/Project/ML# 
