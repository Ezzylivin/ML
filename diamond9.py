# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v85.2 - "The Sovereign Hydrator"
# 🛠 FIX: Added Strategy Hydration. If 'strategies' list is empty but 'comboConfig' exists,
# the bot now auto-generates strategy objects so the loop actually runs.

import os, json, logging, traceback, math, threading, warnings, joblib, time
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, UpdateOne

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("System")

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data/optimizer_results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 🟢 MONGODB CONNECTION
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/test")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_database() 
bots_collection = db['bots']

logger.info(f"✅ Connected to MongoDB: {db.name}")

DEFAULT_TAKER_FEE = 0.0006
SLIPPAGE_BPS = 2.0 
MAX_TOTAL_RISK = 0.30 

# 🛑 GRACEFUL SHUTDOWN SIGNAL
shutdown_event = threading.Event()

app = FastAPI(title="Sovereign Executive v85.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛑 SYSTEM EVENTS
@app.on_event("shutdown")
def shutdown_event_handler():
    logger.warning("🛑 System Shutdown Initiated. Stopping all bots...")
    shutdown_event.set()

active_live_bots = {}
bots_lock = threading.Lock()

# --- 2. DYNAMIC STRATEGY REGISTRY ---

def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    """
    Computes signal based on the LAST CLOSED CANDLE.
    df_slice must include the completed candle at .iloc[-1]
    """
    try:
        code = strategy_conf.get('code')
        p = strategy_conf.get('params', {})
        weight = float(strategy_conf.get('weight', 1.0))
        ml_confirm = strategy_conf.get('mlConfirm', True)

        def get_p(key, default): return float(p.get(key, default))

        ml_gate_sig = 0
        
        # 🧠 ML Gating Logic
        if ml_model is not None and ml_confirm:
            try:
                window = df_slice.tail(5).select_dtypes(include=[np.number])
                if hasattr(ml_model, "n_features_in_") and window.shape[1] == ml_model.n_features_in_:
                    probs = ml_model.predict_proba(window)[:, 1]
                    p_long = np.mean(probs)
                    if p_long > ml_thresh: ml_gate_sig = 1
                    elif p_long < (1 - ml_thresh): ml_gate_sig = -1
                    else: ml_gate_sig = 0 
                else: ml_gate_sig = 0 
            except Exception: ml_gate_sig = 0

        sig = 0
        if len(df_slice) < 3: return 0
        
        # Use .iloc[-1] as the candle that JUST closed
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

    except Exception: return 0.0

# --- 3. RISK MANAGER ---
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
        self.peak = float(capital)
        self.dd_series = [0]
        self.current_equity = float(capital)
        
        self.max_daily_loss_pct = float(max_daily_loss)
        self.max_trades_per_day = int(max_trades_per_day)
        self.today_start_equity = float(capital)
        self.daily_trades_count = 0
        self.last_day_reset = datetime.now(timezone.utc).date()

    def check_daily_reset(self, current_time_str):
        current_date = datetime.fromisoformat(current_time_str).date()
        if current_date > self.last_day_reset:
            self.last_day_reset = current_date
            self.today_start_equity = self.current_equity
            self.daily_trades_count = 0
            return True 
        return False

    def is_trading_allowed(self):
        if self.daily_trades_count >= self.max_trades_per_day:
            return False, f"Daily trade limit ({self.max_trades_per_day}) reached."
        daily_pnl_pct = ((self.current_equity - self.today_start_equity) / self.today_start_equity) * 100
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return False, f"Daily loss limit (-{self.max_daily_loss_pct}%) hit. PnL: {daily_pnl_pct:.2f}%"
        return True, "OK"

    def step_equity(self, price):
        self.current_equity = self.cash + sum(p['qty'] * price for p in self.positions)
        self.peak = max(self.peak, self.current_equity)
        dd = (self.peak - self.current_equity) / self.peak if self.peak > 0 else 0
        self.dd_series.append(dd)
        return self.current_equity

    def handle(self, signal, price, low, high, time, atr, tsl_mult, slippage_bps=0):
        self.check_daily_reset(time)
        orders = []
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        
        slippage_rate = slippage_bps / 10000.0
        # Execution price (Base Fill)
        fill_price = price * (1 + slippage_rate) if signal == 1 else price * (1 - slippage_rate)

        current_risk_exposure = len(self.positions) * self.base_risk
        if (current_risk_exposure + self.base_risk) > MAX_TOTAL_RISK: signal = 0 
        allowed, reason = self.is_trading_allowed()
        if not allowed and signal == 1: signal = 0 

        # --- ENTRY ---
        if signal == 1 and len(self.positions) < self.max_layers:
            dist = (atr * tsl_mult) if atr > 0 else (fill_price * 0.05)
            if dist <= 0: dist = fill_price * 0.05
            qty = (balance_metric * self.base_risk) / dist
            
            cost = qty * fill_price
            entry_fee = cost * self.fee
            
            if self.cash >= (cost + entry_fee):
                self.positions.append({
                    'qty': qty, 'entry': fill_price, 'time': time, 
                    'peak': fill_price, 
                    'stop_loss': fill_price - dist 
                })
                self.cash -= (cost + entry_fee)
                self.daily_trades_count += 1 
                orders.append(('BUY', qty))
        
        # --- EXIT (Reversal) ---
        elif signal == -1 and self.positions:
            q = sum(p['qty'] for p in self.positions)
            self.close_all(fill_price, time, "SIGNAL_REVERSAL")
            orders.append(('SELL', q))
            return orders, reason

        # --- TRAILING STOP ---
        dist = (atr * tsl_mult) if atr > 0 else (price * 0.05)
        remaining = []
        for p in self.positions:
            p['peak'] = max(p['peak'], price) 
            stop_level = p['peak'] - dist
            
            # 🛑 CRITICAL FIX: GAP DETECTION LOGIC
            # If Candle Open (passed as 'price' in Execution Phase) is BELOW Stop Level,
            # we gapped down. Fill at Open, not Stop Level.
            if low <= stop_level:
                # Did we gap past the stop?
                gap_down = price < stop_level 
                base_fill = price if gap_down else stop_level
                
                # Apply slippage to the worse price
                stop_fill = base_fill * (1 - slippage_rate)
                
                # Sanity check: Fill cannot be better than Low
                # (Simulates liquidity drying up inside the candle)
                stop_fill = min(stop_fill, low * (1 + slippage_rate)) 
                
                self.close_position(p, stop_fill, time, 'TSL_STOP')
                orders.append(('SELL', p['qty']))
            else:
                remaining.append(p)
        
        self.positions = remaining
        return orders, reason

    def close_all(self, price, time, reason):
        for p in self.positions: self.close_position(p, price, time, reason)
        self.positions = []

    def close_position(self, p, price, time, reason):
        proceeds = p['qty'] * price
        exit_fee = proceeds * self.fee
        entry_fee = p['entry'] * p['qty'] * self.fee
        pnl = proceeds - exit_fee - (p['entry'] * p['qty']) - entry_fee
        self.cash += (proceeds - exit_fee)
        self.trades.append({
            'entryTime': p['time'], 'exitTime': time, 'entryPrice': p['entry'],
            'exitPrice': price, 'qty': p['qty'], 'profit': pnl, 'type': reason, 'side': 'long'
        })

# --- 5. LIVE TRADING BOT ENGINE ---

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.user_id = config.get('userId', 'anon_user')
        
        if 'botId' in config:
            self.bot_id = config['botId']
        else:
            clean_symbol = self.symbol.replace('/', '-')
            self.bot_id = f"{self.user_id}_{clean_symbol}_{self.tf}"
            
        self.is_paper = config.get('mode', 'paper') == 'paper'
        self.last_processed_ts = 0
        
        # --- 🛡️ AUTO-HYDRATION LOGIC (Fixes empty strategies array) ---
        # If frontend sends isCombo=True but empty strategies, we build them from strategyCodes
        is_combo = config.get('isCombo', False)
        strats = config.get('strategies', [])
        
        if is_combo and not strats:
            combo_conf = config.get('comboConfig', {})
            codes = combo_conf.get('strategyCodes', [])
            global_params = config.get('params', {})
            
            hydrated_strats = []
            for code in codes:
                hydrated_strats.append({
                    'code': code,
                    'params': global_params, # Inherit global params
                    'weight': 1.0,
                    'mlConfirm': True
                })
            
            self.config['strategies'] = hydrated_strats
            print(f"✅ [Hydrator] Auto-generated {len(hydrated_strats)} strategies from Combo Config.")
        # -------------------------------------------------------------

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
        
        exchange_id = config.get('exchange', 'coinbase').lower()
        if not hasattr(ccxt, exchange_id): raise ValueError(f"Exchange '{exchange_id}' not supported.")
        
        exchange_class = getattr(ccxt, exchange_id)
        try:
            auth = {'enableRateLimit': True}
            if config.get('apiKey'):
                # 🛡️ MASK SECRETS in logs/memory dump if possible
                auth.update({'apiKey': config['apiKey'], 'secret': config['secret']})
                
            self.exchange = exchange_class(auth)
            
            # For paper trading, try to fetch real price to set initial balance logic if needed
            if self.is_paper and not config.get('initialBalance'):
                try: 
                    self.exchange.fetch_ticker(self.symbol)
                    config['initialBalance'] = 10000 
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
            base_risk=risk_pct, risk_mode=risk_mode,
            max_daily_loss=config.get('maxDailyLoss', 5.0),
            max_trades_per_day=config.get('maxTradesPerDay', 20)
        )
        
        # 🟢 CRITICAL RECOVERY FIX
        # Recover positions then IMMEDIATELY update equity with live price
        self._load_state_from_mongo()
        try:
            # Sync equity immediately on restart
            ticker = self.exchange.fetch_ticker(self.symbol)
            if ticker and 'last' in ticker:
                self.manager.step_equity(ticker['last'])
                self.logger.info(f"💰 Equity synced to Market: ${self.manager.current_equity:.2f}")
        except: 
            self.logger.warning("⚠️ Could not sync equity on startup (Network issue?)")

        self.is_running = False

    def _load_model(self, name):
        if not name: return None
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        return joblib.load(path) if os.path.exists(path) else None

    def _load_state_from_mongo(self):
        try:
            doc = bots_collection.find_one({"botId": self.bot_id})
            if doc:
                if 'currentBalance' in doc:
                    self.manager.cash = float(doc['currentBalance'])
                    # Temporary set until step_equity corrects it
                    self.manager.current_equity = self.manager.cash 
                if 'activePositions' in doc: self.manager.positions = doc['activePositions']
                if 'tradeHistory' in doc: self.manager.trades = doc['tradeHistory']
                self.logger.info(f"💾 State recovered. Cash: ${self.manager.cash:.2f}")
        except Exception as e:
            self.logger.error(f"⚠️ Mongo Recovery Failed: {e}")

    def _save_state_to_mongo(self, latest_ts):
        try:
            total_trades = len(self.manager.trades)
            wins = len([t for t in self.manager.trades if t['profit'] > 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            update_data = {
                "currentBalance": self.manager.current_equity,
                "status": "running",
                "activePositions": self.manager.positions,
                "tradeHistory": self.manager.trades,
                "lastActive": datetime.now(),
                "performanceMetrics": {
                    "totalTrades": total_trades,
                    "winRate": win_rate,
                    "totalProfit": self.manager.current_equity - self.manager.initial_capital,
                    "maxDrawdown": max(self.manager.dd_series) * 100 if self.manager.dd_series else 0
                }
            }
            bots_collection.update_one({"botId": self.bot_id}, {"$set": update_data}, upsert=True)
        except Exception as e:
            self.logger.error(f"⚠️ Mongo Save Failed: {e}")

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"STATUS | 🚀 Bot Active. User: {self.user_id} | ID: {self.bot_id}")
        
        # 🛑 SHUTDOWN CHECK & WATCHDOG
        while self.is_running and not shutdown_event.is_set():
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=300)
                latest_ts = ohlcv[-1][0]
                
                if latest_ts <= self.last_processed_ts:
                    time.sleep(10); continue

                # 🛡️ LIVE SAFETY: DROP INCOMPLETE CANDLE
                # We slice [:-1] to ensure we ONLY calc indicators on closed candles
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

                atr_col = df.filter(like='ATR').columns[-1]
                adx_col = df.filter(like='ADX').columns[0]
                
                # METRICS FOR DISPLAY (From the just-closed candle)
                price = df['close'].iloc[-1]
                low = df['low'].iloc[-1]
                high = df['high'].iloc[-1]
                current_adx = df[adx_col].iloc[-1] 
                current_rsi = df['RSI_14'].iloc[-1]
                
                report = []
                report.append(f"--- 📊 MARKET SNAPSHOT [{df.index[-1]}] ---")
                
                trend_status = "Weak/Choppy" if current_adx < 25 else "Strong Trend"
                report.append(f"• Price: ${price:.2f}")
                report.append(f"• Trend (ADX): {current_adx:.1f} ({trend_status})")
                current_exposure = len(self.manager.positions) * self.manager.base_risk * 100
                report.append(f"• Risk Exposure: {current_exposure:.1f}% / {MAX_TOTAL_RISK*100}% Cap")

                report.append("--- 🗳️ STRATEGY VOTES ---")
                
                if self.min_adx > 0 and current_adx < self.min_adx:
                    combined_sig = 0
                    report.append(f"⛔ TRADE BLOCKED: ADX {current_adx:.1f} is below minimum {self.min_adx}.")
                else:
                    strats = self.config.get('strategies', [])
                    votes = []
                    for s in strats:
                        # 🛡️ Pass full DF up to current closed candle
                        vote = compute_signal(df, s, self.ml_model, 0.5)
                        votes.append(vote)
                        vote_str = "🟢 BUY" if vote > 0 else ("🔴 SELL" if vote < 0 else "⚪ WAIT")
                        report.append(f"   {vote_str} | Strategy: {s.get('code')}")

                    combined_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
                    final_verdict = "🟢 BUY" if combined_sig == 1 else ("🔴 SELL" if combined_sig == -1 else "⚪ HOLD")
                    report.append(f"⚖️  FINAL VERDICT: {final_verdict} (Score: {sum(votes)})")

                safe_atr = df[atr_col].iloc[-1]
                orders, risk_msg = self.manager.handle(combined_sig, price, low, high, 
                                                     df.index[-1].isoformat(), safe_atr, self.tsl_mult)
                
                if risk_msg != "OK": report.append(f"⚠️ RISK LIMIT: {risk_msg}")

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
                self._save_state_to_mongo(latest_ts)
                
                for line in report:
                    self.logger.info(f"DECISION | {line}")

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(60)

    def stop(self): self.is_running = False

# --- 6. ENDPOINTS ---

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    try:
        body = await request.json()
        bot_id = body.get('botId')
        if not bot_id and body.get('userId'):
            clean_symbol = body['symbol'].replace('/', '-')
            bot_id = f"{body['userId']}_{clean_symbol}_{body['timeframe']}"
            body['botId'] = bot_id

        if not bot_id: return JSONResponse({"error": "botId required"}, 400)
        
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
    bot_id = body.get('botId')
    
    if not bot_id and 'userId' in body:
        userId = body['userId']
        stopped_count = 0
        with bots_lock:
            to_remove = [bid for bid in active_live_bots if bid.startswith(userId)]
            for bid in to_remove:
                active_live_bots[bid].stop()
                del active_live_bots[bid]
                bots_collection.update_one({"botId": bid}, {"$set": {"status": "stopped", "stoppedAt": datetime.now()}})
                stopped_count += 1
        return {"status": "stopped", "count": stopped_count}

    if not bot_id: return JSONResponse({"error": "botId required"}, 400)
    
    with bots_lock:
        if bot_id in active_live_bots:
            active_live_bots[bot_id].stop()
            del active_live_bots[bot_id]
            bots_collection.update_one({"botId": bot_id}, {"$set": {"status": "stopped", "stoppedAt": datetime.now()}})
            return {"status": "stopped"}
    return JSONResponse({"error": "Bot not found"}, 404)

@app.get('/api/bot/status')
async def get_bot_status(botId: str = None, userId: str = None):
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

    trades = target_bot.manager.trades
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['profit'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    dd_series = target_bot.manager.dd_series
    max_dd = max(dd_series) * 100 if dd_series else 0

    return {
        "status": "running",
        "symbol": target_bot.symbol,
        "timeframe": target_bot.tf,
        "currentBalance": target_bot.manager.current_equity,
        "positions": target_bot.manager.positions,
        "trades": trades, 
        "performanceMetrics": {
            "totalTrades": total_trades,
            "winRate": round(win_rate, 2),
            "netProfit": target_bot.manager.current_equity - target_bot.manager.initial_capital,
            "maxDrawdown": round(max_dd, 2)
        }
    }

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
        # 🟢 FIX: PRE-CALCULATE SIGNALS
        # New Logic: Calculate signal at i, execute at i+1 (Open).
        
        # Initialize signal state
        prev_sig = 0
        
        for i in range(50, len(df)):
            # 1. EXECUTION PHASE (At Open of Candle i)
            # Execute the signal determined at the END of Candle i-1
            p_open = df['open'].iloc[i]
            p_close = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            
            # ATR from previous completed candle (Safe)
            atr_val = df[atr_col].iloc[i-1] 

            # Handle Trades (Fills at Open, Stops check Low/High of current candle)
            mgr.handle(prev_sig, p_open, low, high, df.index[i].isoformat(), atr_val, tsl_mult, slippage_bps)
            
            # 2. DECISION PHASE (At Close of Candle i)
            # Determine signal for NEXT candle (i+1)
            
            # Check Trend on COMPLETED candle i
            adx_val = df[adx_col].iloc[i] 
            
            if min_adx > 0 and adx_val < min_adx:
                new_sig = 0 
            else:
                # Compute using data up to i (Completed)
                votes = [compute_signal(df.iloc[:i+1], s, ml_model, body.get('mlThreshold', 0.5)) for s in strats]
                new_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
            # Update state for next iteration
            prev_sig = new_sig
            
            # 3. RECORDING PHASE (End of Candle i)
            # Update equity curve using Close price
            curve.append({'timestamp': df.index[i].isoformat(), 'balance': mgr.step_equity(p_close)})

        final_bal = curve[-1]['balance']
        total_ret = ((final_bal - body.get('initialBalance', 1000)) / body.get('initialBalance', 1000)) * 100
        trades = len(mgr.trades)
        wins = len([t for t in mgr.trades if t['profit'] > 0])
        win_rate = (wins / trades * 100) if trades > 0 else 0
        periods_per_year = 8760
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
