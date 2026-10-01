# File: /root/Project/ML/diamond.py
# 🚀 UPGRADE: v75.0 - "The Integrated Sovereign"
# 🛠 FEATURES: Live/Paper/Backtest Parity, Full Param Support, SQLite Persistence.

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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

PROJECT_ROOT = "/root/Project/ML"
MODEL_DIR = os.path.join(PROJECT_ROOT, "app/models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data/optimizer_results")
DB_PATH = os.path.join(PROJECT_ROOT, "trading_state.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_TAKER_FEE = 0.0006
GLOBAL_MODEL_CACHE = {}

app = FastAPI(title="Sovereign Executive v75.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_live_bots = {}

# --- 2. PERSISTENCE LAYER ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS bots (bot_id TEXT PRIMARY KEY, cash REAL, last_ts INTEGER)')
    cur.execute('CREATE TABLE IF NOT EXISTS positions (bot_id TEXT, qty REAL, entry REAL, time TEXT, peak REAL)')
    conn.commit()
    conn.close()

init_db()

# --- 3. DYNAMIC STRATEGY REGISTRY ---
def compute_signal(df_slice, strategy_conf, ml_model=None, ml_thresh=0.5):
    try:
        code = strategy_conf.get('code')
        weight = float(strategy_conf.get('weight', 1.0))
        ml_confirm = strategy_conf.get('mlConfirm', True)

        ml_gate_sig = 0
        if ml_model is not None and ml_confirm:
            window = df_slice.tail(5).select_dtypes(include=[np.number])
            if len(window) >= 5:
                p_long = np.mean(ml_model.predict_proba(window)[:, 1])
                if p_long > ml_thresh: ml_gate_sig = 1
                elif p_long < (1 - ml_thresh): ml_gate_sig = -1

        sig = 0
        t_row = df_slice.iloc[-1]
        t_prev = df_slice.iloc[-2] if len(df_slice) > 1 else t_row
        
        if code == 'sma_crossover':
            if t_row['fast_sma'] > t_row['slow_sma'] and t_prev['fast_sma'] <= t_prev['slow_sma']: sig = 1
            elif t_row['fast_sma'] < t_row['slow_sma'] and t_prev['fast_sma'] >= t_prev['slow_sma']: sig = -1
        elif code == 'macd_crossover':
            if t_row['MACD'] > t_row['MACDs'] and t_prev['MACD'] <= t_prev['MACDs']: sig = 1
            elif t_row['MACD'] < t_row['MACDs'] and t_prev['MACD'] >= t_prev['MACDs']: sig = -1
        elif code == 'psar_flip_signal':
            if pd.notna(t_row.get('PSARl')) and pd.isna(t_prev.get('PSARl')): sig = 1
            elif pd.notna(t_row.get('PSARs')) and pd.isna(t_prev.get('PSARs')): sig = -1

        final_sig = np.where((sig != 0) & (sig == ml_gate_sig), sig, 0) if (ml_model and ml_confirm) else sig
        return float(final_sig * weight)
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

    def handle(self, signal, price, time, atr, tsl_mult):
        orders = []
        balance_metric = self.current_equity if self.risk_mode == "dynamic" else self.initial_capital
        
        if signal == 1 and len(self.positions) < self.max_layers:
            dist = (atr * tsl_mult) if atr > 0 else (price * 0.05)
            if dist <= 0: dist = price * 0.05
            qty = (balance_metric * self.base_risk) / dist
            
            if self.cash >= (qty * price):
                self.positions.append({'qty': qty, 'entry': price, 'time': time, 'peak': price})
                self.cash -= (qty * price)
                orders.append(('BUY', qty))
        
        elif signal == -1 and self.positions:
            q = sum(p['qty'] for p in self.positions)
            self.close_all(price, time, "SIGNAL_REVERSAL")
            orders.append(('SELL', q))
            return orders

        dist = (atr * tsl_mult) if atr > 0 else (price * 0.05)
        remaining = []
        for p in self.positions:
            p['peak'] = max(p['peak'], price)
            if price <= (p['peak'] - dist):
                pnl = (price * (1 - self.fee) - p['entry']) * p['qty']
                self.cash += (p['entry'] * p['qty']) + pnl
                self.trades.append({'entryTime': p['time'], 'exitTime': time, 'profit': pnl, 'type': 'TSL_STOP'})
                orders.append(('SELL', p['qty']))
            else:
                remaining.append(p)
        self.positions = remaining
        return orders

    def close_all(self, price, time, reason):
        for p in self.positions:
            pnl = (price * (1 - self.fee) - p['entry']) * p['qty']
            self.cash += (p['entry'] * p['qty']) + pnl
            self.trades.append({'entryTime': p['time'], 'exitTime': time, 'profit': pnl, 'type': reason})
        self.positions = []

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
        
        # Extract params for live execution
        self.params = config.get('params', {})
        self.tsl_mult = float(self.params.get('tslAtrMult', 3.5))
        self.min_adx = float(self.params.get('minAdxLevel', 0))
        
        self.exchange = ccxt.coinbase({
            'apiKey': config.get('apiKey', ''), 
            'secret': config.get('secret', ''),
            'enableRateLimit': True
        })
        
        self.ml_model = self._load_model(config.get('mlModel'))
        
        # Initialize Manager with Correct Risk Settings
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
            logger.info(f"✨ Bot Recovered. Equity: {self.manager.current_equity}")
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
        logger.info(f"🚀 Bot Started: {self.bot_id} | Mode: {'PAPER' if self.is_paper else 'REAL'}")
        
        while self.is_running:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=100)
                latest_ts = ohlcv[-1][0]
                if latest_ts <= self.last_processed_ts:
                    time.sleep(20); continue

                df = pd.DataFrame(ohlcv[:-1], columns=['ts','open','high','low','close','volume'])
                df.set_index(pd.to_datetime(df['ts'], unit='ms', utc=True), inplace=True)

                # Indicators matching Backtest
                df.ta.sma(length=10, append=True, col_names="fast_sma")
                df.ta.sma(length=50, append=True, col_names="slow_sma")
                df.ta.macd(append=True); df.ta.psar(append=True)
                df.ta.atr(length=14, append=True, col_names="ATR_14")
                df.ta.adx(length=14, append=True, col_names=("ADX", "DMP", "DMN"))

                atr = df['ATR_14'].iloc[-1]
                adx = df['ADX'].iloc[-1]
                price = df['close'].iloc[-1]
                
                # ADX Filter
                if self.min_adx > 0 and adx < self.min_adx:
                    combined_sig = 0
                else:
                    votes = [compute_signal(df, s, self.ml_model, 0.5) for s in self.config.get('strategies', [])]
                    combined_sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)

                # Execution
                orders = self.manager.handle(combined_sig, price, df.index[-1].isoformat(), atr, self.tsl_mult)
                
                if not self.is_paper:
                    for action, qty in orders:
                        safe_qty = float(self.exchange.amount_to_precision(self.symbol, qty))
                        self.exchange.create_market_order(self.symbol, action.lower(), safe_qty)
                        self.reconcile_balance()

                self.manager.step_equity(price)
                self.last_processed_ts = latest_ts
                self._save_state(latest_ts)
                logger.info(f"✅ Candle Processed: {df.index[-1]} | Signal: {combined_sig}")

            except Exception as e:
                logger.error(f"Bot Loop Error: {traceback.format_exc()}")
                time.sleep(60)

    def _save_state(self, last_ts):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("REPLACE INTO bots VALUES (?, ?, ?)", (self.bot_id, self.manager.cash, last_ts))
        cur.execute("DELETE FROM positions WHERE bot_id=?", (self.bot_id,))
        for p in self.manager.positions:
            cur.execute("INSERT INTO positions VALUES (?, ?, ?, ?, ?)", (self.bot_id, p['qty'], p['entry'], p['time'], p['peak']))
        conn.commit(); conn.close()

    def stop(self): self.is_running = False

# --- 6. ENDPOINTS ---

@app.post('/api/bot/start')
async def start_live_bot(request: Request, bg: BackgroundTasks):
    body = await request.json()
    bot_id = f"{body['symbol']}_{body['timeframe']}"
    if bot_id in active_live_bots: return JSONResponse({"error": "Bot already active"}, 400)
    
    bot = LiveExecutiveBot(body)
    bg.add_task(bot.sync_and_trade)
    active_live_bots[bot_id] = bot
    return {"status": "started", "bot_id": bot_id}

@app.post('/api/bot/stop')
async def stop_live_bot(request: Request):
    body = await request.json()
    bot_id = f"{body['symbol']}_{body['timeframe']}"
    if bot_id in active_live_bots:
        active_live_bots[bot_id].stop()
        del active_live_bots[bot_id]
        return {"status": "stopped"}
    return JSONResponse({"error": "Bot not found"}, 404)

# 🛠 REPLACE YOUR EXISTING handle_backtest WITH THIS BLOCK
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

        # 1. Indicators
        df.ta.sma(length=10, append=True, col_names="fast_sma")
        df.ta.sma(length=50, append=True, col_names="slow_sma")
        df.ta.macd(append=True); df.ta.psar(append=True)
        df.ta.atr(length=14, append=True) # Don't force name, let safe filter find it
        df.ta.adx(length=14, append=True)

        # 2. Safety: Find ATR/ADX regardless of naming convention
        atr_col = df.filter(like='ATR').columns[-1]
        adx_col = df.filter(like='ADX').columns[0]
        
        # 3. Prevent Lookahead (Shift Indicators)
        df['ATR_SAFE'] = df[atr_col].shift(1)
        df['ADX_SAFE'] = df[adx_col].shift(1)
        
        ml_model = None
        if body.get('mlMode') == 'on':
            path = os.path.join(MODEL_DIR, f"{body['mlModel']}.joblib")
            if os.path.exists(path): ml_model = joblib.load(path)

        strats = body.get('strategies', []) or [{"code": body.get('code'), "params": body.get('params', {})}]
        
        # 4. Extract UI Parameters correctly
        tsl_mult = float(params.get('tslAtrMult', 3.5))
        min_adx = float(params.get('minAdxLevel', 0))
        risk_mode = body.get('riskManagementMode', 'static')
        risk_pct = float(body.get('riskPercentage', 1)) / 100.0
        max_layers = int(body.get('maxPyramiding', 3))

        mgr = PrecisionPyramidManager(
            capital=body.get('initialBalance', 1000),
            fee=DEFAULT_TAKER_FEE,
            max_layers=max_layers,
            base_risk=risk_pct,
            risk_mode=risk_mode
        )
        
        curve = []

        # 5. Execution Loop
        for i in range(50, len(df)):
            p = df['close'].iloc[i]
            adx_val = df['ADX_SAFE'].iloc[i]
            
            # ADX Filter Logic
            if min_adx > 0 and adx_val < min_adx:
                sig = 0 
            else:
                votes = [compute_signal(df.iloc[:i+1], s, ml_model, body.get('mlThreshold', 0.5)) for s in strats]
                sig = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
            
            # 🚀 PASS TSL_MULT, NOT STD
            mgr.handle(sig, p, df.index[i].isoformat(), df['ATR_SAFE'].iloc[i], tsl_mult)
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
        
        return JSONResponse(content={"combinedResult": res, **res})

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
