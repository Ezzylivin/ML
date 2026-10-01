import time
import logging
import pandas as pd
import pandas_ta as ta
import ccxt
import os
import traceback
import numpy as np
from datetime import datetime, timezone

# Internal Project Imports
from .config2 import bots_collection, shutdown_event, MODEL_STORAGE_DIR, LOG_DIR, ML_CONFIG
from .utils2 import recursive_clean, normalize_positions
from .logging_handlers2 import MongoLogHandler
from .manager2 import PrecisionPyramidManager
from app.predictors.model_factory import ModelFactory

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.bot_id = config.get('botId')
        
        # 1. Logging Setup
        self.logger = logging.getLogger(self.bot_id)
        self.logger.setLevel(logging.INFO)
        if not any(isinstance(h, MongoLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        # 2. Exchange & Risk Manager
        self.exchange = getattr(ccxt, config.get('exchange', 'coinbase'))({'enableRateLimit': True})
        self.params = config.get('params', {})
        
        self.manager = PrecisionPyramidManager(
            capital=float(config.get('capitalAllocation', 1000)),
            symbol=self.symbol,
            base_risk=0.01, 
            **self.params # Pass all gates/thresholds directly
        )

        # 3. ML Predictor Initialization
        self.model_type = self.params.get('model_type', 'stacking')
        self.predictor = ModelFactory.load_model(
            model_type=self.model_type, 
            bot_id=self.bot_id
        )
        
        self.is_running = False
        self.last_processed_ts = 0
        self.last_prob = 0.5 # Track conviction for dashboard

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"🚀 NEO-V7 Modular Executive Online: {self.bot_id} | Brain: {self.model_type}")

        while self.is_running and not shutdown_event.is_set():
            try:
                # --- 🔵 STEP 1: FETCH & FEATURE ENGINEERING ---
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=100)
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                
                # Standard Feature Set
                df.ta.rsi(length=14, append=True, col_names=("rsi",))
                df.ta.atr(length=14, append=True, col_names=("atr",))
                df.ta.adx(append=True, col_names=("adx", "dmp", "dmn"))
                df.fillna(0, inplace=True)

                current_ts = ohlcv[-2][0]
                current_price = ohlcv[-1][4]
                current_adx = df['adx'].iloc[-1]

                # --- 🔵 STEP 2: INTRABAR SAFETY ---
                if self.manager.positions:
                    self.manager.check_exit(
                        high=df['high'].iloc[-1], 
                        low=df['low'].iloc[-1], 
                        time=datetime.now(timezone.utc).isoformat()
                    )

                if current_ts <= self.last_processed_ts:
                    time.sleep(15); continue

                # --- 🔵 STEP 3: ML INFERENCE ---
                lookback = int(self.params.get('lookback', 50))
                state = df.tail(lookback)
                
                # prob is the probability (0.0 to 1.0)
                prob = self.predictor.predict_direction(state)
                self.last_prob = prob # Store for DB update
                
                # Gates
                long_gate = self.params.get('long_threshold', 0.65)
                short_gate = self.params.get('short_threshold', 0.35)
                exit_gate = self.params.get('exit_threshold', 0.50)
                adx_min = self.params.get('minAdxLevel', 25)

                self.logger.info(f"🧠 Inference: Prob={prob:.2f} | ADX={current_adx:.1f}")

                # --- 🔵 STEP 4: PROBABILISTIC EXECUTION ---
                final_sig = 0
                
                if not self.manager.positions and current_adx >= adx_min:
                    if prob >= long_gate: final_sig = 1
                    elif prob <= short_gate: final_sig = -1
                
                elif self.manager.positions:
                    # FIX: Manager positions are list of dicts, access correctly
                    active_pos = self.manager.positions[0]
                    active_side = 1 if active_pos['side'] == 'long' else -1
                    
                    if (active_side == 1 and prob < exit_gate) or (active_side == -1 and prob > exit_gate):
                        self.manager._close_position(current_price, datetime.now(timezone.utc).isoformat(), "AI Confidence Drop")

                # 🟢 CRITICAL: Passing signal_prob allows the manager to scale size
                self.manager.handle(
                    signal=final_sig, 
                    price=current_price, 
                    time=datetime.now(timezone.utc).isoformat(),
                    signal_prob=prob, # <--- THE MISSING LINK
                    atr=df['atr'].iloc[-2],
                    adx=current_adx
                )

                self.manager.step_equity(current_price, timestamp=datetime.now(timezone.utc).isoformat())
                self.last_processed_ts = current_ts
                self._update_db_status(current_price)

            except Exception as e:
                self.logger.error(f"❌ Execution Crash: {traceback.format_exc()}")
                time.sleep(30)

    def _update_db_status(self, price):
        """Syncs the Live Bot's ML-driven state to MongoDB."""
        bots_collection.update_one(
            {"botId": self.bot_id},
            {"$set": {
                "currentBalance": round(float(self.manager.current_equity), 2),
                "totalTrades": len(self.manager.trades),
                "activePositions": normalize_positions(self.manager.positions),
                "lastProb": round(float(self.last_prob), 4),
                "lastActive": datetime.now(timezone.utc)
            }}, upsert=True
        )

    def stop(self):
        self.is_running = False
