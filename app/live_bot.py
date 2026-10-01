import time
import logging
import joblib
import pandas as pd
import pandas_ta as ta
import ccxt
import os
import traceback
import numpy as np
from datetime import datetime, timezone

from .config import bots_collection, shutdown_event, MODEL_DIR, LOG_DIR
from .utils import recursive_clean, normalize_positions
from .logging_handlers import MongoLogHandler
from .strategies import compute_signal, aggregate_signals, log_requirements
from .manager import PrecisionPyramidManager

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT').replace('-', '/')
        self.tf = config.get('timeframe', '1h')
        self.bot_id = config.get('botId')
        
        self.logger = logging.getLogger(self.bot_id)
        self.logger.setLevel(logging.INFO)
        if not any(isinstance(h, MongoLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        self.exchange = getattr(ccxt, config.get('exchange', 'coinbase'))({'enableRateLimit': True})
        
        # 🟢 UPGRADE: Initialize with account risk and dynamic parameters
        self.manager = PrecisionPyramidManager(
            capital=float(config.get('capitalAllocation', 1000)),
            symbol=self.symbol,
            base_risk=0.01, 
            commission=float(config.get('params', {}).get('commission', 0.001)),
            slippage=float(config.get('params', {}).get('slippage', 0.0005)),
            minAdxLevel=float(config.get('params', {}).get('minAdxLevel', 20))
        )
        
        self.is_running = False
        self.last_processed_ts = 0

    def sync_and_trade(self):
        self.is_running = True
        self.logger.info(f"🚀 Bot v93.5 Upgrade Online: {self.bot_id}")

        while self.is_running and not shutdown_event.is_set():
            try:
                # 1. Fetch High-Limit Data
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=200)
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                
                # 🟢 UPGRADE: Standardize Indicator names for Strategy Registry
                df.ta.rsi(length=14, append=True, col_names=("rsi",))
                df.ta.sma(length=10, append=True, col_names=("fast_sma",))
                df.ta.sma(length=50, append=True, col_names=("slow_sma",))
                df.ta.atr(length=14, append=True, col_names=("atr",))
                df.ta.adx(append=True, col_names=("adx", "dmp", "dmn"))
                df.fillna(0, inplace=True)

                current_ts = ohlcv[-2][0]
                current_price = ohlcv[-1][4]
                current_adx = df['adx'].iloc[-1]
                
                # 🟢 UPGRADE: Constant Stop-Loss Monitoring (Intrabar)
                # This ensures we exit if SL is hit even between candle closes
                if self.manager.positions:
                    self.manager.check_exit(
                        high=df['high'].iloc[-1], 
                        low=df['low'].iloc[-1], 
                        time=datetime.now(timezone.utc).isoformat()
                    )

                # Skip if we already processed this candle's entry signal
                if current_ts <= self.last_processed_ts:
                    time.sleep(30); continue

                # 2. Strategy Processing
                strategies = self.config.get("strategies", [])
                votes = [compute_signal(df, s) for s in strategies]
                
                # 🟢 UPGRADE: Multi-Layer Aggregation with Regime Veto
                decision_data = aggregate_signals(
                    votes, 
                    adx_val=current_adx, 
                    combination_rule=self.config.get('params', {}).get('combinationRule', 'OR'),
                    minAdxLevel=float(self.config.get('params', {}).get('minAdxLevel', 20))
                )
                
                final_sig = 0
                if decision_data['decision'] == 'LONG': final_sig = 1
                elif decision_data['decision'] == 'SHORT': final_sig = -1

                # 3. Execution using Risk-Based Sizing
                self.manager.handle(
                    signal=final_sig, 
                    price=current_price, 
                    time=datetime.now(timezone.utc).isoformat(),
                    atr=df['atr'].iloc[-2], # Use confirmed ATR from previous candle
                    adx=current_adx,
                    tslAtrMult=float(self.config.get('params', {}).get('tslAtrMult', 3.0))
                )

                self.manager.step_equity(current_price, timestamp=datetime.now(timezone.utc).isoformat())
                self.last_processed_ts = current_ts
                self._update_db_status(current_price)

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(30)

    def _update_db_status(self, price):
        trades = self.manager.trades
        bots_collection.update_one(
            {"botId": self.bot_id},
            {"$set": {
                "currentBalance": round(float(self.manager.current_equity), 2),
                "totalTrades": len(trades),
                "activePositions": normalize_positions(self.manager.positions),
                "equityCurve": recursive_clean(self.manager.equity_curve[-200:]),
                "lastActive": datetime.now(timezone.utc)
            }}, upsert=True
        )

    def stop(self):
        self.is_running = False
