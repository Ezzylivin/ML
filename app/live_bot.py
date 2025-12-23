import logging
import time
import traceback
import pandas as pd
import ccxt  # Kept for compatibility, though we inject LocalExchange
from datetime import datetime
from config import bots_collection, SHUTDOWN_EVENT, LOG_DIR, MODEL_DIR
from utils import recursive_clean, normalize_positions
from bot_logging import MongoLogHandler, log_monitoring_phase
from strategies import log_intent, log_requirements, compute_signal
from manager import PrecisionPyramidManager

# Constants for strategy parameters
DEFAULT_OHLCV_LIMIT = 100
ATR_MULTIPLIER = 3.0
RSI_PERIOD = 14
FAST_SMA_LENGTH = 10
SLOW_SMA_LENGTH = 50
EMA_LENGTH = 20

class LiveExecutiveBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get("symbol", "BTC/USDT").replace("-", "/")
        self.tf = config.get("timeframe", "1h")
        self.user_id = config.get("userId", "anon")
        self.bot_id = config.get("botId", f"{self.user_id}_{self.symbol.replace('/', '-')}")
        self.is_paper = config.get("mode", "paper") == "paper"
        self.force_close = config.get("forceCloseOnStop", False)
        self.last_processed_ts = 0

        # Initialize structured logging
        self.logger = logging.getLogger(self.bot_id)
        # Ensure MongoHandler is attached if available (simulated here via bot_logging.py)
        if not any(isinstance(handler, MongoLogHandler) for handler in self.logger.handlers):
            self.logger.addHandler(MongoLogHandler(bots_collection, self.bot_id))

        # File-based logging for persistence
        try:
            import os
            if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
                os.makedirs(LOG_DIR, exist_ok=True)
                fh = logging.FileHandler(os.path.join(LOG_DIR, f"{self.bot_id}.log"))
                fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
                self.logger.addHandler(fh)
        except Exception as e:
            print(f"Logger initialization error: {e}")
            self.logger.error(f"Logger initialization error: {e}")

        # Initialize the exchange
        # CRITICAL UPDATE: logic to accept the injected LocalCSVExchange or fallback to CCXT
        if 'exchange_instance' in config:
             self.exchange = config['exchange_instance']
             self.logger.info("Using Injected Local Exchange (Deep Dive Mode)")
        else:
            try:
                exchange_name = config.get("exchange", "coinbase").lower()
                self.exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
                if not self.is_paper and config.get("apiKey") and config.get("secret"):
                    self.exchange.apiKey = config["apiKey"]
                    self.exchange.secret = config["secret"]
            except AttributeError as e:
                self.logger.error(f"Exchange initialization error: {e}")
                raise ValueError(f"Invalid exchange: {exchange_name}")

        # Initialize the trading manager
        try:
            self.manager = PrecisionPyramidManager(capital=config.get("capitalAllocation", 1000))
            # In a real DB scenario, we would load state here. 
            # For this run, we rely on the in-memory manager initialized above.
            if hasattr(bots_collection, "find_one"):
                saved_data = bots_collection.find_one({"botId": self.bot_id})
                if saved_data:
                    self.manager.cash = float(saved_data.get("cash", self.manager.cash))
                    self.manager.positions = normalize_positions(saved_data.get("activePositions", []))
        except Exception as e:
            self.logger.error(f"Failed to load bot state from database: {traceback.format_exc()}")

        # Handle ML model loading
        self.ml_model = None
        # (ML logic preserved but skipped for this specific CSV run unless model exists)

    def sync_and_trade(self):
        """Main loop that fetches market data, evaluates strategy signals, and executes trades."""
        self.logger.info(f"Bot {self.bot_id} started.")
        self.is_running = True

        while self.is_running and not SHUTDOWN_EVENT.is_set():
            try:
                # Fetch OHLCV data and validate it
                try:
                    ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.tf, limit=DEFAULT_OHLCV_LIMIT)
                    if not ohlcv or len(ohlcv) < 2: # Need at least 2 candles for prev/curr logic
                        self.logger.warning("Fetch error: invalid OHLCV data.")
                        time.sleep(1)
                        continue
                except Exception as e:
                    self.logger.error(f"Failed to fetch market data: {e}")
                    time.sleep(5)
                    continue

                # Build DataFrame
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                df['ts'] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index('ts', inplace=True)

                try:
                    # Apply indicators
                    # Note: We rely on the Shim or pandas_ta being present
                    df.ta.rsi(length=RSI_PERIOD, append=True)
                    df.ta.atr(length=14, append=True)
                except Exception as e:
                    self.logger.error(f"Error applying indicators: {e}")

                # Parse last known timestamp from the 2nd to last candle (closed candle)
                # In live trading, we often look at the just-closed candle.
                current_candle_time = int(ohlcv[-1][0])
                
                # Check if we have processed this timestamp already
                if current_candle_time <= self.last_processed_ts:
                    # Simulation mode needs to sleep less, real mode needs to sleep more
                    time.sleep(0.5) 
                    continue

                current_price = df['close'].iloc[-1]
                
                # Evaluate Signals
                strategies = self.config.get("strategies", [])
                log_intent(self.logger, strategies, current_price)

                vote = 0
                for s in strategies:
                    v, r = compute_signal(df, s)
                    vote += v
                    if v != 0: 
                         self.logger.info(f"Strategy {s.get('code')}: Signal {v} ({r})")

                final_sig = 1 if vote > 0 else (-1 if vote < 0 else 0)

                # Execute via Manager
                orders, msg = self.manager.handle(
                    final_sig, 
                    current_price, 
                    df['low'].iloc[-1], 
                    df['high'].iloc[-1], 
                    datetime.now().isoformat(), # Use current system time for trade log
                    df['ATRr_14'].iloc[-1] if 'ATRr_14' in df else 0.0, # Handle TA naming variance
                    3.0
                )

                if orders:
                    self.logger.info(f"EXECUTED: {orders}")
                elif msg != "OK":
                    self.logger.info(f"Manager Info: {msg}")

                # Save/Update State
                self.manager.step_equity(current_price)
                self.last_processed_ts = current_candle_time
                self.latest_candles = df.tail(50).reset_index().to_dict(orient='records')
                
                # Database Update (Safe wrap)
                if hasattr(bots_collection, "update_one"):
                    bots_collection.update_one(
                        {"botId": self.bot_id},
                        {"$set": {
                            "currentBalance": self.manager.current_equity,
                            "activePositions": normalize_positions(self.manager.positions),
                            "lastActive": datetime.now()
                        }},
                        upsert=True
                    )

            except Exception as e:
                self.logger.error(f"Loop Error: {traceback.format_exc()}")
                time.sleep(1)

    def stop(self):
        self.is_running = False
        self.logger.info(f"Bot {self.bot_id} stopping...")
