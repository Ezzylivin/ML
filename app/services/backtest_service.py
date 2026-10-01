import traceback
import logging
import os
import ccxt
import pandas as pd
import pandas_ta as ta

# Adjust imports to work whether run from root or app folder
try:
    from manager import PrecisionPyramidManager
    from config import MODEL_DIR
except ImportError:
    from ..manager import PrecisionPyramidManager
    from ..config import MODEL_DIR

# Set up logging
logger = logging.getLogger("BacktestService")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

async def execute_backtest(
    symbol: str = 'BTC-USD', 
    timeframe: str = '1h', 
    limit: int = 1000, 
    initial_balance: float = 1000.0,
    risk_percentage: float = 1.0,
    risk_mode: str = 'static'
) -> dict:
    """
    Core backtesting logic decoupled from the API layer.
    """
    try:
        # Normalize symbol for Coinbase (BTC-USD -> BTC/USD)
        clean_symbol = symbol.replace('-', '/')
        
        # --- CONNECT TO COINBASE ---
        exchange_config = {
            'timeout': 30000,
            'enableRateLimit': True,
        }
        
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        
        if api_key and api_secret:
            exchange_config.update({'apiKey': api_key, 'secret': api_secret})
            logger.info(f"Fetching {limit} candles for {clean_symbol} (Authenticated)...")
        else:
            logger.info(f"Fetching {limit} candles for {clean_symbol} (Public)...")

        try:
            exchange = ccxt.coinbase(exchange_config)
            ohlcv = exchange.fetch_ohlcv(clean_symbol, timeframe, limit=limit)
        except ccxt.NetworkError as e:
            raise RuntimeError(f"Network error connecting to Coinbase: {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Exchange error (check symbol/keys): {e}")
        except Exception as e:
            raise RuntimeError(f"CCXT Error: {e}")

        # Validate Data
        if not ohlcv or not all(len(row) == 6 for row in ohlcv):
            raise ValueError("Invalid or insufficient OHLCV data fetched from exchange.")

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index(pd.to_datetime(df['timestamp'], unit='ms', utc=True), inplace=True)

        # Apply Indicators
        try:
            df.ta.sma(length=10, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.macd(append=True)
            df.ta.rsi(length=14, append=True)
            if 'ATR' not in df.columns:
                 df.ta.atr(length=14, append=True)
        except Exception as e:
            logger.error(f"Indicator calculation failed: {traceback.format_exc()}")
            raise RuntimeError(f"Indicator computation error: {str(e)}")

        # Initialize Manager
        real_risk = float(risk_percentage) / 100
        mgr = PrecisionPyramidManager(
            capital=initial_balance,
            risk_mode=risk_mode,
            base_risk=real_risk
        )

        # Simulation Loop
        curve = []
        start_index = 50 # Warmup period for indicators
        
        for i in range(start_index, len(df)):
            candle = df.iloc[i]
            close_price = candle['close']
            
            # Safe ATR access
            atr_val = candle.get('ATRr_14', candle.get('ATR', 0))
            if pd.isna(atr_val): atr_val = 0

            try:
                # Placeholder Signal: 1 (Replace with strategy logic later)
                signal = 1 
                
                mgr.handle(
                    signal=signal, 
                    close=close_price, 
                    low=candle['low'], 
                    high=candle['high'], 
                    time=candle.name.isoformat(), 
                    atr=atr_val, 
                    risk_reward=3.0
                )
                
                curve.append({
                    'timestamp': candle.name.isoformat(), 
                    'balance': mgr.step_equity(close_price)
                })
            except Exception as e:
                logger.warning(f"Loop error at {candle.name}: {e}")
                continue

        # Calculate Metrics
        final_balance = curve[-1]['balance'] if curve else initial_balance
        total_return = (final_balance - initial_balance) / initial_balance * 100

        return {
            "metrics": {
                "totalReturn": round(total_return, 2),
                "finalBalance": round(final_balance, 2)
            },
            "equityCurve": curve
        }

    except Exception as e:
        logger.error(f"Backtest execution failed: {traceback.format_exc()}")
        raise # Re-raise to let the caller handle it
