import os
import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
import csv # New import for logging

# --- Local Imports ---
from feature_engineering import engineer_features
from predict_service import predict, load_model

# --- Configuration ---
CONFIG = {
    'exchange': 'coinbase',
    'symbol': 'ETH/USD',
    'model_name': 'eth',
    'timeframe': '1h',
    'candles_to_fetch': 300,
    'confidence_threshold': 0.65,
    'log_file': 'paper_trades.csv' # New setting for our log file
}

# --- Bot State ---
bot_state = {
    'in_position': False
}

# --- NEW LOGGING FUNCTION ---
def log_trade(timestamp, action, price, confidence):
    """Appends a trade record to the log file."""
    log_file = CONFIG['log_file']
    # Check if file exists to write header only once
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'action', 'price', 'confidence']) # Write header
        
        # Write the trade data
        writer.writerow([timestamp, action, price, f"{confidence:.2%}"])

def get_utc_timestamp_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def wait_for_next_candle(timeframe: str):
    # ... (this function is unchanged) ...
    now_ms = get_utc_timestamp_ms()
    timeframe_seconds = 3600
    next_candle_start_s = (now_ms // 1000 // timeframe_seconds + 1) * timeframe_seconds
    wait_seconds = next_candle_start_s - (now_ms // 1000)
    if wait_seconds > 0 and wait_seconds <= timeframe_seconds:
        print(f"Waiting {wait_seconds} seconds for the next '{timeframe}' candle to close...")
        time.sleep(wait_seconds)

def main():
    # ... (startup messages are unchanged) ...
    print(f"--- Starting Live Paper Trading Bot for {CONFIG['symbol']} ---")
    exchange = getattr(ccxt, CONFIG['exchange'])()
    print(f"Pre-loading model '{CONFIG['model_name']}'...")
    try:
        load_model(CONFIG['model_name'])
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file for '{CONFIG['model_name']}' not found. Please train the model first.")
        return

    while True:
        try:
            wait_for_next_candle(CONFIG['timeframe'])
            
            # ... (data fetching, feature calculation, and prediction are unchanged) ...
            print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Fetching latest candles...")
            ohlcv = exchange.fetch_ohlcv(CONFIG['symbol'], CONFIG['timeframe'], limit=CONFIG['candles_to_fetch'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            latest_candle = df.iloc[-1]
            latest_price = latest_candle['close']
            print(f"Latest candle time: {latest_candle['datetime']}, Close price: {latest_price}")
            print("Calculating features...")
            df_processed = engineer_features(df.copy())
            for lag in [1, 2, 3]:
                df_processed[f'target_lag{lag}'] = df_processed['target'].shift(lag)
            df_processed.fillna(0, inplace=True)
            latest_features_row = df_processed.iloc[-1]
            print("Getting a prediction...")
            model = load_model(CONFIG['model_name'])
            feature_names = model.get_booster().feature_names
            features_for_prediction = latest_features_row[feature_names].tolist()
            result = predict(CONFIG['model_name'], features_for_prediction)
            signal = result['prediction']
            confidence = result['confidence']
            print(f"  > Model Prediction: {signal} (Confidence: {confidence:.2%})")

            # --- UPDATED TRADING LOGIC WITH LOGGING ---
            print("Executing trading logic...")
            if signal == 1 and confidence > CONFIG['confidence_threshold'] and not bot_state['in_position']:
                print(f"--- ACTION: BUY {CONFIG['symbol']} at {latest_price} ---")
                log_trade(latest_candle['datetime'], 'BUY', latest_price, confidence) # Log the trade
                bot_state['in_position'] = True

            elif signal == -1 and confidence > CONFIG['confidence_threshold'] and bot_state['in_position']:
                print(f"--- ACTION: SELL {CONFIG['symbol']} at {latest_price} ---")
                log_trade(latest_candle['datetime'], 'SELL', latest_price, confidence) # Log the trade
                bot_state['in_position'] = False
            
            else:
                print("  > Action: HOLD.")
            
            print("--- Cycle complete. Awaiting next candle. ---")

        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()
