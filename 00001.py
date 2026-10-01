import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import ccxt
from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)

# Allow Vercel frontend to access this API
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration ---
SUPPORTED_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", 
    "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "LTC-USD", 
    "DOT-USD", "SHIB-USD", "UNI-USD"
]

TOP_5_SYMBOLS = ["BTC-USD", "ETH-USD"]

FRED_SERIES_IDS = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL"
}

# --- 🚀 CACHING SYSTEM ---
# Stores data in memory: { "BTC-USD_ONE_HOUR": { "data": [...], "timestamp": 1234567890 } }
CACHE = {}
CACHE_TTL = 60  # Cache lives for 60 seconds

def get_from_cache(key):
    """Retrieve data if it exists and is fresh."""
    if key in CACHE:
        entry = CACHE[key]
        if time.time() - entry['timestamp'] < CACHE_TTL:
            print(f"⚡ Served {key} from CACHE")
            return entry['data']
    return None

def save_to_cache(key, data):
    """Save data to memory with current timestamp."""
    CACHE[key] = {
        "data": data,
        "timestamp": time.time()
    }

# --- Helper to Fix NaN JSON Crash ---
def clean_nan(obj):
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

# --- Data Fetching ---
def get_fred_data(start_date, end_date, api_key):
    # Check Cache for Macro Data
    cache_key = "MACRO_DATA"
    cached = get_from_cache(cache_key)
    if cached is not None: return pd.DataFrame(cached)

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    all_series_data = {}
    
    for name, series_id in FRED_SERIES_IDS.items():
        try:
            params = {
                "series_id": series_id, "api_key": api_key, "file_type": "json",
                "observation_start": start_date.strftime('%Y-%m-%d'),
                "observation_end": end_date.strftime('%Y-%m-%d'),
            }
            response = requests.get(base_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json().get('observations', [])
                df = pd.DataFrame(data)[['date', 'value']]
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                all_series_data[name] = df['value']
        except:
            all_series_data[name] = pd.Series(dtype=float)
    
    macro_df = pd.DataFrame(all_series_data)
    macro_df.ffill(inplace=True)
    
    # Save formatted dict to cache to avoid re-parsing
    save_to_cache(cache_key, macro_df.to_dict())
    return macro_df

def get_single_symbol_data(symbol, start_time, end_time, granularity="ONE_DAY"):
    try:
        exchange = ccxt.coinbase()
        normalized = symbol.replace('-', '/')
        
        timeframe_map = {'ONE_DAY': '1d', 'ONE_HOUR': '1h', 'SIX_HOUR': '6h'}
        tf = timeframe_map.get(granularity, '1d')
        
        # Limit fetch to speed it up (300 candles is plenty for a quick chart)
        ohlcv = exchange.fetch_ohlcv(normalized, tf, limit=300)
        
        if not ohlcv: return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('time', inplace=True)
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        df['start'] = (df['timestamp'] / 1000).astype(int)
        
        return df
    except Exception as e:
        print(f"[CCXT ERROR] {symbol}: {e}")
        return None

def get_top_market_data():
    fred_api_key = os.environ.get("FRED_API_KEY")
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)
    
    all_crypto_data = {}
    
    # Use cached macro data if available inside get_fred_data
    macro_df = get_fred_data(start_time, end_time, fred_api_key) if fred_api_key else pd.DataFrame()
    if isinstance(macro_df, dict): macro_df = pd.DataFrame(macro_df) # Rehydrate from cache if needed

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_single_symbol_data, sym, start_time, end_time, "ONE_DAY"): sym for sym in TOP_5_SYMBOLS}
        for future in futures:
            sym = futures[future]
            try:
                df = future.result()
                if df is not None:
                    if not macro_df.empty:
                        df = df.join(macro_df, how='left').ffill()
                    all_crypto_data[sym] = df.replace({np.nan: None})
            except: pass
    return all_crypto_data

# --- API ENDPOINTS ---

@app.route('/api/symbols', methods=['GET'])
def get_symbols():
    return jsonify(SUPPORTED_SYMBOLS)

@app.route('/api/candles', methods=['GET'])
def get_candles():
    symbol = request.args.get('product_id', 'BTC-USD')
    granularity = request.args.get('granularity', 'ONE_HOUR')
    
    # 🚀 CHECK CACHE FIRST
    cache_key = f"{symbol}_{granularity}"
    cached_data = get_from_cache(cache_key)
    if cached_data:
        return jsonify(cached_data)

    end_time = datetime.now(timezone.utc)
    days_back = 14 if granularity == 'ONE_HOUR' else 300
    start_time = end_time - timedelta(days=days_back)

    df = get_single_symbol_data(symbol, start_time, end_time, granularity)
    
    if df is None or df.empty:
        return jsonify([])
    
    df = df.replace({np.nan: None})
    df_reset = df.reset_index()
    records = []
    for _, row in df_reset.iterrows():
        records.append({
            "start": int(row['start']),
            "time": row['time'].isoformat(),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
    
    final_data = clean_nan(records)
    
    # 🚀 SAVE TO CACHE
    save_to_cache(cache_key, final_data)
    
    return jsonify(final_data)

@app.route('/api/data', methods=['GET'])
def get_market_data_api():
    # Cache the heavy homepage load too
    cache_key = "HOME_DATA"
    cached_data = get_from_cache(cache_key)
    if cached_data: return jsonify(cached_data)

    try:
        data_dict = get_top_market_data()
        json_payload = {}
        for symbol, df in data_dict.items():
            df_reset = df.reset_index()
            records = df_reset.to_dict(orient='records')
            
            cleaned_records = []
            for r in records:
                r['time'] = r['time'].isoformat() if isinstance(r['time'], pd.Timestamp) else str(r['time'])
                cleaned_records.append(r)
                
            json_payload[symbol] = clean_nan(cleaned_records)
        
        save_to_cache(cache_key, json_payload)
        return jsonify(json_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
