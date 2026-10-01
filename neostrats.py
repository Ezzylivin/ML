import pandas as pd
import pandas_ta as ta
import numpy as np
import optuna
import os, sys, logging, time, json, glob
from colorama import Fore, Style, init
from numba import njit

# 🟢 INITIALIZATION
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format=f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TurboDragnet")

# Global Settings
USER_SETTINGS = {
    "commission": 0.0006,
    "slippage": 0.001,
    "train_df": None
}

# --- 🟢 1. HELPER FUNCTIONS ---
def get_input(prompt, default):
    v = input(f"{Fore.YELLOW}{prompt} [{default}]: ").strip()
    return v if v else default

def find_data_directory():
    search_paths = ["data", "Project/ML/data", "/root/Project/ML/data", "../data", "."]
    for path in search_paths:
        if os.path.isdir(path) and glob.glob(os.path.join(path, "*.csv")): return path
    return None

def select_target_file(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"\n{Fore.YELLOW}📂 DATA FOUND IN: {data_dir}")
    print(f"{Fore.WHITE}{'-'*40}")
    default_idx = 0
    for i, f in enumerate(files):
        fname = os.path.basename(f)
        if "BTC" in fname and "1h" in fname: default_idx = i
        color = Fore.GREEN if "BTC" in fname else Fore.WHITE
        print(f"{color}[{i}] {fname}")
    print(f"{Fore.WHITE}{'-'*40}")
    try:
        sel = input(f"{Fore.CYAN}👉 Select File ID [Default {default_idx}]: ").strip()
        idx = int(sel) if sel else default_idx
        return files[idx]
    except:
        logger.error("❌ Invalid selection.")
        sys.exit()

def load_csv_data(file_path):
    """
    v72.0 Upgrade: Smart Timestamp Detection (ms vs s vs string)
    """
    if not os.path.exists(file_path): return None
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        
        # 2. Normalize Columns
        df.columns = [c.lower() for c in df.columns]
        time_col = next((c for c in df.columns if c in ['timestamp', 'time', 'date', 'datetime']), None)
        
        if not time_col:
            print("❌ No time column found!")
            return None
            
        # 3. SMART CONVERSION 🧠
        # Check the first value to decide how to parse
        first_val = df[time_col].iloc[0]
        
        if isinstance(first_val, (int, float, np.number)):
            if first_val > 1e11: # > 100 Billion implies Milliseconds (e.g. 1600000000000)
                df[time_col] = pd.to_datetime(df[time_col], unit='ms', utc=True)
            else: # Likely Seconds (e.g. 1600000000)
                df[time_col] = pd.to_datetime(df[time_col], unit='s', utc=True)
        else:
            # Fallback to string parsing
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')

        # 4. Finalize
        df.rename(columns={time_col: 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return None

# --- 🟢 2. STRATEGY LIBRARY ---
def calc_strategy_signal(df, code, params):
    close = df['close']
    high = df['high']
    low = df['low']
    signal = pd.Series(0, index=df.index)

    try:
        if code == "sma_crossover":
            fast = ta.sma(close, length=params['fast_sma'])
            slow = ta.sma(close, length=params['slow_sma'])
            signal[fast > slow] = 1
            signal[fast < slow] = -1

        elif code == "rsi_threshold":
            rsi = ta.rsi(close, length=params['length'])
            signal[rsi < params['limit']] = 1
            signal[rsi > (100 - params['limit'])] = -1
            
        elif code == "macd_crossover":
            macd = ta.macd(close, fast=params['fast'], slow=params['slow'], signal=params['sig'])
            if macd is not None:
                macd_line = macd.iloc[:, 0]
                sig_line = macd.iloc[:, 2]
                signal[macd_line > sig_line] = 1
                signal[macd_line < sig_line] = -1

        elif code == "bollinger_bands":
            bb = ta.bbands(close, length=params['length'], std=params['std'])
            if bb is not None:
                lower = bb.iloc[:, 0]
                upper = bb.iloc[:, 2]
                signal[close < lower] = 1
                signal[close > upper] = -1

        elif code == "atr_breakout":
            atr = ta.atr(high, low, close, length=14)
            baseline = ta.sma(close, length=20)
            upper = baseline + (atr * params['mult'])
            lower = baseline - (atr * params['mult'])
            signal[close > upper] = 1
            signal[close < lower] = -1

        elif code == "stochastic_cross":
            stoch = ta.stoch(high, low, close)
            if stoch is not None:
                k, d = stoch.iloc[:, 0], stoch.iloc[:, 1]
                signal[(k > d) & (k < 30)] = 1 
                signal[(k < d) & (k > 70)] = -1

        elif code == "ichimoku_cloud":
            ichi = ta.ichimoku(high, low, close)[0]
            if ichi is not None:
                span_a, span_b = ichi.iloc[:, 0], ichi.iloc[:, 1]
                signal[(close > span_a) & (close > span_b)] = 1
                signal[(close < span_a) & (close < span_b)] = -1

        elif code == "psar_signal":
            psar = ta.psar(high, low, close)
            if psar is not None:
                long_col = next((c for c in psar.columns if "PSARl" in c), None)
                short_col = next((c for c in psar.columns if "PSARs" in c), None)
                if long_col: signal[psar[long_col] > 0] = 1
                if short_col: signal[psar[short_col] > 0] = -1

        elif code == "obv_trend":
            obv = ta.obv(close, df['volume'])
            obv_sma = ta.sma(obv, length=20)
            signal[obv > obv_sma] = 1
            signal[obv < obv_sma] = -1

        elif code == "rsi_divergence":
            rsi = ta.rsi(close, length=14)
            price_low = low.rolling(10).min()
            rsi_low = rsi.rolling(10).min()
            signal[(low == price_low) & (rsi > rsi_low) & (rsi < 40)] = 1
            price_high = high.rolling(10).max()
            rsi_high = rsi.rolling(10).max()
            signal[(high == price_high) & (rsi < rsi_high) & (rsi > 60)] = -1

    except Exception: return pd.Series(0, index=df.index)
    return signal.fillna(0)

def generate_consensus_signal(df, active_strategies, rule, adx_level):
    # 1. ADX Filter
    trend_ok = pd.Series(1, index=df.index)
    if adx_level > 0:
        if not any("adx" in c.lower() for c in df.columns): 
            try: df.ta.adx(append=True)
            except: pass
        adx_cols = [c for c in df.columns if "adx" in c.lower()]
        if adx_cols:
            trend_ok = (df[adx_cols[0]] > adx_level).astype(int)

    # 2. Strategy Signals
    raw_signals = []
    for strat in active_strategies:
        s_sig = calc_strategy_signal(df, strat['code'], strat['params'])
        raw_signals.append(s_sig)

    if not raw_signals: return None

    # 3. Vote Logic
    sig_df = pd.concat(raw_signals, axis=1).fillna(0)
    vote_sum = sig_df.sum(axis=1)
    
    final_signal = pd.Series(0, index=df.index)
    if rule == 'Majority':
        final_signal[vote_sum > 0] = 1
        final_signal[vote_sum < 0] = -1
    else: # Unanimous
        n = len(active_strategies)
        final_signal[vote_sum == n] = 1
        final_signal[vote_sum == -n] = -1

    return (final_signal * trend_ok).values 

# --- ⚡ 3. THE NUMBA ENGINE ---
@njit(fastmath=True)
def fast_backtest(prices, highs, lows, signals, comm, ts_pct, sl_pct, tp_pct):
    balance = 1000.0
    position = 0 
    entry_price = 0.0
    best_price = 0.0 
    trades_count = 0
    max_balance = 1000.0
    max_dd = 0.0
    
    for i in range(1, len(prices)):
        price = prices[i]
        high = highs[i]
        low = lows[i]
        signal = signals[i-1] 
        
        # EXIT LOGIC
        if position == 1:
            if high > best_price: best_price = high
            stop_price = best_price * (1 - ts_pct)
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
            
            if low < sl_price: 
                balance *= (sl_price / entry_price) * (1 - comm)
                position = 0
            elif low < stop_price: 
                balance *= (stop_price / entry_price) * (1 - comm)
                position = 0
            elif high > tp_price: 
                balance *= (tp_price / entry_price) * (1 - comm)
                position = 0

        elif position == -1:
            if low < best_price: best_price = low
            stop_price = best_price * (1 + ts_pct)
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
            
            if high > sl_price:
                balance *= (entry_price / sl_price) * (1 - comm)
                position = 0
            elif high > stop_price:
                balance *= (entry_price / stop_price) * (1 - comm)
                position = 0
            elif low < tp_price:
                balance *= (entry_price / tp_price) * (1 - comm)
                position = 0

        # ENTRY LOGIC
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                best_price = price
                balance *= (1 - comm)
                trades_count += 1
            elif signal == -1:
                position = -1
                entry_price = price
                best_price = price
                balance *= (1 - comm)
                trades_count += 1
                
        elif position == 1 and signal == -1:
            balance *= (price / entry_price) * (1 - comm)
            position = -1
            entry_price = price
            best_price = price
            balance *= (1 - comm)
            trades_count += 1
            
        elif position == -1 and signal == 1:
            balance *= (entry_price / price) * (1 - comm)
            position = 1
            entry_price = price
            best_price = price
            balance *= (1 - comm)
            trades_count += 1

        if balance > max_balance: max_balance = balance
        if max_balance > 0:
            dd = (max_balance - balance) / max_balance
            if dd > max_dd: max_dd = dd

    return balance, trades_count, max_dd

# --- 🟢 4. OPTIMIZATION OBJECTIVE ---
def objective(trial):
    df_train = USER_SETTINGS['train_df']
    
    num_strats = trial.suggest_int("num_strategies", 1, 3)
    rule = trial.suggest_categorical("rule", ["Majority", "Unanimous"])
    adx_level = trial.suggest_int("adx", 0, 30)
    
    ts_pct = trial.suggest_float("trailing_stop", 0.01, 0.10)
    sl_pct = trial.suggest_float("stop_loss", 0.02, 0.15)
    tp_pct = trial.suggest_float("take_profit", 0.05, 0.30)
    
    active_strats = []
    available = ["sma_crossover", "rsi_threshold", "macd_crossover", "bollinger_bands", "atr_breakout", "stochastic_cross", "ichimoku_cloud", "psar_signal", "obv_trend", "rsi_divergence"]
    
    for i in range(num_strats):
        code = trial.suggest_categorical(f"s{i}_code", available)
        p = {}
        if code == "sma_crossover":
            p = {"fast_sma": trial.suggest_int(f"s{i}_f", 5, 50), "slow_sma": trial.suggest_int(f"s{i}_s", 60, 200)}
        elif code == "rsi_threshold":
            p = {"length": 14, "limit": trial.suggest_int(f"s{i}_rlim", 20, 45)}
        elif code == "macd_crossover":
            p = {"fast": 12, "slow": 26, "sig": 9}
        elif code == "bollinger_bands":
            p = {"length": 20, "std": trial.suggest_float(f"s{i}_std", 1.5, 3.0)}
        elif code == "atr_breakout":
            p = {"mult": trial.suggest_float(f"s{i}_mult", 1.5, 5.0)}
        
        active_strats.append({"code": code, "params": p})

    prices = df_train['close'].values
    highs = df_train['high'].values
    lows = df_train['low'].values
    signals = generate_consensus_signal(df_train, active_strats, rule, adx_level)
    
    if signals is None: return -100

    comm = USER_SETTINGS['commission'] + USER_SETTINGS['slippage']
    final_balance, trades, max_dd = fast_backtest(prices, highs, lows, signals, comm, ts_pct, sl_pct, tp_pct)
    
    if trades < 5: return -100

    roi = ((final_balance - 1000.0) / 1000.0) * 100
    score = roi - (max_dd * 100 * 2) 
    
    trial.set_user_attr("roi", roi)
    trial.set_user_attr("dd", max_dd * 100)
    trial.set_user_attr("trades", trades)
    trial.set_user_attr("params", active_strats)
    
    return score

# --- 🟢 5. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"{Fore.MAGENTA}{'='*60}\n🚀 TURBO DRAGNET v72.0 - SMART DATE FIX\n{'='*60}")
    
    data_dir = find_data_directory()
    target_file = select_target_file(data_dir)
    FULL_DF = load_csv_data(target_file)
    
    if FULL_DF is None or FULL_DF.empty:
        print("❌ Critical Error: Could not load data.")
        sys.exit()

    min_date = FULL_DF.index.min().date()
    max_date = FULL_DF.index.max().date()
    
    print(f"\n{Fore.CYAN}📅 AVAILABLE DATA RANGE:")
    print(f"   Start: {Fore.GREEN}{min_date}")
    print(f"   End:   {Fore.GREEN}{max_date}")
    print(f"{Fore.WHITE}{'-'*40}")
    
    default_split = min_date + (max_date - min_date) * 0.7
    t_start = get_input("Training Start Date", str(min_date))
    t_end = get_input("Training End Date  ", str(default_split))
    
    TRIALS = int(get_input("Trials", "1000"))
    USER_SETTINGS['commission'] = float(get_input("Commission (e.g. 0.0006)", "0.0006"))
    USER_SETTINGS['slippage'] = float(get_input("Slippage (e.g. 0.001)", "0.001"))
    
    try:
        USER_SETTINGS['train_df'] = FULL_DF.loc[t_start:t_end].copy()
        print(f"{Fore.GREEN}✅ Loaded {len(USER_SETTINGS['train_df'])} training candles.")
        
        print(f"{Fore.YELLOW}⚙️ Compiling Numba Engine...")
        dummy = np.array([100.0, 101.0, 99.0] * 10)
        fast_backtest(dummy, dummy, dummy, np.zeros(30), 0.001, 0.05, 0.05, 0.10)
        print(f"{Fore.GREEN}✅ Engine Ready.")
        
    except Exception as e: 
        print(f"{Fore.RED}❌ Setup Error: {e}")
        sys.exit()

    print(f"{Fore.CYAN}🔥 Running {TRIALS} Simulations...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS, n_jobs=1)
    
    print(f"\n{Fore.MAGENTA}{'='*60}\n🏆 ELITE STRATEGIES (Sorted by ROI)\n{'='*60}")
    
    candidates = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("roi", -99) > 0]
    candidates.sort(key=lambda x: x.user_attrs.get("roi", -99), reverse=True)
    
    for i, t in enumerate(candidates[:5]):
        print(f"\n{Fore.GREEN}💎 Rank #{i+1} | ROI: {t.user_attrs['roi']:.2f}% | DD: {t.user_attrs['dd']:.2f}% | Trades: {t.user_attrs['trades']}")
        print(f"   Risk: TS={t.params['trailing_stop']:.1%} | SL={t.params['stop_loss']:.1%} | TP={t.params['take_profit']:.1%}")
        print(f"   Strats: {json.dumps(t.user_attrs['params'], indent=2)}")
