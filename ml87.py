# ============================================================
#  Trading ML Server v22.0 (Full Realism Edition)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas_ta as ta
import uvicorn

# ============================================================
#  GLOBAL INDICATOR CACHE
# ============================================================
_indicator_cache = {}

def get_cached_df(key):
    return _indicator_cache.get(key)

def set_cached_df(key, df):
    _indicator_cache[key] = df

# ============================================================
#  SAFE COLUMN FINDER
# ============================================================
def find_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) == 1:
        return cols[0]
    for c in cols:
        if c == prefix:
            return c
    return cols[0] if cols else None

# ============================================================
#  CONFIG MODELS
# ============================================================
class StrategyConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class BotConfig(BaseModel):
    symbol: str
    timeframe: str
    interval_minutes: int
    strategies: List[StrategyConfig]
    regime: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    slippage_pct: float = 0.0005
    fee_pct: float = 0.0004

class BacktestConfig(BaseModel):
    symbol: str
    timeframe: str
    years: int = 1
    strategies: List[StrategyConfig]
    regime: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    slippage_pct: float = 0.0005
    fee_pct: float = 0.0004
    initial_balance: float = 1000.0
    allocation_pct: float = 1.0

# ============================================================
#  DATA LOADING
# ============================================================
def load_efficient_data(symbol: str, timeframe: str, years: int = 1) -> pd.DataFrame:
    period_map = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "4h": "730d",
        "1d": f"{365 * years}d"
    }
    yf_interval = timeframe if timeframe != "4h" else "1h"
    df = yf.download(symbol, interval=yf_interval, period=period_map.get(timeframe, "365d"), auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df.set_index("timestamp", inplace=True)
    if timeframe == "4h":
        df = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return df

# ============================================================
#  UNIFIED INDICATOR ENGINE
# ============================================================
def compute_indicators(df: pd.DataFrame, config: BacktestConfig | BotConfig) -> pd.DataFrame:
    key = (config.symbol, config.timeframe, str(config.dict()))
    cached = get_cached_df(key)
    if cached is not None:
        return cached.copy()
    data = df.copy()
    for s in config.strategies:
        if s.type == "RSI":
            rlen = s.params.get("length", 14)
            data.ta.rsi(length=rlen, append=True)
        elif s.type == "MACD":
            fast = s.params.get("fast", 12)
            slow = s.params.get("slow", 26)
            sig = s.params.get("signal", 9)
            data.ta.macd(fast=fast, slow=slow, signal=sig, append=True)
        elif s.type == "SMA":
            length = s.params.get("length", 20)
            data.ta.sma(length=length, append=True)
        elif s.type == "MA_Trend_Crossover":
            f = s.params.get("fast_length", 20)
            s_ = s.params.get("slow_length", 50)
            data.ta.sma(length=f, append=True)
            data.ta.sma(length=s_, append=True)
        elif s.type == "PSAR":
            af = s.params.get("af",0.02)
            max_af = s.params.get("max_af",0.2)
            data.ta.psar(af=af,max_af=max_af,append=True)
        elif s.type == "CCI":
            length = s.params.get("length",20)
            data.ta.cci(length=length,append=True)
        elif s.type == "Stochastic":
            k = s.params.get("k",14)
            d = s.params.get("d",3)
            smooth = s.params.get("smooth_k",3)
            data.ta.stoch(k=k,d=d,smooth_k=smooth,append=True)
        elif s.type in ["ADXSlope","Engulfing","BreakoutRetest"]:
            length = s.params.get("atr_length",14)
            data.ta.atr(length=length,append=True)
    if config.regime is not None and config.regime.get("type") == "ADX":
        length = config.regime.get("length",14)
        data.ta.adx(length=length, append=True)
    set_cached_df(key,data)
    return data.copy()

# ============================================================
#  STRATEGY SIGNAL ENGINE
# ============================================================
def generate_ta_signals(df: pd.DataFrame, config: BacktestConfig | BotConfig):
    df = df.copy()
    df["ta_signal"] = 0
    for strat in config.strategies:
        stype = strat.type
        p = strat.params
        if stype == "RSI": df = _sig_rsi(df,p)
        elif stype == "MACD": df = _sig_macd(df,p)
        elif stype == "SMA": df = _sig_sma(df,p)
        elif stype == "MA_Trend_Crossover": df = _sig_ma_crossover(df,p)
        elif stype == "PSAR": df = _sig_psar(df,p)
        elif stype == "CCI": df = _sig_cci(df,p)
        elif stype == "Stochastic": df = _sig_stoch(df,p)
        elif stype == "BreakoutRetest": df = _sig_breakout_retest(df,p)
        elif stype == "ADXSlope": df = _sig_adx_slope(df,p)
        elif stype == "Engulfing": df = _sig_engulfing(df,p)
    return df

# ----------------------- Strategy implementations -----------------------
def _sig_rsi(df,p):
    length = p.get("length",14)
    os = p.get("oversold",30)
    ob = p.get("overbought",70)
    rsi_col = find_col(df,f"RSI_{length}")
    if rsi_col is None: return df
    df.loc[df[rsi_col]<os,"ta_signal"]=1
    df.loc[df[rsi_col]>ob,"ta_signal"]=-1
    return df

def _sig_macd(df,p):
    fast = p.get("fast",12)
    slow = p.get("slow",26)
    sig = p.get("signal",9)
    macd = find_col(df,f"MACD_{fast}_{slow}_{sig}")
    macds = find_col(df,f"MACDs_{fast}_{slow}_{sig}")
    if macd is None or macds is None: return df
    buy = (df[macd]>df[macds]) & (df[macd].shift(1)<=df[macds].shift(1))
    sell = (df[macd]<df[macds]) & (df[macd].shift(1)>=df[macds].shift(1))
    df.loc[buy,"ta_signal"]=1
    df.loc[sell,"ta_signal"]=-1
    return df

def _sig_sma(df,p):
    length = p.get("length",20)
    sma = find_col(df,f"SMA_{length}")
    if sma is None: return df
    df.loc[df["close"]>df[sma],"ta_signal"]=1
    df.loc[df["close"]<df[sma],"ta_signal"]=-1
    return df

def _sig_ma_crossover(df,p):
    f = p.get("fast_length",20)
    s = p.get("slow_length",50)
    sma_f = find_col(df,f"SMA_{f}")
    sma_s = find_col(df,f"SMA_{s}")
    if sma_f is None or sma_s is None: return df
    buy = (df[sma_f]>df[sma_s]) & (df[sma_f].shift(1)<=df[sma_s].shift(1))
    sell = (df[sma_f]<df[sma_s]) & (df[sma_f].shift(1)>=df[sma_s].shift(1))
    df.loc[buy,"ta_signal"]=1
    df.loc[sell,"ta_signal"]=-1
    return df

def _sig_psar(df,p):
    psarl = find_col(df,"PSARl")
    psars = find_col(df,"PSARs")
    if psarl is None or psars is None: return df
    df.loc[df[psarl]>0,"ta_signal"]=1
    df.loc[df[psars]>0,"ta_signal"]=-1
    return df

def _sig_cci(df,p):
    length = p.get("length",20)
    low_t = p.get("oversold",-100)
    high_t = p.get("overbought",100)
    cci = find_col(df,f"CCI_{length}")
    if cci is None: return df
    df.loc[df[cci]<low_t,"ta_signal"]=1
    df.loc[df[cci]>high_t,"ta_signal"]=-1
    return df

def _sig_stoch(df,p):
    k_col = find_col(df,"STOCHk")
    d_col = find_col(df,"STOCHd")
    if k_col is None or d_col is None: return df
    buy = (df[k_col]>df[d_col]) & (df[k_col].shift(1)<=df[d_col].shift(1)) & (df[k_col]<20)
    sell = (df[k_col]<df[d_col]) & (df[k_col].shift(1)>=df[d_col].shift(1)) & (df[k_col]>80)
    df.loc[buy,"ta_signal"]=1
    df.loc[sell,"ta_signal"]=-1
    return df

def _sig_breakout_retest(df,p):
    atr_len = p.get("atr_length",14)
    mult = p.get("atr_mult",2.0)
    atr = find_col(df,f"ATR_{atr_len}")
    if atr is None: return df
    df["ema20"] = df["close"].ewm(span=20).mean()
    upper = df["ema20"] + df[atr]*mult
    lower = df["ema20"] - df[atr]*mult
    df.loc[df["close"]>upper,"ta_signal"]=1
    df.loc[df["close"]<lower,"ta_signal"]=-1
    return df

def _sig_adx_slope(df,p):
    length = p.get("length",14)
    slope = p.get("slope",0.1)
    adx = find_col(df,f"ADX_{length}")
    if adx is None: return df
    df["adx_slope"] = df[adx].diff()
    df.loc[df["adx_slope"]>slope,"ta_signal"]=1
    df.loc[df["adx_slope"]<-slope,"ta_signal"]=-1
    return df

def _sig_engulfing(df,p):
    body_prev = abs(df["close"].shift(1)-df["open"].shift(1))
    body_now = abs(df["close"]-df["open"])
    bull = (df["close"].shift(1)<df["open"].shift(1)) & (df["close"]>df["open"]) & (body_now>body_prev)
    bear = (df["close"].shift(1)>df["open"].shift(1)) & (df["close"]<df["open"]) & (body_now>body_prev)
    df.loc[bull,"ta_signal"]=1
    df.loc[bear,"ta_signal"]=-1
    return df

# ============================================================
#  UNIFIED TRADE & EXECUTION ENGINE
# ============================================================
class Position:
    def __init__(self, side:str, entry_price:float, capital:float, fee_pct:float, slippage_pct:float):
        self.side = side
        self.entry_price = entry_price
        self.capital = capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.qty = capital/entry_price
        self.entry_fee = capital*fee_pct
        self.entry_time = None
        self.exit_time = None

    def exit(self, exit_price:float):
        if self.side=="long":
            exit_price*=(1-self.slippage_pct)
            pnl_value = (exit_price-self.entry_price)*self.qty
        else:
            exit_price*=(1+self.slippage_pct)
            pnl_value = (self.entry_price-exit_price)*self.qty
        gross_value = self.capital+pnl_value
        exit_fee = gross_value*self.fee_pct
        net_profit = pnl_value-self.entry_fee-exit_fee
        return {"side":self.side,"entryPrice":self.entry_price,"exitPrice":exit_price,
                "qty":self.qty,"entryFee":self.entry_fee,"exitFee":exit_fee,"profit":net_profit}

class ExecutionEngine:
    def __init__(self, capital:float, fee_pct:float, slippage_pct:float):
        self.initial_capital = capital
        self.balance = capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.position: Position|None = None
        self.trades = []

    def enter(self, side:str, price:float, timestamp):
        if self.position is not None: return
        if side=="long": entry_price = price*(1+self.slippage_pct)
        else: entry_price = price*(1-self.slippage_pct)
        self.position = Position(side,entry_price,self.balance,self.fee_pct,self.slippage_pct)
        self.position.entry_time = timestamp

    def exit(self, price:float, timestamp):
        if self.position is None: return
        result = self.position.exit(price)
        result["entryTime"] = self.position.entry_time
        result["exitTime"] = timestamp
        self.balance += result["profit"]
        self.trades.append(result)
        self.position = None

    def force_close_last(self, last_price:float, timestamp):
        if self.position is not None:
            self.exit(last_price, timestamp)

# ============================================================
#  BACKTEST ENGINE
# ============================================================
def run_backtest(df: pd.DataFrame, config: BacktestConfig):
    if df.empty: return {"error":"No data"}
    exe = ExecutionEngine(config.initial_balance, config.fee_pct, config.slippage_pct)
    equity_curve=[]
    last_time=None
    for i in range(1,len(df)):
        row = df.iloc[i]
        sig = row["ta_signal"]
        price = float(row["close"])
        timestamp = row.name
        last_time = timestamp
        equity_curve.append({"timestamp":timestamp.isoformat(),"balance":exe.balance})
        if exe.position is None:
            if sig==1: exe.enter("long",price,timestamp)
            elif sig==-1: exe.enter("short",price,timestamp)
        else:
            if exe.position.side=="long" and sig==-1: exe.exit(price,timestamp)
            elif exe.position.side=="short" and sig==1: exe.exit(price,timestamp)
    if last_time is not None: exe.force_close_last(df["close"].iloc[-1],last_time)
    final_balance = exe.balance
    total_return = ((final_balance-config.initial_balance)/config.initial_balance*100)
    wins = [t for t in exe.trades if t["profit"]>0]
    exec_trades = exe.trades
    losses = [t for t in exec_trades if t["profit"]<0]
    max_dd=0
    peak=exe.initial_capital
    for e in equity_curve:
        b=e["balance"]
        peak=max(peak,b)
        dd=(peak-b)/peak*100
        max_dd=max(max_dd,dd)
    results={"metrics":{"totalReturn":total_return,"finalBalance":final_balance,"totalTrades":len(exe.trades),
                        "winningTrades":len(wins),"losingTrades":len(losses),
                        "winRate":(len(wins)/len(exe.trades)*100 if exe.trades else 0),
                        "maxDrawdown":max_dd},
             "tradeBreakdown":exe.trades,
             "equityCurve":equity_curve,
             "candleData":df[["open","high","low","close","volume"]].reset_index().to_dict("records")}
    return results

# ============================================================
#  PAPER TRADING BOT + API
# ============================================================
class PaperTradeConfig:
    initial_balance: float = 10000
    fee_pct: float = 0.0005
    slippage_pct: float = 0.0002

engine = ExecutionEngine(PaperTradeConfig.initial_balance, PaperTradeConfig.fee_pct, PaperTradeConfig.slippage_pct)
app = FastAPI()

@app.get("/backtest")
def api_backtest(symbol:str,timeframe:str="1h",years:int=1):
    df=load_efficient_data(symbol,timeframe,years)
    if df.empty: raise HTTPException(status_code=404,detail="Data not found")
    config=BacktestConfig(symbol=symbol,timeframe=timeframe,strategies=[StrategyConfig(type="RSI")])
    df=compute_indicators(df,config)
    df=generate_ta_signals(df,config)
    return run_backtest(df,config)

# ============================================================
#  RUN SERVER
# ============================================================
if __name__=="__main__":
    uvicorn.run("trading_ml_server_v22:app",host="0.0.0.0",port=8000,reload=True)
