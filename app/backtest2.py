import pandas as pd
import pandas_ta as ta
import os
import asyncio
import logging
import numpy as np
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("BacktestEngine")

class RawModelAdapter:
    def __init__(self, model_payload):
        if isinstance(model_payload, dict):
            self.model = model_payload.get('model')
            self.feature_names = model_payload.get('feature_names', [])
            self.scaler = model_payload.get('scaler', None)
        else:
            self.model = model_payload
            self.feature_names = []
            self.scaler = None

    def predict_direction(self, df):
        try:
            last_row = df.iloc[[-1]].copy()

            # 🟢 AUTO-FIX: Map Backtester columns to Model columns
            # The model might want "BBL_20_2.0_2.0" but we have "BBL_20_2.0"
            if self.feature_names:
                # 1. Identify what we have
                available_cols = list(last_row.columns)
                
                # 2. Identify what is missing
                missing = [f for f in self.feature_names if f not in available_cols]
                
                # 3. Try to fuzzy match the missing ones
                if missing:
                    # Find our BBL/BBU columns
                    our_bbl = next((c for c in available_cols if c.startswith("BBL")), None)
                    our_bbu = next((c for c in available_cols if c.startswith("BBU")), None)
                    
                    # Find model's BBL/BBU requirements
                    model_bbl = next((f for f in missing if f.startswith("BBL")), None)
                    model_bbu = next((f for f in missing if f.startswith("BBU")), None)

                    # Rename if found
                    if our_bbl and model_bbl:
                        last_row.rename(columns={our_bbl: model_bbl}, inplace=True)
                    if our_bbu and model_bbu:
                        last_row.rename(columns={our_bbu: model_bbu}, inplace=True)

                # 4. Check again
                # If still missing, we really can't proceed
                final_missing = [f for f in self.feature_names if f not in last_row.columns]
                if final_missing:
                    # print(f"❌ Still missing: {final_missing}") # Debug line
                    return 0.5 
                
                X = last_row[self.feature_names]
            else:
                # Fallback for legacy models
                cols_to_exclude = ['datetime', 'timestamp', 'time', 'date', 'target']
                X = last_row.drop(columns=[c for c in cols_to_exclude if c in last_row.columns], errors='ignore')

            # Scale
            if self.scaler:
                X = self.scaler.transform(X)

            # Predict
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                if len(probs) == 3:
                    return probs[2]
                else:
                    return probs[1]
            else:
                return float(self.model.predict(X)[0])
                
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return 0.5

    def predict_direction_batch(self, df):
        """Vectorized predict: one buy-probability per row, in a single
        predict_proba call. Each row's prediction uses only that row's features
        (same as predict_direction on an expanding slice), so this is
        mathematically identical to the per-bar loop but O(n) instead of O(n^2)."""
        try:
            frame = df.copy()
            if self.feature_names:
                available = list(frame.columns)
                missing = [f for f in self.feature_names if f not in available]
                if missing:
                    our_bbl = next((c for c in available if c.startswith("BBL")), None)
                    our_bbu = next((c for c in available if c.startswith("BBU")), None)
                    model_bbl = next((f for f in missing if f.startswith("BBL")), None)
                    model_bbu = next((f for f in missing if f.startswith("BBU")), None)
                    if our_bbl and model_bbl:
                        frame.rename(columns={our_bbl: model_bbl}, inplace=True)
                    if our_bbu and model_bbu:
                        frame.rename(columns={our_bbu: model_bbu}, inplace=True)
                final_missing = [f for f in self.feature_names if f not in frame.columns]
                if final_missing:
                    return np.full(len(df), 0.5)
                X = frame[self.feature_names]
            else:
                cols_to_exclude = ['datetime', 'timestamp', 'time', 'date', 'target']
                X = frame.drop(columns=[c for c in cols_to_exclude if c in frame.columns], errors='ignore')

            if self.scaler:
                X = self.scaler.transform(X)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                col = 2 if proba.shape[1] == 3 else 1
                return proba[:, col]
            return np.asarray(self.model.predict(X), dtype=float)
        except Exception as e:
            logger.error(f"Batch Prediction Error: {e}")
            return np.full(len(df), 0.5)

class Backtester:
    def __init__(self, config: dict):
        self.config = config # Keep original for reference
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        
        # 🟢 Use the exact keys from your Pydantic model
        self.start_str = config.get('startDate')
        self.end_str = config.get('endDate')
        
        # Convert for internal logic
        self.start_date = pd.to_datetime(self.start_str).tz_localize(None) if self.start_str else None
        self.end_date = pd.to_datetime(self.end_str).tz_localize(None) if self.end_str else None
        
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.model_name = config.get('mlModel')
        self.params = config.get('params', {})
        self.ml_conf_threshold = float(config.get('ml_confidence_threshold', 0.10))

    async def load_data(self):
        """
        Refreshes data from US-based exchanges if local file is missing or outdated.
        """
        try:
            # 🟢 CALL THE FETCH-ON-DEMAND LOGIC
            # We 'await' this because ensure_full_data is an async function
            from main4 import ensure_full_data 
            df = await ensure_full_data(
                self.symbol, 
                self.timeframe, 
                self.start_str, 
                self.end_str
            )
            
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Data gap could not be filled.")
            
            # Ensure index is clean
            df.index = df.index.tz_localize(None)
            # FIX: slice to the requested window. ensure_full_data returns the FULL
            # cached history, so without this every backtest ran over all data and
            # silently ignored startDate/endDate (identical results for any range).
            if self.start_date is not None:
                df = df[df.index >= self.start_date]
            if self.end_date is not None:
                df = df[df.index <= self.end_date]
            return df
            
        except Exception as e:
            logger.error(f"Data Sync Error: {e}")
            raise HTTPException(status_code=500, detail=f"Data Sync Failed: {str(e)}")

    def calculate_indicators(self, df):
        if len(df) < 50: return df 
        
        # --- MATCHING YOUR TRAINING LOGIC ---
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if st is not None:
            df['st_trend'] = st.iloc[:, 1]
        else:
            df['st_trend'] = 0

        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx.iloc[:, 0]
        else:
            df['adx'] = 0
            
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
            
        return df.dropna()

    # ---- ML batch inference (correct, crash-safe) -------------------------
    def _feature_frame(self, df, feats):
        frame = df.copy()
        missing = [f for f in feats if f not in frame.columns]
        if missing:
            our_bbl = next((c for c in frame.columns if c.startswith("BBL")), None)
            our_bbu = next((c for c in frame.columns if c.startswith("BBU")), None)
            model_bbl = next((f for f in missing if f.startswith("BBL")), None)
            model_bbu = next((f for f in missing if f.startswith("BBU")), None)
            if our_bbl and model_bbl: frame.rename(columns={our_bbl: model_bbl}, inplace=True)
            if our_bbu and model_bbu: frame.rename(columns={our_bbu: model_bbu}, inplace=True)
        final_missing = [f for f in feats if f not in frame.columns]
        if final_missing:
            return None
        return frame[feats]

    def _expert_batch(self, model_type, df):
        """Per-row buy-probability for one expert over the whole frame. Always
        returns a 1-D array of length len(df); degrades to 0.5 on any failure."""
        fa = ModelFactory.load_model(model_type, self.symbol, self.timeframe)
        if fa is None:
            return np.full(len(df), 0.5)
        try:
            if hasattr(fa.model, "input_shape"):  # keras transformer -> windowed batch
                import tensorflow as tf
                ff = self._feature_frame(df, fa.feature_names)
                if ff is None:
                    return np.full(len(df), 0.5)
                arr = ff.values.astype('float32')
                if fa.norm_mean is not None and fa.norm_std is not None:
                    arr = (arr - fa.norm_mean) / fa.norm_std
                N, W = len(arr), 50
                out = np.full(N, 0.5)
                if N >= W:
                    idx = np.arange(W)[None, :] + np.arange(N - W + 1)[:, None]
                    windows = arr[idx]  # (M, W, F)
                    preds = fa.model(tf.convert_to_tensor(windows), training=False).numpy()
                    p = preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]
                    out[W - 1:] = p
                return out
            ff = self._feature_frame(df, fa.feature_names)
            if ff is None:
                return np.full(len(df), 0.5)
            if hasattr(fa.model, "predict_proba"):
                proba = fa.model.predict_proba(ff)
                col = 2 if proba.shape[1] == 3 else 1
                return np.asarray(proba[:, col]).ravel()
            return np.asarray(fa.model.predict(ff), dtype=float).ravel()
        except Exception as e:
            logger.error(f"expert batch {model_type} failed: {e}")
            return np.full(len(df), 0.5)

    def _ml_probs(self, df):
        """Return a 1-D array (len==len(df)) of buy-probabilities for the
        configured model, or None. Rebuilds the council for the stacking judge."""
        fa = ModelFactory.load_model(self.model_name, self.symbol, self.timeframe)
        if fa is None:
            return None
        try:
            if getattr(fa, 'is_meta_model', False):
                xgb = self._expert_batch('xgboost', df)
                rf  = self._expert_batch('randomforest', df)
                tfb = self._expert_batch('transformer', df)
                stack = np.column_stack([xgb, rf, tfb])
                proba = fa.model.predict_proba(stack)
                col = 2 if proba.shape[1] == 3 else 1
                return np.asarray(proba[:, col]).ravel()
            return np.asarray(self._expert_batch(self.model_name, df)).ravel()
        except Exception as e:
            logger.error(f"_ml_probs failed: {e}")
            return None

    def _prepare_frame(self, df):
        """FIX #18: build the FULL FEATURE_COLUMNS set with the SAME engine the
        live bot and trainer use (apply_mega_features), so the ML council
        receives all ~25 features instead of the ~7 the old calculate_indicators
        built — which forced every expert to 0.5 and pinned the regime to RANGE.
        Falls back to the basic indicators if the shared engine is unavailable."""
        try:
            from app.verify.engineer_and_train import apply_mega_features
            out, _ = apply_mega_features(df.copy())
            if out is not None and len(out) >= 10 and 'sma_50' in out.columns and 'rsi' in out.columns:
                return out
            logger.warning("apply_mega_features returned too little; using basic indicators.")
        except Exception as e:
            logger.error(f"apply_mega_features failed ({e}); using basic indicators.")
        return self.calculate_indicators(df)

    def _strategy_votes(self, df, strategies):
        """Vectorized per-bar (votes, weighted_votes) for the configured strategies,
        mirroring live StrategyBrain.calculate_signals so the backtest tests the
        SAME signal the bot trades — not a fixed sma50/rsi rule. O(strategies) work,
        not O(bars). Returns two 1-D arrays of length len(df)."""
        n = len(df)
        close = df['close']; high = df['high']; low = df['low']
        vol = df['volume'] if 'volume' in df.columns else pd.Series(np.ones(n), index=df.index)
        price = close.values.astype(float)
        votes = np.zeros(n); weighted = np.zeros(n)

        def clip(a, lo, hi):
            return np.clip(np.nan_to_num(a, nan=lo), lo, hi)

        try: ema20 = ta.ema(close, length=20).values.astype(float)
        except Exception: ema20 = price.copy()
        try:
            bb = ta.bbands(close, length=20, std=2.0)
            bb_l = bb.iloc[:, 0].values.astype(float); bb_m = bb.iloc[:, 1].values.astype(float); bb_u = bb.iloc[:, 2].values.astype(float)
        except Exception:
            bb_l = price * 0.99; bb_m = price.copy(); bb_u = price * 1.01

        for strat in (strategies or []):
            if not isinstance(strat, dict):
                continue
            code = strat.get('code'); p = strat.get('params', {}) or {}
            raw = np.zeros(n); conf = np.full(n, 0.5)
            try:
                if code == 'rsi_threshold':
                    r = ta.rsi(close, length=int(p.get('rsi_length', 14))).values.astype(float)
                    dist = np.minimum(np.abs(r - 30), np.abs(r - 70)); conf = clip(1.0 - dist / 40, 0.1, 1.0)
                    raw = np.where(r < p.get('oversold', 30), 1.0, np.where(r > p.get('overbought', 70), -1.0, 0.0))
                elif code == 'sma_crossover':
                    f = ta.sma(close, length=int(p.get('fast_sma', 50))).values.astype(float)
                    s = ta.sma(close, length=int(p.get('slow_sma', 200))).values.astype(float)
                    gap = np.abs(f - s) / np.where(s == 0, np.nan, s); conf = clip(0.5 + gap * 25, 0.1, 1.0); raw = np.where(f > s, 1.0, -1.0)
                elif code == 'macd_crossover':
                    m = ta.macd(close, fast=int(p.get('fast', 12)))
                    macd_line = m.iloc[:, 0].values.astype(float); hist = m.iloc[:, 1].values.astype(float); sigl = m.iloc[:, 2].values.astype(float)
                    conf = clip(np.abs(hist) / (price * 0.0005), 0.1, 1.0); raw = np.where(macd_line > sigl, 1.0, -1.0)
                elif code == 'supertrend':
                    st = ta.supertrend(high, low, close)
                    st_val = st.iloc[:, 0].values.astype(float); st_dir = st.iloc[:, 1].values.astype(float)
                    dist = np.abs(price - st_val) / np.where(price == 0, np.nan, price); conf = clip(0.5 + dist * 10, 0.1, 1.0); raw = np.where(st_dir == 1, 1.0, -1.0)
                elif code == 'bb_fade':
                    bw = np.maximum(bb_u - bb_l, price * 0.001)
                    raw = np.where(price < bb_l, 1.0, np.where(price > bb_u, -1.0, 0.0))
                    conf = np.where(price < bb_l, clip(0.5 + (bb_l - price) / bw * 2, 0.1, 1.0),
                            np.where(price > bb_u, clip(0.5 + (price - bb_u) / bw * 2, 0.1, 1.0), 0.25))
                elif code == 'atr_breakout':
                    atr = ta.atr(high, low, close).values.astype(float)
                    target = ema20 + atr * float(p.get('multiplier', 1.5))
                    dist = np.abs(price - target) / np.where(target == 0, np.nan, target); conf = clip(0.5 + dist * 5, 0.1, 1.0); raw = np.where(price > target, 1.0, -1.0)
                elif code == 'pa_breakout':
                    hh = high.rolling(int(p.get('lookback', 20))).max().values.astype(float)
                    dist = np.abs(price - hh) / np.where(hh == 0, np.nan, hh); conf = clip(0.5 + dist * 5, 0.1, 1.0); raw = np.where(price >= hh, 1.0, -1.0)
                elif code == 'vol_profile':
                    vma = ta.sma(vol, length=int(p.get('vol_ma', 20))).values.astype(float)
                    ratio = vol.values.astype(float) / np.where(vma == 0, np.nan, vma * float(p.get('threshold', 1.5)))
                    conf = clip(ratio, 0.1, 1.0); raw = np.where(ratio >= 1.0, np.where(price > bb_m, 1.0, -1.0), 0.0)
                elif code == 'stoch':
                    stk = ta.stoch(high, low, close).iloc[:, 0].values.astype(float)
                    dist = np.minimum(np.abs(stk - 20), np.abs(stk - 80)); conf = clip(1.0 - dist / 40, 0.1, 1.0)
                    raw = np.where(stk < 20, 1.0, np.where(stk > 80, -1.0, 0.0))
                elif code == 'ema_cloud':
                    fe = ta.ema(close, length=int(p.get('fast_ema', 9))).values.astype(float)
                    se = ta.ema(close, length=int(p.get('slow_ema', 21))).values.astype(float)
                    gap = np.abs(fe - se) / np.where(se == 0, np.nan, se); conf = clip(0.5 + gap * 50, 0.1, 1.0); raw = np.where(fe > se, 1.0, -1.0)
                # ---- NEW INPUT: cross-asset / regime (need a 'btc_close' column) ----
                elif code == 'btc_regime':
                    if 'btc_close' not in df.columns:
                        continue
                    bc = df['btc_close'].values.astype(float)
                    bt_ema = pd.Series(bc).ewm(span=int(p.get('span', 50)), adjust=False).mean().values
                    raw = np.where(bc > bt_ema, 1.0, -1.0)     # risk-on only when BTC trends up
                    conf = np.full(n, 0.6)
                elif code == 'rel_strength':
                    if 'btc_close' not in df.columns:
                        continue
                    lb = int(p.get('lookback', 20))
                    sym_ret = pd.Series(price).pct_change(lb).values
                    btc_ret = pd.Series(df['btc_close'].values.astype(float)).pct_change(lb).values
                    diff = sym_ret - btc_ret
                    raw = np.where(diff > 0, 1.0, -1.0)         # long when outperforming BTC
                    conf = clip(0.5 + np.abs(diff) * 5, 0.1, 1.0)
                # ---- NEW INPUT: perp funding (need a 'funding' column) ----
                elif code == 'funding_extreme':
                    if 'funding' not in df.columns:
                        continue
                    fnd = df['funding'].values.astype(float)
                    thr = float(p.get('threshold', 0.0005))    # per-interval funding extreme
                    raw = np.where(fnd > thr, -1.0, np.where(fnd < -thr, 1.0, 0.0))  # fade the crowd
                    conf = clip(0.5 + np.abs(fnd) / (thr * 4.0), 0.1, 1.0)
                else:
                    continue
            except Exception as e:
                logger.warning(f"backtest strategy {code} failed: {e}")
                continue
            raw = np.nan_to_num(raw, nan=0.0); conf = np.nan_to_num(conf, nan=0.5)
            votes += raw; weighted += raw * conf
        return votes, weighted

    def _simulate(self, df, ml_probs):
        """FIX #17/#19/#20: faithful simulation that mirrors the live engine —
        ATR-based take-profit / stop-loss / trailing stop, taker fees on entry
        and exit, risk-based position sizing, a DIRECTIONAL ML confidence gate
        using the configured thresholds, and a forced close of any position
        still open at the end of the data. Same result shape as before."""
        from app.config2 import DEFAULT_TAKER_FEE
        cfg       = self.config
        risk_pct  = float(cfg.get('risk_percentage', cfg.get('riskPercentage', 1.0)) or 1.0)
        thr_long  = float(cfg.get('mlThresholdLong', 0.55) or 0.55)
        thr_short = float(cfg.get('mlThresholdShort', 0.55) or 0.55)
        tp_mult   = float(self.params.get('atr_tp_mult', cfg.get('atrTpMultiplier', 3.0)) or 3.0)
        sl_mult   = float(self.params.get('atr_sl_mult', cfg.get('atrSlMultiplier', 1.5)) or 1.5)
        fee_rate  = float(DEFAULT_TAKER_FEE)
        direction = (self.params.get('trade_direction') or cfg.get('trade_direction') or 'BOTH')
        ml_active = ml_probs is not None and self.model_name not in (None, '', 'off')
        # FIX #2: realistic execution costs. Slippage = adverse fill on entry AND
        # exit (stops fill worse than the trigger). Funding = per-bar borrow carry
        # on shorts and any margin position.
        slip       = float(cfg.get('slippageBps', 5.0)) / 10000.0
        is_margin  = bool(cfg.get('enable_shorting') or cfg.get('use_margin'))
        funding_pb = float(cfg.get('fundingRatePerBar', 0.00005))  # ~0.005%/bar

        closes = df['close'].values.astype(float)
        highs  = df['high'].values.astype(float)
        lows   = df['low'].values.astype(float)
        idx    = df.index
        sma50  = df['sma_50'].values.astype(float) if 'sma_50' in df.columns else closes
        rsi    = df['rsi'].values.astype(float) if 'rsi' in df.columns else np.full(len(df), 50.0)
        atr_a  = df['atr'].values.astype(float) if 'atr' in df.columns else (closes * 0.01)

        # Test the SAME strategy signal the live bot trades (vectorized
        # StrategyBrain), not a fixed sma50/rsi rule.
        strategies = cfg.get('strategies', []) or []
        n_strats = max(1, len(strategies))
        rule = (cfg.get('comboConfig', {}) or {}).get('combinationRule') or self.params.get('hybridMode') or cfg.get('hybridMode') or 'OR'
        min_weighted = float(cfg.get('minWeightedSignal', 0.3) or 0.3)
        cooldown_bars = int(cfg.get('minBarsBetweenTrades', 0) or 0)
        votes_arr, weighted_arr = self._strategy_votes(df, strategies)

        balance = self.initial_balance
        position = None
        trades, equity_curve = [], []
        entries = wins = 0
        signal_has_reset = True      # anti-churn: require sig to flatten before re-entry
        last_exit_bar = -10**9

        for i in range(len(df)):
            price = closes[i]

            # ---- manage an open position: TP / SL / trailing stop ----
            if position is not None:
                if position['type'] == 'long':
                    position['tsl'] = max(position['tsl'], price - position['trail'])
                    hit_tp = highs[i] >= position['tp']
                    hit_sl = lows[i]  <= position['tsl']
                else:
                    position['tsl'] = min(position['tsl'], price + position['trail'])
                    hit_tp = lows[i]  <= position['tp']
                    hit_sl = highs[i] >= position['tsl']
                if hit_tp or hit_sl:
                    # Conservative: if both are touched in one bar, assume the stop
                    # hit first (intrabar order is unknown).
                    use_sl     = hit_sl
                    exit_price = position['tsl'] if use_sl else position['tp']
                    # FIX #2: slippage pushes the fill against us on exit.
                    exit_fill  = exit_price * (1 - slip) if position['type'] == 'long' else exit_price * (1 + slip)
                    if position['type'] == 'long':
                        gross = (exit_fill - position['entry']) * position['size']
                    else:
                        gross = (position['entry'] - exit_fill) * position['size']
                    fee  = abs(position['size'] * exit_fill) * fee_rate
                    fund = (abs(position['size'] * position['entry']) * funding_pb * max(0, i - position.get('entry_bar', i))
                            if (position['type'] == 'short' or is_margin) else 0.0)
                    net = gross - fee - fund
                    balance += net
                    wins += 1 if net > 0 else 0
                    trades.append({'type': 'close_' + position['type'], 'price': round(exit_fill, 2),
                                   'time': str(idx[i]), 'balance': round(balance, 2),
                                   'pnl': round(net, 2), 'reason': 'SL' if use_sl else 'TP'})
                    equity_curve.append({"time": str(idx[i]), "balance": round(balance, 2)})
                    position = None
                    last_exit_bar = i

            # ---- look for an entry only when flat ----
            if position is None:
                atr = atr_a[i]
                v = votes_arr[i]; wv = weighted_arr[i]
                sig = 0
                if rule == 'AND':
                    if v >= n_strats and wv > 0:    sig = 1
                    elif v <= -n_strats and wv < 0: sig = -1
                else:
                    if v > 0 and wv >= min_weighted:    sig = 1
                    elif v < 0 and wv <= -min_weighted: sig = -1

                # FIX #19: directional ML confidence gate using configured thresholds.
                if sig != 0 and ml_active:
                    p = float(ml_probs[i])
                    gate = (p >= thr_long) if sig == 1 else ((1.0 - p) >= thr_short)
                    if not gate:
                        sig = 0

                if sig == 1 and direction == 'SHORT': sig = 0
                if sig == -1 and direction == 'LONG': sig = 0

                # Anti-churn (mirrors live market_gate / signal_has_reset): after a
                # close, require the signal to flatten once before re-entering the
                # same continuous signal; also honor an optional bar cooldown.
                if sig == 0:
                    signal_has_reset = True
                can_enter = signal_has_reset and (i - last_exit_bar) >= cooldown_bars

                if sig != 0 and can_enter and atr > 0 and np.isfinite(atr):
                    stop_dist = atr * sl_mult
                    size      = (balance * (risk_pct / 100.0)) / stop_dist if stop_dist > 0 else 0.0
                    notional  = size * price
                    if notional > balance:      # spot: no leverage in the backtest
                        size     = balance / price
                        notional = size * price
                    if size > 0:
                        # FIX #2: slippage pushes the entry fill against us too.
                        entry_fill = price * (1 + slip) if sig == 1 else price * (1 - slip)
                        balance -= notional * fee_rate     # entry fee
                        if sig == 1:
                            tp = entry_fill + atr * tp_mult; sl = entry_fill - atr * sl_mult
                        else:
                            tp = entry_fill - atr * tp_mult; sl = entry_fill + atr * sl_mult
                        position = {'type': 'long' if sig == 1 else 'short', 'entry': entry_fill,
                                    'size': size, 'tp': tp, 'sl': sl, 'tsl': sl,
                                    'trail': atr * sl_mult, 'entry_bar': i}
                        entries += 1
                        signal_has_reset = False
                        trades.append({'type': 'buy' if sig == 1 else 'sell', 'price': round(entry_fill, 2),
                                       'time': str(idx[i]), 'balance': round(balance, 2)})

        # FIX #20: close any position still open at the last bar
        if position is not None:
            price = closes[-1]
            exit_fill = price * (1 - slip) if position['type'] == 'long' else price * (1 + slip)
            gross = ((exit_fill - position['entry']) if position['type'] == 'long'
                     else (position['entry'] - exit_fill)) * position['size']
            fund = (abs(position['size'] * position['entry']) * funding_pb * max(0, (len(df) - 1) - position.get('entry_bar', len(df) - 1))
                    if (position['type'] == 'short' or is_margin) else 0.0)
            net = gross - abs(position['size'] * exit_fill) * fee_rate - fund
            balance += net
            wins += 1 if net > 0 else 0
            trades.append({'type': 'close_' + position['type'], 'price': round(exit_fill, 2),
                           'time': str(idx[-1]), 'balance': round(balance, 2),
                           'pnl': round(net, 2), 'reason': 'EOD'})
            equity_curve.append({"time": str(idx[-1]), "balance": round(balance, 2)})
            position = None

        chart_df = df.reset_index().rename(columns={'index': 'time', 'datetime': 'time', 'timestamp': 'time'})
        chart_df['time'] = (pd.to_datetime(chart_df['time']).astype('int64') // 10**9)
        candle_data = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records')

        roi = ((balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = round((wins / entries) * 100, 1) if entries else 0.0
        return {
            "status": "success",
            "metrics": {
                "final_balance": round(balance, 2),
                "roi": round(roi, 2),
                "total_trades": entries,        # FIX #20: real entry count, not len(trades)//2
                "win_rate": win_rate,
            },
            "candleData": candle_data,
            "trades": trades,
            "equityCurve": equity_curve,
            "initialBalance": self.initial_balance,
        }

    async def run(self):
        try:
            df = await self.load_data()
            # FIX #10: indicators + batch ML inference are heavy synchronous CPU.
            # Off-load them to a worker thread so a long backtest cannot freeze the
            # shared event loop (which would otherwise stop live bots' TP/SL
            # reactions and every endpoint until the backtest finished).
            df = await asyncio.to_thread(self._prepare_frame, df)
            
            if len(df) < 10:
                return {"status": "failed", "error": "Not enough data", "metrics": {"roi": -100}}

            # Batch ML inference up front. _ml_probs returns a 1-D array of
            # length len(df) (or None). Replaces the old double-wrapped adapter
            # path, which returned a 0-d scalar for the meta model and crashed
            # at ml_probs[i] ("0-dimensional array indexed").
            ml_probs = None
            if self.model_name and self.model_name != "off":
                ml_probs = await asyncio.to_thread(self._ml_probs, df)  # FIX #10: off event loop
            if ml_probs is not None:
                ml_probs = np.asarray(ml_probs).ravel()
                if len(ml_probs) != len(df):
                    logger.warning(f"ml_probs len {len(ml_probs)} != {len(df)}; disabling ML gate")
                    ml_probs = None

            # FIX #17/#19/#20: run the faithful simulation (ATR TP/SL/trailing
            # stop, fees, risk-based sizing, directional ML gate, final-position
            # close) off the event loop instead of the old opposite-signal,
            # no-fee, 100%-compounding loop.
            return await asyncio.to_thread(self._simulate, df, ml_probs)
        except Exception as e:
            logger.error(f"Backtest Error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e), "metrics": {"roi": -100}}
