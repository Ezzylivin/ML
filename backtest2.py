import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import logging
import tensorflow as tf  
from fastapi import HTTPException
from app.predictors.model_factory import ModelFactory

# ============================================================
# 🔧 FIX #1: Import centralized constants from config2
# ============================================================
from app.config2 import DEFAULT_TAKER_FEE, FEATURE_COLUMNS

# ============================================================
# 🔧 FIX #2: Import RawModelAdapter from its own module
# ============================================================
# OLD: RawModelAdapter was defined here in backtest2.py, causing a
#      circular import: ModelFactory → backtest2 → ModelFactory
# NEW: Lives in app/predictors/model_adapter.py. Both files import it cleanly.
from app.predictors.model_adapter import RawModelAdapter

logger = logging.getLogger("BacktestEngine")


class Backtester:
    def __init__(self, config: dict):
        self.config = config 
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        self.start_str = config.get('startDate')
        self.end_str = config.get('endDate')
        self.params = config.get('params', {})
        self.initial_balance = float(config.get('initialBalance', 1000))
        self.combination_rule = config.get('combinationRule', 'OR').upper()

        p = self.params
        self.tp_pct = float(p.get('take_profit', 0.13))
        self.ts_pct = float(p.get('trailing_stop', 0.086))
        
        self.strategies = config.get('strategies', [])
        raw_risk = float(config.get('riskPercentage', config.get('risk_percentage', 100.0)))
        self.risk_mult = float(raw_risk) / 100.0
        
        self.ml_limit_long = float(config.get('mlThresholdLong', 0.80))
        self.ml_limit_short = float(config.get('mlThresholdShort', 0.80))
        self.model_name = config.get('mlModel', 'stacking')

        # ============================================================
        # 🔧 FIX #3: Centralized fee rate
        # ============================================================
        # OLD: Hardcoded 0.0006 (0.06%) inline — 10x less than live engine
        # NEW: Uses the same constant as main4.py
        self.fee_rate = DEFAULT_TAKER_FEE

    def calculate_indicators(self, df):
        if len(df) < 200: return df 
        
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None: df['adx'] = adx_df.iloc[:, 0]
            
        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None:
            df['BBL_20_2.0_2.0'] = bb.iloc[:, 0]
            df['BBU_20_2.0_2.0'] = bb.iloc[:, 2]

        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['st_trend'] = st.iloc[:, 1] if st is not None else 0

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None: df['STOCHk_14_3_3'] = stoch.iloc[:, 0]
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD_12_26_9'] = macd.iloc[:, 0]
            df['MACDs_12_26_9'] = macd.iloc[:, 2]

        df['pa_high'] = df['high'].rolling(window=20).max()
        df['pa_low'] = df['low'].rolling(window=20).min()
        df['vol_ma'] = ta.sma(df['volume'], length=20)

        df['adx_logic'] = np.where(df['adx'] > 25, 1, 0)
        df['atr_logic'] = (df['atr'] / df['close']) * 1000
        df['sma_logic'] = np.where(df['close'] > df['sma_200'], 1, -1)

        return df.dropna()

    def get_signal(self, row, strategies):
        votes = 0
        for strat in strategies:
            code = strat.get('code')
            p = strat.get('params', {})
            
            if code == "stoch":
                k_val = row.get('STOCHk_14_3_3', 50)
                if k_val < 20: votes += 1
                elif k_val > 80: votes -= 1
            elif code in ["bb_fade", "bollinger_bands"]:
                lower = row.get('BBL_20_2.0_2.0', 0)
                upper = row.get('BBU_20_2.0_2.0', 999999)
                if row['close'] < lower: votes += 1
                elif row['close'] > upper: votes -= 1
            elif code == "rsi_threshold":
                rsi = row.get('rsi', 50)
                if rsi < p.get('oversold', 30): votes += 1
                elif rsi > p.get('overbought', 70): votes -= 1
            elif code == "sma_crossover":
                if row.get('sma_50', 0) > row.get('sma_200', 0): votes += 1
                else: votes -= 1
            elif code == "supertrend":
                if row.get('st_trend', 0) == 1: votes += 1
                elif row.get('st_trend', 0) == -1: votes -= 1
            elif code == "macd_crossover":
                if row.get('MACD_12_26_9', 0) > row.get('MACDs_12_26_9', 0): votes += 1
                else: votes -= 1
            elif code == "atr_breakout":
                upper = row.get('ema_20', 0) + (row.get('atr', 0) * p.get('multiplier', 1.5))
                lower = row.get('ema_20', 0) - (row.get('atr', 0) * p.get('multiplier', 1.5))
                if row['close'] > upper: votes += 1
                elif row['close'] < lower: votes -= 1
            elif code == "ema_cloud":
                if row.get('ema_9', 0) > row.get('ema_21', 0): votes += 1
                else: votes -= 1
            elif code == "pa_breakout":
                if row['close'] >= row.get('pa_high', 999999): votes += 1
                elif row['close'] <= row.get('pa_low', 0): votes -= 1
            elif code == "vol_profile":
                if row.get('volume', 0) > (row.get('vol_ma', 0) * p.get('threshold', 1.5)):
                    votes += (1 if row['close'] > row.get('sma_50', 0) else -1)

        if self.combination_rule == "AND":
            if votes >= len(strategies): return 1
            if votes <= -len(strategies): return -1
            return 0
        else: 
            if votes > 0: return 1
            if votes < 0: return -1
            return 0

    async def load_data(self):
        from datetime import timedelta
        fetch_start = (pd.to_datetime(self.start_str) - timedelta(days=15)).strftime('%Y-%m-%d')
        try:
            from main4 import ensure_full_data 
            df = await ensure_full_data(self.symbol, self.timeframe, fetch_start, self.end_str)
            if df is None or df.empty:
                raise HTTPException(status_code=400, detail="Data gap could not be filled.")
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Data Sync Error: {e}")
            raise HTTPException(status_code=500, detail=f"Data Sync Failed: {str(e)}")

    async def run(self):
        try:
            # 1. DATA SETUP
            df_raw = await self.load_data()
            df_full = self.calculate_indicators(df_raw)
            user_start = pd.to_datetime(self.start_str).replace(tzinfo=None)
            
            df_slice = df_full[df_full.index >= (user_start - pd.Timedelta(hours=60))].copy()
            if df_slice.empty: return {"status": "failed", "error": "No data found."}
            
            # 2. 🚀 TURBO BATCH PRE-CALCULATION
            if self.model_name and self.model_name != "off":
                import time
                batch_start_time = time.time()
                
                from app.predictors.stacking_predictor import StackingPredictor
                ml_engine = StackingPredictor(symbol=self.symbol, timeframe=self.timeframe)
                
                feature_cols = ml_engine.experts['transformer'].feature_names
                numpy_matrix = df_slice[feature_cols].values.astype('float32')
                
                transformer_input_batch = []
                for i in range(50, len(df_slice)):
                    transformer_input_batch.append(numpy_matrix[i-49 : i+1])
                
                logger.info(f"📡 Sending {len(transformer_input_batch)} samples to Transformer in ONE batch...")
                
                transformer_model = ml_engine.experts['transformer'].model
                
                # 🔧 FIX: Apply normalization to batch (matches training)
                # The turbo batch bypasses RawModelAdapter, so we must
                # normalize manually using the same stats saved during training.
                transformer_adapter = ml_engine.experts['transformer']
                batch_array = np.array(transformer_input_batch)
                if transformer_adapter.norm_mean is not None and transformer_adapter.norm_std is not None:
                    batch_array = (batch_array - transformer_adapter.norm_mean) / transformer_adapter.norm_std
                    logger.info("📐 Applied normalization to Transformer batch")
                
                full_batch_tensor = tf.convert_to_tensor(batch_array.astype('float32'), dtype=tf.float32)
                
                all_transformer_preds = transformer_model(full_batch_tensor, training=False).numpy()
                
                transformer_scores = [float(p[1] if p.shape[0] > 1 else p[0]) for p in all_transformer_preds]
                logger.info(f"✅ Transformer Batch Complete in {time.time() - batch_start_time:.2f}s")

                ai_scores = []
                total = len(transformer_scores)
                for i in range(total):
                    if i % 2000 == 0:
                        logger.info(f"📊 Final Stacking: {i}/{total}")

                    score = ml_engine.predict_direction(
                        transformer_input_batch[i], 
                        precalc_transformer=transformer_scores[i]
                    )
                    ai_scores.append(score)
                
                logger.info(f"✅ AI CRUNCH DONE in {time.time() - batch_start_time:.2f}s")
                
                df_final = df_slice.iloc[50:].copy()
                df_final['ai_conf'] = ai_scores
            else:
                df_final = df_slice[df_slice.index >= user_start].copy()
                df_final['ai_conf'] = 1.0

            # 3. HIGH-SPEED TRADING LOOP
            balance, position, entry_price = self.initial_balance, None, 0
            tp_price, tsl_price = 0, 0
            equity_curve, trades, vetoed_logs = [], [], []

            peak_balance = float(self.initial_balance)
            max_drawdown = 0.0

            pending_vetoes = []
            veto_stats = {"saved": 0, "missed": 0}

            # ============================================================
            # 🔧 FIX #4: Capture position size at ENTRY, not at exit
            # ============================================================
            # OLD: position_size_fiat = balance * self.risk_mult (at exit time)
            # This meant size changed if balance changed between entry/exit.
            # NEW: Store it when we enter, use the stored value at exit.
            entry_position_size_fiat = 0

            rows = df_final.to_dict('records')
            timestamps = (df_final.index.astype('int64') // 10**9).tolist()
            iso_timestamps = df_final.index.strftime('%Y-%m-%dT%H:%M:%S.000Z').tolist()

            for i in range(len(rows)):
                row = rows[i]
                current_time_unix = timestamps[i]
                current_time_iso = iso_timestamps[i]
                
                signal = self.get_signal(row, self.strategies)
                conf_score = row['ai_conf']
                limit = self.ml_limit_short if row['close'] < row.get('sma_200', 0) else self.ml_limit_long
                gate_passed = conf_score >= limit

                surviving_vetoes = []
                for v in pending_vetoes:
                    outcome = None
                    if v["type"] == "long":
                        if row['low'] <= v["sl"]: outcome = "saved"
                        elif row['high'] >= v["tp"]: outcome = "missed"
                    else:
                        if row['high'] >= v["sl"]: outcome = "saved"
                        elif row['low'] <= v["tp"]: outcome = "missed"
                    
                    if outcome:
                        veto_stats[outcome] += 1
                    else:
                        surviving_vetoes.append(v)
                pending_vetoes = surviving_vetoes

                if signal != 0 and not gate_passed and position is None:
                    trade_type = "long" if signal == 1 else "short"
                    
                    hypo_tp = row['close'] * (1 + self.tp_pct) if signal == 1 else row['close'] * (1 - self.tp_pct)
                    hypo_sl = row['close'] * (1 - self.ts_pct) if signal == 1 else row['close'] * (1 + self.ts_pct)
                    
                    pending_vetoes.append({
                        "type": trade_type,
                        "tp": hypo_tp,
                        "sl": hypo_sl
                    })

                    if i % 5 == 0:
                        vetoed_logs.append({
                            "time": current_time_iso, "signal": trade_type.capitalize(),
                            "conf_score": round(conf_score, 4), "limit": round(limit, 4), "price": row['close']
                        })

                # ============================================================
                # 🔧 FIX #5: Exit logic — record equity on close, no skip
                # ============================================================
                # OLD: `continue` after closing a position skipped the equity
                # curve append at the bottom. Trades that closed never got
                # their new balance recorded that candle → visual gaps.
                # NEW: Use a flag instead of continue.
                did_close_position = False

                if position:
                    exit_p, exit_r = None, None
                    
                    if position == 'long':
                        if row['low'] <= tsl_price: exit_p, exit_r = tsl_price, "Trailing Stop"
                        elif row['high'] >= tp_price: exit_p, exit_r = tp_price, "Take Profit"
                        elif signal == -1: exit_p, exit_r = row['close'], "Signal Flip"
                    else:
                        if row['high'] >= tsl_price: exit_p, exit_r = tsl_price, "Trailing Stop"
                        elif row['low'] <= tp_price: exit_p, exit_r = tp_price, "Take Profit"
                        elif signal == 1: exit_p, exit_r = row['close'], "Signal Flip"

                    if exit_p:
                        # 🔧 FIX #4b: Use the size captured at entry
                        pnl_v = (exit_p - entry_price) / entry_price if position == 'long' else (entry_price - exit_p) / entry_price
                        
                        gross_profit = entry_position_size_fiat * pnl_v
                        
                        # 🔧 FIX #3b: Use self.fee_rate from config2
                        # OLD: hardcoded 0.0006
                        entry_fee = entry_position_size_fiat * self.fee_rate
                        exit_fee = (entry_position_size_fiat + gross_profit) * self.fee_rate
                        net_profit = gross_profit - (entry_fee + exit_fee)
                        
                        balance += net_profit
                        
                        if balance > peak_balance:
                            peak_balance = balance
                        
                        current_dd = (peak_balance - balance) / peak_balance
                        if current_dd > max_drawdown:
                            max_drawdown = current_dd
                        
                        trades.append({
                            "type": "exit", "reason": exit_r, "price": round(exit_p, 2),
                            "time": current_time_iso, "pnl": round(pnl_v * 100, 2),
                            "balance": round(balance, 2)
                        })
                        position = None
                        entry_position_size_fiat = 0
                        did_close_position = True
                        # 🔧 FIX #5: Removed `continue` — fall through to equity append
                    
                    # Only update trailing stop if we didn't just close
                    if not did_close_position and position:
                        if position == 'long':
                            new_tsl = row['high'] * (1 - self.ts_pct)
                            if new_tsl > tsl_price: tsl_price = new_tsl
                        else:
                            new_tsl = row['low'] * (1 + self.ts_pct)
                            if new_tsl < tsl_price: tsl_price = new_tsl

                # Don't enter on the same bar we just exited (conservative, prevents look-ahead)
                if not did_close_position and position is None and gate_passed and signal != 0:
                    position = 'long' if signal == 1 else 'short'
                    entry_price = row['close']
                    tp_price = entry_price * (1 + self.tp_pct) if position == 'long' else entry_price * (1 - self.tp_pct)
                    tsl_price = entry_price * (1 - self.ts_pct) if position == 'long' else entry_price * (1 + self.ts_pct)
                    
                    # 🔧 FIX #4c: Capture position size at entry
                    entry_position_size_fiat = balance * self.risk_mult
                    
                    trades.append({
                        "type": "buy" if position == 'long' else "sell",
                        "price": entry_price, "time": current_time_iso,
                        "ai_score": round(conf_score, 4), "gate_limit": round(limit, 4)
                    })

                # 🔧 FIX #5b: This now always runs, even on bars where we closed
                if i % 4 == 0:
                    equity_curve.append({"time": current_time_unix, "balance": round(balance, 2)})

            # 🎯 SYNC-LOCK CHART DATA
            chart_candles = df_final.copy()
            chart_candles['time'] = timestamps
            chart_data_json = chart_candles.tail(1000)[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

            # 🚀 1. Calculate AI Shield Metrics
            total_resolved = veto_stats["saved"] + veto_stats["missed"]
            ai_accuracy = (veto_stats["saved"] / total_resolved * 100) if total_resolved > 0 else 0.0

            # 🚀 2. Calculate Win Rate & Profit Factor natively in Python
            wins = 0
            losses = 0
            gross_total_profit = 0.0
            gross_total_loss = 0.0
            trade_pnls = []

            for t in trades:
                if t.get("type") == "exit":
                    trade_pnl = float(t.get("pnl", 0))
                    if trade_pnl > 0:
                        wins += 1
                        gross_total_profit += trade_pnl
                    elif trade_pnl < 0:
                        losses += 1
                        gross_total_loss += abs(trade_pnl)

            total_closed = wins + losses
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
            profit_factor = (gross_total_profit / gross_total_loss) if gross_total_loss > 0 else (99.9 if gross_total_profit > 0 else 0.0)

            # System Quality Number (SQN)
            sqn = 0.0
            if len(trade_pnls) > 0:
                mean_pnl = np.mean(trade_pnls)
                std_pnl = np.std(trade_pnls)
                if std_pnl > 0:
                    sqn = (mean_pnl / std_pnl) * np.sqrt(len(trade_pnls))

            # 🚀 3. Calculate Sharpe Ratio
            returns = []
            for i in range(1, len(equity_curve)):
                prev = equity_curve[i-1]["balance"]
                curr = equity_curve[i]["balance"]
                if prev > 0:
                    returns.append((curr - prev) / prev)

            sharpe = 0.0
            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    # Annualizing based on roughly hourly steps
                    sharpe = (mean_return / std_return) * np.sqrt(365 * 24)
                    
            metrics = {
                "finalBalance": round(balance, 2),
                "roi": round(((balance - self.initial_balance) / self.initial_balance) * 100, 2),
                "totalTrades": len(trades),
                "netProfit": round(balance - self.initial_balance, 2),
                "maxDrawdown": round(max_drawdown * 100, 2),
                "max_drawdown": round(max_drawdown * 100, 2),
                "winRate": round(win_rate, 1),
                "wins": wins,
                "losses": losses,
                "profitFactor": round(profit_factor, 2),
                "aiShieldAccuracy": round(ai_accuracy, 1),
                "saved": veto_stats["saved"],
                "missed": veto_stats["missed"],
                "sharpeRatio": round(sharpe, 2),
                "sharpe_ratio": round(sharpe, 2),
                "sqn": round(sqn, 2)
            }

            return {
                "status": "success",
                "metrics": metrics,
                "candleData": chart_data_json,
                "trades": trades,
                "equityCurve": equity_curve,
                "vetoed_signals": vetoed_logs[-500:],
                "initialBalance": self.initial_balance
            }
        except Exception as e:
            logger.error(f"❌ Backtest Runtime Error: {e}")
            return {"status": "failed", "error": str(e)}

    # ============================================================
    # 🔧 FIX #6: Removed dead method run_with_precalculated_data
    # ============================================================
    # OLD: This method accepted df_final but ignored it completely,
    # just calling self.run() which re-fetches everything.
    # It was dead code that could mislead someone into thinking
    # pre-calculated data was being used. Removed entirely.
