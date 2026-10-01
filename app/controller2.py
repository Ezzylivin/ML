import asyncio
import logging
import traceback
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 🟢 INTERNAL PATH PROTECTION
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.backtest2 import execute_backtest
from app.predictors.model_factory import ModelFactory

logger = logging.getLogger("SystemController")

class SystemController:
    def __init__(self):
        self.model_factory = ModelFactory()

    async def run_backtest_logic(self, **data):
        """
        Executes the backtest logic.
        Now performs a 'Safety Merge' to ensure top-level strategy config
        is correctly passed down to the engine's 'params' dictionary.
        """
        symbol = data.get('symbol', 'BTC-USD')
        
        # 🟢 1. PARAMETER SAFETY MERGE
        # Ensure 'params' exists
        if 'params' not in data or data['params'] is None:
            data['params'] = {}
            
        params = data['params']
        
        # List of "Hybrid Strategy" fields that might exist at the top level (from UI)
        # We force them into the 'params' dict so execute_backtest can find them.
        hybrid_fields = [
            'trend_strategy', 
            'range_strategy', 
            'ml_confidence_threshold', 
            'trade_direction',
            'regime_threshold'
        ]
        
        for field in hybrid_fields:
            if field in data:
                # If the UI sent it at the top level, inject it into params
                if field not in params:
                    params[field] = data[field]
                    logger.info(f"⚙️ Controller: Injected {field}={data[field]} into params")

        model_type = params.get("model_type", "XGBoost")
        combination_rule = data.get('combinationRule', 'OR')
        
        # 🟢 UI SYNC: Immediately reset heartbeat
        try:
            with open("engine_heartbeat.txt", "w") as f:
                f.write(f"{symbol} | 0.1% | Initializing Sovereign Engine...")
        except:
            pass

        # 2. Prepare Config (Legacy Consensus Logic)
        model_config = {
            "model_type": model_type,
            "lookback": int(params.get("lookback", 50)),
            "thresholds": {
                "long": float(params.get("long_threshold", 0.65)),
                "short": float(params.get("short_threshold", 0.35)),
                "exit": float(params.get("exit_threshold", 0.50))
            },
            "consensus_rule": combination_rule,
            "consensus_threshold": float(params.get("consensus_pct", 0.5)) 
        }
        data['model_config'] = model_config

        try:
            # 3. Load the bridged model
            # We load it here to ensure it's ready for the Engine
            if model_type != "None":
                try:
                    ml_model = self.model_factory.load_model(model_type, symbol=symbol)
                    data['ml_model'] = ml_model
                except Exception as e:
                    logger.warning(f"⚠️ Could not load ML model {model_type}: {e}")

            # 4. 🗳️ VOTER LOGIC PRE-PROCESSING
            if combination_rule == "CONSENSUS":
                logger.info(f"⚖️ Applying Consensus Filter: {model_config['consensus_threshold']*100}% majority required.")

            # 5. Execute in Thread (CPU intensive task)
            # We pass **data, which now contains the updated 'params'
            if asyncio.iscoroutinefunction(execute_backtest):
                raw_result = await execute_backtest(**data)
            else:
                raw_result = await asyncio.to_thread(execute_backtest, **data)

            # 6. Truncate for UI Performance
            MAX_TRADES = 5000
            if "trades" in raw_result and len(raw_result["trades"]) > MAX_TRADES:
                raw_result["trades"] = raw_result["trades"][-MAX_TRADES:]

            return self._clean_for_json(raw_result)

        except Exception as e:
            logger.error(f"❌ Controller Crash: {traceback.format_exc()}")
            with open("engine_heartbeat.txt", "w") as f:
                f.write(f"{symbol} | 0% | CRASHED: {str(e)}")
            return {"status": "error", "msg": str(e)}

    def _clean_for_json(self, data):
        if data is None: return None
        if isinstance(data, (bool, str)): return data
        if isinstance(data, (float, np.floating)):
            return 0.0 if (np.isnan(data) or np.isinf(data)) else float(data)
        if isinstance(data, (int, np.integer)): return int(data)
        if isinstance(data, (list, tuple)): return [self._clean_for_json(i) for i in data]
        if isinstance(data, dict): return {k: self._clean_for_json(v) for k, v in data.items()}
        if isinstance(data, (datetime, pd.Timestamp)): return data.isoformat()
        return str(data)

system_controller = SystemController()
