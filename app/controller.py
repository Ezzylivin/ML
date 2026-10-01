import asyncio
import logging
import time
import traceback
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- Imports ---
try:
    from .backtest import execute_backtest
    from .ml import get_available_models_logic
    from .config import bots_collection, MODEL_DIR, OPTIMIZER_DIR
    from .manager import PrecisionPyramidManager
except ImportError:
    from backtest import execute_backtest
    from ml import get_available_models_logic
    from config import bots_collection, MODEL_DIR, OPTIMIZER_DIR
    from manager import PrecisionPyramidManager

logger = logging.getLogger("SystemController")
logger.setLevel(logging.INFO)

class SystemController:
    def __init__(self):
        self.active_bots = {}

    def _clean_for_json(self, data):
        """
        Recursively converts all data into strictly valid JSON types.
        """
        if data is None: return None
        if isinstance(data, (bool, str)): return data
        
        # Handle Numbers
        if isinstance(data, (float, np.floating)):
            return float(data) if not (np.isnan(data) or np.isinf(data)) else 0
        if isinstance(data, (int, np.integer)):
            return int(data)

        # Handle Lists (Optimized)
        if isinstance(data, (list, tuple)):
            return [self._clean_for_json(i) for i in data]

        # Handle Dictionaries
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
            
        # Handle Complex Objects
        if isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        
        if hasattr(data, 'to_dict'):
            return self._clean_for_json(data.to_dict())

        return str(data)

    async def run_backtest_logic(self, **kwargs):
        """
        Main entry point for Backtests. Ensures candleData is preserved.
        """
        logger.info(f"📉 Starting Backtest for: {kwargs.get('symbol', 'BTC-USD')}")
        
        # Normalize params
        initial_balance = kwargs.pop('initialBalance', kwargs.get('initial_balance', 1000.0))
        kwargs['initial_balance'] = float(initial_balance)
        
        if 'riskPercentage' in kwargs: kwargs['risk_percentage'] = kwargs.pop('riskPercentage')
        if 'riskManagementMode' in kwargs: kwargs['risk_mode'] = kwargs.pop('riskManagementMode')

        try:
            # Execute engine
            if asyncio.iscoroutinefunction(execute_backtest):
                raw_result = await execute_backtest(**kwargs)
            else:
                raw_result = await asyncio.to_thread(execute_backtest, **kwargs)
            
            # 🟢 DEBUG: Verify keys found in raw results
            logger.info(f"✅ Result keys: {list(raw_result.keys())}")
            logger.info(f"✅ Found {len(raw_result.get('candleData', []))} bars.")

            # Sanitize and Return
            return self._clean_for_json(raw_result)

        except Exception as e:
            logger.error(f"❌ Backtest Logic Crashed: {traceback.format_exc()}")
            return {"metrics": {"roi": 0}, "candleData": [], "trades": [], "equityCurve": []}

    async def get_optimizer_winners(self):
        """Loads top alpha strategies from the results directory."""
        winners = []
        target_dir = os.path.abspath(OPTIMIZER_DIR)
        
        if not os.path.exists(target_dir): return []

        try:
            files = [f for f in os.listdir(target_dir) if f.endswith(".json")]
            for filename in files:
                with open(os.path.join(target_dir, filename), 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    roi = metrics.get('roi', metrics.get('totalReturn', 0))
                    winners.append({
                        "botId": filename.replace('.json', ''),
                        "symbol": data.get('symbol', 'BTC-USD'),
                        "roi": float(roi),
                        "config": data.get("config", {}) or data
                    })
            winners.sort(key=lambda x: x['roi'], reverse=True)
            return winners[:20]
        except Exception as e:
            logger.error(f"Winner fetch failed: {e}")
            return []

system_controller = SystemController()
