import asyncio
import traceback
import logging
from app.backtest2 import execute_backtest

# Set logging to see internal engine messages
logging.basicConfig(level=logging.INFO)

async def debug_payload():
    # This mirrors EXACTLY what your integrated Backtests.jsx sends
    mock_data = {
        "symbol": "BTC-USD",
        "timeframe": "1h",
        "startDate": "2025-01-01",
        "endDate": "2026-01-01",
        "initialBalance": 1000,
        "risk_percentage": 1.0,
        "params": {
            "model_type": "stacking",
            "long_threshold": 0.65,
            "short_threshold": 0.35,
            "minAdxLevel": 25,
            "lookback": 50,
            "tslAtrMult": 3.0,
            "trendFilterPeriod": 200,
            "commission": 0.0006,
            "slippage": 0.0001
        }
    }
    
    print("\n🚀 SIMULATING MASTER BACKTEST PAYLOAD...")
    try:
        # Step 1: Map frontend name to backend name (Bridge Check)
        mock_data['initial_capital'] = mock_data.pop('initialBalance')
        
        # Step 2: Execute Engine
        res = await execute_backtest(**mock_data)
        
        print("\n✅ CONTRACT VALIDATED")
        print(f"Status: {res.get('status')}")
        print(f"Keys Returned: {list(res.keys())}")
        
        if "metrics" in res:
            m = res['metrics']
            print(f"--- RESULTS ---")
            print(f"Final ROI: {m.get('roi', 0):.2f}%")
            print(f"Win Rate:  {m.get('winRate', 0):.2f}%")
            print(f"Trades:    {m.get('totalTrades', 0)}")
            print(f"Drawdown:  {m.get('maxDrawdown', 0):.2f}%")
            
    except Exception as e:
        print("\n❌ ENGINE CRASHED")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(debug_payload())
