import asyncio
from app.backtest2 import execute_backtest

async def test():
    print("🧪 Testing Engine Manually...")
    # Small window to ensure speed
    res = await execute_backtest(
        symbol='BTC-USD',
        startDate='2025-08-01',
        endDate='2025-08-07',
        model_type='stacking',
        initialBalance=1000,
        params={
            "long_threshold": 0.55,  # 🟢 LOWER these for the test 
            "short_threshold": 0.45, # 🟢 to force the AI to be "braver"
            "minAdxLevel": 10        # 🟢 Lower ADX so it doesn't veto trades
        }
   )
    print(f"🏁 Result: {res.get('status')} | ROI: {res.get('metrics', {}).get('roi')}")
    print(f"📊 Trades: {len(res.get('trades', []))}")
if __name__ == "__main__":
    asyncio.run(test())
