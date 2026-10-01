import asyncio
import ccxt.async_support as ccxt

async def test_exchange(exchange_id, api_key, secret):
    print(f"\n🧪 Testing {exchange_id.upper()} Connection...")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
    })
    
    try:
        # 1. Test Read Permissions
        balance = await exchange.fetch_balance()
        print(f"✅ READ SUCCESS: Total USD Balance: ${balance.get('USD', {}).get('total', 0.0)}")
        
        # 2. Test Trading Permissions (Checking capabilities)
        if exchange.has['createOrder']:
            print("✅ TRADE SUCCESS: API Key has execution permissions.")
        else:
            print("❌ TRADE FAILED: API Key cannot create orders.")
            
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {str(e)}")
    finally:
        await exchange.close()

async def main():
    # Insert your actual keys here to test
    COINBASE_KEY = "your_coinbase_key_here"
    COINBASE_SECRET = "your_coinbase_secret_here"
    
    KRAKEN_KEY = "your_kraken_key_here"
    KRAKEN_SECRET = "your_kraken_secret_here"
    
    await test_exchange('coinbase', COINBASE_KEY, COINBASE_SECRET)
    await test_exchange('kraken', KRAKEN_KEY, KRAKEN_SECRET)

if __name__ == "__main__":
    asyncio.run(main())
