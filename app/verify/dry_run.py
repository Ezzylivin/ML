import json
import os
import time

# --- SIMULATION SETTINGS ---
INITIAL_BALANCE = 1000.0  
RISK_PER_TRADE = 0.01  # Risk 1% of balance ($10)
LOG_FILE = "app/discovery/trade_log.json"

def dispatch_trade(symbol, bias, confidence):
    dna_path = f"app/discovery/dna_{symbol.lower().replace('-', '_')}.json"
    if not os.path.exists(dna_path):
        print(f"❌ Error: No DNA found for {symbol}")
        return

    with open(dna_path, 'r') as f:
        dna = json.load(f)

    # 1. THE CONFIDENCE GATE
    threshold = dna['short_threshold'] if bias == "SHORT" else dna['long_threshold']
    print(f"\n📡 DISPATCHER CHECK: {symbol} {bias}")
    print(f"Council Score: {confidence:.2f} | Strategy Target: {threshold:.2f}")

    if confidence < threshold:
        print("💤 STATUS: Confidence below target. Trade Vetoed.")
        return

    # 2. THE POSITION SIZER (Risk Management)
    # Formula: Risk Amount / Stop Loss Distance
    risk_usd = INITIAL_BALANCE * RISK_PER_TRADE
    sl_percent = dna['stop_loss']
    position_size_usd = risk_usd / sl_percent

    print(f"✅ GATE PASSED: Executing Dry-Run")
    print("-" * 50)
    print(f"💰 Account Balance  : ${INITIAL_BALANCE:.2f}")
    print(f"🛡️  Risk Amount       : ${risk_usd:.2f}")
    print(f"📊 Position Size    : ${position_size_usd:.2f}")
    print(f"🎯 Take Profit      : {dna['take_profit']:.2%}")
    print(f"🛑 Stop Loss        : {sl_percent:.2%}")

    # 3. LOG THE TRADE
    trade = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "symbol": symbol,
        "bias": bias,
        "entry_score": confidence,
        "size_usd": round(position_size_usd, 2),
        "status": "OPEN"
    }

    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f: log = json.load(f)
    
    log.append(trade)
    with open(LOG_FILE, 'w') as f: json.dump(log, f, indent=4)
    print(f"💾 Trade Logged: {LOG_FILE}")

if __name__ == "__main__":
    # Test with current SOL Alpha (Bias: SHORT, Confidence: 0.71)
    dispatch_trade("SOL-USD", "SHORT", 0.71)
