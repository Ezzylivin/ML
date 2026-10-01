import json
import os

def dispatch_sol():
    dna_path = "app/discovery/dna_sol_usd.json"
    
    if not os.path.exists(dna_path):
        print("❌ Error: No DNA found. Run evolution first.")
        return

    with open(dna_path, 'r') as f:
        dna = json.load(f)

    print("\n🎯 SOL-USD SHORT DISPATCH READY")
    print("=" * 45)
    print(f"Confidence Target : {dna['short_threshold']:.2%}")
    print(f"Risk Management   : SL {dna['stop_loss']:.2%} | TP {dna['take_profit']:.2%}")
    print(f"Technical Filter  : RSI({dna['rsi_period']})")
    print("-" * 45)
    print("🚀 DISPATCH STATUS: Awaiting Final API Confirmation...")

if __name__ == "__main__":
    dispatch_sol()
